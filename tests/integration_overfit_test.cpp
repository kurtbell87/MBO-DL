// integration_overfit_test.cpp — TDD RED phase: end-to-end integration overfit tests
// Spec: .kit/docs/integration-overfit.md
//
// Validates the full pipeline on real MBO data:
//   .dbn.zst → book_builder → feature_encoder → trajectory_builder → subsample N=32
//     → overfit MLP (>=99%, <=500 epochs)
//     → overfit CNN (>=99%, <=500 epochs)
//     → overfit GBT (>=99%, <=1000 rounds)
//
// Test categories:
//   1. Data validation (file loads, snapshot count/quality, timestamps)
//   2. Feature encoding (shape, finiteness)
//   3. Trajectory (length, label validity, label distribution)
//   4. Overfit (MLP, CNN, GBT achieve >=99% on N=32; no NaN; determinism)

#include <gtest/gtest.h>

#include "book_builder.hpp"
#include "feature_encoder.hpp"
#include "oracle_labeler.hpp"
#include "trajectory_builder.hpp"
#include "mlp_model.hpp"
#include "cnn_model.hpp"
#include "gbt_model.hpp"
#include "training_loop.hpp"     // overfit_mlp, OverfitResult
#include "gbt_features.hpp"      // GBT_FEATURE_DIM, compute_gbt_features

#include <torch/torch.h>

// databento C++ API — reads .dbn.zst files
#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>
#include <databento/enums.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

// ===========================================================================
// Constants from the spec
// ===========================================================================
namespace {

// Data file
const std::string DATA_DIR  = "DATA/GLBX-20260207-L953CAPU5B";
const std::string DATA_FILE = DATA_DIR + "/glbx-mdp3-20220103.mbo.dbn.zst";

// Instrument
constexpr uint32_t INSTRUMENT_ID = 13615;  // MESM2

// Session window: RTH 09:30 - 10:00 ET on 2022-01-03
// 2022-01-03 00:00:00 ET = 2022-01-03 05:00:00 UTC = 1641186000 seconds since epoch
constexpr uint64_t NS_PER_SEC  = 1'000'000'000ULL;
constexpr uint64_t NS_PER_MIN  = 60ULL * NS_PER_SEC;
constexpr uint64_t NS_PER_HOUR = 3600ULL * NS_PER_SEC;

// Midnight ET 2022-01-03 in UTC nanoseconds
constexpr uint64_t MIDNIGHT_ET_NS = 1641186000ULL * NS_PER_SEC;
constexpr uint64_t RTH_OPEN_NS    = MIDNIGHT_ET_NS + 9ULL * NS_PER_HOUR + 30ULL * NS_PER_MIN;
constexpr uint64_t RTH_CLOSE_NS   = MIDNIGHT_ET_NS + 10ULL * NS_PER_HOUR;  // 10:00 ET

// Warm-up from 09:29:00 ET
constexpr uint64_t WARMUP_NS = MIDNIGHT_ET_NS + 9ULL * NS_PER_HOUR + 29ULL * NS_PER_MIN;

// Oracle parameters
constexpr int    HORIZON       = 100;   // 10 seconds
constexpr int    TARGET_TICKS  = 10;    // 2.50 points
constexpr int    STOP_TICKS    = 5;     // 1.25 points
constexpr int    TP_TICKS      = 20;    // 5.00 points
constexpr float  ORACLE_TICK   = 0.25f;

// Sampling
constexpr int N_OVERFIT = 32;

// Training (neural)
constexpr int   MAX_EPOCHS_NEURAL = 500;
constexpr float LR_NEURAL         = 1e-3f;
constexpr float TARGET_ACC        = 0.99f;

// Training (GBT)
constexpr int   MAX_ROUNDS_GBT = 1000;
constexpr float TARGET_ACC_GBT = 0.99f;

constexpr int NUM_CLASSES = 5;

}  // anonymous namespace

// ===========================================================================
// Shared test fixture — loads data once for all integration tests
// ===========================================================================
class IntegrationOverfitTest : public ::testing::Test {
protected:
    // Shared across all tests in this fixture. Populated once by SetUpTestSuite.
    static std::vector<BookSnapshot> snapshots_;
    static std::vector<TrainingSample> trajectory_;
    static std::vector<TrainingSample> subsampled_;
    static bool data_loaded_;

    static void SetUpTestSuite() {
        if (data_loaded_) return;

        // --- Check data file exists ---
        ASSERT_TRUE(std::filesystem::exists(DATA_FILE))
            << "Data file not found: " << DATA_FILE
            << ". Place the MBO data file at the expected path.";

        // --- Build order book from .dbn.zst ---
        BookBuilder builder(INSTRUMENT_ID);

        // Read all MBO events from the file and feed to builder.
        // We process from the start of the file to capture pre-market warm-up data.
        databento::DbnFileStore store{std::filesystem::path(DATA_FILE)};

        // Process all MBO messages in the file using the iterator API
        while (const auto* record = store.NextRecord()) {
            if (const auto* mbo = record->GetIf<databento::MboMsg>()) {
                // Extract ts_event as uint64_t nanoseconds
                uint64_t ts_ns = static_cast<uint64_t>(
                    mbo->hd.ts_event.time_since_epoch().count());
                // Extract flags as uint8_t for BookBuilder interface
                uint8_t flags_raw = static_cast<uint8_t>(
                    mbo->flags.Raw());
                builder.process_event(
                    ts_ns,
                    mbo->order_id,
                    mbo->hd.instrument_id,
                    static_cast<char>(mbo->action),
                    static_cast<char>(mbo->side),
                    mbo->price,
                    mbo->size,
                    flags_raw
                );
            }
        }

        // Emit snapshots for 09:30 - 10:00 ET window
        // Start from warmup (09:29) so builder has time to populate the book
        snapshots_ = builder.emit_snapshots(WARMUP_NS, RTH_CLOSE_NS);

        // Filter to only 09:30+ snapshots (the builder already does RTH filtering,
        // but our warmup start is 09:29 — builder clamps to RTH open internally)
        // So snapshots_ should already be 09:30+.

        data_loaded_ = !snapshots_.empty();

        if (!data_loaded_) {
            GTEST_SKIP() << "No snapshots emitted — cannot run integration tests.";
            return;
        }

        // --- Build trajectory ---
        trajectory_ = build_trajectory(snapshots_, HORIZON);

        // --- Subsample N=32 ---
        int traj_len = static_cast<int>(trajectory_.size());
        int k = traj_len / N_OVERFIT;
        ASSERT_GT(k, 0) << "Trajectory too short to subsample N=" << N_OVERFIT;

        // Primary sampling: every k-th sample starting at offset 0
        subsampled_.clear();
        subsampled_.reserve(N_OVERFIT);
        for (int i = 0; i < N_OVERFIT; ++i) {
            subsampled_.push_back(trajectory_[i * k]);
        }

        // Validate label distribution (>=3 of 4 classes, no class >80%, no class 4)
        std::map<int, int> label_counts;
        for (const auto& s : subsampled_) {
            label_counts[s.label]++;
        }

        int classes_present = static_cast<int>(label_counts.size());
        bool any_class_over_80 = false;
        for (auto& [cls, cnt] : label_counts) {
            if (static_cast<float>(cnt) / N_OVERFIT > 0.80f) {
                any_class_over_80 = true;
            }
        }

        // If distribution fails, retry with offset k/2
        if (classes_present < 3 || any_class_over_80) {
            int offset = k / 2;
            subsampled_.clear();
            for (int i = 0; i < N_OVERFIT; ++i) {
                int idx = offset + i * k;
                if (idx < traj_len) {
                    subsampled_.push_back(trajectory_[idx]);
                }
            }
        }
    }
};

// Static member definitions
std::vector<BookSnapshot> IntegrationOverfitTest::snapshots_;
std::vector<TrainingSample> IntegrationOverfitTest::trajectory_;
std::vector<TrainingSample> IntegrationOverfitTest::subsampled_;
bool IntegrationOverfitTest::data_loaded_ = false;

// ===========================================================================
// DATA VALIDATION TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 1: File loads — book_builder successfully reads the .dbn.zst file
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, FileLoads) {
    ASSERT_TRUE(data_loaded_)
        << "BookBuilder must successfully read the .dbn.zst file and emit snapshots.";
    EXPECT_FALSE(snapshots_.empty())
        << "emit_snapshots() should return a non-empty vector of BookSnapshots.";
}

// ---------------------------------------------------------------------------
// Test 2: Snapshot count — at least 17,000 snapshots in 30-minute RTH window
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, SnapshotCount) {
    ASSERT_TRUE(data_loaded_);

    // 30 minutes × 60 seconds × 10 snapshots/sec = 18,000 expected
    // Allow margin: at least 17,000
    EXPECT_GE(static_cast<int>(snapshots_.size()), 17000)
        << "Expected ~18,000 snapshots in 30-minute RTH window, got "
        << snapshots_.size();

    // Upper bound sanity check (should not exceed 18,001)
    EXPECT_LE(static_cast<int>(snapshots_.size()), 18001)
        << "Snapshot count exceeds theoretical maximum for 30-minute window";
}

// ---------------------------------------------------------------------------
// Test 3: Snapshot quality — all snapshots have mid_price > 0, spread >= 0,
//         no crossed book (best_bid < best_ask)
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, SnapshotQuality) {
    ASSERT_TRUE(data_loaded_);

    for (size_t i = 0; i < snapshots_.size(); ++i) {
        const auto& snap = snapshots_[i];

        // mid_price > 0
        EXPECT_GT(snap.mid_price, 0.0f)
            << "Snapshot " << i << ": mid_price must be > 0, got " << snap.mid_price;

        // spread >= 0
        EXPECT_GE(snap.spread, 0.0f)
            << "Snapshot " << i << ": spread must be >= 0, got " << snap.spread;

        // No crossed book: best_bid < best_ask
        // Only check if both sides are populated (non-zero price)
        if (snap.bids[0][0] > 0.0f && snap.asks[0][0] > 0.0f) {
            EXPECT_LT(snap.bids[0][0], snap.asks[0][0])
                << "Snapshot " << i << ": crossed book detected — "
                << "best_bid=" << snap.bids[0][0]
                << " >= best_ask=" << snap.asks[0][0];
        }
    }
}

// ---------------------------------------------------------------------------
// Test 4: Timestamps in range — all snapshots within 09:30:00 - 10:00:00 ET
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, TimestampsInRange) {
    ASSERT_TRUE(data_loaded_);

    for (size_t i = 0; i < snapshots_.size(); ++i) {
        uint64_t ts = snapshots_[i].timestamp;

        EXPECT_GE(ts, RTH_OPEN_NS)
            << "Snapshot " << i << ": timestamp before RTH open 09:30 ET";
        EXPECT_LT(ts, RTH_CLOSE_NS)
            << "Snapshot " << i << ": timestamp at or after RTH close 10:00 ET";
    }
}

// ===========================================================================
// FEATURE ENCODING TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 5: Feature shape — all encoded features have exactly 194 dimensions
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, FeatureShape) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_FALSE(trajectory_.empty());

    // Check the first, middle, and last trajectory samples
    std::vector<size_t> indices = {0, trajectory_.size() / 2, trajectory_.size() - 1};
    for (size_t idx : indices) {
        const auto& sample = trajectory_[idx];

        // Each window should have W snapshots
        ASSERT_EQ(static_cast<int>(sample.window.size()), W)
            << "Sample " << idx << ": window should have W=" << W << " snapshots";

        // Each snapshot should have FEATURE_DIM features
        for (int t = 0; t < W; ++t) {
            EXPECT_EQ(static_cast<int>(sample.window[t].size()), FEATURE_DIM)
                << "Sample " << idx << ", timestep " << t
                << ": feature vector should have " << FEATURE_DIM << " dimensions";
        }
    }
}

// ---------------------------------------------------------------------------
// Test 6: No NaN in features — all feature values are finite
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, NoNanInFeatures) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_FALSE(subsampled_.empty());

    for (size_t i = 0; i < subsampled_.size(); ++i) {
        const auto& sample = subsampled_[i];
        for (int t = 0; t < W; ++t) {
            for (int f = 0; f < FEATURE_DIM; ++f) {
                EXPECT_TRUE(std::isfinite(sample.window[t][f]))
                    << "Subsample " << i << ", t=" << t << ", f=" << f
                    << ": feature value is not finite ("
                    << sample.window[t][f] << ")";
            }
        }
    }
}

// ===========================================================================
// TRAJECTORY TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 7: Trajectory length — trajectory_length >= 32
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, TrajectoryLength) {
    ASSERT_TRUE(data_loaded_);

    int traj_len = static_cast<int>(trajectory_.size());
    EXPECT_GE(traj_len, N_OVERFIT)
        << "Trajectory must have at least " << N_OVERFIT
        << " samples to subsample, got " << traj_len;

    // Also verify the expected formula:
    // trajectory_length = len(snapshots) - horizon - W + 1
    int expected_len = static_cast<int>(snapshots_.size()) - HORIZON - W + 1;
    EXPECT_EQ(traj_len, expected_len)
        << "Trajectory length should match formula: snapshots("
        << snapshots_.size() << ") - horizon(" << HORIZON << ") - W("
        << W << ") + 1 = " << expected_len;
}

// ---------------------------------------------------------------------------
// Test 8: Labels valid — all labels in {0, 1, 2, 3}, no label == 4
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, LabelsValid) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_FALSE(subsampled_.empty());

    std::set<int> valid_labels = {0, 1, 2, 3};

    for (size_t i = 0; i < subsampled_.size(); ++i) {
        int label = subsampled_[i].label;

        EXPECT_TRUE(valid_labels.count(label))
            << "Subsample " << i << ": label=" << label
            << " is invalid. Must be in {0, 1, 2, 3}.";

        EXPECT_NE(label, 4)
            << "Subsample " << i << ": label=4 (REVERSE) must never be generated.";
    }
}

// ---------------------------------------------------------------------------
// Test 9: Label distribution — >=3 of 4 classes present, no class >80%
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, LabelDistribution) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_.size()), N_OVERFIT)
        << "Subsampled set must have exactly " << N_OVERFIT << " samples";

    std::map<int, int> counts;
    for (const auto& s : subsampled_) {
        counts[s.label]++;
    }

    // At least 3 of 4 classes present
    int classes_present = static_cast<int>(counts.size());
    EXPECT_GE(classes_present, 3)
        << "Need >= 3 of 4 classes present in subsample, got " << classes_present;

    // No class > 80%
    for (auto& [cls, cnt] : counts) {
        float pct = static_cast<float>(cnt) / static_cast<float>(N_OVERFIT);
        EXPECT_LE(pct, 0.80f)
            << "Class " << cls << " has " << cnt << "/" << N_OVERFIT
            << " = " << (pct * 100.0f) << "%, which exceeds 80%";
    }

    // No class 4 (already checked in LabelsValid, but verify in distribution too)
    EXPECT_EQ(counts.count(4), 0u)
        << "Class 4 (REVERSE) must not appear in subsample distribution";
}

// ===========================================================================
// OVERFIT TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 10: MLP overfit — reaches >=99% accuracy within 500 epochs on N=32
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, MLPOverfit) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_.size()), N_OVERFIT);

    OverfitResult result = overfit_mlp(
        subsampled_,
        MAX_EPOCHS_NEURAL,
        TARGET_ACC,
        LR_NEURAL
    );

    EXPECT_TRUE(result.success)
        << "MLP must overfit N=" << N_OVERFIT << " to >= 99% accuracy. "
        << "Got " << (result.final_accuracy * 100.0f) << "% after "
        << MAX_EPOCHS_NEURAL << " epochs.";

    EXPECT_GE(result.final_accuracy, TARGET_ACC)
        << "MLP final accuracy " << (result.final_accuracy * 100.0f)
        << "% < target " << (TARGET_ACC * 100.0f) << "%";

    if (result.success) {
        EXPECT_GT(result.epochs_to_target, 0)
            << "Should take at least 1 epoch to reach target";
        EXPECT_LE(result.epochs_to_target, MAX_EPOCHS_NEURAL)
            << "Must reach target within " << MAX_EPOCHS_NEURAL << " epochs";
    }
}

// ---------------------------------------------------------------------------
// Test 11: CNN overfit — reaches >=99% accuracy within 500 epochs on N=32
//
// NOTE: overfit_cnn() does not exist yet. This test calls the expected
// function signature. The GREEN phase must implement it (likely in
// training_loop.hpp or a new header).
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, CNNOverfit) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_.size()), N_OVERFIT);

    // overfit_cnn should have the same interface as overfit_mlp:
    //   OverfitResult overfit_cnn(samples, max_epochs, target_accuracy, lr)
    OverfitResult result = overfit_cnn(
        subsampled_,
        MAX_EPOCHS_NEURAL,
        TARGET_ACC,
        LR_NEURAL
    );

    EXPECT_TRUE(result.success)
        << "CNN must overfit N=" << N_OVERFIT << " to >= 99% accuracy. "
        << "Got " << (result.final_accuracy * 100.0f) << "% after "
        << MAX_EPOCHS_NEURAL << " epochs.";

    EXPECT_GE(result.final_accuracy, TARGET_ACC)
        << "CNN final accuracy " << (result.final_accuracy * 100.0f)
        << "% < target " << (TARGET_ACC * 100.0f) << "%";

    if (result.success) {
        EXPECT_GT(result.epochs_to_target, 0);
        EXPECT_LE(result.epochs_to_target, MAX_EPOCHS_NEURAL);
    }
}

// ---------------------------------------------------------------------------
// Test 12: GBT overfit — reaches >=99% accuracy within 1000 rounds on N=32
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, GBTOverfit) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_.size()), N_OVERFIT);

    GBTOverfitResult result = overfit_gbt(
        subsampled_,
        MAX_ROUNDS_GBT,
        TARGET_ACC_GBT
    );

    EXPECT_TRUE(result.success)
        << "GBT must overfit N=" << N_OVERFIT << " to >= 99% accuracy. "
        << "Got " << (result.accuracy * 100.0f) << "% after "
        << result.rounds << " rounds.";

    EXPECT_GE(result.accuracy, TARGET_ACC_GBT)
        << "GBT final accuracy " << (result.accuracy * 100.0f)
        << "% < target " << (TARGET_ACC_GBT * 100.0f) << "%";
}

// ---------------------------------------------------------------------------
// Test 13: No NaN during training — no NaN/Inf in any loss value during
//          any model's training
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, NoNanDuringTraining) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_.size()), N_OVERFIT);

    // --- MLP loss finiteness ---
    {
        OverfitResult mlp_result = overfit_mlp(
            subsampled_, MAX_EPOCHS_NEURAL, TARGET_ACC, LR_NEURAL);
        EXPECT_TRUE(std::isfinite(mlp_result.final_loss))
            << "MLP final loss is not finite: " << mlp_result.final_loss;
        EXPECT_TRUE(std::isfinite(mlp_result.final_accuracy))
            << "MLP final accuracy is not finite: " << mlp_result.final_accuracy;
    }

    // --- CNN loss finiteness ---
    {
        OverfitResult cnn_result = overfit_cnn(
            subsampled_, MAX_EPOCHS_NEURAL, TARGET_ACC, LR_NEURAL);
        EXPECT_TRUE(std::isfinite(cnn_result.final_loss))
            << "CNN final loss is not finite: " << cnn_result.final_loss;
        EXPECT_TRUE(std::isfinite(cnn_result.final_accuracy))
            << "CNN final accuracy is not finite: " << cnn_result.final_accuracy;
    }

    // --- GBT (accuracy finiteness — GBT doesn't track loss per round) ---
    {
        GBTOverfitResult gbt_result = overfit_gbt(
            subsampled_, MAX_ROUNDS_GBT, TARGET_ACC_GBT);
        EXPECT_TRUE(std::isfinite(gbt_result.accuracy))
            << "GBT accuracy is not finite: " << gbt_result.accuracy;
    }
}

// ---------------------------------------------------------------------------
// Test 14: Deterministic — two runs with seed=42 produce identical final
//          accuracy for all models
// ---------------------------------------------------------------------------
TEST_F(IntegrationOverfitTest, Deterministic) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_.size()), N_OVERFIT);

    // --- MLP determinism ---
    {
        OverfitResult run1 = overfit_mlp(
            subsampled_, MAX_EPOCHS_NEURAL, TARGET_ACC, LR_NEURAL);
        OverfitResult run2 = overfit_mlp(
            subsampled_, MAX_EPOCHS_NEURAL, TARGET_ACC, LR_NEURAL);

        EXPECT_FLOAT_EQ(run1.final_accuracy, run2.final_accuracy)
            << "MLP: two runs with seed=42 must produce identical accuracy";
        EXPECT_FLOAT_EQ(run1.final_loss, run2.final_loss)
            << "MLP: two runs with seed=42 must produce identical loss";
    }

    // --- CNN determinism ---
    {
        OverfitResult run1 = overfit_cnn(
            subsampled_, MAX_EPOCHS_NEURAL, TARGET_ACC, LR_NEURAL);
        OverfitResult run2 = overfit_cnn(
            subsampled_, MAX_EPOCHS_NEURAL, TARGET_ACC, LR_NEURAL);

        EXPECT_FLOAT_EQ(run1.final_accuracy, run2.final_accuracy)
            << "CNN: two runs with seed=42 must produce identical accuracy";
        EXPECT_FLOAT_EQ(run1.final_loss, run2.final_loss)
            << "CNN: two runs with seed=42 must produce identical loss";
    }

    // --- GBT determinism ---
    {
        GBTOverfitResult run1 = overfit_gbt(
            subsampled_, MAX_ROUNDS_GBT, TARGET_ACC_GBT);
        GBTOverfitResult run2 = overfit_gbt(
            subsampled_, MAX_ROUNDS_GBT, TARGET_ACC_GBT);

        EXPECT_FLOAT_EQ(run1.accuracy, run2.accuracy)
            << "GBT: two runs with seed=42 must produce identical accuracy";
    }
}
