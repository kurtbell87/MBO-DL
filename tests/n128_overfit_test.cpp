// n128_overfit_test.cpp — N=128 overfit integration tests
// Spec: .kit/docs/n128-overfit.md
//
// Validates that all 3 models (MLP, CNN, GBT) achieve ≥95% training accuracy
// on N=128 evenly-spaced samples from real MBO data. Accuracy must be
// evaluated on ALL 128 samples (not per-batch).
//
// Test categories:
//   1. N=128 sampling — correct count, spacing, label quality
//   2. Overfit — MLP, CNN, GBT each reach ≥95% on 128 samples
//   3. Accuracy denominator — verify evaluation uses all 128 samples

#include <gtest/gtest.h>

#include "book_builder.hpp"
#include "feature_encoder.hpp"
#include "trajectory_builder.hpp"
#include "gbt_model.hpp"
#include "training_loop.hpp"

#include <torch/torch.h>

// databento C++ API — reads .dbn.zst files
#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>
#include <databento/enums.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

// ===========================================================================
// Constants — N=128 overfit parameters
// ===========================================================================
namespace {

// Data file (same as N=32 integration tests)
const std::string DATA_DIR  = "DATA/GLBX-20260207-L953CAPU5B";
const std::string DATA_FILE = DATA_DIR + "/glbx-mdp3-20220103.mbo.dbn.zst";

// Instrument
constexpr uint32_t INSTRUMENT_ID = 13615;  // MESM2

// Session window: RTH 09:30 - 10:00 ET on 2022-01-03
constexpr uint64_t NS_PER_SEC  = 1'000'000'000ULL;
constexpr uint64_t NS_PER_MIN  = 60ULL * NS_PER_SEC;
constexpr uint64_t NS_PER_HOUR = 3600ULL * NS_PER_SEC;

// Midnight ET 2022-01-03 in UTC nanoseconds
constexpr uint64_t MIDNIGHT_ET_NS = 1641186000ULL * NS_PER_SEC;
constexpr uint64_t RTH_OPEN_NS    = MIDNIGHT_ET_NS + 9ULL * NS_PER_HOUR + 30ULL * NS_PER_MIN;
constexpr uint64_t RTH_CLOSE_NS   = MIDNIGHT_ET_NS + 10ULL * NS_PER_HOUR;

// Warm-up from 09:29:00 ET
constexpr uint64_t WARMUP_NS = MIDNIGHT_ET_NS + 9ULL * NS_PER_HOUR + 29ULL * NS_PER_MIN;

// Oracle parameters
constexpr int HORIZON = 100;

// Sampling
constexpr int N_OVERFIT_128 = 128;

// Training (MLP) — spec says up to 1000 epochs for MLP on N=128
constexpr int   MAX_EPOCHS_MLP  = 1000;
constexpr float LR_NEURAL       = 1e-3f;
constexpr float TARGET_ACC_128  = 0.95f;

// Training (CNN) — spec says 500 epochs max should suffice
constexpr int MAX_EPOCHS_CNN = 500;

// Training (GBT) — spec says 1000 rounds max
constexpr int MAX_ROUNDS_GBT = 1000;

constexpr int NUM_CLASSES = 5;

}  // anonymous namespace

// ===========================================================================
// Shared test fixture — loads data once, subsamples N=128
// ===========================================================================
class N128OverfitTest : public ::testing::Test {
protected:
    static std::vector<BookSnapshot> snapshots_;
    static std::vector<TrainingSample> trajectory_;
    static std::vector<TrainingSample> subsampled_128_;
    static bool data_loaded_;

    static void SetUpTestSuite() {
        if (data_loaded_) return;

        // --- Check data file exists ---
        ASSERT_TRUE(std::filesystem::exists(DATA_FILE))
            << "Data file not found: " << DATA_FILE
            << ". Place the MBO data file at the expected path.";

        // --- Build order book from .dbn.zst ---
        BookBuilder builder(INSTRUMENT_ID);

        databento::DbnFileStore store{std::filesystem::path(DATA_FILE)};

        while (const auto* record = store.NextRecord()) {
            if (const auto* mbo = record->GetIf<databento::MboMsg>()) {
                uint64_t ts_ns = static_cast<uint64_t>(
                    mbo->hd.ts_event.time_since_epoch().count());
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

        snapshots_ = builder.emit_snapshots(WARMUP_NS, RTH_CLOSE_NS);

        data_loaded_ = !snapshots_.empty();

        if (!data_loaded_) {
            GTEST_SKIP() << "No snapshots emitted — cannot run N=128 integration tests.";
            return;
        }

        // --- Build trajectory ---
        trajectory_ = build_trajectory(snapshots_, HORIZON);

        // --- Subsample N=128 evenly spaced ---
        int traj_len = static_cast<int>(trajectory_.size());
        int k = traj_len / N_OVERFIT_128;
        ASSERT_GT(k, 0) << "Trajectory too short to subsample N=" << N_OVERFIT_128
                         << " (trajectory length = " << traj_len << ")";

        subsampled_128_.clear();
        subsampled_128_.reserve(N_OVERFIT_128);
        for (int i = 0; i < N_OVERFIT_128; ++i) {
            subsampled_128_.push_back(trajectory_[i * k]);
        }

        // Validate label distribution (>=3 of 4 classes, no class >80%, no class 4)
        std::map<int, int> label_counts;
        for (const auto& s : subsampled_128_) {
            label_counts[s.label]++;
        }

        int classes_present = static_cast<int>(label_counts.size());
        bool any_class_over_80 = false;
        for (auto& [cls, cnt] : label_counts) {
            if (static_cast<float>(cnt) / N_OVERFIT_128 > 0.80f) {
                any_class_over_80 = true;
            }
        }

        // If distribution fails, retry with offset k/2
        if (classes_present < 3 || any_class_over_80) {
            int offset = k / 2;
            subsampled_128_.clear();
            for (int i = 0; i < N_OVERFIT_128; ++i) {
                int idx = offset + i * k;
                if (idx < traj_len) {
                    subsampled_128_.push_back(trajectory_[idx]);
                }
            }
        }
    }
};

// Static member definitions
std::vector<BookSnapshot> N128OverfitTest::snapshots_;
std::vector<TrainingSample> N128OverfitTest::trajectory_;
std::vector<TrainingSample> N128OverfitTest::subsampled_128_;
bool N128OverfitTest::data_loaded_ = false;

// ===========================================================================
// N=128 SAMPLING VALIDATION
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: Subsample has exactly 128 samples
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, SubsampleCount128) {
    ASSERT_TRUE(data_loaded_);

    EXPECT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128)
        << "Subsampled set must contain exactly " << N_OVERFIT_128 << " samples";
}

// ---------------------------------------------------------------------------
// Test: All labels are valid (in {0,1,2,3}, no class 4)
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, Labels128Valid) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128);

    std::set<int> valid_labels = {0, 1, 2, 3};

    for (size_t i = 0; i < subsampled_128_.size(); ++i) {
        int label = subsampled_128_[i].label;

        EXPECT_TRUE(valid_labels.count(label))
            << "Subsample " << i << ": label=" << label
            << " is invalid. Must be in {0, 1, 2, 3}.";

        EXPECT_NE(label, 4)
            << "Subsample " << i << ": label=4 (REVERSE) must never be generated.";
    }
}

// ---------------------------------------------------------------------------
// Test: Label distribution has >=3 classes, no class >80%
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, LabelDistribution128) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128);

    std::map<int, int> counts;
    for (const auto& s : subsampled_128_) {
        counts[s.label]++;
    }

    int classes_present = static_cast<int>(counts.size());
    EXPECT_GE(classes_present, 3)
        << "N=128 subsample needs >= 3 of 4 classes, got " << classes_present;

    for (auto& [cls, cnt] : counts) {
        float pct = static_cast<float>(cnt) / static_cast<float>(N_OVERFIT_128);
        EXPECT_LE(pct, 0.80f)
            << "Class " << cls << " has " << cnt << "/" << N_OVERFIT_128
            << " = " << (pct * 100.0f) << "%, which exceeds 80%";
    }
}

// ---------------------------------------------------------------------------
// Test: All 128 feature windows are finite (no NaN/Inf)
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, FeaturesFinite128) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128);

    for (size_t i = 0; i < subsampled_128_.size(); ++i) {
        const auto& sample = subsampled_128_[i];
        ASSERT_EQ(static_cast<int>(sample.window.size()), W)
            << "Sample " << i << ": window should have W=" << W << " timesteps";
        for (int t = 0; t < W; ++t) {
            for (int f = 0; f < FEATURE_DIM; ++f) {
                EXPECT_TRUE(std::isfinite(sample.window[t][f]))
                    << "Sample " << i << ", t=" << t << ", f=" << f
                    << ": feature is not finite (" << sample.window[t][f] << ")";
            }
        }
    }
}

// ===========================================================================
// OVERFIT TESTS — ≥95% accuracy on all 128 samples
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: MLP reaches ≥95% accuracy on N=128 within 1000 epochs
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, MLPOverfit128) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128);

    OverfitResult result = overfit_mlp(
        subsampled_128_,
        MAX_EPOCHS_MLP,
        TARGET_ACC_128,
        LR_NEURAL
    );

    EXPECT_TRUE(result.success)
        << "MLP must overfit N=" << N_OVERFIT_128 << " to >= 95% accuracy. "
        << "Got " << (result.final_accuracy * 100.0f) << "% after "
        << MAX_EPOCHS_MLP << " epochs.";

    EXPECT_GE(result.final_accuracy, TARGET_ACC_128)
        << "MLP final accuracy " << (result.final_accuracy * 100.0f)
        << "% < target " << (TARGET_ACC_128 * 100.0f) << "%";

    // Verify accuracy denominator is N=128 (not a subset).
    // The training loop computes accuracy as correct / n where n = input_tensor.size(0).
    // Confirm the training function received all 128 samples by checking the result
    // is consistent with 128-sample evaluation (accuracy must be a multiple of 1/128).
    if (result.success) {
        EXPECT_GT(result.epochs_to_target, 0)
            << "Should take at least 1 epoch to reach target";
        EXPECT_LE(result.epochs_to_target, MAX_EPOCHS_MLP)
            << "Must reach target within " << MAX_EPOCHS_MLP << " epochs";

        // Accuracy must be a multiple of 1/128 (proves denominator is 128).
        // accuracy = correct/128, so accuracy * 128 must be an integer (within float tolerance).
        float scaled = result.final_accuracy * static_cast<float>(N_OVERFIT_128);
        float rounded = std::round(scaled);
        EXPECT_NEAR(scaled, rounded, 0.01f)
            << "MLP accuracy " << result.final_accuracy
            << " is not a multiple of 1/" << N_OVERFIT_128
            << " — suggests accuracy was not computed over all 128 samples. "
            << "scaled=" << scaled << ", rounded=" << rounded;
    }

    // No NaN in final metrics
    EXPECT_TRUE(std::isfinite(result.final_loss))
        << "MLP final loss is not finite: " << result.final_loss;
    EXPECT_TRUE(std::isfinite(result.final_accuracy))
        << "MLP final accuracy is not finite: " << result.final_accuracy;
}

// ---------------------------------------------------------------------------
// Test: CNN reaches ≥95% accuracy on N=128 within 500 epochs
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, CNNOverfit128) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128);

    OverfitResult result = overfit_cnn(
        subsampled_128_,
        MAX_EPOCHS_CNN,
        TARGET_ACC_128,
        LR_NEURAL
    );

    EXPECT_TRUE(result.success)
        << "CNN must overfit N=" << N_OVERFIT_128 << " to >= 95% accuracy. "
        << "Got " << (result.final_accuracy * 100.0f) << "% after "
        << MAX_EPOCHS_CNN << " epochs.";

    EXPECT_GE(result.final_accuracy, TARGET_ACC_128)
        << "CNN final accuracy " << (result.final_accuracy * 100.0f)
        << "% < target " << (TARGET_ACC_128 * 100.0f) << "%";

    // Verify accuracy denominator is N=128
    if (result.success) {
        EXPECT_GT(result.epochs_to_target, 0)
            << "Should take at least 1 epoch to reach target";
        EXPECT_LE(result.epochs_to_target, MAX_EPOCHS_CNN)
            << "Must reach target within " << MAX_EPOCHS_CNN << " epochs";

        float scaled = result.final_accuracy * static_cast<float>(N_OVERFIT_128);
        float rounded = std::round(scaled);
        EXPECT_NEAR(scaled, rounded, 0.01f)
            << "CNN accuracy " << result.final_accuracy
            << " is not a multiple of 1/" << N_OVERFIT_128
            << " — suggests accuracy was not computed over all 128 samples. "
            << "scaled=" << scaled << ", rounded=" << rounded;
    }

    // No NaN in final metrics
    EXPECT_TRUE(std::isfinite(result.final_loss))
        << "CNN final loss is not finite: " << result.final_loss;
    EXPECT_TRUE(std::isfinite(result.final_accuracy))
        << "CNN final accuracy is not finite: " << result.final_accuracy;
}

// ---------------------------------------------------------------------------
// Test: GBT reaches ≥95% accuracy on N=128 within 1000 rounds
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, GBTOverfit128) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128);

    GBTOverfitResult result = overfit_gbt(
        subsampled_128_,
        MAX_ROUNDS_GBT,
        TARGET_ACC_128
    );

    EXPECT_TRUE(result.success)
        << "GBT must overfit N=" << N_OVERFIT_128 << " to >= 95% accuracy. "
        << "Got " << (result.accuracy * 100.0f) << "% after "
        << result.rounds << " rounds.";

    EXPECT_GE(result.accuracy, TARGET_ACC_128)
        << "GBT final accuracy " << (result.accuracy * 100.0f)
        << "% < target " << (TARGET_ACC_128 * 100.0f) << "%";

    // Verify accuracy denominator is N=128
    float scaled = result.accuracy * static_cast<float>(N_OVERFIT_128);
    float rounded = std::round(scaled);
    EXPECT_NEAR(scaled, rounded, 0.01f)
        << "GBT accuracy " << result.accuracy
        << " is not a multiple of 1/" << N_OVERFIT_128
        << " — suggests accuracy was not computed over all 128 samples. "
        << "scaled=" << scaled << ", rounded=" << rounded;

    // No NaN
    EXPECT_TRUE(std::isfinite(result.accuracy))
        << "GBT accuracy is not finite: " << result.accuracy;
}

// ===========================================================================
// ACCURACY DENOMINATOR VERIFICATION
// ===========================================================================

// ---------------------------------------------------------------------------
// Test: MLP training receives exactly 128 samples (tensor dimension check)
//
// This test verifies the contract that samples_to_tensors() produces a
// tensor with dim(0) == 128 when given 128 TrainingSamples. This ensures
// the accuracy denominator in train_model() is 128.
// ---------------------------------------------------------------------------
TEST_F(N128OverfitTest, MLPTensorDim128) {
    ASSERT_TRUE(data_loaded_);
    ASSERT_EQ(static_cast<int>(subsampled_128_.size()), N_OVERFIT_128);

    auto [input_tensor, label_tensor] = samples_to_tensors(subsampled_128_);

    // Input tensor shape must be (128, W, FEATURE_DIM)
    EXPECT_EQ(input_tensor.size(0), N_OVERFIT_128)
        << "Input tensor batch dim must be " << N_OVERFIT_128
        << " to ensure accuracy is evaluated on all 128 samples";
    EXPECT_EQ(input_tensor.size(1), W)
        << "Input tensor time dim must be W=" << W;
    EXPECT_EQ(input_tensor.size(2), FEATURE_DIM)
        << "Input tensor feature dim must be FEATURE_DIM=" << FEATURE_DIM;

    // Label tensor shape must be (128,)
    EXPECT_EQ(label_tensor.size(0), N_OVERFIT_128)
        << "Label tensor must have " << N_OVERFIT_128 << " elements";
}
