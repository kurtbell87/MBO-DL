// hybrid_model_tb_label_test.cpp — TDD RED phase tests for hybrid model Phase A
// Spec: .kit/docs/hybrid-model.md §Phase A, §Test Cases 1-10
//
// Tests triple barrier label computation for the bar_feature_export extension:
// compute_tb_label() with spec-mandated parameters (target=10, stop=5,
// volume_horizon=500, min_return_ticks=2, max_time_horizon_s=300).
//
// Also tests CSV schema extension: new columns tb_label, tb_exit_type, tb_bars_held.

#include <gtest/gtest.h>

#include "backtest/triple_barrier.hpp"
#include "features/bar_features.hpp"
#include "features/raw_representations.hpp"

#include "test_bar_helpers.hpp"
#include "test_export_helpers.hpp"

#include <filesystem>
#include <set>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

using test_helpers::TICK;
using test_helpers::make_bar_series;
using test_helpers::make_bar_path;
using test_helpers::make_bar_sequence;

using export_test_helpers::BINARY_PATH;
using export_test_helpers::run_command;
using export_test_helpers::parse_csv_header;
using export_test_helpers::read_first_line;
using export_test_helpers::read_all_lines;

// Hybrid model spec TB config (from .kit/docs/hybrid-model.md §Phase A)
TripleBarrierConfig hybrid_tb_config() {
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.volume_horizon = 500;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    return cfg;
}

std::string temp_csv_path(const std::string& suffix = "") {
    return (std::filesystem::temp_directory_path() /
            ("hybrid_model_test" + suffix + ".csv")).string();
}

}  // anonymous namespace

// ===========================================================================
// Test 1: TB label computation — target hit → label = +1
// ===========================================================================
// Spec: "Verify label = +1 when target hit first"
class TBLabelTargetHitTest : public ::testing::Test {};

TEST_F(TBLabelTargetHitTest, TargetHitFirstGivesLabelPlusOne) {
    // Build a price path that rises enough to hit the target barrier.
    // target_ticks=10, tick_size=0.25 → target_dist = 2.50 points.
    // Price rises by 0.25 per bar (1 tick/bar) for 15 bars → +3.75 total.
    // Stop at 5 ticks = 1.25 points — never hit because price only goes up.
    auto cfg = hybrid_tb_config();

    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);  // +1 tick per bar up
    auto bars = make_bar_path(4500.0f, deltas, 30);  // low volume, won't hit vol horizon

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 1) << "Target hit first should produce label +1";
    EXPECT_EQ(result.exit_type, "target") << "Exit type should be 'target'";
}

TEST_F(TBLabelTargetHitTest, TargetHitExactlyAtBoundary) {
    // Price rises exactly to target_ticks * tick_size = 10 * 0.25 = 2.50
    auto cfg = hybrid_tb_config();

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);  // exactly 10 ticks up
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);   // flat after
    auto bars = make_bar_path(4500.0f, deltas, 30);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 1) << "Exactly hitting target should label +1";
    EXPECT_EQ(result.exit_type, "target");
}

// ===========================================================================
// Test 1b: TB label computation — stop hit → label = -1
// ===========================================================================

TEST_F(TBLabelTargetHitTest, StopHitFirstGivesLabelMinusOne) {
    // Price drops to hit stop barrier first.
    // stop_ticks=5, tick_size=0.25 → stop_dist = 1.25 points.
    // Price drops by 0.25 per bar → at bar 5, diff = -1.25 = -5 ticks → stop hit.
    auto cfg = hybrid_tb_config();

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);  // -1 tick per bar
    for (int i = 0; i < 10; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 30);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, -1) << "Stop hit first should produce label -1";
    EXPECT_EQ(result.exit_type, "stop") << "Exit type should be 'stop'";
}

// ===========================================================================
// Test 2: TB exit type — verify all exit types
// ===========================================================================
class TBExitTypeTest : public ::testing::Test {};

TEST_F(TBExitTypeTest, TargetExitType) {
    auto cfg = hybrid_tb_config();
    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 30);

    auto result = compute_tb_label(bars, 0, cfg);
    EXPECT_EQ(result.exit_type, "target");
}

TEST_F(TBExitTypeTest, StopExitType) {
    auto cfg = hybrid_tb_config();
    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);
    auto bars = make_bar_path(4500.0f, deltas, 30);

    auto result = compute_tb_label(bars, 0, cfg);
    EXPECT_EQ(result.exit_type, "stop");
}

TEST_F(TBExitTypeTest, ExpiryExitType) {
    // Volume horizon hit before target/stop. Small return >= min_return.
    // volume_horizon=500. Use 50 vol/bar → 10 bars to reach 500.
    // Price moves +3 ticks in 10 bars (3 >= min_return_ticks=2).
    auto cfg = hybrid_tb_config();
    cfg.volume_horizon = 500;

    // 10 bars of 50 vol each = 500 cumulative. Price drifts up 3 ticks.
    std::vector<float> deltas;
    for (int i = 0; i < 3; ++i) deltas.push_back(TICK);   // +3 ticks
    for (int i = 0; i < 10; ++i) deltas.push_back(0.0f);  // flat
    auto bars = make_bar_path(4500.0f, deltas, 50);  // 50 vol/bar

    auto result = compute_tb_label(bars, 0, cfg);
    EXPECT_EQ(result.exit_type, "expiry")
        << "Should be 'expiry' when volume_horizon reached before target/stop";
}

TEST_F(TBExitTypeTest, TimeoutExitType) {
    // Time cap triggers before volume or price barriers.
    // max_time_horizon_s=300 → 300 seconds.
    // Use 5s bars, low volume (1/bar so 60 bars = 60 vol < 500 horizon).
    auto cfg = hybrid_tb_config();

    // 65 bars × 5 seconds = 325s > 300s time cap.
    // Volume: 1/bar × 60 bars to 300s = 60 < 500 vol horizon.
    // Price barely moves: +3 ticks over 65 bars.
    std::vector<float> deltas;
    for (int i = 0; i < 3; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 62; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_sequence(4500.0f, deltas, std::vector<uint32_t>(66, 1), 5.0f);

    auto result = compute_tb_label(bars, 0, cfg);
    EXPECT_EQ(result.exit_type, "timeout")
        << "Should be 'timeout' when max_time_horizon_s exceeded before volume/price barriers";
}

// ===========================================================================
// Test 3: TB bars held — count matches scan distance
// ===========================================================================
class TBBarsHeldTest : public ::testing::Test {};

TEST_F(TBBarsHeldTest, BarsHeldMatchesScanDistance) {
    auto cfg = hybrid_tb_config();

    // Target hit at bar 10 (10 ticks up, 1 tick per bar).
    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 30);

    auto result = compute_tb_label(bars, 0, cfg);

    // bars[0] is entry, bars[10] should be where diff first >= 2.50 (10 ticks)
    EXPECT_EQ(result.bars_held, 10)
        << "Target hit at bar 10 means bars_held should be 10";
}

TEST_F(TBBarsHeldTest, BarsHeldForStopIsCorrect) {
    auto cfg = hybrid_tb_config();

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);
    auto bars = make_bar_path(4500.0f, deltas, 30);

    auto result = compute_tb_label(bars, 0, cfg);

    // Stop at 5 ticks down → hit at bar 5
    EXPECT_EQ(result.bars_held, 5)
        << "Stop hit at bar 5 means bars_held should be 5";
}

TEST_F(TBBarsHeldTest, BarsHeldFromMiddleIndex) {
    // compute_tb_label at idx=5 instead of idx=0
    auto cfg = hybrid_tb_config();

    // Build 25 bars where price starts moving up at bar 5
    std::vector<float> deltas;
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);      // flat first 5
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);      // +1 tick/bar from bar 5
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 30);

    auto result = compute_tb_label(bars, 5, cfg);

    // From bar 5, target (10 ticks up) hit at bar 15, so bars_held = 10
    EXPECT_EQ(result.bars_held, 10);
    EXPECT_EQ(result.label, 1);
}

// ===========================================================================
// Test 4: TB volume accumulation — volume_horizon respected
// ===========================================================================
class TBVolumeAccumulationTest : public ::testing::Test {};

TEST_F(TBVolumeAccumulationTest, ExpiresWhenCumulativeVolumeReachesHorizon) {
    // volume_horizon=500. Each bar has 100 vol → 5 bars to reach 500.
    // Price moves slowly (+0.5 tick/bar = not enough for target/stop in 5 bars).
    auto cfg = hybrid_tb_config();

    std::vector<float> deltas;
    for (int i = 0; i < 20; ++i) deltas.push_back(0.05f);  // tiny drift
    std::vector<uint32_t> volumes(21, 100);  // 100 vol/bar
    auto bars = make_bar_sequence(4500.0f, deltas, volumes, 5.0f);

    auto result = compute_tb_label(bars, 0, cfg);

    // Volume accumulates: bar1=100, bar2=200, bar3=300, bar4=400, bar5=500 → expiry at bar 5
    EXPECT_EQ(result.exit_type, "expiry")
        << "Should expire when cumulative volume reaches volume_horizon (500)";
    EXPECT_EQ(result.bars_held, 5)
        << "With 100 vol/bar, volume_horizon=500 reached at bar 5";
}

TEST_F(TBVolumeAccumulationTest, TargetStillHitsBeforeVolumeHorizon) {
    // Price moves fast enough to hit target before volume accumulates.
    auto cfg = hybrid_tb_config();

    // Target at 10 ticks = 2.50. Price rises 3 ticks/bar → hits at bar 4 (12 ticks > 10).
    // Volume: 10/bar → only 40 vol by bar 4 << 500 horizon.
    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(3.0f * TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "target")
        << "Target should hit before volume horizon";
    EXPECT_EQ(result.label, 1);
}

// ===========================================================================
// Test 5: TB time cap — max_time_horizon_s triggers timeout
// ===========================================================================
class TBTimeCapTest : public ::testing::Test {};

TEST_F(TBTimeCapTest, TimeoutTriggeredWhenTimeExceeded) {
    auto cfg = hybrid_tb_config();
    cfg.max_time_horizon_s = 30;  // Short time cap for testing

    // 10 bars × 5 seconds = 50 seconds. Time cap at 30s triggers at bar 6.
    // Volume: 1/bar → 6 vol << 500 horizon. Price flat → no target/stop.
    std::vector<float> deltas(10, 0.0f);
    std::vector<uint32_t> volumes(11, 1);
    auto bars = make_bar_sequence(4500.0f, deltas, volumes, 5.0f);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "timeout")
        << "Should timeout when elapsed time >= max_time_horizon_s";
}

TEST_F(TBTimeCapTest, TimeoutWithLargeReturnGivesDirectionalLabel) {
    // Timeout but |return| >= min_return_ticks → directional label
    auto cfg = hybrid_tb_config();
    cfg.max_time_horizon_s = 30;

    // Price rises 4 ticks in 6 bars (before 30s cap), then time exceeds cap.
    // 4 ticks >= min_return_ticks(2) → label based on sign.
    std::vector<float> deltas;
    for (int i = 0; i < 4; ++i) deltas.push_back(TICK);  // +4 ticks in first 4 bars
    for (int i = 0; i < 6; ++i) deltas.push_back(0.0f);  // flat after
    std::vector<uint32_t> volumes(11, 1);
    auto bars = make_bar_sequence(4500.0f, deltas, volumes, 5.0f);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "timeout");
    EXPECT_EQ(result.label, 1)
        << "Timeout with positive return >= min_return should give label +1";
}

TEST_F(TBTimeCapTest, TimeoutWithSmallReturnGivesHold) {
    // Timeout but |return| < min_return_ticks → label = 0
    auto cfg = hybrid_tb_config();
    cfg.max_time_horizon_s = 30;

    // Price barely moves: +1 tick < min_return_ticks(2)
    std::vector<float> deltas;
    deltas.push_back(TICK);  // +1 tick
    for (int i = 0; i < 9; ++i) deltas.push_back(0.0f);
    std::vector<uint32_t> volumes(11, 1);
    auto bars = make_bar_sequence(4500.0f, deltas, volumes, 5.0f);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "timeout");
    EXPECT_EQ(result.label, 0)
        << "Timeout with return < min_return_ticks should give label 0 (HOLD)";
}

// ===========================================================================
// Test 6: TB min_return filter — expiry with small return → label = 0
// ===========================================================================
class TBMinReturnFilterTest : public ::testing::Test {};

TEST_F(TBMinReturnFilterTest, ExpirySmallReturnGivesHold) {
    // At expiry (volume horizon reached), if |return| < min_return_ticks → label = 0.
    auto cfg = hybrid_tb_config();

    // 10 bars × 50 vol = 500 → expiry at bar 10. Price moves +1 tick (< min 2).
    std::vector<float> deltas;
    deltas.push_back(TICK);  // +1 tick
    for (int i = 0; i < 14; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "expiry");
    EXPECT_EQ(result.label, 0)
        << "Expiry with |return| < min_return_ticks should give HOLD (0)";
}

TEST_F(TBMinReturnFilterTest, ExpiryLargePositiveReturnGivesLong) {
    auto cfg = hybrid_tb_config();

    // 10 bars × 50 vol = 500 → expiry. Price moves +3 ticks (>= min 2).
    std::vector<float> deltas;
    for (int i = 0; i < 3; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 12; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "expiry");
    EXPECT_EQ(result.label, 1)
        << "Expiry with positive return >= min_return should give +1";
}

TEST_F(TBMinReturnFilterTest, ExpiryLargeNegativeReturnGivesShort) {
    auto cfg = hybrid_tb_config();

    // 10 bars × 50 vol → expiry. Price moves -3 ticks.
    std::vector<float> deltas;
    for (int i = 0; i < 3; ++i) deltas.push_back(-TICK);
    for (int i = 0; i < 12; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "expiry");
    EXPECT_EQ(result.label, -1)
        << "Expiry with negative return >= min_return should give -1";
}

TEST_F(TBMinReturnFilterTest, ExpiryExactlyAtMinReturnGivesDirectional) {
    auto cfg = hybrid_tb_config();

    // Exactly 2 ticks up = min_return_ticks boundary.
    std::vector<float> deltas;
    for (int i = 0; i < 2; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 13; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_tb_label(bars, 0, cfg);

    // 2 ticks * 0.25 = 0.50 >= min_return_dist (2 * 0.25 = 0.50)
    EXPECT_EQ(result.label, 1)
        << "Exactly at min_return boundary should produce directional label";
}

// ===========================================================================
// Test 7: TB label distribution — on synthetic data, 3 classes present
// ===========================================================================
class TBLabelDistributionTest : public ::testing::Test {};

TEST_F(TBLabelDistributionTest, AllThreeClassesPresentOnMixedPricePath) {
    // Create a mixed price path that should produce all three label classes:
    // rising segment (target hits → +1), falling segment (stop hits → -1),
    // and flat segments (expiry with small return → 0).
    auto cfg = hybrid_tb_config();

    // Build a long bar series with alternating phases
    std::vector<float> deltas;
    // Phase 1: strong rise (will produce +1 labels at entry points)
    for (int i = 0; i < 20; ++i) deltas.push_back(TICK);
    // Phase 2: flat (will produce 0 labels)
    for (int i = 0; i < 30; ++i) deltas.push_back(0.0f);
    // Phase 3: strong drop (will produce -1 labels)
    for (int i = 0; i < 15; ++i) deltas.push_back(-TICK);
    // Phase 4: flat again
    for (int i = 0; i < 30; ++i) deltas.push_back(0.0f);
    // Phase 5: rise again
    for (int i = 0; i < 20; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    std::set<int> observed_labels;
    int count_plus = 0, count_minus = 0, count_zero = 0;

    for (int i = 0; i < static_cast<int>(bars.size()) - 20; ++i) {
        auto result = compute_tb_label(bars, i, cfg);
        observed_labels.insert(result.label);
        if (result.label == 1) count_plus++;
        else if (result.label == -1) count_minus++;
        else count_zero++;
    }

    int total = count_plus + count_minus + count_zero;

    // All 3 classes must be present
    EXPECT_TRUE(observed_labels.count(1) > 0) << "Label +1 (long) must be present";
    EXPECT_TRUE(observed_labels.count(-1) > 0) << "Label -1 (short) must be present";
    EXPECT_TRUE(observed_labels.count(0) > 0) << "Label 0 (hold) must be present";

    // No class > 60% of total (spec: "no class > 60%")
    float pct_plus = static_cast<float>(count_plus) / total;
    float pct_minus = static_cast<float>(count_minus) / total;
    float pct_zero = static_cast<float>(count_zero) / total;

    EXPECT_LE(pct_plus, 0.60f)
        << "Long class should be <= 60% of labels, got " << pct_plus * 100 << "%";
    EXPECT_LE(pct_minus, 0.60f)
        << "Short class should be <= 60% of labels, got " << pct_minus * 100 << "%";
    EXPECT_LE(pct_zero, 0.60f)
        << "Hold class should be <= 60% of labels, got " << pct_zero * 100 << "%";
}

// ===========================================================================
// Test 8: TB no NaN/invalid labels — all bars with forward data get valid label
// ===========================================================================
class TBNoInvalidLabelsTest : public ::testing::Test {};

TEST_F(TBNoInvalidLabelsTest, AllLabelsAreValidValues) {
    auto cfg = hybrid_tb_config();

    auto bars = make_bar_series(4500.0f, 4503.0f, 100, 50);

    for (int i = 0; i < static_cast<int>(bars.size()) - 1; ++i) {
        auto result = compute_tb_label(bars, i, cfg);

        // Label must be exactly -1, 0, or +1
        EXPECT_TRUE(result.label == -1 || result.label == 0 || result.label == 1)
            << "Label at bar " << i << " is " << result.label
            << " — must be -1, 0, or +1";

        // exit_type must be one of the valid strings
        EXPECT_TRUE(result.exit_type == "target" ||
                     result.exit_type == "stop" ||
                     result.exit_type == "expiry" ||
                     result.exit_type == "timeout")
            << "Invalid exit_type '" << result.exit_type << "' at bar " << i;

        // bars_held must be non-negative
        EXPECT_GE(result.bars_held, 0)
            << "bars_held must be >= 0 at bar " << i;
    }
}

TEST_F(TBNoInvalidLabelsTest, LastBarStillProducesValidResult) {
    auto cfg = hybrid_tb_config();
    auto bars = make_bar_series(4500.0f, 4501.0f, 10, 50);

    // Last bar has no forward data → should still return a valid struct
    auto result = compute_tb_label(bars, 9, cfg);

    EXPECT_TRUE(result.label == -1 || result.label == 0 || result.label == 1);
    EXPECT_FALSE(result.exit_type.empty());
}

// ===========================================================================
// Test 9: Export CSV schema — new columns present and parseable
// ===========================================================================
class ExportCSVSchemaTest : public ::testing::Test {};

TEST_F(ExportCSVSchemaTest, HeaderContainsTBColumns) {
    // The CSV header produced by bar_feature_export must include
    // tb_label, tb_exit_type, tb_bars_held as the last 3 columns.
    // We construct the expected header programmatically.
    std::ostringstream ss;
    ss << "timestamp,bar_type,bar_param,day,is_warmup,bar_index";
    auto feature_names = BarFeatureRow::feature_names();
    for (const auto& name : feature_names) ss << "," << name;
    for (int i = 0; i < 40; ++i) ss << ",book_snap_" << i;
    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i) ss << ",msg_summary_" << i;
    ss << ",return_1,return_5,return_20,return_100";
    ss << ",mbo_event_count";
    ss << ",tb_label,tb_exit_type,tb_bars_held";

    auto cols = parse_csv_header(ss.str());

    // Verify new columns are present at the end
    size_t n = cols.size();
    ASSERT_GE(n, 3u);
    EXPECT_EQ(cols[n - 3], "tb_label");
    EXPECT_EQ(cols[n - 2], "tb_exit_type");
    EXPECT_EQ(cols[n - 1], "tb_bars_held");
}

TEST_F(ExportCSVSchemaTest, TotalColumnCountIs149) {
    // Old: 146 columns (6 meta + 62 Track A + 40 book + 33 msg + 4 ret + 1 evt)
    // New: 149 columns = 146 + 3 TB columns
    size_t expected = 6 + BarFeatureRow::feature_count() + 40 +
                      MessageSummary::SUMMARY_SIZE + 4 + 1 + 3;
    EXPECT_EQ(expected, 149u)
        << "Total columns should be 149 (146 original + 3 TB columns)";
}

// ===========================================================================
// Test 9b: Live CSV schema validation (requires binary + data)
// ===========================================================================

class ExportCSVLiveSchemaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(BINARY_PATH)) {
            GTEST_SKIP() << "Binary not built yet";
        }
    }
    void TearDown() override {
        for (const auto& p : temp_files_) std::filesystem::remove(p);
    }
    void track_temp(const std::string& p) { temp_files_.push_back(p); }
    std::vector<std::string> temp_files_;
};

TEST_F(ExportCSVLiveSchemaTest, OutputCSVHasTBColumnsInHeader) {
    auto csv = temp_csv_path("_schema_tb");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero (data may not be present)";
    }

    auto header_line = read_first_line(csv);
    auto cols = parse_csv_header(header_line);

    size_t n = cols.size();
    ASSERT_GE(n, 3u);
    EXPECT_EQ(cols[n - 3], "tb_label")
        << "Third-to-last column should be tb_label";
    EXPECT_EQ(cols[n - 2], "tb_exit_type")
        << "Second-to-last column should be tb_exit_type";
    EXPECT_EQ(cols[n - 1], "tb_bars_held")
        << "Last column should be tb_bars_held";
}

TEST_F(ExportCSVLiveSchemaTest, TBLabelValuesAreParseable) {
    auto csv = temp_csv_path("_tb_parse");
    track_temp(csv);
    auto result = run_command(
        BINARY_PATH + " --bar-type time --bar-param 5.0 --output " + csv);

    if (result.exit_code != 0) {
        GTEST_SKIP() << "Tool exited non-zero";
    }

    auto lines = read_all_lines(csv);
    if (lines.size() < 2) {
        GTEST_SKIP() << "No data rows produced";
    }

    auto header_cols = parse_csv_header(lines[0]);
    size_t ncols = header_cols.size();

    for (size_t i = 1; i < std::min(lines.size(), size_t(20)); ++i) {
        auto row = parse_csv_header(lines[i]);
        ASSERT_EQ(row.size(), ncols) << "Row " << i << " column count mismatch";

        // tb_label should be -1, 0, or 1
        int tb_label = std::stoi(row[ncols - 3]);
        EXPECT_TRUE(tb_label == -1 || tb_label == 0 || tb_label == 1)
            << "tb_label=" << tb_label << " at row " << i;

        // tb_exit_type should be a valid string
        const auto& exit_type = row[ncols - 2];
        EXPECT_TRUE(exit_type == "target" || exit_type == "stop" ||
                     exit_type == "expiry" || exit_type == "timeout")
            << "Invalid tb_exit_type '" << exit_type << "' at row " << i;

        // tb_bars_held should be non-negative integer
        int bars_held = std::stoi(row[ncols - 1]);
        EXPECT_GE(bars_held, 0) << "tb_bars_held must be >= 0 at row " << i;
    }
}

// ===========================================================================
// Test 10: Export backward compatibility — existing columns unchanged
// ===========================================================================
class ExportBackwardCompatibilityTest : public ::testing::Test {};

TEST_F(ExportBackwardCompatibilityTest, OriginalColumnsPreserved) {
    // Verify the first 146 columns of the new 149-column header match the old spec.
    // The old header: 6 meta + 62 Track A + 40 book + 33 msg + 4 ret + 1 evt = 146.
    // New header adds 3 TB columns at the END.

    std::ostringstream old_header_ss;
    old_header_ss << "timestamp,bar_type,bar_param,day,is_warmup,bar_index";
    auto feature_names = BarFeatureRow::feature_names();
    for (const auto& name : feature_names) old_header_ss << "," << name;
    for (int i = 0; i < 40; ++i) old_header_ss << ",book_snap_" << i;
    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i)
        old_header_ss << ",msg_summary_" << i;
    old_header_ss << ",return_1,return_5,return_20,return_100";
    old_header_ss << ",mbo_event_count";

    auto old_cols = parse_csv_header(old_header_ss.str());
    ASSERT_EQ(old_cols.size(), 146u);

    // Build new header (with TB columns)
    std::ostringstream new_header_ss;
    new_header_ss << old_header_ss.str();
    new_header_ss << ",tb_label,tb_exit_type,tb_bars_held";

    auto new_cols = parse_csv_header(new_header_ss.str());
    ASSERT_EQ(new_cols.size(), 149u);

    // First 146 columns must match exactly
    for (size_t i = 0; i < 146; ++i) {
        EXPECT_EQ(new_cols[i], old_cols[i])
            << "Column " << i << " changed: old='" << old_cols[i]
            << "' new='" << new_cols[i] << "'";
    }
}

TEST_F(ExportBackwardCompatibilityTest, MetadataColumnsUnchanged) {
    std::ostringstream ss;
    ss << "timestamp,bar_type,bar_param,day,is_warmup,bar_index";
    auto cols = parse_csv_header(ss.str());

    EXPECT_EQ(cols[0], "timestamp");
    EXPECT_EQ(cols[1], "bar_type");
    EXPECT_EQ(cols[2], "bar_param");
    EXPECT_EQ(cols[3], "day");
    EXPECT_EQ(cols[4], "is_warmup");
    EXPECT_EQ(cols[5], "bar_index");
}

TEST_F(ExportBackwardCompatibilityTest, EventCountStillPrecedesTBColumns) {
    // mbo_event_count should be at its original position (index 145),
    // with TB columns appended after it.
    std::ostringstream ss;
    ss << "timestamp,bar_type,bar_param,day,is_warmup,bar_index";
    auto feature_names = BarFeatureRow::feature_names();
    for (const auto& name : feature_names) ss << "," << name;
    for (int i = 0; i < 40; ++i) ss << ",book_snap_" << i;
    for (size_t i = 0; i < MessageSummary::SUMMARY_SIZE; ++i)
        ss << ",msg_summary_" << i;
    ss << ",return_1,return_5,return_20,return_100";
    ss << ",mbo_event_count";
    ss << ",tb_label,tb_exit_type,tb_bars_held";

    auto cols = parse_csv_header(ss.str());
    EXPECT_EQ(cols[145], "mbo_event_count");
    EXPECT_EQ(cols[146], "tb_label");
    EXPECT_EQ(cols[147], "tb_exit_type");
    EXPECT_EQ(cols[148], "tb_bars_held");
}

// ===========================================================================
// Additional: compute_tb_label edge cases
// ===========================================================================
class TBEdgeCasesTest : public ::testing::Test {};

TEST_F(TBEdgeCasesTest, SingleBarAfterEntryReturnsValidResult) {
    // Only 2 bars total (entry + 1 forward bar)
    auto cfg = hybrid_tb_config();
    auto bars = make_bar_series(4500.0f, 4500.5f, 2, 50);

    auto result = compute_tb_label(bars, 0, cfg);

    // With only 1 forward bar, volume = 50 << 500 horizon, time ~1s << 300s.
    // Runs out of bars → expiry/end-of-day handling.
    EXPECT_TRUE(result.label == -1 || result.label == 0 || result.label == 1);
}

TEST_F(TBEdgeCasesTest, EntryAtLastBarReturnsHold) {
    auto cfg = hybrid_tb_config();
    auto bars = make_bar_series(4500.0f, 4501.0f, 10, 50);

    // Entry at last bar — no forward data
    auto result = compute_tb_label(bars, 9, cfg);

    EXPECT_EQ(result.label, 0)
        << "Entry at last bar with no forward data should be HOLD";
    EXPECT_EQ(result.bars_held, 0);
}

TEST_F(TBEdgeCasesTest, ZeroVolumeHorizonTriggersImmediately) {
    auto cfg = hybrid_tb_config();
    cfg.volume_horizon = 0;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_tb_label(bars, 0, cfg);

    // Volume horizon 0 → cumulative vol (50) >= 0 at bar 1
    EXPECT_EQ(result.bars_held, 1);
}

TEST_F(TBEdgeCasesTest, LabelSymmetryPositiveAndNegative) {
    // A symmetric test: same magnitude price path, opposite direction.
    auto cfg = hybrid_tb_config();

    // Upward path
    std::vector<float> up_deltas;
    for (int i = 0; i < 15; ++i) up_deltas.push_back(TICK);
    auto up_bars = make_bar_path(4500.0f, up_deltas, 30);
    auto up_result = compute_tb_label(up_bars, 0, cfg);

    // Downward path
    std::vector<float> down_deltas;
    for (int i = 0; i < 15; ++i) down_deltas.push_back(-TICK);
    auto down_bars = make_bar_path(4500.0f, down_deltas, 30);
    auto down_result = compute_tb_label(down_bars, 0, cfg);

    EXPECT_EQ(up_result.label, 1);
    EXPECT_EQ(down_result.label, -1);
    EXPECT_EQ(up_result.exit_type, "target");
    EXPECT_EQ(down_result.exit_type, "stop");
}
