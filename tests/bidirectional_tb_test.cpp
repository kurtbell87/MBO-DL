// bidirectional_tb_test.cpp — Tests for bidirectional triple barrier labels
// Spec: .kit/docs/bidirectional-label-export.md
//
// Tests bidirectional triple barrier labeling: two independent barrier races per bar.
// Long race: does price hit +target before -stop?
// Short race: does price hit -target before +stop? (mirrored geometry)
// Label = +1 if only long wins, -1 if only short wins, 0 if neither or both.

#include <gtest/gtest.h>

#include "backtest/triple_barrier.hpp"

#include "test_bar_helpers.hpp"

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

// Default bidirectional config matching the spec's default geometry:
// target=10, stop=5, volume_horizon=500, tick_size=0.25
TripleBarrierConfig make_bidir_config() {
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.volume_horizon = 500;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = true;  // NEW: enable bidirectional mode
    return cfg;
}

}  // anonymous namespace

// ===========================================================================
// Config: bidirectional flag exists and defaults to true
// ===========================================================================
class BidirectionalConfigTest : public ::testing::Test {};

TEST_F(BidirectionalConfigTest, BidirectionalFieldExists) {
    TripleBarrierConfig cfg{};
    // Spec: "Default to bidirectional=true for new exports"
    EXPECT_TRUE(cfg.bidirectional)
        << "TripleBarrierConfig::bidirectional must default to true";
}

TEST_F(BidirectionalConfigTest, BidirectionalCanBeDisabled) {
    TripleBarrierConfig cfg{};
    cfg.bidirectional = false;
    EXPECT_FALSE(cfg.bidirectional);
}

// ===========================================================================
// Result: new fields exist in BidirectionalTBResult
// ===========================================================================
class BidirectionalResultTest : public ::testing::Test {};

TEST_F(BidirectionalResultTest, ResultHasLongTriggeredField) {
    BidirectionalTBResult result{};
    EXPECT_FALSE(result.long_triggered);
}

TEST_F(BidirectionalResultTest, ResultHasShortTriggeredField) {
    BidirectionalTBResult result{};
    EXPECT_FALSE(result.short_triggered);
}

TEST_F(BidirectionalResultTest, ResultHasBothTriggeredField) {
    BidirectionalTBResult result{};
    EXPECT_FALSE(result.both_triggered);
}

TEST_F(BidirectionalResultTest, ResultHasLabelField) {
    BidirectionalTBResult result{};
    EXPECT_EQ(result.label, 0);
}

TEST_F(BidirectionalResultTest, ResultHasExitTypeField) {
    BidirectionalTBResult result{};
    EXPECT_TRUE(result.exit_type.empty());
}

TEST_F(BidirectionalResultTest, ResultHasBarsHeldField) {
    BidirectionalTBResult result{};
    EXPECT_EQ(result.bars_held, 0);
}

// ===========================================================================
// T1: Symmetric Price Move — Both Races Trigger → HOLD
// Spec §T1: price rises AND falls target_ticks within the window
// ===========================================================================
class T1_BothRacesTrigger : public ::testing::Test {};

TEST_F(T1_BothRacesTrigger, BothRacesTriggerGivesHold) {
    // Price path: rises 10 ticks (= target), then drops 20 ticks (net -10 from entry),
    // hitting both +target and -target within the window.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;  // large so volume doesn't expire early

    // +10 ticks (target) then -20 ticks (passes through entry and hits -10 = short target)
    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);   // +10 ticks from entry
    for (int i = 0; i < 20; ++i) deltas.push_back(-TICK);  // back down to -10 from entry

    auto bars = make_bar_path(4500.0f, deltas, 10);  // 10 vol/bar, well under 5000 horizon

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 0)
        << "Both races triggering should produce HOLD (label=0)";
    EXPECT_TRUE(result.both_triggered)
        << "tb_both_triggered should be true when both races hit their targets";
    EXPECT_TRUE(result.long_triggered)
        << "Long race should have triggered (price hit +target)";
    EXPECT_TRUE(result.short_triggered)
        << "Short race should have triggered (price hit -target)";
}

TEST_F(T1_BothRacesTrigger, BothTriggeredExitTypeIsBoth) {
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 20; ++i) deltas.push_back(-TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // Spec §Parquet Schema: exit_type "both" when both races trigger
    EXPECT_EQ(result.exit_type, "both");
}

// ===========================================================================
// T2: Clean Long Signal — Only Long Race Triggers → +1
// Spec §T2: price rises target_ticks without falling stop_ticks
// ===========================================================================
class T2_CleanLongSignal : public ::testing::Test {};

TEST_F(T2_CleanLongSignal, OnlyLongRaceTriggersGivesPlusOne) {
    // Price rises 10 ticks steadily, never drops 5 ticks from entry.
    // Long race: +10 hit → triggered
    // Short race: price never drops 10 ticks, and stop (+5 from entry) hit first → not triggered
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);  // +15 ticks steady rise
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 1)
        << "Only long race triggering should produce label +1";
    EXPECT_FALSE(result.both_triggered)
        << "Both triggered should be false when only long wins";
    EXPECT_TRUE(result.long_triggered)
        << "Long race should have triggered";
    EXPECT_FALSE(result.short_triggered)
        << "Short race should NOT have triggered";
}

TEST_F(T2_CleanLongSignal, ExitTypeIsLongTarget) {
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // Spec §Parquet Schema: exit_type "long_target"
    EXPECT_EQ(result.exit_type, "long_target");
}

// ===========================================================================
// T3: Clean Short Signal — Only Short Race Triggers → -1
// Spec §T3: price falls target_ticks without rising stop_ticks
// ===========================================================================
class T3_CleanShortSignal : public ::testing::Test {};

TEST_F(T3_CleanShortSignal, OnlyShortRaceTriggersGivesMinusOne) {
    // Price falls 10 ticks (= target_ticks) steadily. Never rises 5 ticks.
    // Short race: -10 hit → triggered
    // Long race: price never rises 10, and long stop (-5 from entry) hit first → not triggered
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(-TICK);  // -15 ticks steady drop
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, -1)
        << "Only short race triggering should produce label -1";
    EXPECT_FALSE(result.both_triggered);
    EXPECT_FALSE(result.long_triggered)
        << "Long race should NOT have triggered";
    EXPECT_TRUE(result.short_triggered)
        << "Short race should have triggered";
}

TEST_F(T3_CleanShortSignal, ExitTypeIsShortTarget) {
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(-TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "short_target");
}

TEST_F(T3_CleanShortSignal, ShortRequiresFullTargetTicks) {
    // Spec: "short label requires the full 10-tick downward move"
    // Price drops only 5 ticks (= stop_ticks) — NOT enough for short target.
    // This is the core fix: under old labeling this would be -1 (stop hit);
    // under bidirectional, a 5-tick drop is NOT a short signal.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    // Price drops exactly 5 ticks then stays flat (within ±window).
    std::vector<float> deltas;
    for (int i = 0; i < 5; ++i) deltas.push_back(-TICK);  // -5 ticks
    for (int i = 0; i < 30; ++i) deltas.push_back(0.0f);  // flat
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // -5 ticks is NOT enough for short target (which needs -10 ticks).
    // Long race: stop hit at -5 → long not triggered
    // Short race: only -5, needs -10 → short not triggered (and long stop=+5 not hit either since price went down)
    // Actually: short race checks if price drops target_ticks (10) before rising stop_ticks (5).
    // Price drops 5 then flat. Short target (-10) not reached. Short stop (+5 from entry) not reached since price only goes DOWN.
    // → neither triggered → label=0
    EXPECT_FALSE(result.short_triggered)
        << "5-tick drop must NOT trigger short race (needs 10 ticks)";
    EXPECT_EQ(result.label, 0)
        << "5-tick drop should be HOLD under bidirectional, not -1";
}

// ===========================================================================
// T4: Neither Race Triggers — Expiry → HOLD
// Spec §T4: price moves less than stop_ticks in either direction
// ===========================================================================
class T4_NeitherTriggers : public ::testing::Test {};

TEST_F(T4_NeitherTriggers, NeitherRaceTriggersGivesHold) {
    // Price barely moves (< stop_ticks in either direction) before volume expiry.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 500;

    // Price drifts +2 ticks (< stop=5 and << target=10) over 10 bars × 50 vol = 500 expiry
    std::vector<float> deltas;
    for (int i = 0; i < 2; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 13; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 0)
        << "Neither race triggering should produce HOLD (label=0)";
    EXPECT_FALSE(result.both_triggered);
    EXPECT_FALSE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
}

TEST_F(T4_NeitherTriggers, NeitherExitTypeIsNeither) {
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 500;

    std::vector<float> deltas;
    for (int i = 0; i < 2; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 13; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // Spec §Parquet Schema: exit_type "neither" when neither race triggers
    EXPECT_EQ(result.exit_type, "neither");
}

TEST_F(T4_NeitherTriggers, FlatPriceGivesHold) {
    // Completely flat price path
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 500;

    std::vector<float> deltas(15, 0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 0);
    EXPECT_FALSE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
    EXPECT_FALSE(result.both_triggered);
}

// ===========================================================================
// T5 (revised): Long Stopped, Short Not Reached — Both Fail → HOLD
// Spec §T5: 5-tick drop is NOT a short signal under bidirectional
// ===========================================================================
class T5_LongStoppedShortNotReached : public ::testing::Test {};

TEST_F(T5_LongStoppedShortNotReached, FiveTickDropThenRecoveryGivesHold) {
    // Spec §T5 revised: price drops stop_ticks (5) then rises
    // Long race: stop hit at -5 → long target NOT triggered
    // Short race: price dropped 5 (not 10) then rose → short stop hit at +5 → short target NOT triggered
    // Expected: label=0 (neither triggered)
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    // Drop 5 ticks
    for (int i = 0; i < 5; ++i) deltas.push_back(-TICK);
    // Rise 10 ticks from the low (= +5 from entry)
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // Long race: price drops 5 ticks → long stop hit at bar 5 → long NOT triggered
    EXPECT_FALSE(result.long_triggered)
        << "Long race should fail: stop hit at -5 ticks before target (+10)";

    // Short race: price drops 5 ticks (not enough for -10 target), then rises.
    // At some point price rises to +5 from entry → short stop (+5) hit → short NOT triggered
    EXPECT_FALSE(result.short_triggered)
        << "Short race should fail: only dropped 5 ticks (need 10), then stop hit";

    EXPECT_EQ(result.label, 0)
        << "Old -1 label must become 0 under bidirectional (5-tick drop insufficient)";
    EXPECT_FALSE(result.both_triggered);
}

TEST_F(T5_LongStoppedShortNotReached, OldLabelWouldBeMinus1) {
    // Verify the old (unidirectional) labeling gives -1 for the same scenario,
    // confirming this is a behavioral change.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 5; ++i) deltas.push_back(-TICK);
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    // Old function: compute_tb_label (long-perspective only)
    TripleBarrierConfig old_cfg{};
    old_cfg.target_ticks = 10;
    old_cfg.stop_ticks = 5;
    old_cfg.volume_horizon = 5000;
    old_cfg.tick_size = 0.25f;

    auto old_result = compute_tb_label(bars, 0, old_cfg);

    // Under old labeling: price drops 5 ticks → stop hit → label = -1
    EXPECT_EQ(old_result.label, -1)
        << "Old (long-perspective) labeling should give -1 for 5-tick drop";
}

// ===========================================================================
// T6: Short Signal Requires Full Target Move
// Spec §T6: price falls exactly target_ticks (10) before rising stop_ticks (5)
// ===========================================================================
class T6_ShortFullTarget : public ::testing::Test {};

TEST_F(T6_ShortFullTarget, FullTargetDropGivesShortLabel) {
    // Price drops exactly 10 ticks before rising 5 ticks.
    // Long race: stop hit at -5 ticks → not triggered
    // Short race: target hit at -10 ticks → triggered
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);  // -10 ticks
    for (int i = 0; i < 10; ++i) deltas.push_back(0.0f);   // flat after
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, -1)
        << "Full 10-tick drop should produce short label (-1)";
    EXPECT_TRUE(result.short_triggered)
        << "Short race should trigger on 10-tick drop";
    EXPECT_FALSE(result.long_triggered)
        << "Long race should NOT trigger (stop hit at -5)";
    EXPECT_FALSE(result.both_triggered);
}

TEST_F(T6_ShortFullTarget, ExactBoundaryShortTarget) {
    // Price drops exactly target_ticks * tick_size = 10 * 0.25 = 2.50 points
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 5000;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, -1);
    EXPECT_TRUE(result.short_triggered);
}

// ===========================================================================
// T7: Parameterized Geometry — Non-Default Target/Stop
// Spec §T7: target=15, stop=3
// ===========================================================================
class T7_ParameterizedGeometry : public ::testing::Test {};

TEST_F(T7_ParameterizedGeometry, NonDefaultGeometryLongWins) {
    // target=15, stop=3. Long race checks +15/-3, short checks -15/+3.
    // Price rises 15 ticks without dropping 3.
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 15;
    cfg.stop_ticks = 3;
    cfg.volume_horizon = 5000;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = true;

    std::vector<float> deltas;
    for (int i = 0; i < 20; ++i) deltas.push_back(TICK);  // +20 ticks
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 1) << "Long should win with 15-tick target hit";
    EXPECT_TRUE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered)
        << "Short needs -15 ticks, price only went up";
}

TEST_F(T7_ParameterizedGeometry, NonDefaultGeometryShortWins) {
    // target=15, stop=3. Price drops 15 ticks without rising 3.
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 15;
    cfg.stop_ticks = 3;
    cfg.volume_horizon = 5000;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = true;

    std::vector<float> deltas;
    for (int i = 0; i < 20; ++i) deltas.push_back(-TICK);  // -20 ticks
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, -1) << "Short should win with 15-tick target hit";
    EXPECT_FALSE(result.long_triggered)
        << "Long needs +15 ticks, price only went down";
    EXPECT_TRUE(result.short_triggered);
}

TEST_F(T7_ParameterizedGeometry, TightStopBothStopped) {
    // target=15, stop=3. Price oscillates ±4 ticks.
    // Both races get stopped (long stop at -3, short stop at +3).
    // Neither target reached → label=0.
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 15;
    cfg.stop_ticks = 3;
    cfg.volume_horizon = 5000;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = true;

    // Price rises 4 ticks (stops short race at +3) then drops 8 (stops long race at -3)
    std::vector<float> deltas;
    for (int i = 0; i < 4; ++i) deltas.push_back(TICK);   // +4 ticks → short stop triggered
    for (int i = 0; i < 8; ++i) deltas.push_back(-TICK);  // -4 from entry → long stop triggered
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 0) << "Both stops hit → neither wins → HOLD";
    EXPECT_FALSE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
    EXPECT_FALSE(result.both_triggered);
}

// ===========================================================================
// T8: Expiry Behavior — Volume and Time Barriers
// Spec §T8: price stays within ±(stop-1) ticks for entire window
// ===========================================================================
class T8_ExpiryBehavior : public ::testing::Test {};

TEST_F(T8_ExpiryBehavior, VolumeExpiryWithSmallMoveGivesHold) {
    // Price stays within ±(5-1)=±4 ticks. Volume horizon reached.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 500;

    // Move +3 ticks (< stop=5 and << target=10), then stay flat
    // 10 bars × 50 vol = 500 → volume expiry
    std::vector<float> deltas;
    for (int i = 0; i < 3; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 12; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 0)
        << "Volume expiry with small price move should give HOLD";
    EXPECT_FALSE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
}

TEST_F(T8_ExpiryBehavior, TimeExpiryWithSmallMoveGivesHold) {
    // Time expires before volume horizon or price barriers.
    auto cfg = make_bidir_config();
    cfg.max_time_horizon_s = 30;  // short time cap
    cfg.volume_horizon = 50000;   // won't hit

    // +2 ticks drift in 10 bars × 5s = 50s > 30s cap. Low volume.
    std::vector<float> deltas;
    for (int i = 0; i < 2; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 8; ++i) deltas.push_back(0.0f);
    std::vector<uint32_t> volumes(11, 1);
    auto bars = make_bar_sequence(4500.0f, deltas, volumes, 5.0f);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.label, 0)
        << "Time expiry with small move should give HOLD";
    EXPECT_FALSE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
}

TEST_F(T8_ExpiryBehavior, ExpiryReportedForBothRaces) {
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 500;

    std::vector<float> deltas(15, 0.0f);  // completely flat
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(result.exit_type, "neither")
        << "Exit type should be 'neither' when no race triggers before expiry";
}

// ===========================================================================
// T9: Both-Triggered Frequency Diagnostic
// Spec §T9: diagnostic column counts high-volatility bars
// ===========================================================================
class T9_BothTriggeredDiagnostic : public ::testing::Test {};

TEST_F(T9_BothTriggeredDiagnostic, BothTriggeredRateOnKnownData) {
    // Create 10 scenarios: 5 where both races trigger, 5 where neither does.
    // Compute both-triggered rate = 50%.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;  // won't expire early

    int both_count = 0;
    int total = 0;

    // 5 bars with both-triggered: price rises +target then falls to -target
    for (int k = 0; k < 5; ++k) {
        std::vector<float> deltas;
        for (int i = 0; i < 10; ++i) deltas.push_back(TICK);   // +10 = target
        for (int i = 0; i < 20; ++i) deltas.push_back(-TICK);  // -10 from entry = short target
        for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);
        auto bars = make_bar_path(4500.0f + k * 10.0f, deltas, 10);

        auto result = compute_bidirectional_tb_label(bars, 0, cfg);
        if (result.both_triggered) both_count++;
        total++;
    }

    // 5 bars where neither triggers: flat price
    for (int k = 0; k < 5; ++k) {
        std::vector<float> deltas(30, 0.0f);
        auto bars = make_bar_path(4500.0f + k * 10.0f, deltas, 10);

        auto result = compute_bidirectional_tb_label(bars, 0, cfg);
        if (result.both_triggered) both_count++;
        total++;
    }

    EXPECT_EQ(total, 10);
    EXPECT_EQ(both_count, 5)
        << "Expect 5/10 = 50% both-triggered rate";

    float rate = static_cast<float>(both_count) / static_cast<float>(total);
    EXPECT_FLOAT_EQ(rate, 0.5f);
}

// ===========================================================================
// T10: Backward Compatibility — Old Mode Still Works
// Spec §T10: bidirectional=false reproduces old-style labels exactly
// ===========================================================================
class T10_BackwardCompatibility : public ::testing::Test {};

TEST_F(T10_BackwardCompatibility, OldModeReproducesLongLabel) {
    // bidirectional=false should produce identical results to compute_tb_label()
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.volume_horizon = 5000;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = false;  // OLD mode

    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto old_result = compute_tb_label(bars, 0, cfg);
    auto new_result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(new_result.label, old_result.label)
        << "bidirectional=false must match compute_tb_label exactly";
    EXPECT_EQ(new_result.exit_type, old_result.exit_type);
    EXPECT_EQ(new_result.bars_held, old_result.bars_held);
}

TEST_F(T10_BackwardCompatibility, OldModeReproducesShortLabel) {
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.volume_horizon = 5000;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = false;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto old_result = compute_tb_label(bars, 0, cfg);
    auto new_result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(new_result.label, old_result.label);
    EXPECT_EQ(new_result.exit_type, old_result.exit_type);
    EXPECT_EQ(new_result.bars_held, old_result.bars_held);
}

TEST_F(T10_BackwardCompatibility, OldModeReproducesExpiryHold) {
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.volume_horizon = 500;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = false;

    // Expiry with small return → HOLD
    std::vector<float> deltas;
    deltas.push_back(TICK);  // +1 tick < min_return
    for (int i = 0; i < 14; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    auto old_result = compute_tb_label(bars, 0, cfg);
    auto new_result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(new_result.label, old_result.label);
    EXPECT_EQ(new_result.exit_type, old_result.exit_type);
    EXPECT_EQ(new_result.bars_held, old_result.bars_held);
}

TEST_F(T10_BackwardCompatibility, OldModeReproducesTimeout) {
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 100;
    cfg.stop_ticks = 100;
    cfg.volume_horizon = 50000;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 30;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = false;

    std::vector<float> deltas;
    deltas.push_back(TICK);
    for (int i = 0; i < 9; ++i) deltas.push_back(0.0f);
    std::vector<uint32_t> volumes(11, 1);
    auto bars = make_bar_sequence(4500.0f, deltas, volumes, 5.0f);

    auto old_result = compute_tb_label(bars, 0, cfg);
    auto new_result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_EQ(new_result.label, old_result.label);
    EXPECT_EQ(new_result.exit_type, old_result.exit_type);
    EXPECT_EQ(new_result.bars_held, old_result.bars_held);
}

TEST_F(T10_BackwardCompatibility, OldModeOverManyBarsMatchesExactly) {
    // Run both old and new (bidirectional=false) over many bars, verify all match.
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.volume_horizon = 500;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = false;

    // Mixed price path
    std::vector<float> deltas;
    for (int i = 0; i < 20; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 30; ++i) deltas.push_back(0.0f);
    for (int i = 0; i < 15; ++i) deltas.push_back(-TICK);
    for (int i = 0; i < 30; ++i) deltas.push_back(0.0f);
    for (int i = 0; i < 20; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    for (int i = 0; i < static_cast<int>(bars.size()) - 20; ++i) {
        auto old_result = compute_tb_label(bars, i, cfg);
        auto new_result = compute_bidirectional_tb_label(bars, i, cfg);

        EXPECT_EQ(new_result.label, old_result.label)
            << "Label mismatch at bar " << i;
        EXPECT_EQ(new_result.bars_held, old_result.bars_held)
            << "bars_held mismatch at bar " << i;
    }
}

// ===========================================================================
// Independence of Races — races see the same price sequence independently
// ===========================================================================
class RaceIndependence : public ::testing::Test {};

TEST_F(RaceIndependence, LongRaceDoesNotConsumeShortPath) {
    // Verify both races observe the same price sequence.
    // Price rises 10 ticks (long target hit) then falls 15 ticks (net -5 from entry).
    // Long race: triggered at +10.
    // Short race: also sees the -10 from entry (since it crossed through entry downward).
    // But: short race has its OWN barrier check. The short stop is at +stop_ticks (5) from entry.
    // The price rose +10 first → short stop was hit at +5 → short race LOST.
    // So only long wins.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);   // +10 ticks → long target
    for (int i = 0; i < 25; ++i) deltas.push_back(-TICK);  // -15 from high → -5 from entry
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_TRUE(result.long_triggered)
        << "Long race should trigger at +10";
    EXPECT_FALSE(result.short_triggered)
        << "Short race should NOT trigger: stop (+5) was hit before target (-10)";
    EXPECT_EQ(result.label, 1);
}

TEST_F(RaceIndependence, ShortRaceDoesNotConsumeeLongPath) {
    // Mirror: price drops 10 (short target) then rises 15 (+5 from entry).
    // Short race: triggered at -10.
    // Long race: long stop at -5 was hit before long target (+10) → long lost.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);  // -10 ticks → short target
    for (int i = 0; i < 25; ++i) deltas.push_back(TICK);   // +15 from low → +5 from entry
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_FALSE(result.long_triggered)
        << "Long race: stop (-5) hit before target (+10)";
    EXPECT_TRUE(result.short_triggered)
        << "Short race should trigger at -10";
    EXPECT_EQ(result.label, -1);
}

TEST_F(RaceIndependence, BothRacesSeeFullPriceSequence) {
    // Price oscillates: +10 then -20 then +15
    // Long race: +10 hit → triggered
    // Short race: price drops to -10 from entry (after the +10 rise then -20 fall) →
    //   but short stop (+5) was already hit when price rose to +5, so short lost.
    // Wait — let me reconsider. The races are evaluated independently. Each scans the
    // SAME price sequence from entry forward. They don't interact.
    // Short race: scan from entry forward. The first thing that happens is price rises.
    // Short race checks: did price hit -target (10) before +stop (5)?
    // Price rises +1, +2, +3, +4, +5 → short STOP HIT at +5 → short NOT triggered.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);   // rise → short stop at +5
    for (int i = 0; i < 20; ++i) deltas.push_back(-TICK);  // fall through
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);   // rise back
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_TRUE(result.long_triggered)
        << "Long race: target (+10) hit at bar 10";
    EXPECT_FALSE(result.short_triggered)
        << "Short race: stop (+5) hit at bar 5, before target (-10) could be reached";
    EXPECT_EQ(result.label, 1);
}

// ===========================================================================
// Edge Cases
// ===========================================================================
class BidirectionalEdgeCases : public ::testing::Test {};

TEST_F(BidirectionalEdgeCases, EntryAtLastBarGivesHold) {
    auto cfg = make_bidir_config();
    auto bars = make_bar_series(4500.0f, 4501.0f, 10, 50);

    auto result = compute_bidirectional_tb_label(bars, 9, cfg);

    EXPECT_EQ(result.label, 0)
        << "Entry at last bar with no forward data should be HOLD";
    EXPECT_EQ(result.bars_held, 0);
    EXPECT_FALSE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
    EXPECT_FALSE(result.both_triggered);
}

TEST_F(BidirectionalEdgeCases, SingleForwardBarNoTrigger) {
    auto cfg = make_bidir_config();
    auto bars = make_bar_series(4500.0f, 4500.5f, 2, 50);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // 2 ticks move, neither target (10) nor stop (5) hit
    EXPECT_TRUE(result.label == -1 || result.label == 0 || result.label == 1);
    EXPECT_FALSE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
}

TEST_F(BidirectionalEdgeCases, SymmetricStopTicksEqualsTargetTicks) {
    // When stop_ticks == target_ticks, geometry is symmetric.
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 10;  // equal to target
    cfg.volume_horizon = 50000;
    cfg.min_return_ticks = 2;
    cfg.max_time_horizon_s = 300;
    cfg.tick_size = 0.25f;
    cfg.bidirectional = true;

    // Price rises 10 → both long target and short stop hit simultaneously at the same bar
    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // Long race: target (+10) hit → triggered
    // Short race: stop (+10) hit → NOT triggered (stop before target)
    EXPECT_TRUE(result.long_triggered);
    EXPECT_FALSE(result.short_triggered);
    EXPECT_EQ(result.label, 1);
}

TEST_F(BidirectionalEdgeCases, LabelAlwaysValidAcrossManyBars) {
    // Over a synthetic price path, verify every bidirectional label is valid.
    auto cfg = make_bidir_config();

    std::vector<float> deltas;
    for (int i = 0; i < 20; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 30; ++i) deltas.push_back(0.0f);
    for (int i = 0; i < 15; ++i) deltas.push_back(-TICK);
    for (int i = 0; i < 30; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 50);

    for (int i = 0; i < static_cast<int>(bars.size()) - 1; ++i) {
        auto result = compute_bidirectional_tb_label(bars, i, cfg);

        EXPECT_TRUE(result.label == -1 || result.label == 0 || result.label == 1)
            << "Invalid label " << result.label << " at bar " << i;

        EXPECT_TRUE(result.exit_type == "long_target" ||
                     result.exit_type == "short_target" ||
                     result.exit_type == "both" ||
                     result.exit_type == "neither" ||
                     result.exit_type == "long_expiry" ||
                     result.exit_type == "short_expiry" ||
                     result.exit_type == "timeout" ||
                     result.exit_type == "expiry" ||
                     // backward compat exit types
                     result.exit_type == "target" ||
                     result.exit_type == "stop")
            << "Invalid exit_type '" << result.exit_type << "' at bar " << i;

        EXPECT_GE(result.bars_held, 0) << "bars_held must be >= 0 at bar " << i;

        // Consistency: if both_triggered, then both long and short must be triggered
        if (result.both_triggered) {
            EXPECT_TRUE(result.long_triggered)
                << "both_triggered=true but long_triggered=false at bar " << i;
            EXPECT_TRUE(result.short_triggered)
                << "both_triggered=true but short_triggered=false at bar " << i;
        }

        // Consistency: label logic matches triggered flags
        if (result.long_triggered && !result.short_triggered) {
            EXPECT_EQ(result.label, 1)
                << "Only long triggered → label must be +1 at bar " << i;
        }
        if (result.short_triggered && !result.long_triggered) {
            EXPECT_EQ(result.label, -1)
                << "Only short triggered → label must be -1 at bar " << i;
        }
        if (result.both_triggered) {
            EXPECT_EQ(result.label, 0)
                << "Both triggered → label must be 0 at bar " << i;
        }
        if (!result.long_triggered && !result.short_triggered) {
            EXPECT_EQ(result.label, 0)
                << "Neither triggered → label must be 0 at bar " << i;
        }
    }
}

TEST_F(BidirectionalEdgeCases, BarsHeldConsistentWithTrigger) {
    // When a race triggers, bars_held should be positive.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    std::vector<float> deltas;
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_GT(result.bars_held, 0)
        << "When a race triggers, bars_held must be positive";
    // Specifically, target hit at bar 10 (10 ticks × 1 tick/bar)
    EXPECT_EQ(result.bars_held, 10);
}

TEST_F(BidirectionalEdgeCases, MiddleBarEntryPointWorks) {
    // compute_bidirectional_tb_label at idx=5
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    std::vector<float> deltas;
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);       // flat first 5
    for (int i = 0; i < 15; ++i) deltas.push_back(TICK);       // +15 ticks from bar 5
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 5, cfg);

    EXPECT_EQ(result.label, 1);
    EXPECT_TRUE(result.long_triggered);
    EXPECT_EQ(result.bars_held, 10);
}

// ===========================================================================
// Function existence and independence
// ===========================================================================
class FunctionContract : public ::testing::Test {};

TEST_F(FunctionContract, ComputeBidirectionalTBLabelExists) {
    // Verify the function exists and is callable.
    auto cfg = make_bidir_config();
    auto bars = make_bar_series(4500.0f, 4501.0f, 10, 50);

    // This should compile if the function exists
    BidirectionalTBResult result = compute_bidirectional_tb_label(bars, 0, cfg);
    (void)result;  // suppress unused warning
}

TEST_F(FunctionContract, OldComputeTbLabelStillWorks) {
    // Spec: compute_bidirectional_tb_label() is independent of compute_tb_label()
    // Both functions must coexist without shared mutable state.
    TripleBarrierConfig cfg{};
    cfg.target_ticks = 10;
    cfg.stop_ticks = 5;
    cfg.volume_horizon = 500;
    cfg.tick_size = 0.25f;

    auto bars = make_bar_series(4500.0f, 4503.0f, 30, 50);

    // Call both in sequence — neither should affect the other
    auto old1 = compute_tb_label(bars, 0, cfg);
    cfg.bidirectional = true;
    auto bidir = compute_bidirectional_tb_label(bars, 0, cfg);
    cfg.bidirectional = false;
    auto old2 = compute_tb_label(bars, 0, cfg);

    // Old results should be identical regardless of bidirectional calls between them
    EXPECT_EQ(old1.label, old2.label);
    EXPECT_EQ(old1.exit_type, old2.exit_type);
    EXPECT_EQ(old1.bars_held, old2.bars_held);
}

// ===========================================================================
// Parquet column contract (spec §Parquet Schema Changes)
// ===========================================================================
class ParquetSchemaContract : public ::testing::Test {};

TEST_F(ParquetSchemaContract, NewColumnsInSchema) {
    // The Parquet export should include these new columns.
    // We test this by verifying the result struct has the right fields,
    // since the Parquet writer reads from this struct.

    BidirectionalTBResult result{};

    // These fields MUST exist for the Parquet writer to produce the right columns:
    // tb_both_triggered (float64: 0.0 or 1.0)
    EXPECT_EQ(result.both_triggered, false);

    // tb_long_triggered (float64: 0.0 or 1.0)
    EXPECT_EQ(result.long_triggered, false);

    // tb_short_triggered (float64: 0.0 or 1.0)
    EXPECT_EQ(result.short_triggered, false);

    // Existing fields still present:
    EXPECT_EQ(result.label, 0);
    EXPECT_TRUE(result.exit_type.empty());
    EXPECT_EQ(result.bars_held, 0);
}

// ===========================================================================
// Asymmetric barrier geometry — target > stop (the normal case)
// ===========================================================================
class AsymmetricBarriers : public ::testing::Test {};

TEST_F(AsymmetricBarriers, LongRaceBarriersAreCorrect) {
    // Long race: target = +target_ticks * tick_size, stop = -stop_ticks * tick_size
    // With target=10, stop=5: long target = +2.50, long stop = -1.25
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    // Price rises 2.50 (10 ticks) → long target hit
    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(TICK);
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);
    EXPECT_TRUE(result.long_triggered);
    EXPECT_EQ(result.bars_held, 10);
}

TEST_F(AsymmetricBarriers, ShortRaceBarriersAreMirrored) {
    // Short race: target = -target_ticks * tick_size, stop = +stop_ticks * tick_size
    // With target=10, stop=5: short target = -2.50, short stop = +1.25
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    // Price drops 2.50 (10 ticks) → short target hit
    std::vector<float> deltas;
    for (int i = 0; i < 10; ++i) deltas.push_back(-TICK);
    for (int i = 0; i < 5; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);
    EXPECT_TRUE(result.short_triggered);
    EXPECT_EQ(result.bars_held, 10);
}

TEST_F(AsymmetricBarriers, ShortStopIsAtPlusStopTicks) {
    // Short race stop = +stop_ticks (5) from entry.
    // Price rises 5 ticks without dropping 10 → short STOPPED (not triggered).
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    std::vector<float> deltas;
    for (int i = 0; i < 6; ++i) deltas.push_back(TICK);  // +6 ticks → past short stop (+5)
    for (int i = 0; i < 20; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    // Short race: stop (+5) hit at bar 5, target (-10) never reached
    EXPECT_FALSE(result.short_triggered)
        << "Short race should be stopped at +5 ticks from entry";
}

TEST_F(AsymmetricBarriers, LongStopIsAtMinusStopTicks) {
    // Long race stop = -stop_ticks (5) from entry.
    // Price drops 5 ticks without rising 10 → long STOPPED.
    auto cfg = make_bidir_config();
    cfg.volume_horizon = 50000;

    std::vector<float> deltas;
    for (int i = 0; i < 6; ++i) deltas.push_back(-TICK);  // -6 ticks → past long stop (-5)
    for (int i = 0; i < 20; ++i) deltas.push_back(0.0f);
    auto bars = make_bar_path(4500.0f, deltas, 10);

    auto result = compute_bidirectional_tb_label(bars, 0, cfg);

    EXPECT_FALSE(result.long_triggered)
        << "Long race should be stopped at -5 ticks from entry";
}
