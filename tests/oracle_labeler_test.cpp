// oracle_labeler_test.cpp — TDD RED phase tests for oracle_labeler
// Spec: .kit/docs/oracle-trajectory.md
//
// Tests the oracle_label() function that generates ground-truth action labels
// using future price data (lookahead oracle). The function is stateless per-call
// and returns an action from {0=HOLD, 1=ENTER_LONG, 2=ENTER_SHORT, 3=EXIT}.
// Action 4 (REVERSE) is never generated.

#include <gtest/gtest.h>
#include "oracle_labeler.hpp"
#include "book_builder.hpp"  // BookSnapshot struct

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers — synthetic snapshot construction for oracle tests
// ---------------------------------------------------------------------------
namespace {

constexpr float TICK = 0.25f;
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

// Build a vector of snapshots with linearly interpolated mid_price.
// mid_price[0] = start_mid, mid_price[N-1] = end_mid, linearly spaced.
// All other fields are set to reasonable defaults.
std::vector<BookSnapshot> make_price_series(int count, float start_mid,
                                             float end_mid) {
    std::vector<BookSnapshot> snaps(count);
    for (int i = 0; i < count; ++i) {
        float frac = (count > 1) ? static_cast<float>(i) / (count - 1) : 0.0f;
        float mid = start_mid + frac * (end_mid - start_mid);
        snaps[i].mid_price = mid;
        snaps[i].spread = TICK;
        snaps[i].bids[0][0] = mid - TICK / 2.0f;
        snaps[i].bids[0][1] = 10.0f;
        snaps[i].asks[0][0] = mid + TICK / 2.0f;
        snaps[i].asks[0][1] = 10.0f;
        snaps[i].time_of_day = 9.5f;
        snaps[i].timestamp = static_cast<uint64_t>(i) * 100'000'000ULL;
    }
    return snaps;
}

// Build snapshots where mid_price follows an explicit path of per-step deltas.
// mid_price[0] = start_mid, mid_price[k] = mid_price[k-1] + deltas[k-1].
// Extra snapshots beyond the deltas array hold the last price (flat).
std::vector<BookSnapshot> make_price_path(float start_mid,
                                           const std::vector<float>& deltas,
                                           int total_count) {
    std::vector<BookSnapshot> snaps(total_count);
    float mid = start_mid;
    for (int i = 0; i < total_count; ++i) {
        if (i > 0 && i - 1 < static_cast<int>(deltas.size())) {
            mid += deltas[i - 1];
        }
        snaps[i].mid_price = mid;
        snaps[i].spread = TICK;
        snaps[i].bids[0][0] = mid - TICK / 2.0f;
        snaps[i].bids[0][1] = 10.0f;
        snaps[i].asks[0][0] = mid + TICK / 2.0f;
        snaps[i].asks[0][1] = 10.0f;
        snaps[i].time_of_day = 9.5f;
        snaps[i].timestamp = static_cast<uint64_t>(i) * 100'000'000ULL;
    }
    return snaps;
}

// Build snapshots with constant mid_price
std::vector<BookSnapshot> make_flat_series(int count, float mid = 4500.0f) {
    return make_price_series(count, mid, mid);
}

}  // anonymous namespace

// ===========================================================================
// Test 1: Flat + price rises to target → ENTER_LONG
// ===========================================================================
TEST(OracleLabelTest, FlatPriceRisesToTarget_EnterLong) {
    // With default params: target_ticks=10, stop_ticks=5, tick=0.25
    // target threshold = 10 * 0.25 = 2.50 points
    // stop threshold   = 5 * 0.25  = 1.25 points
    //
    // Create a series where price rises by 2.50 within horizon=100.
    // mid[0]=4500.00, mid rises to 4502.50 (=+10 ticks) at snapshot ~50.
    // Price never drops below -1.25 (=-5 ticks) from start.
    float start = 4500.0f;
    float target_price = start + 10 * TICK;  // 4502.50

    // Build snapshots: t=0 at start, straight line up to target at t=50,
    // then stay there until horizon=100.
    int total = 200;  // enough for t=0 + horizon=100
    std::vector<float> deltas;
    // Rise by target_ticks * tick_size over 50 steps
    float step = (target_price - start) / 50.0f;
    for (int i = 0; i < 50; ++i) deltas.push_back(step);
    // Stay flat for the rest
    auto snaps = make_price_path(start, deltas, total);

    int result = oracle_label(snaps, /*t=*/0, /*position_state=*/0,
                              /*entry_price=*/NaN);

    EXPECT_EQ(result, 1) << "Flat + price rises to target → ENTER_LONG (1)";
}

// ===========================================================================
// Test 2: Flat + price falls to target → ENTER_SHORT
// ===========================================================================
TEST(OracleLabelTest, FlatPriceFallsToTarget_EnterShort) {
    float start = 4500.0f;
    float target_price = start - 10 * TICK;  // 4497.50

    int total = 200;
    std::vector<float> deltas;
    float step = (target_price - start) / 50.0f;
    for (int i = 0; i < 50; ++i) deltas.push_back(step);
    auto snaps = make_price_path(start, deltas, total);

    int result = oracle_label(snaps, /*t=*/0, /*position_state=*/0,
                              /*entry_price=*/NaN);

    EXPECT_EQ(result, 2) << "Flat + price falls to target → ENTER_SHORT (2)";
}

// ===========================================================================
// Test 3: Flat + price hits stop before target → HOLD
// ===========================================================================
TEST(OracleLabelTest, FlatPriceHitsStopBeforeTarget_Hold) {
    // Price drops by stop_ticks (5 ticks = 1.25 points) before rising to
    // target_ticks (10 ticks = 2.50 points).
    float start = 4500.0f;
    int total = 200;

    std::vector<float> deltas;
    // Drop by 1.25 points over 10 steps (hits -stop_ticks for upward target)
    float drop_step = -5.0f * TICK / 10.0f;
    for (int i = 0; i < 10; ++i) deltas.push_back(drop_step);
    // Then rise sharply (but too late — stop already hit)
    float rise_step = 15.0f * TICK / 40.0f;
    for (int i = 0; i < 40; ++i) deltas.push_back(rise_step);

    auto snaps = make_price_path(start, deltas, total);

    int result = oracle_label(snaps, 0, 0, NaN);

    EXPECT_EQ(result, 0) << "Flat + stop hit before target → HOLD (0)";
}

// ===========================================================================
// Test 4: Flat + no threshold hit → HOLD
// ===========================================================================
TEST(OracleLabelTest, FlatNoThresholdHit_Hold) {
    // Price stays within both thresholds for the entire horizon.
    // Neither ±target_ticks nor ±stop_ticks reached.
    float start = 4500.0f;
    int total = 200;

    // Oscillate within ±3 ticks (0.75 points) — below stop_ticks (5 ticks)
    std::vector<float> deltas;
    for (int i = 0; i < 100; ++i) {
        float d = 3.0f * TICK * std::sin(2.0f * 3.14159f * i / 20.0f) / 20.0f;
        deltas.push_back(d);
    }
    auto snaps = make_price_path(start, deltas, total);

    // Verify we didn't accidentally cross any threshold
    for (int k = 1; k <= 100; ++k) {
        float delta_ticks = (snaps[k].mid_price - start) / TICK;
        ASSERT_LT(std::abs(delta_ticks), 5.0f)
            << "Test setup error: price moved too much at step " << k;
    }

    int result = oracle_label(snaps, 0, 0, NaN);

    EXPECT_EQ(result, 0) << "Flat + no threshold hit → HOLD (0)";
}

// ===========================================================================
// Test 5: Long + take profit hit → EXIT
// ===========================================================================
TEST(OracleLabelTest, LongTakeProfitHit_Exit) {
    // position_state = +1 (long), entry_price = 4500.0
    // take_profit_ticks = 20 → need future PnL >= 20 ticks = 5.0 points
    // future_pnl_ticks = +1 * (mid[t+k] - entry_price) / tick
    // Need mid[t+k] >= entry_price + 20*0.25 = 4505.0
    float entry = 4500.0f;
    float take_profit_price = entry + 20 * TICK;  // 4505.0

    int total = 200;
    std::vector<float> deltas;
    float step = (take_profit_price - entry) / 60.0f;
    for (int i = 0; i < 60; ++i) deltas.push_back(step);
    auto snaps = make_price_path(entry, deltas, total);

    int result = oracle_label(snaps, 0, /*position_state=*/1,
                              /*entry_price=*/entry);

    EXPECT_EQ(result, 3) << "Long + take profit hit → EXIT (3)";
}

// ===========================================================================
// Test 6: Long + stop loss hit → EXIT
// ===========================================================================
TEST(OracleLabelTest, LongStopLossHit_Exit) {
    // position_state = +1 (long), entry_price = 4500.0
    // stop_ticks = 5 → need future PnL <= -5 ticks = -1.25 points
    // Need mid[t+k] <= entry_price - 5*0.25 = 4498.75
    float entry = 4500.0f;
    float stop_price = entry - 5 * TICK;  // 4498.75

    int total = 200;
    std::vector<float> deltas;
    float step = (stop_price - entry) / 30.0f;
    for (int i = 0; i < 30; ++i) deltas.push_back(step);
    auto snaps = make_price_path(entry, deltas, total);

    int result = oracle_label(snaps, 0, 1, entry);

    EXPECT_EQ(result, 3) << "Long + stop loss hit → EXIT (3)";
}

// ===========================================================================
// Test 7: Long + hold (neither TP nor SL hit)
// ===========================================================================
TEST(OracleLabelTest, LongHold) {
    // Price stays between entry-stop and entry+take_profit for entire horizon.
    float entry = 4500.0f;
    int total = 200;

    // Price stays flat at entry — no PnL movement
    auto snaps = make_flat_series(total, entry);

    int result = oracle_label(snaps, 0, 1, entry);

    EXPECT_EQ(result, 0) << "Long + neither TP nor SL → HOLD (0)";
}

// ===========================================================================
// Test 8: Short + take profit hit → EXIT
// ===========================================================================
TEST(OracleLabelTest, ShortTakeProfitHit_Exit) {
    // position_state = -1 (short), entry_price = 4500.0
    // take_profit_ticks = 20
    // future_pnl_ticks = -1 * (mid[t+k] - entry_price) / tick
    // Need -1 * (mid[t+k] - 4500.0) / 0.25 >= 20
    // → mid[t+k] <= 4500.0 - 5.0 = 4495.0
    float entry = 4500.0f;
    float take_profit_price = entry - 20 * TICK;  // 4495.0

    int total = 200;
    std::vector<float> deltas;
    float step = (take_profit_price - entry) / 60.0f;
    for (int i = 0; i < 60; ++i) deltas.push_back(step);
    auto snaps = make_price_path(entry, deltas, total);

    int result = oracle_label(snaps, 0, /*position_state=*/-1,
                              /*entry_price=*/entry);

    EXPECT_EQ(result, 3) << "Short + take profit hit → EXIT (3)";
}

// ===========================================================================
// Test 9: Short + stop loss hit → EXIT
// ===========================================================================
TEST(OracleLabelTest, ShortStopLossHit_Exit) {
    // position_state = -1 (short), entry_price = 4500.0
    // stop_ticks = 5
    // future_pnl_ticks = -1 * (mid[t+k] - entry) / tick
    // Need future_pnl_ticks <= -5 → mid[t+k] >= entry + 5*0.25 = 4501.25
    float entry = 4500.0f;
    float stop_price = entry + 5 * TICK;  // 4501.25

    int total = 200;
    std::vector<float> deltas;
    float step = (stop_price - entry) / 30.0f;
    for (int i = 0; i < 30; ++i) deltas.push_back(step);
    auto snaps = make_price_path(entry, deltas, total);

    int result = oracle_label(snaps, 0, -1, entry);

    EXPECT_EQ(result, 3) << "Short + stop loss hit → EXIT (3)";
}

// ===========================================================================
// Test 10: Oracle never returns REVERSE (4)
// ===========================================================================
TEST(OracleLabelTest, NeverReturnsReverse) {
    // Test across all position states and various price movements.
    float start = 4500.0f;
    int total = 200;

    // Scenario A: flat, price rises
    auto snaps_up = make_price_series(total, start, start + 10 * TICK);
    EXPECT_NE(oracle_label(snaps_up, 0, 0, NaN), 4)
        << "REVERSE must never be returned (flat, rising)";

    // Scenario B: flat, price falls
    auto snaps_down = make_price_series(total, start, start - 10 * TICK);
    EXPECT_NE(oracle_label(snaps_down, 0, 0, NaN), 4)
        << "REVERSE must never be returned (flat, falling)";

    // Scenario C: long, price rises (take profit)
    auto snaps_long_tp = make_price_series(total, start, start + 20 * TICK);
    EXPECT_NE(oracle_label(snaps_long_tp, 0, 1, start), 4)
        << "REVERSE must never be returned (long, TP)";

    // Scenario D: long, price falls (stop)
    auto snaps_long_sl = make_price_series(total, start, start - 5 * TICK);
    EXPECT_NE(oracle_label(snaps_long_sl, 0, 1, start), 4)
        << "REVERSE must never be returned (long, SL)";

    // Scenario E: short, price falls (take profit)
    auto snaps_short_tp = make_price_series(total, start, start - 20 * TICK);
    EXPECT_NE(oracle_label(snaps_short_tp, 0, -1, start), 4)
        << "REVERSE must never be returned (short, TP)";

    // Scenario F: short, price rises (stop)
    auto snaps_short_sl = make_price_series(total, start, start + 5 * TICK);
    EXPECT_NE(oracle_label(snaps_short_sl, 0, -1, start), 4)
        << "REVERSE must never be returned (short, SL)";

    // Scenario G: flat, no movement
    auto snaps_flat = make_flat_series(total, start);
    EXPECT_NE(oracle_label(snaps_flat, 0, 0, NaN), 4)
        << "REVERSE must never be returned (flat, no movement)";
}

// ===========================================================================
// Test 10b: Oracle return values are always in {0, 1, 2, 3}
// ===========================================================================
TEST(OracleLabelTest, ReturnValueAlwaysInValidRange) {
    float start = 4500.0f;
    int total = 200;

    // Flat scenarios
    auto snaps_flat = make_flat_series(total, start);
    int r1 = oracle_label(snaps_flat, 0, 0, NaN);
    EXPECT_GE(r1, 0);
    EXPECT_LE(r1, 3);

    // Long hold
    int r2 = oracle_label(snaps_flat, 0, 1, start);
    EXPECT_GE(r2, 0);
    EXPECT_LE(r2, 3);

    // Short hold
    int r3 = oracle_label(snaps_flat, 0, -1, start);
    EXPECT_GE(r3, 0);
    EXPECT_LE(r3, 3);
}

// ===========================================================================
// Test 11: Precondition — flat requires NaN entry_price
// ===========================================================================
TEST(OracleLabelTest, PreconditionFlatRequiresNaNEntryPrice) {
    // If position_state == 0 but entry_price is a valid number, the function
    // should throw. This is a precondition violation.
    auto snaps = make_flat_series(200, 4500.0f);

    // Passing a valid entry_price while flat should throw.
    EXPECT_THROW(
        oracle_label(snaps, 0, /*position_state=*/0, /*entry_price=*/4500.0f),
        std::invalid_argument
    ) << "Flat position_state with valid entry_price should throw";
}

// ===========================================================================
// Test 12: Precondition — in-position requires valid entry_price
// ===========================================================================
TEST(OracleLabelTest, PreconditionInPositionRequiresValidEntryPrice) {
    auto snaps = make_flat_series(200, 4500.0f);

    // Long position with NaN entry_price
    EXPECT_THROW(
        oracle_label(snaps, 0, /*position_state=*/1, /*entry_price=*/NaN),
        std::invalid_argument
    ) << "Long position with NaN entry_price should throw";

    // Short position with NaN entry_price
    EXPECT_THROW(
        oracle_label(snaps, 0, /*position_state=*/-1, /*entry_price=*/NaN),
        std::invalid_argument
    ) << "Short position with NaN entry_price should throw";
}

// ===========================================================================
// Test 13: Horizon boundary — t + horizon <= snapshots.size()
// ===========================================================================
TEST(OracleLabelTest, HorizonBoundaryEnforced) {
    // With horizon=100 and 150 snapshots, t can be at most 49 (so t+horizon=149 < 150).
    // t=50 → t+horizon=150 = snapshots.size() → should be OK
    // t=51 with horizon=100: t+horizon=151 > 150 → should fail
    auto snaps = make_flat_series(150, 4500.0f);

    // This should throw — not enough data
    EXPECT_THROW(
        oracle_label(snaps, /*t=*/51, 0, NaN, /*horizon=*/100),
        std::invalid_argument
    ) << "t + horizon > snapshots.size() should throw";
}

// ===========================================================================
// Test: Post-condition — flat never gets EXIT
// ===========================================================================
TEST(OracleLabelTest, PostconditionFlatNeverReturnsExit) {
    // When position_state == 0, oracle should never return EXIT (3)
    float start = 4500.0f;
    int total = 200;

    // Try various price paths — none should return EXIT when flat
    auto snaps_up = make_price_series(total, start, start + 20 * TICK);
    EXPECT_NE(oracle_label(snaps_up, 0, 0, NaN), 3)
        << "Flat position should never return EXIT";

    auto snaps_down = make_price_series(total, start, start - 20 * TICK);
    EXPECT_NE(oracle_label(snaps_down, 0, 0, NaN), 3)
        << "Flat position should never return EXIT";

    auto snaps_flat = make_flat_series(total, start);
    EXPECT_NE(oracle_label(snaps_flat, 0, 0, NaN), 3)
        << "Flat position should never return EXIT";
}

// ===========================================================================
// Test: Post-condition — in-position never gets ENTER
// ===========================================================================
TEST(OracleLabelTest, PostconditionInPositionNeverReturnsEnter) {
    float start = 4500.0f;
    int total = 200;

    // Long position — should never return ENTER_LONG (1) or ENTER_SHORT (2)
    auto snaps = make_flat_series(total, start);
    int r1 = oracle_label(snaps, 0, 1, start);
    EXPECT_NE(r1, 1) << "Long position should never return ENTER_LONG";
    EXPECT_NE(r1, 2) << "Long position should never return ENTER_SHORT";

    // Short position — same constraint
    int r2 = oracle_label(snaps, 0, -1, start);
    EXPECT_NE(r2, 1) << "Short position should never return ENTER_LONG";
    EXPECT_NE(r2, 2) << "Short position should never return ENTER_SHORT";
}

// ===========================================================================
// Test: Non-zero t offset — oracle works from middle of snapshot array
// ===========================================================================
TEST(OracleLabelTest, NonZeroTOffset) {
    // t=50 means we start scanning from snapshot[51] through snapshot[150].
    // The price at t=50 is our reference (for flat: delta from mid[t]).
    float start = 4500.0f;
    int total = 200;

    // Price flat for first 50 snapshots, then rises to target after t=50
    std::vector<float> deltas;
    for (int i = 0; i < 50; ++i) deltas.push_back(0.0f);  // flat before t=50
    float step = 10.0f * TICK / 40.0f;
    for (int i = 0; i < 40; ++i) deltas.push_back(step);   // rise after t=50
    auto snaps = make_price_path(start, deltas, total);

    int result = oracle_label(snaps, /*t=*/50, 0, NaN);

    EXPECT_EQ(result, 1) << "Oracle at t=50 should see future rise → ENTER_LONG";
}

// ===========================================================================
// Test: Custom parameters — different horizon and tick thresholds
// ===========================================================================
TEST(OracleLabelTest, CustomParameters) {
    float start = 4500.0f;
    int total = 300;

    // Use horizon=200, target_ticks=5 (1.25 points), stop_ticks=3 (0.75 points)
    // Price rises by 5*0.25 = 1.25 points within 200 snapshots
    auto snaps = make_price_series(total, start, start + 5 * TICK);

    int result = oracle_label(snaps, 0, 0, NaN,
                              /*horizon=*/200,
                              /*target_ticks=*/5,
                              /*stop_ticks=*/3,
                              /*take_profit_ticks=*/10,
                              /*tick_size=*/0.25f);

    EXPECT_EQ(result, 1) << "Custom params: price rises to target → ENTER_LONG";
}

// ===========================================================================
// Test: Flat — price hits both stop and target simultaneously → first hit wins
// ===========================================================================
TEST(OracleLabelTest, FlatFirstThresholdHitWins) {
    // If price rises to exactly +target_ticks and -stop_ticks at the same
    // snapshot (impossible in practice but tests priority), we need to ensure
    // scanning is sequential and the first threshold encountered wins.

    // Price rises to +target_ticks at step 30 (earlier than stop)
    float start = 4500.0f;
    int total = 200;
    std::vector<float> deltas;
    float step = 10.0f * TICK / 30.0f;
    for (int i = 0; i < 30; ++i) deltas.push_back(step);
    // Then drop sharply past -stop_ticks
    float drop = -(10.0f + 5.0f) * TICK / 20.0f;
    for (int i = 0; i < 20; ++i) deltas.push_back(drop);
    auto snaps = make_price_path(start, deltas, total);

    int result = oracle_label(snaps, 0, 0, NaN);

    // Target hit at step 30, stop would hit later → ENTER_LONG
    EXPECT_EQ(result, 1) << "First threshold hit should win";
}

// ===========================================================================
// Test: Default parameter values match spec
// ===========================================================================
TEST(OracleLabelTest, DefaultParametersMatchSpec) {
    // Verify that the default parameters produce expected behavior.
    // Default: horizon=100, target_ticks=10, stop_ticks=5,
    //          take_profit_ticks=20, tick_size=0.25
    //
    // With target_ticks=10 and tick=0.25: threshold = 2.50 points
    // Price rises by exactly 2.50 in 50 steps → should trigger ENTER_LONG
    float start = 4500.0f;
    int total = 200;
    auto snaps = make_price_series(total, start, start + 10 * TICK);

    // Call with only required params (relying on defaults)
    int result = oracle_label(snaps, 0, 0, NaN);

    EXPECT_EQ(result, 1) << "Default params should produce ENTER_LONG for +10 tick rise";
}
