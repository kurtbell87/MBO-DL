// tick_bar_fix_test.cpp — TDD RED phase tests for tick bar construction fix
// Spec: .kit/docs/tick-bar-fix.md
//
// Verifies that tick bars count actual trade events via BookSnapshot::trade_count,
// not just counting snapshots. Also verifies regression safety for other bar types.
//
// RED STATE: These tests will fail to compile because BookSnapshot::trade_count
// does not exist yet. Once the field is added and TickBarBuilder uses it for
// cumulative trade counting, all tests should pass.

#include <gtest/gtest.h>

#include "bars/bar.hpp"
#include "bars/tick_bar_builder.hpp"
#include "bars/time_bar_builder.hpp"
#include "bars/dollar_bar_builder.hpp"
#include "bars/volume_bar_builder.hpp"
#include "bars/bar_factory.hpp"
#include "book_builder.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

// ===========================================================================
// Helpers — synthetic BookSnapshot construction with trade_count
// ===========================================================================
namespace {

constexpr uint64_t NS_PER_SEC  = 1'000'000'000ULL;
constexpr uint64_t NS_PER_HOUR = 3600ULL * NS_PER_SEC;
constexpr uint64_t MIDNIGHT_ET_NS = 1641186000ULL * NS_PER_SEC;  // 2022-01-03 00:00 ET
constexpr uint64_t RTH_OPEN_NS = MIDNIGHT_ET_NS + 9ULL * NS_PER_HOUR + 30ULL * 60 * NS_PER_SEC;
constexpr uint64_t BOUNDARY_100MS = 100'000'000ULL;

constexpr uint64_t rth_ts(uint64_t offset_ms) {
    return RTH_OPEN_NS + offset_ms * 1'000'000ULL;
}

// Build a BookSnapshot with a specified trade_count and optional trade data.
// trade_count: number of action='T' events in this 100ms interval.
// If trade_count > 0, the last trade slot is populated to indicate trade activity.
BookSnapshot make_snapshot_tc(uint64_t timestamp, uint32_t trade_count,
                              float trade_price = 4500.25f,
                              uint32_t trade_size_total = 0,
                              float mid_price = 4500.125f) {
    BookSnapshot snap{};
    snap.timestamp = timestamp;
    snap.mid_price = mid_price;
    snap.spread = 0.25f;
    snap.time_of_day = 9.5f;

    // NEW FIELD — will fail to compile until BookSnapshot gains trade_count
    snap.trade_count = trade_count;

    // Minimal book: 1 level per side
    snap.bids[0][0] = mid_price - 0.125f;
    snap.bids[0][1] = 10.0f;
    snap.asks[0][0] = mid_price + 0.125f;
    snap.asks[0][1] = 10.0f;

    // If trades occurred, populate the last trade slot
    if (trade_count > 0) {
        uint32_t size = (trade_size_total > 0) ? trade_size_total : trade_count;
        snap.trades[TRADE_BUF_LEN - 1][0] = trade_price;
        snap.trades[TRADE_BUF_LEN - 1][1] = static_cast<float>(size);
        snap.trades[TRADE_BUF_LEN - 1][2] = 1.0f;  // buy aggressor
    }

    return snap;
}

// Build a no-trade snapshot (trade_count = 0, no trade data in buffer)
BookSnapshot make_empty_snap(uint64_t timestamp, float mid_price = 4500.125f) {
    return make_snapshot_tc(timestamp, 0, 0.0f, 0, mid_price);
}

// Feed snapshots into a bar builder, collecting emitted bars
std::vector<Bar> feed_all(BarBuilder& builder, const std::vector<BookSnapshot>& snaps) {
    std::vector<Bar> bars;
    for (const auto& snap : snaps) {
        auto bar = builder.on_snapshot(snap);
        if (bar.has_value()) {
            bars.push_back(bar.value());
        }
    }
    return bars;
}

// Sum trade_count across all snapshots
uint32_t sum_trade_counts(const std::vector<BookSnapshot>& snaps) {
    uint32_t total = 0;
    for (const auto& s : snaps) {
        total += s.trade_count;
    }
    return total;
}

// Generate a "day" of snapshots with uniform trade_count per snapshot.
std::vector<BookSnapshot> make_uniform_day(int snap_count, uint32_t tc_per_snap,
                                            uint64_t day_offset_ns = 0) {
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < snap_count; ++i) {
        uint64_t ts = RTH_OPEN_NS + day_offset_ns +
                      static_cast<uint64_t>(i) * BOUNDARY_100MS;
        snaps.push_back(make_snapshot_tc(ts, tc_per_snap));
    }
    return snaps;
}

}  // anonymous namespace

// ===========================================================================
// T1: Trade count field exists on snapshots
// ===========================================================================

class TickBarFixT1 : public ::testing::Test {};

TEST_F(TickBarFixT1, SnapshotHasTradeCountField) {
    // Exit criterion T1: BookSnapshot includes a trade_count field (uint32_t)
    // that counts the number of action='T' MBO events since the previous snapshot.
    BookSnapshot snap{};
    snap.trade_count = 7;
    EXPECT_EQ(snap.trade_count, 7u)
        << "BookSnapshot must have a trade_count field (uint32_t)";
}

TEST_F(TickBarFixT1, TradeCountDefaultsToZero) {
    // A default-constructed BookSnapshot should have trade_count = 0.
    BookSnapshot snap{};
    EXPECT_EQ(snap.trade_count, 0u)
        << "trade_count should default-initialize to 0";
}

// ===========================================================================
// T2: Tick bars count trades, not snapshots
// ===========================================================================

class TickBarCountsTradesTest : public ::testing::Test {};

TEST_F(TickBarCountsTradesTest, CountsTradesNotSnapshots) {
    // CORE TEST — the diagnostic that would have caught the original bug.
    //
    // 10 snapshots, each with trade_count=5 → 50 total trades.
    // threshold=25 → should produce 2 bars.
    //
    // Broken implementation (1 tick per snapshot-with-trade):
    //   tick_count = 10 (one per snapshot) < 25 → 0 bars. WRONG.
    //
    // Fixed implementation (accumulates trade_count):
    //   cumulative_trades = 50, crosses 25 twice → 2 bars. CORRECT.
    TickBarBuilder builder(25);

    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 10; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 5));
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_EQ(bars.size(), 2u)
        << "50 trades (10 snapshots x 5 trades each) / threshold 25 = 2 bars, not "
        << bars.size();
}

TEST_F(TickBarCountsTradesTest, MultiTradesPerSnapshotClosesBar) {
    // A single snapshot with trade_count >= threshold should close a bar immediately.
    TickBarBuilder builder(10);

    auto snap = make_snapshot_tc(rth_ts(0), 10);
    auto result = builder.on_snapshot(snap);

    ASSERT_TRUE(result.has_value())
        << "A single snapshot with trade_count=10 should close bar at threshold 10";
}

TEST_F(TickBarCountsTradesTest, VariableDurationFromTradeClustering) {
    // Bars during high-activity periods should be shorter (fewer snapshots
    // needed to reach threshold) and bars during low-activity periods should
    // be longer (more snapshots needed). This is the key behavioral difference
    // from time bars.
    //
    // Busy phase: 10 snapshots, trade_count=3 each → 30 trades
    // Quiet phase: 30 snapshots, trade_count=1 each → 30 trades
    // threshold=15 → busy bars close every 5 snaps (0.4s), quiet every 15 snaps (1.4s)
    TickBarBuilder builder(15);

    std::vector<BookSnapshot> snaps;
    // Busy: 10 snapshots, trade_count=3
    for (int i = 0; i < 10; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 3));
    }
    // Quiet: 30 snapshots, trade_count=1
    for (int i = 0; i < 30; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts((10 + i) * 100), 1));
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_GE(bars.size(), 3u)
        << "Should produce at least 3 bars from 60 trades at threshold 15";

    // Verify that bar durations are NOT all the same.
    // This is the diagnostic that distinguishes genuine tick bars from time bars.
    float min_dur = bars[0].bar_duration_s;
    float max_dur = bars[0].bar_duration_s;
    for (const auto& bar : bars) {
        min_dur = std::min(min_dur, bar.bar_duration_s);
        max_dur = std::max(max_dur, bar.bar_duration_s);
    }
    EXPECT_GT(max_dur, min_dur)
        << "Bars during busy and quiet periods must have different durations "
        << "(min=" << min_dur << "s, max=" << max_dur << "s). "
        << "If all durations are identical, the builder is counting snapshots, not trades.";
}

// ===========================================================================
// T3: Daily bar count variance > 0
// ===========================================================================

class TickBarDailyVarianceTest : public ::testing::Test {};

TEST_F(TickBarDailyVarianceTest, DifferentActivityLevelsProduceDifferentBarCounts) {
    // Exit criterion T3: Running tick bar export across >= 3 trading days with
    // different activity levels produces non-zero standard deviation in bars-per-day.
    //
    // Day 1 (high activity): 100 snaps × 5 tc = 500 trades → 5 bars at threshold 100
    // Day 2 (medium activity): 100 snaps × 3 tc = 300 trades → 3 bars at threshold 100
    // Day 3 (low activity): 100 snaps × 1 tc = 100 trades → 1 bar at threshold 100
    //
    // bars_per_day_std > 0. ✓
    // Broken impl: tick_count=1 per snap → 100 ticks/day for all 3 → 1 bar each → std=0. FAIL.
    constexpr uint32_t THRESHOLD = 100;
    constexpr int SNAPS_PER_DAY = 100;

    auto day1 = make_uniform_day(SNAPS_PER_DAY, 5);
    auto day2 = make_uniform_day(SNAPS_PER_DAY, 3, 24ULL * NS_PER_HOUR);
    auto day3 = make_uniform_day(SNAPS_PER_DAY, 1, 48ULL * NS_PER_HOUR);

    // Process each day independently (fresh builder per day)
    auto count_bars = [&](const std::vector<BookSnapshot>& snaps) -> int {
        TickBarBuilder builder(THRESHOLD);
        auto bars = feed_all(builder, snaps);
        return static_cast<int>(bars.size());
    };

    int count1 = count_bars(day1);
    int count2 = count_bars(day2);
    int count3 = count_bars(day3);

    // Verify bar counts differ across days
    EXPECT_NE(count1, count2)
        << "Days with different activity levels should produce different bar counts";
    EXPECT_NE(count1, count3)
        << "Days with different activity levels should produce different bar counts";

    // Verify bars_per_day std > 0
    double mean = (count1 + count2 + count3) / 3.0;
    double var = ((count1 - mean) * (count1 - mean) +
                  (count2 - mean) * (count2 - mean) +
                  (count3 - mean) * (count3 - mean)) / 3.0;
    EXPECT_GT(var, 0.0)
        << "bars_per_day variance must be > 0 across days with different activity. "
        << "Got counts: [" << count1 << ", " << count2 << ", " << count3 << "]";
}

// ===========================================================================
// T4: Within-day duration variance > 0
// ===========================================================================

class TickBarDurationVarianceTest : public ::testing::Test {};

TEST_F(TickBarDurationVarianceTest, DurationsVaryWithTradeActivity) {
    // Exit criterion T4: For any tick threshold producing >= 200 bars/day,
    // the p10 and p90 of bar durations must differ.
    //
    // Simulate a day with varying trade density:
    //   Open (20 snaps):  trade_count=4 → 80 trades (busy, short bars)
    //   Lunch (40 snaps): trade_count=1 → 40 trades (quiet, long bars)
    //   Close (20 snaps): trade_count=4 → 80 trades (busy, short bars)
    //   Total: 200 trades, threshold=20
    //
    // Open/close bars: ~5 snapshots each (0.4s duration)
    // Lunch bars: ~20 snapshots each (1.9s duration)
    TickBarBuilder builder(20);

    std::vector<BookSnapshot> snaps;
    // Open: busy
    for (int i = 0; i < 20; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 4));
    }
    // Lunch: quiet
    for (int i = 0; i < 40; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts((20 + i) * 100), 1));
    }
    // Close: busy
    for (int i = 0; i < 20; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts((60 + i) * 100), 4));
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_GE(bars.size(), 5u)
        << "Should produce at least 5 bars from 200 trades at threshold 20";

    // Collect durations and sort
    std::vector<float> durations;
    for (const auto& bar : bars) {
        durations.push_back(bar.bar_duration_s);
    }
    std::sort(durations.begin(), durations.end());

    // p10 and p90
    size_t p10_idx = durations.size() / 10;
    size_t p90_idx = (durations.size() * 9) / 10;
    float p10 = durations[p10_idx];
    float p90 = durations[p90_idx];

    EXPECT_GT(p90, p10)
        << "Bar duration p90 (" << p90 << "s) must exceed p10 (" << p10 << "s). "
        << "Bars should be shorter during busy periods and longer during quiet periods. "
        << "If p10==p90, the builder is effectively a time bar builder.";
}

// ===========================================================================
// T5: Trade count consistency / reconciliation
// ===========================================================================

class TickBarTradeReconciliationTest : public ::testing::Test {};

TEST_F(TickBarTradeReconciliationTest, TotalTradesReconcileWithBarsAndRemainder) {
    // Exit criterion T5: sum(bar_trade_counts) + remainder == total_trade_events.
    // The total trade count from bars must reconcile with the actual trade count
    // from the input stream.
    constexpr uint32_t THRESHOLD = 20;
    TickBarBuilder builder(THRESHOLD);

    // 30 snapshots, trade_count alternates: 2, 3, 2, 3, ...
    // Total trades = 15*2 + 15*3 = 75
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 30; ++i) {
        uint32_t tc = (i % 2 == 0) ? 2 : 3;
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), tc));
    }
    uint32_t expected_total = sum_trade_counts(snaps);
    ASSERT_EQ(expected_total, 75u);

    auto bars = feed_all(builder, snaps);
    auto partial = builder.flush();

    // Sum tick_count across all bars + partial
    uint32_t total_from_bars = 0;
    for (const auto& bar : bars) {
        total_from_bars += bar.tick_count;
    }
    if (partial.has_value()) {
        total_from_bars += partial->tick_count;
    }

    EXPECT_EQ(total_from_bars, expected_total)
        << "Sum of bar tick_counts (" << total_from_bars << ") + partial "
        << "must equal total trade events (" << expected_total << ")";
}

TEST_F(TickBarTradeReconciliationTest, ReconciliationWithVaryingTradeRates) {
    // More complex scenario: varying trade rates including zero-trade gaps.
    constexpr uint32_t THRESHOLD = 30;
    TickBarBuilder builder(THRESHOLD);

    std::vector<BookSnapshot> snaps;
    uint32_t expected_total = 0;
    for (int i = 0; i < 100; ++i) {
        uint32_t tc;
        if (i < 20) tc = 3;                        // busy open
        else if (i < 60) tc = (i % 5 == 0) ? 0 : 1;  // quiet with gaps
        else tc = 2;                                // moderate close
        expected_total += tc;
        if (tc > 0) {
            snaps.push_back(make_snapshot_tc(rth_ts(i * 100), tc));
        } else {
            snaps.push_back(make_empty_snap(rth_ts(i * 100)));
        }
    }

    auto bars = feed_all(builder, snaps);
    auto partial = builder.flush();

    uint32_t total_from_bars = 0;
    for (const auto& bar : bars) {
        total_from_bars += bar.tick_count;
    }
    if (partial.has_value()) {
        total_from_bars += partial->tick_count;
    }

    EXPECT_EQ(total_from_bars, expected_total)
        << "Trade reconciliation must hold across varying trade rates and gaps";
}

// ===========================================================================
// T6: Threshold proportionality
// ===========================================================================

class TickBarProportionalityTest : public ::testing::Test {};

TEST_F(TickBarProportionalityTest, DoubleThresholdApproximatelyHalvesBars) {
    // Exit criterion T6: For two thresholds N and 2N on the same day, the bar
    // count ratio is approximately 2:1 (within 20%).
    constexpr uint32_t N = 50;
    constexpr int SNAP_COUNT = 200;
    constexpr uint32_t TRADES_PER_SNAP = 3;
    // Total trades: 600. N=50 → ~12 bars. 2N=100 → ~6 bars. Ratio ≈ 2.0.

    auto snaps = make_uniform_day(SNAP_COUNT, TRADES_PER_SNAP);

    TickBarBuilder builder_n(N);
    auto bars_n = feed_all(builder_n, snaps);

    TickBarBuilder builder_2n(2 * N);
    auto bars_2n = feed_all(builder_2n, snaps);

    ASSERT_GT(bars_n.size(), 0u) << "Should produce bars at threshold " << N;
    ASSERT_GT(bars_2n.size(), 0u) << "Should produce bars at threshold " << (2 * N);

    double ratio = static_cast<double>(bars_n.size()) / static_cast<double>(bars_2n.size());
    EXPECT_GT(ratio, 1.5)
        << "Bar count ratio N vs 2N should be approximately 2:1 (got " << ratio << ")";
    EXPECT_LT(ratio, 2.5)
        << "Bar count ratio N vs 2N should be approximately 2:1 (got " << ratio << ")";
}

// ===========================================================================
// T10: No empty bars — zero-trade snapshots don't close bars
// ===========================================================================

class TickBarNoEmptyBarsTest : public ::testing::Test {};

TEST_F(TickBarNoEmptyBarsTest, ZeroTradeSnapshotsDoNotCloseBars) {
    // A gap of 20 zero-trade snapshots should not close a bar.
    // The bar stays open, spanning the gap, and only closes when enough
    // trades accumulate from subsequent snapshots.
    TickBarBuilder builder(10);

    std::vector<BookSnapshot> snaps;
    // Phase 1: 3 snapshots, 3 trades each = 9 (below threshold 10)
    for (int i = 0; i < 3; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 3));
    }
    // Gap: 20 snapshots with zero trades
    for (int i = 0; i < 20; ++i) {
        snaps.push_back(make_empty_snap(rth_ts((3 + i) * 100)));
    }
    // Phase 2: 1 snapshot, 1 trade → cumulative 9+1=10 → bar closes
    snaps.push_back(make_snapshot_tc(rth_ts(23 * 100), 1));

    auto bars = feed_all(builder, snaps);

    ASSERT_EQ(bars.size(), 1u)
        << "Only 1 bar should be produced (10 trades total, spanning the gap)";

    // Verify the bar spans the entire duration including the gap
    EXPECT_EQ(bars[0].snapshot_count, 24u)
        << "Bar should include all 24 snapshots (3 + 20 gap + 1)";
}

TEST_F(TickBarNoEmptyBarsTest, AllBarsHavePositiveTradeCount) {
    // No bar should ever have tick_count == 0. Bars with zero trades
    // must not be emitted (T10).
    TickBarBuilder builder(10);

    // Mix of trade and no-trade snapshots
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 60; ++i) {
        uint32_t tc = (i % 3 == 0) ? 3 : 0;  // trades every 3rd snapshot
        if (tc > 0) {
            snaps.push_back(make_snapshot_tc(rth_ts(i * 100), tc));
        } else {
            snaps.push_back(make_empty_snap(rth_ts(i * 100)));
        }
    }

    auto bars = feed_all(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    for (size_t i = 0; i < bars.size(); ++i) {
        EXPECT_GT(bars[i].tick_count, 0u)
            << "Bar " << i << " has tick_count=0 — empty bars must not be emitted";
    }
}

TEST_F(TickBarNoEmptyBarsTest, PurelyZeroTradeStreamProducesNoBars) {
    // A stream consisting entirely of zero-trade snapshots should produce no bars.
    TickBarBuilder builder(10);

    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 50; ++i) {
        snaps.push_back(make_empty_snap(rth_ts(i * 100)));
    }

    auto bars = feed_all(builder, snaps);
    EXPECT_TRUE(bars.empty())
        << "A stream with zero trades should produce no bars";

    auto partial = builder.flush();
    EXPECT_FALSE(partial.has_value())
        << "Flush on zero-trade stream should return nothing";
}

// ===========================================================================
// Carry-over: remainder trades carried to next bar
// ===========================================================================

class TickBarCarryOverTest : public ::testing::Test {};

TEST_F(TickBarCarryOverTest, CarryOverCausesEarlierBarClose) {
    // Spec scope constraint: "DO carry over remainder trades to the next bar."
    //
    // 5 snapshots × 10 trades each = 50 total. Threshold = 25.
    //
    // With carry-over (spec required):
    //   Snap 0: cum=10. Snap 1: cum=20. Snap 2: cum=30 >= 25 → bar 1. carry=5.
    //   Snap 3: cum=5+10=15. Snap 4: cum=15+10=25 >= 25 → bar 2. carry=0.
    //   Result: 2 bars. ✓
    //
    // Without carry-over:
    //   Snap 2: cum=30 >= 25 → bar 1. Reset to 0.
    //   Snap 3: cum=10. Snap 4: cum=20 < 25. No bar 2.
    //   Result: 1 bar + partial. ✗
    TickBarBuilder builder(25);

    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 5; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 10));
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_EQ(bars.size(), 2u)
        << "With carry-over, 5 snaps x 10 tc at threshold 25 should produce 2 bars. "
        << "Bar 1 closes at snap 2 (carry=5), bar 2 closes at snap 4 (5+10+10=25). "
        << "Got " << bars.size() << " bars.";

    // Reconciliation must still hold
    auto partial = builder.flush();
    uint32_t total = 0;
    for (const auto& bar : bars) total += bar.tick_count;
    if (partial.has_value()) total += partial->tick_count;
    EXPECT_EQ(total, 50u) << "All 50 trades must be accounted for";
}

TEST_F(TickBarCarryOverTest, CarryOverAccumulatesAcrossBatches) {
    // Verify carry-over works when snapshots arrive incrementally.
    //
    // Phase 1: 3 snapshots × 9 tc = 27 trades → bar 1 closes (carry=2).
    // Phase 2: 3 snapshots × 8 tc = 24 trades.
    //   With carry: 2+24=26 >= 25 → bar 2 closes.
    //   Without carry: 24 < 25 → no bar 2.
    TickBarBuilder builder(25);

    // Phase 1
    for (int i = 0; i < 3; ++i) {
        auto snap = make_snapshot_tc(rth_ts(i * 100), 9);
        builder.on_snapshot(snap);
    }

    // Phase 2: feed incrementally, track bars
    std::vector<Bar> phase2_bars;
    for (int i = 0; i < 3; ++i) {
        auto snap = make_snapshot_tc(rth_ts((3 + i) * 100), 8);
        auto result = builder.on_snapshot(snap);
        if (result.has_value()) {
            phase2_bars.push_back(result.value());
        }
    }

    // With carry-over: carry(2) + 24 = 26 >= 25 → bar 2 closes in phase 2
    // Without carry-over: 24 < 25 → no bar in phase 2
    ASSERT_EQ(phase2_bars.size(), 1u)
        << "Carry-over of 2 from bar 1 + 24 new trades should close bar 2 at threshold 25";
}

TEST_F(TickBarCarryOverTest, ExactThresholdProducesNoRemainder) {
    // When total trades exactly divide by threshold, no partial bar remains.
    // 10 snapshots × 5 tc = 50 trades. Threshold = 25. → 2 bars, no partial.
    TickBarBuilder builder(25);

    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 10; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 5));
    }

    auto bars = feed_all(builder, snaps);
    EXPECT_EQ(bars.size(), 2u);

    auto partial = builder.flush();
    EXPECT_FALSE(partial.has_value())
        << "50 trades at threshold 25 divides evenly — no partial bar expected";
}

// ===========================================================================
// T7-T9: Regression — time, dollar, volume bars unchanged
// ===========================================================================

class TickBarRegressionTest : public ::testing::Test {};

TEST_F(TickBarRegressionTest, TimeBarIgnoresTradeCount) {
    // Exit criterion T7: TimeBarBuilder should produce identical bars regardless
    // of the trade_count field. Time bars close at fixed time intervals.
    TimeBarBuilder builder(5);  // 5-second bars

    // 50 snapshots at 100ms = 5 seconds → exactly 1 bar
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 50; ++i) {
        // Set varying trade_count — should NOT affect time bar behavior
        uint32_t tc = (i % 5 == 0) ? 10 : 0;
        if (tc > 0) {
            snaps.push_back(make_snapshot_tc(rth_ts(i * 100), tc));
        } else {
            snaps.push_back(make_empty_snap(rth_ts(i * 100)));
        }
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_EQ(bars.size(), 1u)
        << "TimeBarBuilder must produce exactly 1 bar for 5s of data at 5s interval, "
        << "regardless of trade_count values";

    EXPECT_NEAR(bars[0].bar_duration_s, 4.9f, 0.2f)
        << "Time bar duration should be approximately 5 seconds";
}

TEST_F(TickBarRegressionTest, DollarBarIgnoresTradeCount) {
    // Exit criterion T8: DollarBarBuilder behavior unchanged by trade_count field.
    // Dollar bars close on dollar volume, which is computed from trade data in
    // the snapshot buffer, not from trade_count.
    DollarBarBuilder builder(50000.0);

    // 3 snapshots with large trades: price=4500 × size=10 × multiplier=5 = $225,000
    // Each snapshot exceeds threshold → 3 bars (or 1 bar if cumulative reset).
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 3; ++i) {
        auto snap = make_snapshot_tc(rth_ts(i * 100), 5);  // trade_count=5 irrelevant
        snap.trades[TRADE_BUF_LEN - 1][0] = 4500.0f;
        snap.trades[TRADE_BUF_LEN - 1][1] = 10.0f;
        snap.trades[TRADE_BUF_LEN - 1][2] = 1.0f;
        snaps.push_back(snap);
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_GE(bars.size(), 1u)
        << "DollarBarBuilder must still function correctly with trade_count field present";
}

TEST_F(TickBarRegressionTest, VolumeBarIgnoresTradeCount) {
    // Exit criterion T9: VolumeBarBuilder behavior unchanged by trade_count field.
    // Volume bars close on cumulative trade volume (size), not trade_count.
    VolumeBarBuilder builder(50);

    // 10 snapshots, each with trade size 5 → total volume 50 → 1 bar
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 10; ++i) {
        auto snap = make_snapshot_tc(rth_ts(i * 100), 3);  // trade_count=3 irrelevant
        snap.trades[TRADE_BUF_LEN - 1][0] = 4500.25f;
        snap.trades[TRADE_BUF_LEN - 1][1] = 5.0f;
        snap.trades[TRADE_BUF_LEN - 1][2] = (i % 2 == 0) ? 1.0f : -1.0f;
        snaps.push_back(snap);
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_EQ(bars.size(), 1u)
        << "VolumeBarBuilder (threshold=50, 10 snaps x 5 vol) should produce 1 bar";
    EXPECT_EQ(bars[0].volume, 50u)
        << "Volume bar should accumulate trade sizes, not trade_count";
}

// ===========================================================================
// T11: Feature schema unchanged — tick bars produce valid Bar structs
// ===========================================================================

class TickBarSchemaTest : public ::testing::Test {};

TEST_F(TickBarSchemaTest, TickBarProducesValidBarStruct) {
    // Exit criterion T11: The Bar struct fields for tick bars must match the
    // existing schema. Only bar boundaries change, not the column structure.
    TickBarBuilder builder(5);

    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 5; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 1));
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_EQ(bars.size(), 1u);

    const auto& bar = bars[0];

    // Verify all standard Bar fields are populated
    EXPECT_GT(bar.open_ts, 0u) << "open_ts must be set";
    EXPECT_GT(bar.close_ts, 0u) << "close_ts must be set";
    EXPECT_GE(bar.close_ts, bar.open_ts) << "close_ts >= open_ts";
    EXPECT_GT(bar.open_mid, 0.0f) << "open_mid must be set";
    EXPECT_GT(bar.close_mid, 0.0f) << "close_mid must be set";
    EXPECT_GE(bar.high_mid, bar.low_mid) << "high_mid >= low_mid";
    EXPECT_GT(bar.volume, 0u) << "volume must be positive";
    EXPECT_GT(bar.tick_count, 0u) << "tick_count must be positive";
    EXPECT_GT(bar.snapshot_count, 0u) << "snapshot_count must be positive";
    EXPECT_GT(bar.bar_duration_s, 0.0f) << "bar_duration_s must be positive";
    EXPECT_GE(bar.time_of_day, 9.5f) << "time_of_day in RTH range";
    EXPECT_LE(bar.time_of_day, 16.0f) << "time_of_day in RTH range";
}

// ===========================================================================
// Edge Cases
// ===========================================================================

class TickBarFixEdgeCaseTest : public ::testing::Test {};

TEST_F(TickBarFixEdgeCaseTest, EmptyStreamProducesNoBars) {
    TickBarBuilder builder(10);
    std::vector<BookSnapshot> empty;
    auto bars = feed_all(builder, empty);
    EXPECT_TRUE(bars.empty());
    EXPECT_FALSE(builder.flush().has_value());
}

TEST_F(TickBarFixEdgeCaseTest, FlushReturnsPartialBar) {
    TickBarBuilder builder(100);

    // 5 snapshots, 2 trades each → 10 trades (well below threshold 100)
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 5; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 2));
    }

    auto bars = feed_all(builder, snaps);
    EXPECT_TRUE(bars.empty()) << "10 trades < threshold 100 → no completed bars";

    auto partial = builder.flush();
    ASSERT_TRUE(partial.has_value()) << "Flush should return partial bar with 10 trades";
    EXPECT_GT(partial->tick_count, 0u) << "Partial bar must have positive tick_count";
}

TEST_F(TickBarFixEdgeCaseTest, SingleSnapshotMeetsThreshold) {
    // One snapshot with trade_count exactly equal to threshold closes immediately.
    TickBarBuilder builder(5);

    auto snap = make_snapshot_tc(rth_ts(0), 5);
    auto result = builder.on_snapshot(snap);

    ASSERT_TRUE(result.has_value())
        << "Single snapshot with trade_count == threshold should produce a bar";
}

TEST_F(TickBarFixEdgeCaseTest, MixedTradeCountsAccumulateCorrectly) {
    // Snapshots with varying trade_count: 1, 5, 0, 3, 2 → total 11
    // Snap 0 (tc=1): bar starts, cumulative=1
    // Snap 1 (tc=5): cumulative=6
    // Snap 2 (tc=0): cumulative=6 (no advancement)
    // Snap 3 (tc=3): cumulative=9
    // Snap 4 (tc=2): cumulative=11 >= 10 → bar closes
    TickBarBuilder builder(10);

    std::vector<BookSnapshot> snaps;
    uint32_t counts[] = {1, 5, 0, 3, 2};
    for (int i = 0; i < 5; ++i) {
        if (counts[i] > 0) {
            snaps.push_back(make_snapshot_tc(rth_ts(i * 100), counts[i]));
        } else {
            snaps.push_back(make_empty_snap(rth_ts(i * 100)));
        }
    }

    auto bars = feed_all(builder, snaps);
    ASSERT_EQ(bars.size(), 1u) << "11 trades at threshold 10 → 1 bar";

    // Reconciliation
    auto partial = builder.flush();
    uint32_t total = 0;
    for (const auto& bar : bars) total += bar.tick_count;
    if (partial.has_value()) total += partial->tick_count;
    EXPECT_EQ(total, 11u) << "All 11 trades must be accounted for";
}

TEST_F(TickBarFixEdgeCaseTest, BarFactoryCreatesTickBuilderThatUsesTradeCount) {
    // Verify that BarFactory::create("tick", N) produces a builder that uses
    // trade_count for bar boundaries.
    auto builder = BarFactory::create("tick", 10);
    ASSERT_NE(builder, nullptr);

    // 5 snapshots × 3 tc = 15 trades at threshold 10 → 1 bar
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < 5; ++i) {
        snaps.push_back(make_snapshot_tc(rth_ts(i * 100), 3));
    }

    auto bars = feed_all(*builder, snaps);
    ASSERT_GE(bars.size(), 1u)
        << "BarFactory-created tick builder should produce bars based on trade_count";
}
