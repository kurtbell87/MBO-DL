// bar_builder_test.cpp — TDD RED phase tests for Bar construction
// Spec: .kit/docs/bar-construction.md
//
// Tests for Bar struct, BarBuilder interface, VolumeBarBuilder, TickBarBuilder,
// DollarBarBuilder, TimeBarBuilder, BarFactory, encoder adapters (PriceLadderInput,
// MessageSequenceInput), message summary computation, and spread dynamics.
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers that the implementation must provide (spec §Project Structure):
#include "bars/bar.hpp"               // Bar struct, BarBuilder interface, adapters
#include "bars/volume_bar_builder.hpp"
#include "bars/tick_bar_builder.hpp"
#include "bars/dollar_bar_builder.hpp"
#include "bars/time_bar_builder.hpp"
#include "bars/bar_factory.hpp"
#include "data/day_event_buffer.hpp"   // DayEventBuffer, MBOEvent
#include "book_builder.hpp"            // BookSnapshot (existing)

#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

// ===========================================================================
// Helpers — synthetic BookSnapshot construction
// ===========================================================================
namespace {

// Timestamp constants (same conventions as existing book_builder_test.cpp)
constexpr uint64_t NS_PER_SEC  = 1'000'000'000ULL;
constexpr uint64_t NS_PER_HOUR = 3600ULL * NS_PER_SEC;
constexpr uint64_t MIDNIGHT_ET_NS = 1641186000ULL * NS_PER_SEC;  // 2022-01-03 00:00 ET
constexpr uint64_t RTH_OPEN_NS = MIDNIGHT_ET_NS + 9ULL * NS_PER_HOUR + 30ULL * 60 * NS_PER_SEC;
constexpr uint64_t RTH_CLOSE_NS = MIDNIGHT_ET_NS + 16ULL * NS_PER_HOUR;
constexpr uint64_t BOUNDARY_100MS = 100'000'000ULL;

constexpr uint64_t rth_ts(uint64_t offset_ms) {
    return RTH_OPEN_NS + offset_ms * 1'000'000ULL;
}

// Build a BookSnapshot at a given timestamp with configurable fields.
BookSnapshot make_snapshot(uint64_t timestamp, float mid_price = 4500.125f,
                           float spread = 0.25f, float time_of_day = 9.5f) {
    BookSnapshot snap{};
    snap.timestamp = timestamp;
    snap.mid_price = mid_price;
    snap.spread = spread;
    snap.time_of_day = time_of_day;

    // Minimal book: 1 bid, 1 ask
    snap.bids[0][0] = mid_price - spread / 2.0f;
    snap.bids[0][1] = 10.0f;
    snap.asks[0][0] = mid_price + spread / 2.0f;
    snap.asks[0][1] = 10.0f;

    return snap;
}

// Build a snapshot with a trade (simulates a trade event in the snapshot stream).
BookSnapshot make_snapshot_with_trade(uint64_t timestamp, float trade_price,
                                      uint32_t trade_size, char aggressor_side,
                                      float mid_price = 4500.125f) {
    auto snap = make_snapshot(timestamp, mid_price);
    // Place trade at last position (left-padded convention)
    snap.trades[TRADE_BUF_LEN - 1][0] = trade_price;
    snap.trades[TRADE_BUF_LEN - 1][1] = static_cast<float>(trade_size);
    snap.trades[TRADE_BUF_LEN - 1][2] = (aggressor_side == 'B') ? 1.0f : -1.0f;
    return snap;
}

// Build a sequence of snapshots with trades at regular intervals.
std::vector<BookSnapshot> make_trade_sequence(uint64_t start_ts, int count,
                                               uint32_t trade_size_each,
                                               float trade_price = 4500.25f) {
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < count; ++i) {
        uint64_t ts = start_ts + static_cast<uint64_t>(i) * BOUNDARY_100MS;
        auto snap = make_snapshot(ts);
        snap.trades[TRADE_BUF_LEN - 1][0] = trade_price;
        snap.trades[TRADE_BUF_LEN - 1][1] = static_cast<float>(trade_size_each);
        snap.trades[TRADE_BUF_LEN - 1][2] = (i % 2 == 0) ? 1.0f : -1.0f;  // alternate buy/sell
        snaps.push_back(snap);
    }
    return snaps;
}

// Build a sequence of snapshots with varying mid prices for OHLCV testing.
std::vector<BookSnapshot> make_ohlcv_sequence(uint64_t start_ts, int count,
                                               float open_mid = 4500.0f,
                                               float high_delta = 1.0f,
                                               float low_delta = -0.5f,
                                               uint32_t trade_size = 10) {
    std::vector<BookSnapshot> snaps;
    for (int i = 0; i < count; ++i) {
        uint64_t ts = start_ts + static_cast<uint64_t>(i) * BOUNDARY_100MS;
        float mid;
        if (i == 0)              mid = open_mid;
        else if (i == 1)         mid = open_mid + high_delta;  // high
        else if (i == count - 2) mid = open_mid + low_delta;   // low
        else                     mid = open_mid + 0.25f;       // typical

        auto snap = make_snapshot(ts, mid);
        // Add a trade to each snapshot
        snap.trades[TRADE_BUF_LEN - 1][0] = mid;
        snap.trades[TRADE_BUF_LEN - 1][1] = static_cast<float>(trade_size);
        snap.trades[TRADE_BUF_LEN - 1][2] = 1.0f;
        snaps.push_back(snap);
    }
    return snaps;
}

// Feed a sequence of snapshots into a BarBuilder, collecting emitted bars.
std::vector<Bar> feed_snapshots(BarBuilder& builder, const std::vector<BookSnapshot>& snaps) {
    std::vector<Bar> bars;
    for (const auto& snap : snaps) {
        auto bar = builder.on_snapshot(snap);
        if (bar.has_value()) {
            bars.push_back(bar.value());
        }
    }
    return bars;
}

}  // anonymous namespace

// ===========================================================================
// Bar Struct — Data Contract Tests
// ===========================================================================

class BarStructTest : public ::testing::Test {};

TEST_F(BarStructTest, BarHasTemporalFields) {
    Bar bar{};
    bar.open_ts = RTH_OPEN_NS;
    bar.close_ts = RTH_OPEN_NS + 60 * NS_PER_SEC;
    bar.time_of_day = 9.5f;
    bar.bar_duration_s = 60.0f;

    EXPECT_EQ(bar.open_ts, RTH_OPEN_NS);
    EXPECT_EQ(bar.close_ts, RTH_OPEN_NS + 60 * NS_PER_SEC);
    EXPECT_FLOAT_EQ(bar.time_of_day, 9.5f);
    EXPECT_FLOAT_EQ(bar.bar_duration_s, 60.0f);
}

TEST_F(BarStructTest, BarHasOHLCVFields) {
    Bar bar{};
    bar.open_mid = 4500.0f;
    bar.close_mid = 4500.25f;
    bar.high_mid = 4500.50f;
    bar.low_mid = 4499.75f;
    bar.vwap = 4500.125f;
    bar.volume = 100;
    bar.tick_count = 50;
    bar.buy_volume = 60.0f;
    bar.sell_volume = 40.0f;

    EXPECT_FLOAT_EQ(bar.open_mid, 4500.0f);
    EXPECT_FLOAT_EQ(bar.close_mid, 4500.25f);
    EXPECT_FLOAT_EQ(bar.high_mid, 4500.50f);
    EXPECT_FLOAT_EQ(bar.low_mid, 4499.75f);
    EXPECT_FLOAT_EQ(bar.vwap, 4500.125f);
    EXPECT_EQ(bar.volume, 100u);
    EXPECT_EQ(bar.tick_count, 50u);
    EXPECT_FLOAT_EQ(bar.buy_volume, 60.0f);
    EXPECT_FLOAT_EQ(bar.sell_volume, 40.0f);
}

TEST_F(BarStructTest, BarHasBookStateFields) {
    Bar bar{};
    bar.bids[0][0] = 4500.00f;
    bar.bids[0][1] = 10.0f;
    bar.bids[9][0] = 4497.75f;
    bar.bids[9][1] = 2.0f;
    bar.asks[0][0] = 4500.25f;
    bar.asks[0][1] = 8.0f;
    bar.asks[9][0] = 4502.50f;
    bar.asks[9][1] = 1.0f;
    bar.spread = 0.25f;

    EXPECT_FLOAT_EQ(bar.bids[0][0], 4500.00f);
    EXPECT_FLOAT_EQ(bar.bids[9][1], 2.0f);
    EXPECT_FLOAT_EQ(bar.asks[0][0], 4500.25f);
    EXPECT_FLOAT_EQ(bar.asks[9][1], 1.0f);
    EXPECT_FLOAT_EQ(bar.spread, 0.25f);
}

TEST_F(BarStructTest, BarHasSpreadDynamicsFields) {
    Bar bar{};
    bar.max_spread = 0.50f;
    bar.min_spread = 0.25f;
    bar.snapshot_count = 100;

    EXPECT_FLOAT_EQ(bar.max_spread, 0.50f);
    EXPECT_FLOAT_EQ(bar.min_spread, 0.25f);
    EXPECT_EQ(bar.snapshot_count, 100u);
}

TEST_F(BarStructTest, BarHasMBOEventReferenceFields) {
    Bar bar{};
    bar.mbo_event_begin = 1000;
    bar.mbo_event_end = 2000;

    EXPECT_EQ(bar.mbo_event_begin, 1000u);
    EXPECT_EQ(bar.mbo_event_end, 2000u);
}

TEST_F(BarStructTest, BarHasMessageSummaryFields) {
    Bar bar{};
    bar.add_count = 50;
    bar.cancel_count = 30;
    bar.modify_count = 10;
    bar.trade_event_count = 20;
    bar.cancel_add_ratio = 30.0f / (50.0f + 1e-8f);
    bar.message_rate = 110.0f / 60.0f;

    EXPECT_EQ(bar.add_count, 50u);
    EXPECT_EQ(bar.cancel_count, 30u);
    EXPECT_EQ(bar.modify_count, 10u);
    EXPECT_EQ(bar.trade_event_count, 20u);
    EXPECT_GT(bar.cancel_add_ratio, 0.0f);
    EXPECT_GT(bar.message_rate, 0.0f);
}

// ===========================================================================
// BarBuilder Interface Tests
// ===========================================================================

class BarBuilderInterfaceTest : public ::testing::Test {};

TEST_F(BarBuilderInterfaceTest, OnSnapshotReturnsOptionalBar) {
    VolumeBarBuilder builder(100);
    auto snap = make_snapshot(rth_ts(0));
    auto result = builder.on_snapshot(snap);
    // No trade volume in this snapshot → no bar emitted
    EXPECT_FALSE(result.has_value());
}

TEST_F(BarBuilderInterfaceTest, FlushReturnsPartialBar) {
    VolumeBarBuilder builder(100);

    auto snap = make_snapshot_with_trade(rth_ts(0), 4500.25f, 10, 'B');
    builder.on_snapshot(snap);

    auto partial = builder.flush();
    ASSERT_TRUE(partial.has_value());
    EXPECT_LT(partial->volume, 100u);
    EXPECT_GT(partial->volume, 0u);
}

TEST_F(BarBuilderInterfaceTest, FlushReturnsNulloptWhenEmpty) {
    VolumeBarBuilder builder(100);
    auto result = builder.flush();
    EXPECT_FALSE(result.has_value());
}

TEST_F(BarBuilderInterfaceTest, BarBuilderIsPolymorphic) {
    std::unique_ptr<BarBuilder> builder = std::make_unique<VolumeBarBuilder>(100);
    auto snap = make_snapshot(rth_ts(0));
    auto result = builder->on_snapshot(snap);
    EXPECT_FALSE(result.has_value());

    builder = std::make_unique<TickBarBuilder>(50);
    result = builder->on_snapshot(snap);
    EXPECT_FALSE(result.has_value());
}

// ===========================================================================
// VolumeBarBuilder Tests (§4.2)
// ===========================================================================

class VolumeBarBuilderTest : public ::testing::Test {
protected:
    static constexpr uint32_t V = 100;
};

TEST_F(VolumeBarBuilderTest, EmitsBarWhenCumulativeVolumeReachesThreshold) {
    VolumeBarBuilder builder(V);

    // 10 snapshots × 10 contracts each = 100 → should emit a bar
    auto snaps = make_trade_sequence(rth_ts(0), 10, 10);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_EQ(bars.size(), 1u);
    EXPECT_GE(bars[0].volume, V);
}

TEST_F(VolumeBarBuilderTest, BarVolumeIsAtLeastThreshold) {
    VolumeBarBuilder builder(V);

    auto snaps = make_trade_sequence(rth_ts(0), 25, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_GE(bars.size(), 1u);
    for (const auto& bar : bars) {
        EXPECT_GE(bar.volume, V) << "Volume bar must have volume >= threshold";
    }
}

TEST_F(VolumeBarBuilderTest, LargeTradeNotSplit) {
    // "If a single trade crosses the boundary, it belongs to the current bar
    //  (no trade splitting)."
    VolumeBarBuilder builder(V);

    // First snapshot: 90 volume (below threshold)
    auto snap1 = make_snapshot_with_trade(rth_ts(0), 4500.25f, 90, 'B');
    auto result1 = builder.on_snapshot(snap1);
    EXPECT_FALSE(result1.has_value());

    // Second snapshot: 50 volume (cumulative = 140 > 100, but no splitting)
    auto snap2 = make_snapshot_with_trade(rth_ts(100), 4500.25f, 50, 'B');
    auto result2 = builder.on_snapshot(snap2);
    ASSERT_TRUE(result2.has_value());

    // The entire 50-lot trade belongs to this bar (volume = 140, not split to 100 + 40)
    EXPECT_EQ(result2->volume, 140u);
}

TEST_F(VolumeBarBuilderTest, MultipleBarsFromLongStream) {
    VolumeBarBuilder builder(V);

    // 50 snapshots × 10 contracts = 500 total → should produce ~5 bars
    auto snaps = make_trade_sequence(rth_ts(0), 50, 10);
    auto bars = feed_snapshots(builder, snaps);

    EXPECT_GE(bars.size(), 4u);
    EXPECT_LE(bars.size(), 6u);

    for (const auto& bar : bars) {
        EXPECT_GE(bar.volume, V);
    }
}

TEST_F(VolumeBarBuilderTest, FlushReturnsPartialBarBelowThreshold) {
    VolumeBarBuilder builder(V);

    // 5 snapshots × 10 = 50 volume (below V=100)
    auto snaps = make_trade_sequence(rth_ts(0), 5, 10);
    feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    ASSERT_TRUE(partial.has_value());
    EXPECT_EQ(partial->volume, 50u);
    EXPECT_LT(partial->volume, V);
}

TEST_F(VolumeBarBuilderTest, CustomThreshold) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 10);  // 100 total
    auto bars = feed_snapshots(builder, snaps);

    EXPECT_EQ(bars.size(), 2u);
    for (const auto& bar : bars) {
        EXPECT_GE(bar.volume, 50u);
    }
}

// ===========================================================================
// TickBarBuilder Tests (§4.3)
// ===========================================================================

class TickBarBuilderTest : public ::testing::Test {
protected:
    static constexpr uint32_t K = 50;
};

TEST_F(TickBarBuilderTest, EmitsBarWhenTickCountReachesThreshold) {
    TickBarBuilder builder(K);

    auto snaps = make_trade_sequence(rth_ts(0), K, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_EQ(bars.size(), 1u);
    EXPECT_GE(bars[0].tick_count, K);
}

TEST_F(TickBarBuilderTest, CountsDeduplicatedTrades) {
    TickBarBuilder builder(K);

    auto snaps = make_trade_sequence(rth_ts(0), K + 10, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_GE(bars.size(), 1u);
    EXPECT_GE(bars[0].tick_count, K);
}

TEST_F(TickBarBuilderTest, BarTickCountIsAtLeastThreshold) {
    TickBarBuilder builder(K);

    auto snaps = make_trade_sequence(rth_ts(0), K * 3, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_GE(bars.size(), 2u);
    for (const auto& bar : bars) {
        EXPECT_GE(bar.tick_count, K);
    }
}

TEST_F(TickBarBuilderTest, FlushReturnsPartialBar) {
    TickBarBuilder builder(K);

    auto snaps = make_trade_sequence(rth_ts(0), K / 2, 5);
    feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    ASSERT_TRUE(partial.has_value());
    EXPECT_LT(partial->tick_count, K);
    EXPECT_GT(partial->tick_count, 0u);
}

// ===========================================================================
// DollarBarBuilder Tests (§4.4)
// ===========================================================================

class DollarBarBuilderTest : public ::testing::Test {
protected:
    static constexpr double D = 50000.0;
    static constexpr float MULTIPLIER = 5.0f;
};

TEST_F(DollarBarBuilderTest, EmitsBarWhenDollarVolumeReachesThreshold) {
    DollarBarBuilder builder(D);

    // dollar_volume = price × size × multiplier
    // 4500.0 × 10 × 5.0 = 225000 per snapshot > 50000 → 1 snap emits bar
    auto snap = make_snapshot_with_trade(rth_ts(0), 4500.0f, 10, 'B');
    auto result = builder.on_snapshot(snap);

    ASSERT_TRUE(result.has_value());
}

TEST_F(DollarBarBuilderTest, DollarVolumeCalculationUsesMultiplier) {
    // At price 4500, size 1: dollar_vol = 4500 * 1 * 5 = 22500
    // Need 50000 / 22500 ≈ 2.22, so 3 trades of size 1
    DollarBarBuilder builder(D);

    auto snap1 = make_snapshot_with_trade(rth_ts(0), 4500.0f, 1, 'B');
    EXPECT_FALSE(builder.on_snapshot(snap1).has_value());  // 22500 < 50000

    auto snap2 = make_snapshot_with_trade(rth_ts(100), 4500.0f, 1, 'A');
    EXPECT_FALSE(builder.on_snapshot(snap2).has_value());  // 45000 < 50000

    auto snap3 = make_snapshot_with_trade(rth_ts(200), 4500.0f, 1, 'B');
    auto result = builder.on_snapshot(snap3);
    ASSERT_TRUE(result.has_value());  // 67500 >= 50000
}

TEST_F(DollarBarBuilderTest, FlushReturnsPartialBar) {
    DollarBarBuilder builder(D);

    auto snap = make_snapshot_with_trade(rth_ts(0), 4500.0f, 1, 'B');
    builder.on_snapshot(snap);

    auto partial = builder.flush();
    ASSERT_TRUE(partial.has_value());
}

// ===========================================================================
// TimeBarBuilder Tests (§4.5)
// ===========================================================================

class TimeBarBuilderTest : public ::testing::Test {
protected:
    static constexpr uint64_t INTERVAL_S = 60;
};

TEST_F(TimeBarBuilderTest, EmitsBarAtIntervalBoundary) {
    TimeBarBuilder builder(INTERVAL_S);

    // 600 snapshots at 100ms each = 60 seconds total → 1 bar
    auto snaps = make_trade_sequence(rth_ts(0), 600, 1);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_EQ(bars.size(), 1u);
}

TEST_F(TimeBarBuilderTest, BarsAlignedToIntervalBoundaries) {
    TimeBarBuilder builder(INTERVAL_S);

    // 1800 snapshots = 180 seconds = 3 intervals
    auto snaps = make_trade_sequence(rth_ts(0), 1800, 1);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_EQ(bars.size(), 3u);

    for (const auto& bar : bars) {
        EXPECT_NEAR(bar.bar_duration_s, 60.0f, 1.0f);
    }
}

TEST_F(TimeBarBuilderTest, BarsContiguousInTime) {
    TimeBarBuilder builder(INTERVAL_S);

    auto snaps = make_trade_sequence(rth_ts(0), 1200, 1);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_GE(bars.size(), 2u);
    for (size_t i = 1; i < bars.size(); ++i) {
        EXPECT_EQ(bars[i].open_ts, bars[i - 1].close_ts)
            << "Bar " << i << " open_ts must equal previous bar's close_ts";
    }
}

TEST_F(TimeBarBuilderTest, FlushReturnsPartialBar) {
    TimeBarBuilder builder(INTERVAL_S);

    auto snaps = make_trade_sequence(rth_ts(0), 300, 1);
    feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    ASSERT_TRUE(partial.has_value());
    EXPECT_LT(partial->bar_duration_s, static_cast<float>(INTERVAL_S));
}

TEST_F(TimeBarBuilderTest, CustomIntervalSeconds) {
    TimeBarBuilder builder(30);  // 30-second bars

    auto snaps = make_trade_sequence(rth_ts(0), 600, 1);
    auto bars = feed_snapshots(builder, snaps);

    EXPECT_EQ(bars.size(), 2u);
}

// ===========================================================================
// BarFactory Tests
// ===========================================================================

class BarFactoryTest : public ::testing::Test {};

TEST_F(BarFactoryTest, CreateVolumeBuilder) {
    auto builder = BarFactory::create("volume", 100);
    ASSERT_NE(builder, nullptr);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 10);
    auto bars = feed_snapshots(*builder, snaps);
    EXPECT_EQ(bars.size(), 1u);
}

TEST_F(BarFactoryTest, CreateTickBuilder) {
    auto builder = BarFactory::create("tick", 50);
    ASSERT_NE(builder, nullptr);
}

TEST_F(BarFactoryTest, CreateDollarBuilder) {
    auto builder = BarFactory::create("dollar", 50000);
    ASSERT_NE(builder, nullptr);
}

TEST_F(BarFactoryTest, CreateTimeBuilder) {
    auto builder = BarFactory::create("time", 60);
    ASSERT_NE(builder, nullptr);
}

TEST_F(BarFactoryTest, ReturnsNullptrForUnknownType) {
    auto builder = BarFactory::create("unknown", 100);
    EXPECT_EQ(builder, nullptr);
}

// ===========================================================================
// OHLCV Consistency Tests (Validation Gate)
// ===========================================================================

class BarOHLCVConsistencyTest : public ::testing::Test {};

TEST_F(BarOHLCVConsistencyTest, LowMidLessOrEqualOpenCloseHighMid) {
    VolumeBarBuilder builder(50);

    auto snaps = make_ohlcv_sequence(rth_ts(0), 10, 4500.0f, 1.0f, -0.5f, 5);
    auto bars = feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        EXPECT_LE(bar.low_mid, bar.open_mid)
            << "low_mid must be <= open_mid";
        EXPECT_LE(bar.low_mid, bar.close_mid)
            << "low_mid must be <= close_mid";
        EXPECT_GE(bar.high_mid, bar.open_mid)
            << "high_mid must be >= open_mid";
        EXPECT_GE(bar.high_mid, bar.close_mid)
            << "high_mid must be >= close_mid";
        EXPECT_LE(bar.low_mid, bar.high_mid)
            << "low_mid must be <= high_mid";
    }
}

TEST_F(BarOHLCVConsistencyTest, OpenMidIsFirstSnapshotMidPrice) {
    VolumeBarBuilder builder(50);

    float open_price = 4500.0f;
    auto snap1 = make_snapshot_with_trade(rth_ts(0), 4500.25f, 25, 'B', open_price);
    builder.on_snapshot(snap1);

    auto snap2 = make_snapshot_with_trade(rth_ts(100), 4500.25f, 25, 'A', 4500.5f);
    auto result = builder.on_snapshot(snap2);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->open_mid, open_price);
}

TEST_F(BarOHLCVConsistencyTest, CloseMidIsLastSnapshotMidPrice) {
    VolumeBarBuilder builder(50);

    auto snap1 = make_snapshot_with_trade(rth_ts(0), 4500.25f, 25, 'B', 4500.0f);
    builder.on_snapshot(snap1);

    float close_price = 4500.5f;
    auto snap2 = make_snapshot_with_trade(rth_ts(100), 4500.25f, 25, 'A', close_price);
    auto result = builder.on_snapshot(snap2);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->close_mid, close_price);
}

// ===========================================================================
// Non-overlapping Contiguous Bars Tests (Validation Gate)
// ===========================================================================

class BarContiguityTest : public ::testing::Test {};

TEST_F(BarContiguityTest, BarsNonOverlappingAndContiguous) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 100, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_GE(bars.size(), 2u);
    for (size_t i = 1; i < bars.size(); ++i) {
        EXPECT_EQ(bars[i].open_ts, bars[i - 1].close_ts)
            << "Gap between bar " << (i - 1) << " and bar " << i;
        EXPECT_GE(bars[i].open_ts, bars[i - 1].close_ts)
            << "Overlap between bar " << (i - 1) << " and bar " << i;
    }
}

TEST_F(BarContiguityTest, OpenTsIsFirstSnapshotTimestamp) {
    VolumeBarBuilder builder(50);

    uint64_t start = rth_ts(0);
    auto snaps = make_trade_sequence(start, 10, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_EQ(bars.size(), 1u);
    EXPECT_EQ(bars[0].open_ts, start);
}

TEST_F(BarContiguityTest, CloseTsIsLastSnapshotTimestamp) {
    VolumeBarBuilder builder(50);

    uint64_t start = rth_ts(0);
    auto snaps = make_trade_sequence(start, 10, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_EQ(bars.size(), 1u);
    uint64_t expected_close = start + 9 * BOUNDARY_100MS;
    EXPECT_EQ(bars[0].close_ts, expected_close);
}

// ===========================================================================
// Total Volume Conservation Tests (Validation Gate)
// ===========================================================================

class TotalVolumeConservationTest : public ::testing::Test {};

TEST_F(TotalVolumeConservationTest, VolumeBarsTotalVolumeEqualsInput) {
    VolumeBarBuilder builder(50);

    uint32_t trade_size = 7;
    int snap_count = 100;
    auto snaps = make_trade_sequence(rth_ts(0), snap_count, trade_size);
    auto bars = feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    uint64_t total_volume = 0;
    for (const auto& bar : bars) {
        total_volume += bar.volume;
    }

    EXPECT_EQ(total_volume, static_cast<uint64_t>(snap_count) * trade_size);
}

TEST_F(TotalVolumeConservationTest, TickBarsTotalVolumeEqualsInput) {
    TickBarBuilder builder(50);

    uint32_t trade_size = 7;
    int snap_count = 100;
    auto snaps = make_trade_sequence(rth_ts(0), snap_count, trade_size);
    auto bars = feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    uint64_t total_volume = 0;
    for (const auto& bar : bars) {
        total_volume += bar.volume;
    }

    EXPECT_EQ(total_volume, static_cast<uint64_t>(snap_count) * trade_size);
}

TEST_F(TotalVolumeConservationTest, TimeBarsTotalVolumeEqualsInput) {
    TimeBarBuilder builder(60);

    uint32_t trade_size = 7;
    int snap_count = 600;
    auto snaps = make_trade_sequence(rth_ts(0), snap_count, trade_size);
    auto bars = feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    uint64_t total_volume = 0;
    for (const auto& bar : bars) {
        total_volume += bar.volume;
    }

    EXPECT_EQ(total_volume, static_cast<uint64_t>(snap_count) * trade_size);
}

TEST_F(TotalVolumeConservationTest, DollarBarsTotalVolumeEqualsInput) {
    DollarBarBuilder builder(50000.0);

    uint32_t trade_size = 7;
    int snap_count = 100;
    auto snaps = make_trade_sequence(rth_ts(0), snap_count, trade_size);
    auto bars = feed_snapshots(builder, snaps);

    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    uint64_t total_volume = 0;
    for (const auto& bar : bars) {
        total_volume += bar.volume;
    }

    EXPECT_EQ(total_volume, static_cast<uint64_t>(snap_count) * trade_size);
}

// ===========================================================================
// Message Summary Tests (Spec §Message Summary Computation)
// ===========================================================================

class MessageSummaryTest : public ::testing::Test {};

TEST_F(MessageSummaryTest, MessageCountsAccumulated) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        uint32_t total = bar.add_count + bar.cancel_count +
                         bar.modify_count + bar.trade_event_count;
        EXPECT_GT(total, 0u);
    }
}

TEST_F(MessageSummaryTest, CancelAddRatioWithEpsilonGuard) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        float expected_ratio = static_cast<float>(bar.cancel_count) /
                               (static_cast<float>(bar.add_count) + 1e-8f);
        EXPECT_NEAR(bar.cancel_add_ratio, expected_ratio, 1e-6f);
    }
}

TEST_F(MessageSummaryTest, CancelAddRatioSafeWhenAddCountZero) {
    Bar bar{};
    bar.add_count = 0;
    bar.cancel_count = 5;
    bar.cancel_add_ratio = static_cast<float>(bar.cancel_count) /
                           (static_cast<float>(bar.add_count) + 1e-8f);

    EXPECT_FALSE(std::isinf(bar.cancel_add_ratio));
    EXPECT_FALSE(std::isnan(bar.cancel_add_ratio));
}

TEST_F(MessageSummaryTest, MessageRateComputation) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        if (bar.bar_duration_s > 0.0f) {
            float total_msgs = static_cast<float>(bar.add_count + bar.cancel_count +
                                                   bar.modify_count + bar.trade_event_count);
            float expected_rate = total_msgs / bar.bar_duration_s;
            EXPECT_NEAR(bar.message_rate, expected_rate, 1e-4f);
        }
    }
}

// ===========================================================================
// MBO Event Reference Tests (Validation Gate)
// ===========================================================================

class MBOEventReferenceTest : public ::testing::Test {};

TEST_F(MBOEventReferenceTest, MboEventBeginLessThanEnd) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 20, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        if (bar.volume > 0) {
            EXPECT_LT(bar.mbo_event_begin, bar.mbo_event_end);
        }
    }
}

TEST_F(MBOEventReferenceTest, MboEventRangesNonOverlapping) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 40, 5);
    auto bars = feed_snapshots(builder, snaps);

    ASSERT_GE(bars.size(), 2u);
    for (size_t i = 1; i < bars.size(); ++i) {
        EXPECT_GE(bars[i].mbo_event_begin, bars[i - 1].mbo_event_end);
    }
}

// ===========================================================================
// Spread Dynamics Tests
// ===========================================================================

class SpreadDynamicsTest : public ::testing::Test {};

TEST_F(SpreadDynamicsTest, MaxSpreadTrackedAcrossSnapshots) {
    VolumeBarBuilder builder(50);

    auto snap1 = make_snapshot_with_trade(rth_ts(0), 4500.25f, 25, 'B');
    snap1.spread = 0.25f;
    builder.on_snapshot(snap1);

    auto snap2 = make_snapshot_with_trade(rth_ts(100), 4500.25f, 25, 'A');
    snap2.spread = 0.50f;
    auto result = builder.on_snapshot(snap2);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->max_spread, 0.50f);
}

TEST_F(SpreadDynamicsTest, MinSpreadTrackedAcrossSnapshots) {
    VolumeBarBuilder builder(50);

    auto snap1 = make_snapshot_with_trade(rth_ts(0), 4500.25f, 25, 'B');
    snap1.spread = 0.50f;
    builder.on_snapshot(snap1);

    auto snap2 = make_snapshot_with_trade(rth_ts(100), 4500.25f, 25, 'A');
    snap2.spread = 0.25f;
    auto result = builder.on_snapshot(snap2);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->min_spread, 0.25f);
}

TEST_F(SpreadDynamicsTest, SnapshotCountTracked) {
    VolumeBarBuilder builder(50);

    int num_snapshots = 10;
    auto snaps = make_trade_sequence(rth_ts(0), num_snapshots, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    uint32_t total_snap_count = 0;
    for (const auto& bar : bars) {
        total_snap_count += bar.snapshot_count;
    }
    EXPECT_EQ(total_snap_count, static_cast<uint32_t>(num_snapshots));
}

// ===========================================================================
// Bar Duration Tests
// ===========================================================================

class BarDurationTest : public ::testing::Test {};

TEST_F(BarDurationTest, DurationInSecondsMatchesTimestampDelta) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        float expected_duration = static_cast<float>(bar.close_ts - bar.open_ts)
                                  / static_cast<float>(NS_PER_SEC);
        EXPECT_NEAR(bar.bar_duration_s, expected_duration, 0.001f);
    }
}

TEST_F(BarDurationTest, TimeOfDayIsBarCloseInFractionalHoursET) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        EXPECT_GE(bar.time_of_day, 9.5f);
        EXPECT_LE(bar.time_of_day, 16.0f);
    }
}

// ===========================================================================
// Buy/Sell Volume Tests
// ===========================================================================

class BuySellVolumeTest : public ::testing::Test {};

TEST_F(BuySellVolumeTest, BuySellVolumeSumsToTotalVolume) {
    VolumeBarBuilder builder(50);

    auto snaps = make_trade_sequence(rth_ts(0), 10, 5);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        float total = bar.buy_volume + bar.sell_volume;
        EXPECT_NEAR(total, static_cast<float>(bar.volume), 0.01f);
    }
}

// ===========================================================================
// VWAP Tests
// ===========================================================================

class VWAPTest : public ::testing::Test {};

TEST_F(VWAPTest, VWAPIsVolumeWeightedAverage) {
    VolumeBarBuilder builder(50);

    // All trades at same price → VWAP = that price
    auto snaps = make_trade_sequence(rth_ts(0), 10, 5, 4500.25f);
    auto bars = feed_snapshots(builder, snaps);
    auto partial = builder.flush();
    if (partial.has_value()) bars.push_back(partial.value());

    ASSERT_FALSE(bars.empty());
    for (const auto& bar : bars) {
        EXPECT_NEAR(bar.vwap, 4500.25f, 0.01f);
    }
}

// ===========================================================================
// PriceLadderInput Adapter Tests (Validation Gate)
// ===========================================================================

class PriceLadderInputTest : public ::testing::Test {};

TEST_F(PriceLadderInputTest, FromBarProduces20x2Output) {
    Bar bar{};
    for (int i = 0; i < 10; ++i) {
        bar.bids[i][0] = 4500.0f - i * 0.25f;
        bar.bids[i][1] = static_cast<float>(10 - i);
        bar.asks[i][0] = 4500.25f + i * 0.25f;
        bar.asks[i][1] = static_cast<float>(10 - i);
    }
    float mid_price = 4500.125f;

    auto input = PriceLadderInput::from_bar(bar, mid_price);

    for (int i = 0; i < 20; ++i) {
        float price_delta = input.data[i][0];
        float norm_size = input.data[i][1];
        EXPECT_FALSE(std::isnan(price_delta)) << "row " << i << " price_delta is NaN";
        EXPECT_FALSE(std::isnan(norm_size)) << "row " << i << " norm_size is NaN";
    }
}

TEST_F(PriceLadderInputTest, BidOrderReversedInLadder) {
    Bar bar{};
    for (int i = 0; i < 10; ++i) {
        bar.bids[i][0] = 4500.0f - i * 0.25f;
        bar.bids[i][1] = static_cast<float>(10 + i);
    }
    for (int i = 0; i < 10; ++i) {
        bar.asks[i][0] = 4500.25f + i * 0.25f;
        bar.asks[i][1] = static_cast<float>(10 + i);
    }
    float mid_price = 4500.125f;

    auto input = PriceLadderInput::from_bar(bar, mid_price);

    // Row 9 (best bid): price delta should be negative (bid < mid)
    EXPECT_LT(input.data[9][0], 0.0f) << "Best bid delta should be negative";

    // Row 10 (best ask): price delta should be positive (ask > mid)
    EXPECT_GT(input.data[10][0], 0.0f) << "Best ask delta should be positive";
}

// ===========================================================================
// MessageSequenceInput Adapter Tests (Validation Gate)
// ===========================================================================

class MessageSequenceInputTest : public ::testing::Test {};

TEST_F(MessageSequenceInputTest, FromBarProducesCorrectEventSequence) {
    Bar bar{};
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 3;

    DayEventBuffer buf;

    auto input = MessageSequenceInput::from_bar(bar, buf);

    for (const auto& event : input.events) {
        EXPECT_EQ(event.size(), 5u);
    }
}

TEST_F(MessageSequenceInputTest, EventCountMatchesMBORange) {
    Bar bar{};
    bar.mbo_event_begin = 10;
    bar.mbo_event_end = 15;

    DayEventBuffer buf;
    auto input = MessageSequenceInput::from_bar(bar, buf);
    EXPECT_EQ(input.events.size(), 5u);
}

// ===========================================================================
// Edge Cases
// ===========================================================================

class BarEdgeCaseTest : public ::testing::Test {};

TEST_F(BarEdgeCaseTest, EmptySnapshotStreamProducesNoBars) {
    VolumeBarBuilder builder(100);
    std::vector<BookSnapshot> empty;
    auto bars = feed_snapshots(builder, empty);
    EXPECT_TRUE(bars.empty());

    auto flushed = builder.flush();
    EXPECT_FALSE(flushed.has_value());
}

TEST_F(BarEdgeCaseTest, SingleSnapshotWithNoTradeProducesNoBar) {
    VolumeBarBuilder builder(100);
    auto snap = make_snapshot(rth_ts(0));
    auto result = builder.on_snapshot(snap);
    EXPECT_FALSE(result.has_value());
}

TEST_F(BarEdgeCaseTest, VeryLargeVolumeThreshold) {
    VolumeBarBuilder builder(1000000);

    auto snaps = make_trade_sequence(rth_ts(0), 100, 10);
    auto bars = feed_snapshots(builder, snaps);
    EXPECT_TRUE(bars.empty());

    auto partial = builder.flush();
    ASSERT_TRUE(partial.has_value());
    EXPECT_EQ(partial->volume, 1000u);
}

TEST_F(BarEdgeCaseTest, BookStateAtBarCloseReflectsLastSnapshot) {
    VolumeBarBuilder builder(50);

    auto snap1 = make_snapshot_with_trade(rth_ts(0), 4500.25f, 25, 'B');
    snap1.spread = 0.25f;
    snap1.bids[0][0] = 4500.00f;
    snap1.bids[0][1] = 10.0f;
    snap1.asks[0][0] = 4500.25f;
    snap1.asks[0][1] = 8.0f;
    builder.on_snapshot(snap1);

    auto snap2 = make_snapshot_with_trade(rth_ts(100), 4500.25f, 25, 'A');
    snap2.spread = 0.50f;
    snap2.bids[0][0] = 4499.75f;
    snap2.bids[0][1] = 15.0f;
    snap2.asks[0][0] = 4500.25f;
    snap2.asks[0][1] = 12.0f;
    auto result = builder.on_snapshot(snap2);

    ASSERT_TRUE(result.has_value());
    EXPECT_FLOAT_EQ(result->bids[0][0], 4499.75f);
    EXPECT_FLOAT_EQ(result->bids[0][1], 15.0f);
    EXPECT_FLOAT_EQ(result->asks[0][0], 4500.25f);
    EXPECT_FLOAT_EQ(result->asks[0][1], 12.0f);
    EXPECT_FLOAT_EQ(result->spread, 0.50f);
}
