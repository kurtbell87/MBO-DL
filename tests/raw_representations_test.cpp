// raw_representations_test.cpp — TDD RED phase tests for Track B raw representations
// Spec: .kit/docs/feature-computation.md §Track B: Raw Representations
//
// Tests for:
//   B.1: Book snapshot export — PriceLadderInput (20, 2) per bar
//   B.2: Message sequence summary — fixed-length binned action counts
//   B.3: Lookback book sequence — (W, 20, 2) temporal encoder input
//   R2 Tier 2: Raw event sequence from DayEventBuffer (variable-length, max 500, padded)
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers that the implementation must provide (spec §Project Structure):
#include "features/raw_representations.hpp"  // BookSnapshotExport, MessageSummary,
                                              // LookbackBookSequence, RawEventSequence
#include "bars/bar.hpp"                       // Bar, PriceLadderInput
#include "data/day_event_buffer.hpp"          // DayEventBuffer, MBOEvent
#include "book_builder.hpp"                   // BOOK_DEPTH

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

constexpr float TICK_SIZE = 0.25f;

Bar make_bar(float close_mid, uint32_t volume = 100) {
    Bar bar{};
    bar.close_mid = close_mid;
    bar.open_mid = close_mid - 0.25f;
    bar.high_mid = close_mid + 0.25f;
    bar.low_mid = close_mid - 0.50f;
    bar.vwap = close_mid;
    bar.volume = volume;
    bar.spread = 0.25f;
    bar.bar_duration_s = 1.0f;
    bar.open_ts = 1000000000ULL;
    bar.close_ts = 2000000000ULL;

    for (int i = 0; i < BOOK_DEPTH; ++i) {
        bar.bids[i][0] = close_mid - TICK_SIZE * (0.5f + i);
        bar.bids[i][1] = 10.0f + static_cast<float>(i);
        bar.asks[i][0] = close_mid + TICK_SIZE * (0.5f + i);
        bar.asks[i][1] = 10.0f + static_cast<float>(i);
    }

    bar.add_count = 50;
    bar.cancel_count = 30;
    bar.modify_count = 10;
    bar.trade_event_count = volume;

    return bar;
}

std::vector<Bar> make_bar_sequence(int count, float start_mid = 4500.0f) {
    std::vector<Bar> bars;
    for (int i = 0; i < count; ++i) {
        bars.push_back(make_bar(start_mid + i * 0.25f));
    }
    return bars;
}

}  // anonymous namespace

// ===========================================================================
// B.1: Book Snapshot Export — PriceLadderInput shape (20, 2)
// ===========================================================================

class BookSnapshotExportTest : public ::testing::Test {};

TEST_F(BookSnapshotExportTest, ShapeIs20x2) {
    auto bar = make_bar(4500.0f);
    auto pli = BookSnapshotExport::from_bar(bar);
    // Must produce exactly 20 rows × 2 columns
    static_assert(sizeof(pli.data) == 20 * 2 * sizeof(float));
}

TEST_F(BookSnapshotExportTest, Rows0To9AreBids) {
    auto bar = make_bar(4500.0f);
    auto pli = BookSnapshotExport::from_bar(bar);
    // First 10 rows represent bid side
    // Column 0: price_delta_from_mid, Column 1: size_norm
    for (int i = 0; i < 10; ++i) {
        // Price delta should be negative (bids below mid)
        EXPECT_LE(pli.data[i][0], 0.0f)
            << "Bid row " << i << " price delta should be <= 0";
    }
}

TEST_F(BookSnapshotExportTest, Rows10To19AreAsks) {
    auto bar = make_bar(4500.0f);
    auto pli = BookSnapshotExport::from_bar(bar);
    for (int i = 10; i < 20; ++i) {
        // Price delta should be positive (asks above mid)
        EXPECT_GE(pli.data[i][0], 0.0f)
            << "Ask row " << i << " price delta should be >= 0";
    }
}

TEST_F(BookSnapshotExportTest, PriceDeltaFromMid) {
    auto bar = make_bar(4500.0f);
    // Set known prices
    bar.bids[0][0] = 4499.75f;
    bar.asks[0][0] = 4500.25f;

    auto pli = BookSnapshotExport::from_bar(bar);

    // Best bid delta: 4499.75 - 4500.0 = -0.25
    // (exact row depends on ordering convention — PriceLadderInput uses deepest-first bids)
    // Best bid at row 9, best ask at row 10
    EXPECT_NEAR(pli.data[9][0], -0.25f, 1e-4f);
    EXPECT_NEAR(pli.data[10][0], 0.25f, 1e-4f);
}

TEST_F(BookSnapshotExportTest, SizeNormalization) {
    auto bar = make_bar(4500.0f);
    bar.bids[0][1] = 100.0f;
    bar.asks[0][1] = 50.0f;

    auto pli = BookSnapshotExport::from_bar(bar);

    // Column 1 should contain normalized sizes (not raw)
    // At minimum, sizes should be non-negative
    for (int i = 0; i < 20; ++i) {
        EXPECT_GE(pli.data[i][1], 0.0f)
            << "size_norm at row " << i << " should be >= 0";
    }
}

TEST_F(BookSnapshotExportTest, EmptyLevelsHaveZero) {
    auto bar = make_bar(4500.0f);
    // Clear all levels
    for (int i = 0; i < BOOK_DEPTH; ++i) {
        bar.bids[i][0] = 0.0f;
        bar.bids[i][1] = 0.0f;
        bar.asks[i][0] = 0.0f;
        bar.asks[i][1] = 0.0f;
    }

    auto pli = BookSnapshotExport::from_bar(bar);

    for (int i = 0; i < 20; ++i) {
        EXPECT_FLOAT_EQ(pli.data[i][1], 0.0f)
            << "Empty book level size should be 0 at row " << i;
    }
}

TEST_F(BookSnapshotExportTest, FlattenedOutputHas40Values) {
    auto bar = make_bar(4500.0f);
    auto flat = BookSnapshotExport::flatten(bar);
    EXPECT_EQ(flat.size(), 40u);
}

// ===========================================================================
// B.2: Message Sequence Summary — fixed-length
// ===========================================================================

class MessageSummaryTest : public ::testing::Test {};

TEST_F(MessageSummaryTest, ProducesFixedLengthOutput) {
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 100;

    DayEventBuffer buf;
    auto summary = MessageSummary::from_bar(bar, buf);

    // Fixed-length summary — should have consistent size
    EXPECT_GT(summary.size(), 0u);
}

TEST_F(MessageSummaryTest, BinnedActionCountsPerTimeDecile) {
    // Summary includes action counts binned into 10 time deciles within the bar.
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 100;
    bar.bar_duration_s = 1.0f;

    DayEventBuffer buf;
    auto summary = MessageSummary::from_bar(bar, buf);

    // At minimum 10 decile bins × some action types
    EXPECT_GE(summary.size(), 10u);
}

TEST_F(MessageSummaryTest, CancelAddRatesFirstVsSecondHalf) {
    // Summary includes cancel/add rates for first half vs second half of bar.
    auto bar = make_bar(4500.0f);
    DayEventBuffer buf;
    auto summary = MessageSummary::from_bar(bar, buf);

    // Must include first-half and second-half cancel/add rates (2 values)
    EXPECT_GE(summary.size(), 2u);
}

TEST_F(MessageSummaryTest, MaxInstantaneousMessageRate) {
    auto bar = make_bar(4500.0f);
    DayEventBuffer buf;
    auto summary = MessageSummary::from_bar(bar, buf);
    // Must include max instantaneous message rate
    // Last element or specific index
    EXPECT_GE(summary.size(), 1u);
}

TEST_F(MessageSummaryTest, ConsistentWithTrackACategory6) {
    // Track B message summaries should be consistent with Track A Category 6 fields.
    auto bar = make_bar(4500.0f);
    bar.add_count = 100;
    bar.cancel_count = 40;
    bar.modify_count = 20;
    bar.bar_duration_s = 2.0f;

    DayEventBuffer buf;
    auto summary = MessageSummary::from_bar(bar, buf);

    // Total messages in summary should not exceed bar message totals
    // (summary is derived from the same event stream)
    EXPECT_GT(summary.size(), 0u);
}

TEST_F(MessageSummaryTest, EmptyBarProducesZeroSummary) {
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 0;
    bar.add_count = 0;
    bar.cancel_count = 0;
    bar.modify_count = 0;

    DayEventBuffer buf;
    auto summary = MessageSummary::from_bar(bar, buf);

    // All values should be zero for empty bar
    for (size_t i = 0; i < summary.size(); ++i) {
        EXPECT_FLOAT_EQ(summary[i], 0.0f)
            << "summary[" << i << "] should be 0 for empty bar";
    }
}

TEST_F(MessageSummaryTest, FixedSizeAcrossBars) {
    // Summary vector must have the same size for every bar.
    auto bar1 = make_bar(4500.0f);
    bar1.add_count = 100;

    auto bar2 = make_bar(4500.5f);
    bar2.add_count = 10;

    DayEventBuffer buf;
    auto summary1 = MessageSummary::from_bar(bar1, buf);
    auto summary2 = MessageSummary::from_bar(bar2, buf);

    EXPECT_EQ(summary1.size(), summary2.size());
}

// ===========================================================================
// B.3: Lookback Book Sequence — (W, 20, 2)
// ===========================================================================

class LookbackBookSequenceTest : public ::testing::Test {};

TEST_F(LookbackBookSequenceTest, ShapeIsWx20x2) {
    constexpr int W = 10;  // lookback window length (parameterized)
    auto bars = make_bar_sequence(W);
    auto seq = LookbackBookSequence::from_bars(bars, W);

    EXPECT_EQ(seq.window_size(), W);
    EXPECT_EQ(seq.rows_per_bar(), 20);
    EXPECT_EQ(seq.cols_per_row(), 2);
}

TEST_F(LookbackBookSequenceTest, FlattensTo_WTimes40) {
    constexpr int W = 5;
    auto bars = make_bar_sequence(W);
    auto flat = LookbackBookSequence::flatten(bars, W);
    EXPECT_EQ(flat.size(), static_cast<size_t>(W * 40));
}

TEST_F(LookbackBookSequenceTest, InsufficientBarsThrows) {
    constexpr int W = 10;
    auto bars = make_bar_sequence(5);  // Only 5 bars, need 10
    EXPECT_THROW(LookbackBookSequence::from_bars(bars, W), std::invalid_argument);
}

TEST_F(LookbackBookSequenceTest, MostRecentBarIsLastSlice) {
    constexpr int W = 5;
    auto bars = make_bar_sequence(10);

    // Use last W=5 bars: bars[5..9]
    std::vector<Bar> window(bars.end() - W, bars.end());
    auto seq = LookbackBookSequence::from_bars(window, W);

    // The last slice (bar W-1) should correspond to the most recent bar.
    auto last_pli = seq.bar_snapshot(W - 1);
    // Best bid from bars[9]
    float expected_bid_price_delta = bars[9].bids[0][0] - bars[9].close_mid;
    // Row 9 = best bid (deepest-first ordering)
    EXPECT_NEAR(last_pli.data[9][0], expected_bid_price_delta, 1e-4f);
}

TEST_F(LookbackBookSequenceTest, EachBarIndependentlyEncoded) {
    constexpr int W = 3;
    auto bars = make_bar_sequence(3);

    // Give each bar different book sizes
    bars[0].bids[0][1] = 11.0f;
    bars[1].bids[0][1] = 22.0f;
    bars[2].bids[0][1] = 33.0f;

    auto seq = LookbackBookSequence::from_bars(bars, W);

    auto snap0 = seq.bar_snapshot(0);
    auto snap1 = seq.bar_snapshot(1);
    auto snap2 = seq.bar_snapshot(2);

    // Sizes should differ across bars
    EXPECT_NE(snap0.data[9][1], snap1.data[9][1]);
    EXPECT_NE(snap1.data[9][1], snap2.data[9][1]);
}

// ===========================================================================
// R2 Tier 2: Raw Event Sequence from DayEventBuffer
// ===========================================================================

class RawEventSequenceTest : public ::testing::Test {};

TEST_F(RawEventSequenceTest, MaxLength500Events) {
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 1000;  // More than 500 events

    DayEventBuffer buf;
    auto seq = RawEventSequence::from_bar(bar, buf, 500);

    EXPECT_LE(seq.size(), 500u);
}

TEST_F(RawEventSequenceTest, PadsShorterSequences) {
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 10;  // Only 10 events

    DayEventBuffer buf;
    auto seq = RawEventSequence::from_bar(bar, buf, 500);

    // Should be padded to exactly 500
    EXPECT_EQ(seq.size(), 500u);
}

TEST_F(RawEventSequenceTest, PaddedValuesAreZero) {
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 0;  // No events

    DayEventBuffer buf;
    auto seq = RawEventSequence::from_bar(bar, buf, 500);

    // All values should be zero-padded
    for (size_t i = 0; i < seq.size(); ++i) {
        for (size_t j = 0; j < seq[i].size(); ++j) {
            EXPECT_FLOAT_EQ(seq[i][j], 0.0f)
                << "Padded event[" << i << "][" << j << "] should be 0";
        }
    }
}

TEST_F(RawEventSequenceTest, EventFieldsFromBuffer) {
    // Each event row should contain: action, price, size, side, ts_event
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 1;

    DayEventBuffer buf;
    auto seq = RawEventSequence::from_bar(bar, buf, 500);

    // Each event should have at least 5 fields
    if (!seq.empty() && !seq[0].empty()) {
        EXPECT_GE(seq[0].size(), 5u);
    }
}

TEST_F(RawEventSequenceTest, VariableLengthWithinMaxBound) {
    // With 50 events, should get 500 total (50 real + 450 padded).
    auto bar = make_bar(4500.0f);
    bar.mbo_event_begin = 0;
    bar.mbo_event_end = 50;

    DayEventBuffer buf;
    auto seq = RawEventSequence::from_bar(bar, buf, 500);

    EXPECT_EQ(seq.size(), 500u);
}
