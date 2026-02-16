// book_builder_test.cpp — TDD RED phase tests for BookBuilder
// Spec: .kit/docs/book-builder.md
//
// These tests exercise the BookBuilder public interface against synthetic
// MBO events. No real data dependency for unit tests.

#include <gtest/gtest.h>
#include "book_builder.hpp"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers — synthetic MBO event construction
// ---------------------------------------------------------------------------
// The BookBuilder must accept events with at minimum these fields:
//   order_id, instrument_id, action, side, price (int64, 1e-9 scale),
//   size (uint32), flags (uint8), ts_event (uint64 nanos since epoch).
//
// We define a plain struct here. The BookBuilder's process_event() (or
// equivalent) must accept this layout or something convertible from it.
// ---------------------------------------------------------------------------

namespace {

// Flag constants matching Databento spec
constexpr uint8_t F_LAST     = 0x80;
constexpr uint8_t F_SNAPSHOT = 0x20;

// Price helper: convert a human-readable dollar price to fixed-point int64
// at 1e-9 scale.  E.g., 4500.25 → 4500250000000LL
constexpr int64_t to_fixed(double price) {
    return static_cast<int64_t>(price * 1'000'000'000LL);
}

// Instrument IDs
constexpr uint32_t TARGET_ID = 13615;  // MESM2
constexpr uint32_t OTHER_ID  = 10039;  // MESU2

// Timestamp helpers — nanoseconds since epoch
// RTH: 09:30:00 ET through 16:00:00 ET
// Using a reference date of 2022-01-03 (Mon) where:
//   midnight ET (UTC-5) = 2022-01-03T05:00:00Z
//   09:30 ET = 2022-01-03T14:30:00Z
constexpr uint64_t MIDNIGHT_ET_NS  = 1641186000ULL * 1'000'000'000ULL;  // 2022-01-03 00:00 ET
constexpr uint64_t RTH_OPEN_NS     = MIDNIGHT_ET_NS + 9ULL * 3600 * 1'000'000'000ULL
                                                    + 30ULL * 60 * 1'000'000'000ULL;
constexpr uint64_t RTH_CLOSE_NS    = MIDNIGHT_ET_NS + 16ULL * 3600 * 1'000'000'000ULL;

// 100ms in nanoseconds
constexpr uint64_t BOUNDARY_100MS  = 100'000'000ULL;

// Build a timestamp at RTH open + offset_ms milliseconds
constexpr uint64_t rth_ts(uint64_t offset_ms) {
    return RTH_OPEN_NS + offset_ms * 1'000'000ULL;
}

// Pre-market timestamp: 09:29:00 ET + offset_ms
constexpr uint64_t pre_market_ts(uint64_t offset_ms) {
    return RTH_OPEN_NS - 60ULL * 1'000'000'000ULL + offset_ms * 1'000'000ULL;
}

// ---------------------------------------------------------------------------
// MboEvent — minimal synthetic event for feeding BookBuilder
// ---------------------------------------------------------------------------
struct MboEvent {
    uint64_t ts_event;
    uint64_t order_id;
    uint32_t instrument_id;
    char     action;     // 'A','C','M','R','T','F'
    char     side;       // 'A','B','N'
    int64_t  price;      // fixed-point 1e-9
    uint32_t size;
    uint8_t  flags;
};

MboEvent make_add(uint64_t ts, uint64_t order_id, char side, double price,
                  uint32_t size, uint8_t flags = F_LAST,
                  uint32_t instrument_id = TARGET_ID) {
    return {ts, order_id, instrument_id, 'A', side, to_fixed(price), size, flags};
}

MboEvent make_cancel(uint64_t ts, uint64_t order_id, char side, double price,
                     uint32_t size, uint8_t flags = F_LAST,
                     uint32_t instrument_id = TARGET_ID) {
    return {ts, order_id, instrument_id, 'C', side, to_fixed(price), size, flags};
}

MboEvent make_modify(uint64_t ts, uint64_t order_id, char side, double new_price,
                     uint32_t new_size, uint8_t flags = F_LAST,
                     uint32_t instrument_id = TARGET_ID) {
    return {ts, order_id, instrument_id, 'M', side, to_fixed(new_price), new_size, flags};
}

MboEvent make_trade(uint64_t ts, uint64_t order_id, char side, double price,
                    uint32_t size, uint8_t flags = F_LAST,
                    uint32_t instrument_id = TARGET_ID) {
    return {ts, order_id, instrument_id, 'T', side, to_fixed(price), size, flags};
}

MboEvent make_fill(uint64_t ts, uint64_t order_id, char side, double price,
                   uint32_t remaining_size, uint8_t flags = F_LAST,
                   uint32_t instrument_id = TARGET_ID) {
    return {ts, order_id, instrument_id, 'F', side, to_fixed(price), remaining_size, flags};
}

MboEvent make_clear(uint64_t ts, uint8_t flags = F_LAST,
                    uint32_t instrument_id = TARGET_ID) {
    return {ts, 0, instrument_id, 'R', 'N', 0, 0, flags};
}

// Helper: build a minimal book with one bid and one ask, returns the events
std::vector<MboEvent> build_minimal_book(uint64_t ts, double bid_price = 4500.00,
                                         double ask_price = 4500.25) {
    return {
        make_add(ts, 100, 'B', bid_price, 5, 0),          // bid, no F_LAST
        make_add(ts, 200, 'A', ask_price, 3, F_LAST),     // ask, F_LAST
    };
}

// Helper: process a vector of events through a BookBuilder
void feed_events(BookBuilder& builder, const std::vector<MboEvent>& events) {
    for (const auto& e : events) {
        builder.process_event(e.ts_event, e.order_id, e.instrument_id,
                              e.action, e.side, e.price, e.size, e.flags);
    }
}

}  // anonymous namespace

// ===========================================================================
// Test Fixture
// ===========================================================================

class BookBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a BookBuilder targeting MESM2, default RTH session
        builder_ = std::make_unique<BookBuilder>(TARGET_ID);
    }

    std::unique_ptr<BookBuilder> builder_;
};

// ===========================================================================
// 1. Add orders build book
// ===========================================================================
TEST_F(BookBuilderTest, AddOrdersBuildBook) {
    // Feed Add events during pre-market to populate the book,
    // then emit a snapshot at RTH boundary.
    uint64_t ts = pre_market_ts(0);

    // Add 3 bid levels
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'B', to_fixed(4499.75), 5, 0);
    builder_->process_event(ts, 3, TARGET_ID, 'A', 'B', to_fixed(4499.50), 8, F_LAST);

    // Add 3 ask levels
    ts += 1000;
    builder_->process_event(ts, 4, TARGET_ID, 'A', 'A', to_fixed(4500.25), 7, 0);
    builder_->process_event(ts, 5, TARGET_ID, 'A', 'A', to_fixed(4500.50), 3, 0);
    builder_->process_event(ts, 6, TARGET_ID, 'A', 'A', to_fixed(4500.75), 12, F_LAST);

    // Emit snapshots at RTH open
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    const auto& snap = snapshots[0];

    // Best bid = 4500.00, size = 10
    EXPECT_NEAR(snap.bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snap.bids[0][1], 10.0f, 0.01f);

    // Second bid = 4499.75, size = 5
    EXPECT_NEAR(snap.bids[1][0], 4499.75f, 0.01f);
    EXPECT_NEAR(snap.bids[1][1], 5.0f, 0.01f);

    // Third bid = 4499.50, size = 8
    EXPECT_NEAR(snap.bids[2][0], 4499.50f, 0.01f);
    EXPECT_NEAR(snap.bids[2][1], 8.0f, 0.01f);

    // Best ask = 4500.25, size = 7
    EXPECT_NEAR(snap.asks[0][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snap.asks[0][1], 7.0f, 0.01f);

    // Second ask = 4500.50, size = 3
    EXPECT_NEAR(snap.asks[1][0], 4500.50f, 0.01f);
    EXPECT_NEAR(snap.asks[1][1], 3.0f, 0.01f);

    // Third ask = 4500.75, size = 12
    EXPECT_NEAR(snap.asks[2][0], 4500.75f, 0.01f);
    EXPECT_NEAR(snap.asks[2][1], 12.0f, 0.01f);
}

// ===========================================================================
// 1b. Add orders — size aggregation at same price level
// ===========================================================================
TEST_F(BookBuilderTest, AddOrdersAggregateSamePriceLevel) {
    uint64_t ts = pre_market_ts(0);

    // Two bid orders at the same price
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'B', to_fixed(4500.00), 7, 0);
    // One ask
    builder_->process_event(ts, 3, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Aggregated bid size at 4500.00 = 10 + 7 = 17
    EXPECT_NEAR(snapshots[0].bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 17.0f, 0.01f);
}

// ===========================================================================
// 2. Cancel removes order
// ===========================================================================
TEST_F(BookBuilderTest, CancelRemovesOrder) {
    uint64_t ts = pre_market_ts(0);

    // Add a single bid order
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, F_LAST);
    // Add an ask so book isn't empty
    ts += 1000;
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Cancel the bid order
    ts += 1000;
    builder_->process_event(ts, 1, TARGET_ID, 'C', 'B', to_fixed(4500.00), 10, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Bid level at 4500.00 should be gone — best bid should be zero-padded
    EXPECT_NEAR(snapshots[0].bids[0][0], 0.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 0.0f, 0.01f);
}

// ===========================================================================
// 2b. Cancel reduces aggregated size, removes level when zero
// ===========================================================================
TEST_F(BookBuilderTest, CancelReducesSizeRemovesLevelAtZero) {
    uint64_t ts = pre_market_ts(0);

    // Two orders at same price
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'B', to_fixed(4500.00), 7, 0);
    builder_->process_event(ts, 3, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Cancel one order: level size should drop to 7
    ts += 1000;
    builder_->process_event(ts, 1, TARGET_ID, 'C', 'B', to_fixed(4500.00), 10, F_LAST);

    auto snap1 = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snap1.size(), 1u);
    EXPECT_NEAR(snap1[0].bids[0][1], 7.0f, 0.01f);

    // Cancel remaining order: level should be removed entirely
    ts += 1000;
    builder_->process_event(ts, 2, TARGET_ID, 'C', 'B', to_fixed(4500.00), 7, F_LAST);

    auto snap2 = builder_->emit_snapshots(RTH_OPEN_NS + BOUNDARY_100MS,
                                           RTH_OPEN_NS + 2 * BOUNDARY_100MS);
    ASSERT_GE(snap2.size(), 1u);
    EXPECT_NEAR(snap2[0].bids[0][0], 0.0f, 0.01f);
    EXPECT_NEAR(snap2[0].bids[0][1], 0.0f, 0.01f);
}

// ===========================================================================
// 3. Modify updates price and size
// ===========================================================================
TEST_F(BookBuilderTest, ModifyUpdatesPriceAndSize) {
    uint64_t ts = pre_market_ts(0);

    // Add bid at 4500.00 size 10
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Modify order 1: new price 4499.75, new size 15
    ts += 1000;
    builder_->process_event(ts, 1, TARGET_ID, 'M', 'B', to_fixed(4499.75), 15, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Old level at 4500.00 should be gone
    // New best bid should be at 4499.75, size 15
    EXPECT_NEAR(snapshots[0].bids[0][0], 4499.75f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 15.0f, 0.01f);

    // 4500.00 level should not exist (no second level with that price)
    // Since it was the only order at that price, level is removed
    // bids[1] should be zero-padded
    EXPECT_NEAR(snapshots[0].bids[1][0], 0.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[1][1], 0.0f, 0.01f);
}

// ===========================================================================
// 4. Trade appends to trade buffer
// ===========================================================================
TEST_F(BookBuilderTest, TradeAppendsToTradeBuffer) {
    uint64_t ts = pre_market_ts(0);

    // Build a minimal book first
    auto book_events = build_minimal_book(ts);
    feed_events(*builder_, book_events);

    // Send trade events
    ts = rth_ts(10);  // 10ms into RTH
    builder_->process_event(ts, 300, TARGET_ID, 'T', 'B', to_fixed(4500.25), 2, F_LAST);

    ts = rth_ts(50);  // 50ms into RTH
    builder_->process_event(ts, 301, TARGET_ID, 'T', 'A', to_fixed(4500.00), 3, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // With left-padding: 50 slots, 2 real trades at positions [48] and [49]
    // Trade at [48]: price=4500.25, size=2, side=+1.0 (Buy aggressor 'B')
    EXPECT_NEAR(snapshots[0].trades[48][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[48][1], 2.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[48][2], 1.0f, 0.01f);  // 'B' = +1.0

    // Trade at [49]: price=4500.00, size=3, side=-1.0 (Sell aggressor 'A')
    EXPECT_NEAR(snapshots[0].trades[49][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[49][1], 3.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[49][2], -1.0f, 0.01f);  // 'A' = -1.0

    // Padding: first 48 entries should be zero
    for (int i = 0; i < 48; ++i) {
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][0], 0.0f) << "trade[" << i << "] price not zero";
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][1], 0.0f) << "trade[" << i << "] size not zero";
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][2], 0.0f) << "trade[" << i << "] side not zero";
    }
}

// ===========================================================================
// 5. Fill reduces passive order
// ===========================================================================
TEST_F(BookBuilderTest, FillPartiallyReducesOrder) {
    uint64_t ts = pre_market_ts(0);

    // Add bid order: size 10
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Partial fill: remaining size = 6 (filled 4)
    ts += 1000;
    builder_->process_event(ts, 1, TARGET_ID, 'F', 'B', to_fixed(4500.00), 6, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Bid level size should now be 6
    EXPECT_NEAR(snapshots[0].bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 6.0f, 0.01f);
}

TEST_F(BookBuilderTest, FillFullyRemovesOrder) {
    uint64_t ts = pre_market_ts(0);

    // Add bid order: size 10
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Full fill: remaining size = 0
    ts += 1000;
    builder_->process_event(ts, 1, TARGET_ID, 'F', 'B', to_fixed(4500.00), 0, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Bid level should be removed entirely
    EXPECT_NEAR(snapshots[0].bids[0][0], 0.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 0.0f, 0.01f);
}

TEST_F(BookBuilderTest, FillDoesNotAppendToTradeBuffer) {
    uint64_t ts = pre_market_ts(0);

    // Add orders
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Fill (passive side) — should NOT append to trade buffer
    ts += 1000;
    builder_->process_event(ts, 1, TARGET_ID, 'F', 'B', to_fixed(4500.00), 6, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // All 50 trade slots should be zero (no trades recorded)
    for (int i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][0], 0.0f)
            << "Fill should not add to trade buffer, but trade[" << i << "] has non-zero price";
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][1], 0.0f)
            << "Fill should not add to trade buffer, but trade[" << i << "] has non-zero size";
    }
}

// ===========================================================================
// 6. Clear resets book
// ===========================================================================
TEST_F(BookBuilderTest, ClearResetsBook) {
    uint64_t ts = pre_market_ts(0);

    // Populate book
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Clear
    ts += 1000;
    builder_->process_event(ts, 0, TARGET_ID, 'R', 'N', 0, 0, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    // Book is empty — if session filtering skips empty books, we may get 0 snapshots.
    // But the spec says: if book empty at open, skip forward until at least 1 bid+ask.
    // After a clear with no rebuilds, we should get 0 snapshots (all skipped).
    // OR if we get a snapshot, all levels should be zero-padded.
    if (!snapshots.empty()) {
        for (int i = 0; i < 10; ++i) {
            EXPECT_FLOAT_EQ(snapshots[0].bids[i][0], 0.0f)
                << "After clear, bids[" << i << "] price should be 0";
            EXPECT_FLOAT_EQ(snapshots[0].bids[i][1], 0.0f)
                << "After clear, bids[" << i << "] size should be 0";
            EXPECT_FLOAT_EQ(snapshots[0].asks[i][0], 0.0f)
                << "After clear, asks[" << i << "] price should be 0";
            EXPECT_FLOAT_EQ(snapshots[0].asks[i][1], 0.0f)
                << "After clear, asks[" << i << "] size should be 0";
        }
    }
}

// ===========================================================================
// 6b. Clear with F_SNAPSHOT begins snapshot sequence
// ===========================================================================
TEST_F(BookBuilderTest, ClearWithSnapshotFlagBeginsSnapshotSequence) {
    uint64_t ts = pre_market_ts(0);

    // Populate book initially
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Clear with F_SNAPSHOT — resets book and starts snapshot ingestion
    ts += 1000;
    builder_->process_event(ts, 0, TARGET_ID, 'R', 'N', 0, 0, F_SNAPSHOT);

    // Snapshot adds (with F_SNAPSHOT flag)
    ts += 100;
    builder_->process_event(ts, 10, TARGET_ID, 'A', 'B', to_fixed(4501.00), 20, F_SNAPSHOT);
    builder_->process_event(ts, 11, TARGET_ID, 'A', 'A', to_fixed(4501.25), 15,
                            F_SNAPSHOT | F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Book should reflect the snapshot data, not the old data
    EXPECT_NEAR(snapshots[0].bids[0][0], 4501.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 20.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][0], 4501.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][1], 15.0f, 0.01f);
}

// ===========================================================================
// 7. F_LAST batching
// ===========================================================================
TEST_F(BookBuilderTest, BatchProcessingRespectsF_LAST) {
    uint64_t ts = pre_market_ts(0);

    // Send multiple events without F_LAST — intermediate states should not be visible
    // to snapshot emission logic as consistent states.
    // Add bid without F_LAST
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    // Add ask without F_LAST
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, 0);
    // Add another bid with F_LAST — now batch is complete
    builder_->process_event(ts, 3, TARGET_ID, 'A', 'B', to_fixed(4499.75), 8, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // All three orders should be reflected (batch committed atomically)
    EXPECT_NEAR(snapshots[0].bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 10.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[1][0], 4499.75f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[1][1], 8.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][1], 5.0f, 0.01f);
}

TEST_F(BookBuilderTest, IntermediateStateNotVisibleWithoutF_LAST) {
    // This test verifies that snapshot emission only sees the book state
    // after an F_LAST commit, not after intermediate events.

    uint64_t ts = pre_market_ts(0);

    // First batch: establish a book
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Second batch during RTH, starts but doesn't complete (no F_LAST)
    uint64_t ts2 = rth_ts(10);
    builder_->process_event(ts2, 3, TARGET_ID, 'A', 'B', to_fixed(4500.00), 20, 0);
    // Note: no F_LAST sent yet

    // Emit snapshot at the 100ms boundary — should see the first batch state,
    // not the uncommitted second batch
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Bid size at 4500.00 should be 10 (from first batch), not 30
    EXPECT_NEAR(snapshots[0].bids[0][1], 10.0f, 0.01f);
}

// ===========================================================================
// 8. Snapshot emission at 100ms boundaries
// ===========================================================================
TEST_F(BookBuilderTest, SnapshotEmissionAt100msBoundaries) {
    uint64_t ts = pre_market_ts(0);

    // Build book in pre-market
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Emit snapshots for first 500ms of RTH (should get 5 snapshots)
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + 5 * BOUNDARY_100MS);
    ASSERT_EQ(snapshots.size(), 5u);

    // Verify timestamps are at exact 100ms boundaries
    for (size_t i = 0; i < snapshots.size(); ++i) {
        uint64_t expected_ts = RTH_OPEN_NS + i * BOUNDARY_100MS;
        EXPECT_EQ(snapshots[i].timestamp, expected_ts)
            << "Snapshot " << i << " timestamp not at expected 100ms boundary";
    }
}

TEST_F(BookBuilderTest, SnapshotTimestampsAlignedToSessionClock) {
    uint64_t ts = pre_market_ts(0);

    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Feed an event at a non-boundary timestamp during RTH
    uint64_t ts_mid = rth_ts(150);  // 150ms into RTH — between boundaries
    builder_->process_event(ts_mid, 50, TARGET_ID, 'A', 'B', to_fixed(4499.50), 3, F_LAST);

    // Emit from 100ms to 300ms
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS + BOUNDARY_100MS,
                                               RTH_OPEN_NS + 3 * BOUNDARY_100MS);
    ASSERT_EQ(snapshots.size(), 2u);

    // Boundary at 100ms should NOT include the 150ms event
    // Boundary at 200ms SHOULD include it (it arrived before 200ms)
    EXPECT_EQ(snapshots[0].timestamp, RTH_OPEN_NS + BOUNDARY_100MS);
    EXPECT_EQ(snapshots[1].timestamp, RTH_OPEN_NS + 2 * BOUNDARY_100MS);

    // The 200ms snapshot should have the extra bid level at 4499.50
    EXPECT_NEAR(snapshots[1].bids[1][0], 4499.50f, 0.01f);
    EXPECT_NEAR(snapshots[1].bids[1][1], 3.0f, 0.01f);
}

// ===========================================================================
// 9. Level padding (fewer than 10 levels → zero-padded)
// ===========================================================================
TEST_F(BookBuilderTest, LevelPaddingToTen) {
    uint64_t ts = pre_market_ts(0);

    // Add only 2 bid levels and 1 ask level
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'B', to_fixed(4499.75), 5, 0);
    builder_->process_event(ts, 3, TARGET_ID, 'A', 'A', to_fixed(4500.25), 7, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    const auto& snap = snapshots[0];

    // Bids: 2 real levels, then 8 zero-padded
    EXPECT_NEAR(snap.bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snap.bids[1][0], 4499.75f, 0.01f);
    for (int i = 2; i < 10; ++i) {
        EXPECT_FLOAT_EQ(snap.bids[i][0], 0.0f) << "bids[" << i << "] price should be 0 (padded)";
        EXPECT_FLOAT_EQ(snap.bids[i][1], 0.0f) << "bids[" << i << "] size should be 0 (padded)";
    }

    // Asks: 1 real level, then 9 zero-padded
    EXPECT_NEAR(snap.asks[0][0], 4500.25f, 0.01f);
    for (int i = 1; i < 10; ++i) {
        EXPECT_FLOAT_EQ(snap.asks[i][0], 0.0f) << "asks[" << i << "] price should be 0 (padded)";
        EXPECT_FLOAT_EQ(snap.asks[i][1], 0.0f) << "asks[" << i << "] size should be 0 (padded)";
    }
}

// ===========================================================================
// 10. Trade buffer left-padding (fewer than 50 trades)
// ===========================================================================
TEST_F(BookBuilderTest, TradeBufferLeftPadding) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Add 3 trades
    ts = rth_ts(10);
    builder_->process_event(ts, 300, TARGET_ID, 'T', 'B', to_fixed(4500.25), 2, F_LAST);
    ts = rth_ts(30);
    builder_->process_event(ts, 301, TARGET_ID, 'T', 'A', to_fixed(4500.00), 1, F_LAST);
    ts = rth_ts(50);
    builder_->process_event(ts, 302, TARGET_ID, 'T', 'B', to_fixed(4500.25), 5, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Left-padded: first 47 slots are zeros, last 3 are real trades
    for (int i = 0; i < 47; ++i) {
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][0], 0.0f) << "trade[" << i << "] should be zero-padded";
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][1], 0.0f);
        EXPECT_FLOAT_EQ(snapshots[0].trades[i][2], 0.0f);
    }

    // Real trades at positions 47, 48, 49
    EXPECT_NEAR(snapshots[0].trades[47][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[47][1], 2.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[47][2], 1.0f, 0.01f);  // 'B' = +1.0

    EXPECT_NEAR(snapshots[0].trades[48][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[48][1], 1.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[48][2], -1.0f, 0.01f);  // 'A' = -1.0

    EXPECT_NEAR(snapshots[0].trades[49][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[49][1], 5.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[49][2], 1.0f, 0.01f);  // 'B' = +1.0
}

TEST_F(BookBuilderTest, TradeBufferRollsAtFifty) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Feed 52 trades — buffer should contain last 50
    for (uint64_t i = 0; i < 52; ++i) {
        uint64_t trade_ts = rth_ts(i);
        double price = 4500.00 + (i % 2) * 0.25;
        char side = (i % 2 == 0) ? 'B' : 'A';
        builder_->process_event(trade_ts, 1000 + i, TARGET_ID, 'T', side,
                                to_fixed(price), static_cast<uint32_t>(i + 1), F_LAST);
    }

    auto snapshots = builder_->emit_snapshots(rth_ts(52), rth_ts(52) + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // No zero-padding when buffer is full (52 trades, last 50 kept)
    // First trade in buffer should be trade #2 (0-indexed), not trade #0
    // Trade #2: price=4500.00, size=3, side='B' → +1.0
    EXPECT_NEAR(snapshots[0].trades[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[0][1], 3.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[0][2], 1.0f, 0.01f);

    // Last trade in buffer should be trade #51: price=4500.25, size=52, side='A' → -1.0
    EXPECT_NEAR(snapshots[0].trades[49][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[49][1], 52.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[49][2], -1.0f, 0.01f);
}

// ===========================================================================
// 11. Price conversion (fixed-point int64 → float32)
// ===========================================================================
TEST_F(BookBuilderTest, PriceConversionFixedPointToFloat) {
    uint64_t ts = pre_market_ts(0);

    // Use prices that exercise the fixed-point conversion
    // 4500.25 → 4500250000000 (int64 at 1e-9 scale)
    // 4499.75 → 4499750000000
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', 4500250000000LL, 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', 4499750000000LL, 5, F_LAST);

    // Note: bids are descending, so the only bid is 4500.25
    // asks are ascending, so the only ask is 4499.75 — wait, that's crossed.
    // Let's fix: bid < ask
    builder_ = std::make_unique<BookBuilder>(TARGET_ID);
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', 4499750000000LL, 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', 4500250000000LL, 5, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Verify conversion: int64 4499750000000 / 1e9 = 4499.75
    EXPECT_NEAR(snapshots[0].bids[0][0], 4499.75f, 0.01f);
    // Verify conversion: int64 4500250000000 / 1e9 = 4500.25
    EXPECT_NEAR(snapshots[0].asks[0][0], 4500.25f, 0.01f);
}

// ===========================================================================
// 12. Mid price and spread
// ===========================================================================
TEST_F(BookBuilderTest, MidPriceAndSpreadComputation) {
    uint64_t ts = pre_market_ts(0);

    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // mid_price = (4500.00 + 4500.25) / 2 = 4500.125
    EXPECT_NEAR(snapshots[0].mid_price, 4500.125f, 0.01f);

    // spread = 4500.25 - 4500.00 = 0.25 (1 tick for MES)
    EXPECT_NEAR(snapshots[0].spread, 0.25f, 0.01f);
}

TEST_F(BookBuilderTest, MidPriceAndSpreadCarryForwardWhenSideEmpty) {
    uint64_t ts = pre_market_ts(0);

    // Build a book with bid and ask
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Cancel all asks — one side is now empty
    ts = rth_ts(10);
    builder_->process_event(ts, 2, TARGET_ID, 'C', 'A', to_fixed(4500.25), 5, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + 2 * BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 2u);

    // First snapshot (at boundary 0): both sides present
    EXPECT_NEAR(snapshots[0].mid_price, 4500.125f, 0.01f);
    EXPECT_NEAR(snapshots[0].spread, 0.25f, 0.01f);

    // Second snapshot (at boundary 100ms): ask side empty, carry forward
    EXPECT_NEAR(snapshots[1].mid_price, 4500.125f, 0.01f);
    EXPECT_NEAR(snapshots[1].spread, 0.25f, 0.01f);
}

TEST_F(BookBuilderTest, MidPriceAndSpreadDefaultZeroWhenNeverComputed) {
    // If book is empty from the start and somehow we get a snapshot,
    // mid_price and spread should be 0.0
    uint64_t ts = pre_market_ts(0);

    // Add only a bid — no ask exists
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, F_LAST);

    // The spec says: skip forward until at least 1 bid and 1 ask exist.
    // If the implementation still emits a snapshot (e.g., for testing), verify defaults.
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);

    // If no snapshots emitted (correct behavior for empty ask side), that's fine.
    // If a snapshot IS emitted, mid_price and spread must be 0.0
    if (!snapshots.empty()) {
        EXPECT_FLOAT_EQ(snapshots[0].mid_price, 0.0f);
        EXPECT_FLOAT_EQ(snapshots[0].spread, 0.0f);
    }
}

// ===========================================================================
// 13. Instrument filtering
// ===========================================================================
TEST_F(BookBuilderTest, InstrumentFilteringIgnoresOtherInstruments) {
    uint64_t ts = pre_market_ts(0);

    // Add orders for the TARGET instrument
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Add orders for a DIFFERENT instrument — should be ignored
    builder_->process_event(ts, 3, OTHER_ID, 'A', 'B', to_fixed(3000.00), 99, 0);
    builder_->process_event(ts, 4, OTHER_ID, 'A', 'A', to_fixed(3000.25), 88, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Only TARGET_ID data should be in the snapshot
    EXPECT_NEAR(snapshots[0].bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 10.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][1], 5.0f, 0.01f);

    // Verify the other instrument's prices do NOT appear anywhere
    for (int i = 0; i < 10; ++i) {
        EXPECT_NE(snapshots[0].bids[i][0], 3000.00f)
            << "Other instrument bid price leaked into snapshot";
        EXPECT_NE(snapshots[0].asks[i][0], 3000.25f)
            << "Other instrument ask price leaked into snapshot";
    }
}

TEST_F(BookBuilderTest, InstrumentFilteringIgnoresTradesForOtherInstruments) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Trade for another instrument — should be ignored
    ts = rth_ts(10);
    builder_->process_event(ts, 500, OTHER_ID, 'T', 'B', to_fixed(3000.00), 50, F_LAST);

    // Trade for our instrument
    ts = rth_ts(20);
    builder_->process_event(ts, 501, TARGET_ID, 'T', 'B', to_fixed(4500.25), 2, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // Only 1 trade should be in the buffer (at position 49, left-padded)
    EXPECT_NEAR(snapshots[0].trades[49][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].trades[49][1], 2.0f, 0.01f);

    // Position 48 should be zero (the other-instrument trade was filtered out)
    EXPECT_FLOAT_EQ(snapshots[0].trades[48][0], 0.0f);
}

// ===========================================================================
// 14. Gap warning (>5s gap during RTH, carry forward state)
// ===========================================================================
TEST_F(BookBuilderTest, GapCarriesForwardLastKnownState) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // No events for the first 300ms of RTH — snapshots should carry forward
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + 3 * BOUNDARY_100MS);
    ASSERT_EQ(snapshots.size(), 3u);

    // All three snapshots should have the same book state (carried forward)
    for (size_t i = 0; i < snapshots.size(); ++i) {
        EXPECT_NEAR(snapshots[i].bids[0][0], 4500.00f, 0.01f)
            << "Snapshot " << i << " should carry forward bid";
        EXPECT_NEAR(snapshots[i].asks[0][0], 4500.25f, 0.01f)
            << "Snapshot " << i << " should carry forward ask";
    }
}

TEST_F(BookBuilderTest, LargeGapDuringRTHLogsWarning) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Simulate a >5 second gap: emit snapshots over 6 seconds with no events
    uint64_t six_seconds = 6'000'000'000ULL;
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + six_seconds);

    // Should still get 60 snapshots (6s / 100ms)
    EXPECT_EQ(snapshots.size(), 60u);

    // All should carry forward the pre-market book state
    for (const auto& snap : snapshots) {
        EXPECT_NEAR(snap.bids[0][0], 4500.00f, 0.01f);
        EXPECT_NEAR(snap.asks[0][0], 4500.25f, 0.01f);
    }

    // Note: we cannot easily assert that a WARNING was logged in a unit test
    // without injecting a logger. The implementation should log a warning for
    // gaps > 5s during RTH. This test verifies the data behavior (carry forward).
}

// ===========================================================================
// 15. Level ordering assertion (bids descending, asks ascending)
// ===========================================================================
TEST_F(BookBuilderTest, LevelOrderingBidsDescendingAsksAscending) {
    uint64_t ts = pre_market_ts(0);

    // Add bids at various prices (inserted out of order)
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4499.50), 5, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 3, TARGET_ID, 'A', 'B', to_fixed(4499.75), 8, 0);

    // Add asks at various prices (inserted out of order)
    builder_->process_event(ts, 4, TARGET_ID, 'A', 'A', to_fixed(4500.75), 3, 0);
    builder_->process_event(ts, 5, TARGET_ID, 'A', 'A', to_fixed(4500.25), 7, 0);
    builder_->process_event(ts, 6, TARGET_ID, 'A', 'A', to_fixed(4500.50), 12, F_LAST);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    const auto& snap = snapshots[0];

    // Bids: best (highest) first — 4500.00, 4499.75, 4499.50
    EXPECT_NEAR(snap.bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snap.bids[1][0], 4499.75f, 0.01f);
    EXPECT_NEAR(snap.bids[2][0], 4499.50f, 0.01f);

    // Verify descending order
    for (int i = 0; i < 2; ++i) {
        if (snap.bids[i + 1][0] > 0.0f) {
            EXPECT_GE(snap.bids[i][0], snap.bids[i + 1][0])
                << "Bids must be descending: bids[" << i << "] >= bids[" << i + 1 << "]";
        }
    }

    // Asks: best (lowest) first — 4500.25, 4500.50, 4500.75
    EXPECT_NEAR(snap.asks[0][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snap.asks[1][0], 4500.50f, 0.01f);
    EXPECT_NEAR(snap.asks[2][0], 4500.75f, 0.01f);

    // Verify ascending order
    for (int i = 0; i < 2; ++i) {
        if (snap.asks[i + 1][0] > 0.0f) {
            EXPECT_LE(snap.asks[i][0], snap.asks[i + 1][0])
                << "Asks must be ascending: asks[" << i << "] <= asks[" << i + 1 << "]";
        }
    }
}

// ===========================================================================
// 16. Session filtering (snapshots only during RTH)
// ===========================================================================
TEST_F(BookBuilderTest, SessionFilteringOnlyEmitsDuringRTH) {
    uint64_t ts = pre_market_ts(0);  // 09:29:00 ET
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Request snapshots that span pre-market → RTH start
    // Pre-market: 09:29:00 → 09:30:00 (should NOT get snapshots)
    uint64_t pre_rth_start = MIDNIGHT_ET_NS + 9ULL * 3600 * 1'000'000'000ULL
                             + 29ULL * 60 * 1'000'000'000ULL;
    auto snapshots = builder_->emit_snapshots(pre_rth_start, RTH_OPEN_NS);
    EXPECT_EQ(snapshots.size(), 0u) << "No snapshots should be emitted before RTH";
}

TEST_F(BookBuilderTest, SessionFilteringNoSnapshotsAfterRTHClose) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Request snapshots after RTH close
    auto snapshots = builder_->emit_snapshots(RTH_CLOSE_NS, RTH_CLOSE_NS + BOUNDARY_100MS);
    EXPECT_EQ(snapshots.size(), 0u) << "No snapshots should be emitted after RTH close";
}

TEST_F(BookBuilderTest, SessionFilteringProcessesPreMarketEventsForBookState) {
    // Pre-market events should update the book state, even though no snapshots
    // are emitted during pre-market. The RTH snapshot should reflect pre-market state.
    uint64_t ts = pre_market_ts(0);  // 09:29:00 ET

    // Pre-market activity
    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, 0);
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    ts = pre_market_ts(30000);  // 09:29:30 ET
    builder_->process_event(ts, 3, TARGET_ID, 'A', 'B', to_fixed(4499.75), 8, F_LAST);

    // At RTH open, the book should have all pre-market orders
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    EXPECT_NEAR(snapshots[0].bids[0][0], 4500.00f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[0][1], 10.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[1][0], 4499.75f, 0.01f);
    EXPECT_NEAR(snapshots[0].bids[1][1], 8.0f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][0], 4500.25f, 0.01f);
    EXPECT_NEAR(snapshots[0].asks[0][1], 5.0f, 0.01f);
}

TEST_F(BookBuilderTest, EmptyBookAtSessionStartSkipsUntilBidAndAskExist) {
    // Book is empty at 09:30:00 — should skip snapshots until at least 1 bid + 1 ask
    uint64_t ts = rth_ts(250);  // 250ms into RTH — first order arrives

    builder_->process_event(ts, 1, TARGET_ID, 'A', 'B', to_fixed(4500.00), 10, F_LAST);
    // Still no ask — snapshots should be skipped

    ts = rth_ts(450);  // 450ms into RTH — ask arrives
    builder_->process_event(ts, 2, TARGET_ID, 'A', 'A', to_fixed(4500.25), 5, F_LAST);

    // Request snapshots for the first second of RTH
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + 10 * BOUNDARY_100MS);

    // Snapshots at 0ms, 100ms, 200ms, 300ms, 400ms should be skipped (empty book or no ask)
    // First valid snapshot should be at 500ms boundary (after both bid and ask exist)
    ASSERT_FALSE(snapshots.empty());

    // The first emitted snapshot should be at or after 500ms
    EXPECT_GE(snapshots[0].timestamp, RTH_OPEN_NS + 5 * BOUNDARY_100MS);

    // And it should have both bid and ask
    EXPECT_GT(snapshots[0].bids[0][0], 0.0f);
    EXPECT_GT(snapshots[0].asks[0][0], 0.0f);
}

// ===========================================================================
// 17. Time of day (fractional hours since midnight ET)
// ===========================================================================
TEST_F(BookBuilderTest, TimeOfDayComputation) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + BOUNDARY_100MS);
    ASSERT_GE(snapshots.size(), 1u);

    // RTH open = 09:30:00 ET → 9.5 fractional hours since midnight
    EXPECT_NEAR(snapshots[0].time_of_day, 9.5f, 0.001f);
}

TEST_F(BookBuilderTest, TimeOfDayAt100msBoundaries) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Get 10 snapshots (1 second)
    auto snapshots = builder_->emit_snapshots(RTH_OPEN_NS, RTH_OPEN_NS + 10 * BOUNDARY_100MS);
    ASSERT_EQ(snapshots.size(), 10u);

    // 100ms = 0.1/3600 hours = ~0.0000278 hours
    float base_tod = 9.5f;
    for (size_t i = 0; i < snapshots.size(); ++i) {
        float expected_tod = base_tod + static_cast<float>(i * 100) / 3600000.0f;
        EXPECT_NEAR(snapshots[i].time_of_day, expected_tod, 0.0001f)
            << "Snapshot " << i << " time_of_day incorrect";
    }
}

TEST_F(BookBuilderTest, TimeOfDayAtClose) {
    uint64_t ts = pre_market_ts(0);
    auto events = build_minimal_book(ts);
    feed_events(*builder_, events);

    // Snapshot just before close: 15:59:59.900 ET
    uint64_t near_close = RTH_CLOSE_NS - BOUNDARY_100MS;
    auto snapshots = builder_->emit_snapshots(near_close, RTH_CLOSE_NS);
    ASSERT_GE(snapshots.size(), 1u);

    // 15:59:59.900 ET → 15 + 59/60 + 59.9/3600 ≈ 15.99997 hours
    float expected = 15.0f + 59.0f / 60.0f + 59.9f / 3600.0f;
    EXPECT_NEAR(snapshots[0].time_of_day, expected, 0.001f);
}

// ===========================================================================
// BookSnapshot struct layout verification
// ===========================================================================
TEST(BookSnapshotLayoutTest, StructFieldsExist) {
    // Verify that BookSnapshot has the correct fields and dimensions
    BookSnapshot snap{};

    // Timestamp
    snap.timestamp = 0;
    EXPECT_EQ(snap.timestamp, 0u);

    // Bids: 10 levels × (price, size)
    for (int i = 0; i < 10; ++i) {
        snap.bids[i][0] = static_cast<float>(i);
        snap.bids[i][1] = static_cast<float>(i + 10);
    }
    EXPECT_FLOAT_EQ(snap.bids[0][0], 0.0f);
    EXPECT_FLOAT_EQ(snap.bids[9][1], 19.0f);

    // Asks: 10 levels × (price, size)
    for (int i = 0; i < 10; ++i) {
        snap.asks[i][0] = static_cast<float>(i);
        snap.asks[i][1] = static_cast<float>(i + 10);
    }
    EXPECT_FLOAT_EQ(snap.asks[0][0], 0.0f);
    EXPECT_FLOAT_EQ(snap.asks[9][1], 19.0f);

    // Trades: 50 × (price, size, aggressor_side)
    for (int i = 0; i < 50; ++i) {
        snap.trades[i][0] = static_cast<float>(i);
        snap.trades[i][1] = static_cast<float>(i);
        snap.trades[i][2] = 1.0f;
    }
    EXPECT_FLOAT_EQ(snap.trades[49][0], 49.0f);

    // Scalar fields
    snap.mid_price = 4500.125f;
    snap.spread = 0.25f;
    snap.time_of_day = 9.5f;
    EXPECT_FLOAT_EQ(snap.mid_price, 4500.125f);
    EXPECT_FLOAT_EQ(snap.spread, 0.25f);
    EXPECT_FLOAT_EQ(snap.time_of_day, 9.5f);
}

// ===========================================================================
// Integration test — placeholder (requires real data)
// ===========================================================================
TEST(BookBuilderIntegrationTest, DISABLED_ProcessSingleDayFile) {
    // This test requires the real data file:
    //   DATA/GLBX-20260207-L953CAPU5B/glbx-mdp3-20220103.mbo.dbn.zst
    //
    // Enable by removing DISABLED_ prefix when databento-cpp is available
    // and the data file is on disk.
    //
    // The BookBuilder should:
    // 1. Read the file via databento::DbnFileStore
    // 2. Process all MBO events for instrument_id 13615 (MESM2)
    // 3. Emit snapshots during RTH (09:30 - 16:00 ET)
    //
    // Verification:
    // - Non-empty snapshot vector
    // - All snapshots within RTH window
    // - mid_price > 0 for all snapshots
    // - spread >= 0 for all snapshots
    // - No crossed book: bids[0].price < asks[0].price (when both exist)

    constexpr uint32_t MESM2_ID = 13615;
    BookBuilder builder(MESM2_ID);

    // TODO: Load real data file and call builder.process_file() or equivalent
    // auto snapshots = builder.process_file("DATA/GLBX-20260207-L953CAPU5B/glbx-mdp3-20220103.mbo.dbn.zst");

    // ASSERT_FALSE(snapshots.empty());

    // for (const auto& snap : snapshots) {
    //     // All timestamps within RTH
    //     EXPECT_GE(snap.timestamp, RTH_OPEN_NS);
    //     EXPECT_LE(snap.timestamp, RTH_CLOSE_NS);

    //     // mid_price > 0
    //     EXPECT_GT(snap.mid_price, 0.0f);

    //     // spread >= 0
    //     EXPECT_GE(snap.spread, 0.0f);

    //     // No crossed book (when both sides populated)
    //     if (snap.bids[0][0] > 0.0f && snap.asks[0][0] > 0.0f) {
    //         EXPECT_LT(snap.bids[0][0], snap.asks[0][0])
    //             << "Crossed book detected: best bid >= best ask";
    //     }
    // }
}
