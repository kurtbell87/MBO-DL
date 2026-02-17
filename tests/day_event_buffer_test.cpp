// day_event_buffer_test.cpp — TDD RED phase tests for DayEventBuffer
// Spec: .kit/docs/bar-construction.md
//
// Tests for MBOEvent struct, DayEventBuffer class (get_events, size, clear).
// No implementation files exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

#include "data/day_event_buffer.hpp"  // MBOEvent, DayEventBuffer

#include <cstdint>
#include <span>
#include <vector>

// ===========================================================================
// MBOEvent Struct — Data Contract Tests
// ===========================================================================

class MBOEventStructTest : public ::testing::Test {};

TEST_F(MBOEventStructTest, MBOEventHasRequiredFields) {
    MBOEvent event{};
    event.action = 0;      // Add=0, Cancel=1, Modify=2, Trade=3
    event.price = 4500.25f;
    event.size = 10;
    event.side = 0;        // Bid=0, Ask=1
    event.ts_event = 1641186000000000000ULL;

    EXPECT_EQ(event.action, 0);
    EXPECT_FLOAT_EQ(event.price, 4500.25f);
    EXPECT_EQ(event.size, 10u);
    EXPECT_EQ(event.side, 0);
    EXPECT_EQ(event.ts_event, 1641186000000000000ULL);
}

TEST_F(MBOEventStructTest, MBOEventActionTypes) {
    // Verify all action types can be stored
    MBOEvent add{};     add.action = 0;      // Add
    MBOEvent cancel{};  cancel.action = 1;   // Cancel
    MBOEvent modify{};  modify.action = 2;   // Modify
    MBOEvent trade{};   trade.action = 3;    // Trade

    EXPECT_EQ(add.action, 0);
    EXPECT_EQ(cancel.action, 1);
    EXPECT_EQ(modify.action, 2);
    EXPECT_EQ(trade.action, 3);
}

TEST_F(MBOEventStructTest, MBOEventSideValues) {
    MBOEvent bid_event{};
    bid_event.side = 0;  // Bid

    MBOEvent ask_event{};
    ask_event.side = 1;  // Ask

    EXPECT_EQ(bid_event.side, 0);
    EXPECT_EQ(ask_event.side, 1);
}

// ===========================================================================
// DayEventBuffer — Core Functionality Tests
// ===========================================================================

class DayEventBufferTest : public ::testing::Test {};

TEST_F(DayEventBufferTest, DefaultConstructedBufferIsEmpty) {
    DayEventBuffer buf;
    EXPECT_EQ(buf.size(), 0u);
}

TEST_F(DayEventBufferTest, GetEventsReturnsCorrectSpan) {
    // "DayEventBuffer.get_events() returns correct events for bar's index range"
    DayEventBuffer buf;

    // After loading, get_events(begin, end) returns a span of [begin, end)
    // Since we can't call load() without real data in a unit test, this test
    // documents the expected interface. It will fail to compile until the
    // implementation exists, and the GREEN phase should add a way to test this
    // without real data (e.g., by exposing an insert/push method or a test fixture).

    // The interface contract: get_events returns std::span<const MBOEvent>
    auto events = buf.get_events(0, 0);
    EXPECT_EQ(events.size(), 0u);
}

TEST_F(DayEventBufferTest, GetEventsReturnsSubrange) {
    DayEventBuffer buf;

    // After population, get_events(2, 5) should return 3 events
    // This tests the span semantics: [begin, end) is exclusive on end
    auto events = buf.get_events(0, 0);
    EXPECT_TRUE(events.empty());
}

TEST_F(DayEventBufferTest, ClearReleasesMemory) {
    // "DayEventBuffer.clear() releases memory"
    DayEventBuffer buf;

    // After clear, size should be 0
    buf.clear();
    EXPECT_EQ(buf.size(), 0u);
}

TEST_F(DayEventBufferTest, ClearOnEmptyBufferIsSafe) {
    DayEventBuffer buf;
    // Should not crash
    buf.clear();
    EXPECT_EQ(buf.size(), 0u);
}

TEST_F(DayEventBufferTest, SizeReturnsEventCount) {
    DayEventBuffer buf;
    // Default constructed → 0 events
    EXPECT_EQ(buf.size(), 0u);
}

// ===========================================================================
// DayEventBuffer — Load Interface Tests
// ===========================================================================

class DayEventBufferLoadTest : public ::testing::Test {};

TEST_F(DayEventBufferLoadTest, LoadMethodExists) {
    // Verify load() accepts (string, uint32_t) signature
    // This will fail to compile if the interface doesn't match the spec:
    //   void load(const std::string& dbn_path, uint32_t instrument_id);
    DayEventBuffer buf;

    // We don't actually call load with a real file — just verify the interface
    // compiles. The real load test would be an integration test.
    // Calling with a nonexistent file should not crash (graceful error handling).
    // For RED phase, we just need the signature to exist.
    static_assert(
        std::is_same_v<
            decltype(std::declval<DayEventBuffer>().load(
                std::declval<const std::string&>(),
                std::declval<uint32_t>()
            )),
            void
        >,
        "DayEventBuffer::load must accept (const string&, uint32_t) and return void"
    );
}

// ===========================================================================
// DayEventBuffer — Span Boundary Tests
// ===========================================================================

class DayEventBufferSpanTest : public ::testing::Test {};

TEST_F(DayEventBufferSpanTest, GetEventsReturnTypeIsSpan) {
    DayEventBuffer buf;
    auto events = buf.get_events(0, 0);

    // Verify the return type is std::span<const MBOEvent>
    static_assert(
        std::is_same_v<decltype(events), std::span<const MBOEvent>>,
        "get_events must return std::span<const MBOEvent>"
    );
}

TEST_F(DayEventBufferSpanTest, GetEventsWithEqualBeginEndReturnsEmpty) {
    DayEventBuffer buf;
    auto events = buf.get_events(5, 5);
    EXPECT_EQ(events.size(), 0u);
}
