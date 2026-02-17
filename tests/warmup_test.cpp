// warmup_test.cpp — TDD RED phase tests for WarmupTracker
// Spec: .kit/docs/bar-construction.md §Warm-Up State Tracking (§8.6)
//
// Tests for WarmupTracker: is_warmup behavior, boundary conditions,
// EWMA span parameter, and session boundary semantics.
//
// No implementation files exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

#include "features/warmup.hpp"  // WarmupTracker

#include <cstdint>

// ===========================================================================
// WarmupTracker — Core Behavior Tests
// ===========================================================================

class WarmupTrackerTest : public ::testing::Test {
protected:
    WarmupTracker tracker_;
};

TEST_F(WarmupTrackerTest, IsWarmupReturnsTrueForBarIndexBelowSpan) {
    // "Returns true if bar_index < ewma_span (EWMA features not yet stable)"
    // Default ewma_span = 20
    for (int i = 0; i < 20; ++i) {
        EXPECT_TRUE(tracker_.is_warmup(i))
            << "bar_index=" << i << " should be warmup (< 20)";
    }
}

TEST_F(WarmupTrackerTest, IsWarmupReturnsFalseForBarIndexAtOrAboveSpan) {
    // bar_index >= ewma_span → NOT warmup
    EXPECT_FALSE(tracker_.is_warmup(20))
        << "bar_index=20 should NOT be warmup (== ewma_span)";
    EXPECT_FALSE(tracker_.is_warmup(21))
        << "bar_index=21 should NOT be warmup (> ewma_span)";
    EXPECT_FALSE(tracker_.is_warmup(100))
        << "bar_index=100 should NOT be warmup (>> ewma_span)";
}

TEST_F(WarmupTrackerTest, BoundaryAtExactEWMASpan) {
    // Exact boundary: bar_index == ewma_span - 1 is warmup, ewma_span is not
    EXPECT_TRUE(tracker_.is_warmup(19));   // Last warmup bar
    EXPECT_FALSE(tracker_.is_warmup(20));  // First non-warmup bar
}

TEST_F(WarmupTrackerTest, BarIndexZeroIsAlwaysWarmup) {
    EXPECT_TRUE(tracker_.is_warmup(0));
}

// ===========================================================================
// WarmupTracker — Custom EWMA Span Tests
// ===========================================================================

class WarmupTrackerCustomSpanTest : public ::testing::Test {
protected:
    WarmupTracker tracker_;
};

TEST_F(WarmupTrackerCustomSpanTest, CustomSpanParameter) {
    // is_warmup(bar_index, ewma_span) allows custom span
    int custom_span = 10;

    // bar_index < 10 → warmup
    for (int i = 0; i < custom_span; ++i) {
        EXPECT_TRUE(tracker_.is_warmup(i, custom_span))
            << "bar_index=" << i << " should be warmup with span=" << custom_span;
    }

    // bar_index >= 10 → NOT warmup
    EXPECT_FALSE(tracker_.is_warmup(custom_span, custom_span));
    EXPECT_FALSE(tracker_.is_warmup(custom_span + 1, custom_span));
}

TEST_F(WarmupTrackerCustomSpanTest, SpanOfOneOnlyFirstBarIsWarmup) {
    EXPECT_TRUE(tracker_.is_warmup(0, 1));
    EXPECT_FALSE(tracker_.is_warmup(1, 1));
}

TEST_F(WarmupTrackerCustomSpanTest, LargeSpan) {
    int large_span = 200;

    EXPECT_TRUE(tracker_.is_warmup(0, large_span));
    EXPECT_TRUE(tracker_.is_warmup(199, large_span));
    EXPECT_FALSE(tracker_.is_warmup(200, large_span));
}

// ===========================================================================
// WarmupTracker — Default Parameter Tests
// ===========================================================================

class WarmupTrackerDefaultsTest : public ::testing::Test {};

TEST_F(WarmupTrackerDefaultsTest, DefaultEWMASpanIs20) {
    WarmupTracker tracker;

    // With default span (20), bar 19 is warmup, bar 20 is not
    EXPECT_TRUE(tracker.is_warmup(19));
    EXPECT_FALSE(tracker.is_warmup(20));
}

TEST_F(WarmupTrackerDefaultsTest, IsWarmupIsConst) {
    // is_warmup should be a const method (no side effects)
    const WarmupTracker tracker;
    EXPECT_TRUE(tracker.is_warmup(0));
    EXPECT_FALSE(tracker.is_warmup(20));
}

// ===========================================================================
// WarmupTracker — Edge Cases
// ===========================================================================

class WarmupTrackerEdgeCaseTest : public ::testing::Test {};

TEST_F(WarmupTrackerEdgeCaseTest, NegativeBarIndexHandledGracefully) {
    // While bar_index shouldn't be negative in practice (it's an int),
    // the implementation should handle this gracefully.
    // A negative index is always < ewma_span, so should return true.
    WarmupTracker tracker;
    EXPECT_TRUE(tracker.is_warmup(-1));
}

TEST_F(WarmupTrackerEdgeCaseTest, VeryLargeBarIndex) {
    WarmupTracker tracker;
    EXPECT_FALSE(tracker.is_warmup(1000000));
}

TEST_F(WarmupTrackerEdgeCaseTest, SpanOfZeroMeansNoWarmup) {
    // With ewma_span=0, no bars are warmup (bar_index < 0 is never true for
    // non-negative indices). This is an edge case; the implementation may choose
    // to clamp span to >= 1 or handle it as "no warmup needed".
    WarmupTracker tracker;
    // bar_index=0 < ewma_span=0 is false → NOT warmup
    EXPECT_FALSE(tracker.is_warmup(0, 0));
}
