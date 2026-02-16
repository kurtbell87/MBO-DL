// gbt_features_test.cpp — TDD RED phase tests for GBT Feature Engineering
// Spec: .kit/docs/gbt-model.md + ORCHESTRATOR_SPEC.md §5.4, §2.7
//
// Tests the compute_gbt_features() function: 16 hand-crafted features
// from a W=600 snapshot observation window. Includes trade deduplication,
// epsilon guards, edge cases, and no-NaN validation.

#include <gtest/gtest.h>
#include "gbt_features.hpp"
#include "book_builder.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Constants matching spec
// ---------------------------------------------------------------------------
constexpr float TICK_SIZE = 0.25f;
constexpr float EPSILON = 1e-8f;
constexpr float TWO_PI_VAL = 2.0f * 3.14159265358979323846f;

// ---------------------------------------------------------------------------
// Helpers — build synthetic BookSnapshot windows
// ---------------------------------------------------------------------------

// Build a single BookSnapshot with controllable fields
BookSnapshot make_snapshot(float mid_price, float spread, float time_of_day) {
    BookSnapshot snap{};
    snap.mid_price = mid_price;
    snap.spread = spread;
    snap.time_of_day = time_of_day;

    // Set BBO from mid_price and spread
    float half_spread = spread / 2.0f;
    snap.bids[0][0] = mid_price - half_spread;
    snap.bids[0][1] = 10.0f;  // default size
    snap.asks[0][0] = mid_price + half_spread;
    snap.asks[0][1] = 10.0f;

    return snap;
}

// Build a W=600 window of identical snapshots
std::vector<BookSnapshot> make_uniform_window(float mid_price = 4500.125f,
                                               float spread = 0.25f,
                                               float time_of_day = 10.0f) {
    std::vector<BookSnapshot> window(600);
    for (auto& snap : window) {
        snap = make_snapshot(mid_price, spread, time_of_day);
    }
    return window;
}

// Build a snapshot with explicit bid/ask sizes at all 10 levels
BookSnapshot make_snapshot_with_book(
    float mid_price, float spread, float time_of_day,
    const std::array<float, 10>& bid_sizes,
    const std::array<float, 10>& ask_sizes) {

    BookSnapshot snap = make_snapshot(mid_price, spread, time_of_day);

    float half_spread = spread / 2.0f;
    float best_bid = mid_price - half_spread;
    float best_ask = mid_price + half_spread;

    // Fill 10 bid levels, each 1 tick apart
    for (int i = 0; i < 10; ++i) {
        snap.bids[i][0] = best_bid - i * TICK_SIZE;
        snap.bids[i][1] = bid_sizes[i];
    }

    // Fill 10 ask levels, each 1 tick apart
    for (int i = 0; i < 10; ++i) {
        snap.asks[i][0] = best_ask + i * TICK_SIZE;
        snap.asks[i][1] = ask_sizes[i];
    }

    return snap;
}

// Set a trade in a snapshot's trade buffer at a given position
void set_trade(BookSnapshot& snap, int idx, float price, float size, float aggressor_side) {
    snap.trades[idx][0] = price;
    snap.trades[idx][1] = size;
    snap.trades[idx][2] = aggressor_side;
}

}  // anonymous namespace

// ===========================================================================
// Test Fixture
// ===========================================================================
class GBTFeaturesTest : public ::testing::Test {
protected:
    // Standard window for most tests — uniform mid=4500.125, spread=0.25
    std::vector<BookSnapshot> window_;

    void SetUp() override {
        window_ = make_uniform_window();
    }
};

// ===========================================================================
// 1. Feature dimension is 16
// ===========================================================================
TEST(GBTConstantsTest, FeatureDimIs16) {
    EXPECT_EQ(GBT_FEATURE_DIM, 16);
}

// ===========================================================================
// 2. Book imbalance
// ===========================================================================
TEST_F(GBTFeaturesTest, BookImbalance_BidHeavy) {
    // Set bid_sizes = [10, 5, 3, 2, 1, 0, 0, 0, 0, 0] sum=21
    // Set ask_sizes = [5, 3, 2, 1, 1, 0, 0, 0, 0, 0]  sum=12
    // imbalance = (21 - 12) / (21 + 12 + 1e-8) = 9 / 33.0 ≈ 0.27273
    std::array<float, 10> bid_sizes = {10, 5, 3, 2, 1, 0, 0, 0, 0, 0};
    std::array<float, 10> ask_sizes = {5, 3, 2, 1, 1, 0, 0, 0, 0, 0};

    // Set the last snapshot (t = W-1 = 599) with known book sizes
    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, bid_sizes, ask_sizes);

    auto features = compute_gbt_features(window_, 0.0f);

    float sum_bid = 21.0f;
    float sum_ask = 12.0f;
    float expected = (sum_bid - sum_ask) / (sum_bid + sum_ask + EPSILON);
    EXPECT_NEAR(features[0], expected, 1e-5f);
}

TEST_F(GBTFeaturesTest, BookImbalance_AskHeavy) {
    // Reversed: asks > bids → negative imbalance
    std::array<float, 10> bid_sizes = {3, 2, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<float, 10> ask_sizes = {20, 15, 10, 5, 0, 0, 0, 0, 0, 0};

    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, bid_sizes, ask_sizes);

    auto features = compute_gbt_features(window_, 0.0f);

    float sum_bid = 5.0f;
    float sum_ask = 50.0f;
    float expected = (sum_bid - sum_ask) / (sum_bid + sum_ask + EPSILON);
    EXPECT_NEAR(features[0], expected, 1e-5f);
    EXPECT_LT(features[0], 0.0f);  // must be negative
}

TEST_F(GBTFeaturesTest, BookImbalance_Balanced) {
    // Equal sizes on both sides → imbalance ≈ 0
    std::array<float, 10> sizes = {10, 10, 10, 10, 10, 0, 0, 0, 0, 0};

    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, sizes, sizes);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[0], 0.0f, 1e-5f);
}

// ===========================================================================
// 3. Spread in ticks
// ===========================================================================
TEST_F(GBTFeaturesTest, SpreadTicks_OneTick) {
    // spread = 0.25, tick_size = 0.25 → spread_ticks = 1.0
    window_[599].spread = 0.25f;
    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[1], 1.0f, 1e-5f);
}

TEST_F(GBTFeaturesTest, SpreadTicks_TwoTicks) {
    // spread = 0.50 → spread_ticks = 2.0
    window_[599].spread = 0.50f;
    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[1], 2.0f, 1e-5f);
}

TEST_F(GBTFeaturesTest, SpreadTicks_ZeroSpread) {
    // Locked book: spread = 0 → spread_ticks = 0.0
    window_[599].spread = 0.0f;
    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[1], 0.0f, 1e-5f);
}

// ===========================================================================
// 4. Book depth ratio (top 5 levels)
// ===========================================================================
TEST_F(GBTFeaturesTest, BookDepthRatio5) {
    // bid_sizes[0:5] = [10, 8, 6, 4, 2] sum=30
    // ask_sizes[0:5] = [5, 4, 3, 2, 1]  sum=15
    // ratio = 30 / (15 + 1e-8) = 2.0
    std::array<float, 10> bid_sizes = {10, 8, 6, 4, 2, 0, 0, 0, 0, 0};
    std::array<float, 10> ask_sizes = {5, 4, 3, 2, 1, 0, 0, 0, 0, 0};

    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, bid_sizes, ask_sizes);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[2], 2.0f, 1e-4f);
}

// ===========================================================================
// 5. Top level size ratio
// ===========================================================================
TEST_F(GBTFeaturesTest, TopLevelSizeRatio) {
    // bid_size[0] = 15, ask_size[0] = 5
    // ratio = 15 / (5 + 1e-8) = 3.0
    std::array<float, 10> bid_sizes = {15, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<float, 10> ask_sizes = {5, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, bid_sizes, ask_sizes);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[3], 3.0f, 1e-4f);
}

// ===========================================================================
// 6. Mid returns (1s, 5s, 30s, 60s)
// ===========================================================================
TEST_F(GBTFeaturesTest, MidReturn1s) {
    // t = 599, t-10 = 589
    // mid[599] = 4501.00, mid[589] = 4500.00
    // return = (4501.00 - 4500.00) / 0.25 = 4.0 ticks
    window_[589].mid_price = 4500.00f;
    window_[599].mid_price = 4501.00f;

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[4], 4.0f, 1e-4f);
}

TEST_F(GBTFeaturesTest, MidReturn5s) {
    // t = 599, t-50 = 549
    // mid[599] = 4502.00, mid[549] = 4500.00
    // return = (4502.00 - 4500.00) / 0.25 = 8.0 ticks
    window_[549].mid_price = 4500.00f;
    window_[599].mid_price = 4502.00f;

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[5], 8.0f, 1e-4f);
}

TEST_F(GBTFeaturesTest, MidReturn30s) {
    // t = 599, t-300 = 299
    // mid[599] = 4505.00, mid[299] = 4500.00
    // return = (4505.00 - 4500.00) / 0.25 = 20.0 ticks
    window_[299].mid_price = 4500.00f;
    window_[599].mid_price = 4505.00f;

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[6], 20.0f, 1e-4f);
}

TEST_F(GBTFeaturesTest, MidReturn60s) {
    // t = 599, t-599 = 0
    // mid[599] = 4510.00, mid[0] = 4500.00
    // return = (4510.00 - 4500.00) / 0.25 = 40.0 ticks
    window_[0].mid_price = 4500.00f;
    window_[599].mid_price = 4510.00f;

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[7], 40.0f, 1e-4f);
}

TEST_F(GBTFeaturesTest, MidReturn_NegativeReturn) {
    // Price dropped: mid[599] < mid[589]
    window_[589].mid_price = 4502.00f;
    window_[599].mid_price = 4500.00f;

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[4], -8.0f, 1e-4f);  // mid_return_1s
}

TEST_F(GBTFeaturesTest, MidReturn_ZeroReturn) {
    // No price change: all mid_prices identical
    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[4], 0.0f, 1e-4f);  // mid_return_1s
    EXPECT_NEAR(features[5], 0.0f, 1e-4f);  // mid_return_5s
    EXPECT_NEAR(features[6], 0.0f, 1e-4f);  // mid_return_30s
    EXPECT_NEAR(features[7], 0.0f, 1e-4f);  // mid_return_60s
}

// ===========================================================================
// 7. Trade deduplication
// ===========================================================================
TEST_F(GBTFeaturesTest, TradeDeduplication_OverlappingBuffers) {
    // Two consecutive snapshots share the same trade in their buffer.
    // The trade (price=4500.25, size=3, side=+1.0) appears at trades[49]
    // in both snapshot [598] and [599].
    // After dedup, this should count as ONE trade, not two.

    // Place identical trade at trades[49] in both snapshots 598 and 599
    set_trade(window_[598], 49, 4500.25f, 3.0f, 1.0f);
    set_trade(window_[599], 49, 4500.25f, 3.0f, 1.0f);

    auto features = compute_gbt_features(window_, 0.0f);

    // trade_arrival_rate_5s counts unique real trades in [t-50:t+1] / 5.0
    // With dedup, this should be 1 trade → rate = 1/5 = 0.2
    // Without dedup, it would be 2 trades → rate = 2/5 = 0.4
    EXPECT_NEAR(features[10], 0.2f, 1e-4f);
}

TEST_F(GBTFeaturesTest, TradeDeduplication_DistinctTrades) {
    // Two distinct trades in the 1s window [t-10 : t+1] = [589:600]
    // Different sizes → not duplicates
    set_trade(window_[595], 49, 4500.25f, 3.0f, 1.0f);
    set_trade(window_[599], 49, 4500.25f, 5.0f, 1.0f);  // different size

    auto features = compute_gbt_features(window_, 0.0f);

    // trade_imbalance_1s = sum(size * side) for dedup trades in [589:600]
    // Both are buy aggressor (+1.0): imbalance = 3*1 + 5*1 = 8.0
    EXPECT_NEAR(features[8], 8.0f, 1e-4f);
}

TEST_F(GBTFeaturesTest, TradeDeduplication_FilterZeroPadding) {
    // Zero-size trades (padding) should be filtered out after dedup.
    // A zero-padded entry: (0.0, 0.0, 0.0) — present in every snapshot.
    // After dedup + filter, these should not count.

    // All trades in window are zeros (default) — padding only
    auto features = compute_gbt_features(window_, 0.0f);

    // trade_arrival_rate_5s should be 0.0 (no real trades)
    EXPECT_NEAR(features[10], 0.0f, 1e-5f);

    // trade_imbalance_1s should be 0.0
    EXPECT_NEAR(features[8], 0.0f, 1e-5f);

    // trade_imbalance_5s should be 0.0
    EXPECT_NEAR(features[9], 0.0f, 1e-5f);

    // large_trade_flag should be 0.0
    EXPECT_NEAR(features[11], 0.0f, 1e-5f);
}

// ===========================================================================
// 8. Trade imbalance
// ===========================================================================
TEST_F(GBTFeaturesTest, TradeImbalance1s_MixedSides) {
    // Place trades in the 1s window [t-10 : t+1] = snapshots [589:600]
    // Trade 1: size=5, side=+1 (buy)
    // Trade 2: size=3, side=-1 (sell)
    // Net imbalance = 5*(+1) + 3*(-1) = 5 - 3 = 2.0
    set_trade(window_[590], 49, 4500.25f, 5.0f, 1.0f);
    set_trade(window_[595], 49, 4500.00f, 3.0f, -1.0f);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[8], 2.0f, 1e-4f);
}

TEST_F(GBTFeaturesTest, TradeImbalance5s_LargerWindow) {
    // Place trades in the 5s window [t-50 : t+1] = snapshots [549:600]
    // 3 buy trades of size 4, 1 sell trade of size 10
    // imbalance = 3*4 - 10 = 2.0
    set_trade(window_[550], 49, 4500.25f, 4.0f, 1.0f);
    set_trade(window_[560], 49, 4500.25f, 4.0f, 1.0f);
    set_trade(window_[570], 48, 4500.50f, 4.0f, 1.0f);  // different position → distinct
    set_trade(window_[580], 49, 4500.00f, 10.0f, -1.0f);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[9], 2.0f, 1e-4f);
}

// ===========================================================================
// 9. Trade arrival rate
// ===========================================================================
TEST_F(GBTFeaturesTest, TradeArrivalRate5s_KnownCount) {
    // Place 10 distinct trades in [t-50 : t+1] = snapshots [549:600]
    // rate = 10 / 5.0 = 2.0
    for (int i = 0; i < 10; ++i) {
        // Use different prices/sizes to ensure they are distinct after dedup
        set_trade(window_[550 + i * 5], 49,
                  4500.00f + i * 0.25f,
                  static_cast<float>(i + 1),
                  (i % 2 == 0) ? 1.0f : -1.0f);
    }

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[10], 2.0f, 1e-4f);
}

// ===========================================================================
// 10. Large trade flag
// ===========================================================================
TEST_F(GBTFeaturesTest, LargeTradeFlag_AboveThreshold) {
    // Place several trades in the full window to establish a median.
    // Then place one large trade in the 1s window.

    // 5 trades across the window with sizes: 2, 3, 4, 5, 6
    // median = 4, threshold = 2 × 4 = 8
    set_trade(window_[100], 49, 4500.25f, 2.0f, 1.0f);
    set_trade(window_[200], 49, 4500.50f, 3.0f, -1.0f);
    set_trade(window_[300], 49, 4500.75f, 4.0f, 1.0f);
    set_trade(window_[400], 49, 4501.00f, 5.0f, -1.0f);
    set_trade(window_[500], 49, 4501.25f, 6.0f, 1.0f);

    // Large trade in 1s window: size=10 > 8 (2 × median)
    set_trade(window_[595], 48, 4500.00f, 10.0f, 1.0f);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[11], 1.0f, 1e-5f);
}

TEST_F(GBTFeaturesTest, LargeTradeFlag_BelowThreshold) {
    // Same median setup, but trade in 1s window is below threshold.

    // 5 trades with sizes: 2, 3, 4, 5, 6 → median=4, threshold=8
    set_trade(window_[100], 49, 4500.25f, 2.0f, 1.0f);
    set_trade(window_[200], 49, 4500.50f, 3.0f, -1.0f);
    set_trade(window_[300], 49, 4500.75f, 4.0f, 1.0f);
    set_trade(window_[400], 49, 4501.00f, 5.0f, -1.0f);
    set_trade(window_[500], 49, 4501.25f, 6.0f, 1.0f);

    // Trade in 1s window: size=7 < 8 (below threshold)
    set_trade(window_[595], 48, 4500.00f, 7.0f, 1.0f);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[11], 0.0f, 1e-5f);
}

TEST_F(GBTFeaturesTest, LargeTradeFlag_NoTradesInWindow) {
    // No real trades anywhere → flag should be 0.0
    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[11], 0.0f, 1e-5f);
}

// ===========================================================================
// 11. VWAP distance
// ===========================================================================
TEST_F(GBTFeaturesTest, VWAPDistance_KnownTrades) {
    // Place trades in the full window for VWAP computation.
    // Trade 1: price=4500.00, size=10
    // Trade 2: price=4501.00, size=10
    // VWAP = (4500*10 + 4501*10) / (10 + 10) = 45005 / 20 = 4500.50
    // mid_price[t] = 4500.125 (default)
    // vwap_distance = (4500.125 - 4500.50) / 0.25 = -0.375 / 0.25 = -1.5

    set_trade(window_[100], 49, 4500.00f, 10.0f, 1.0f);
    set_trade(window_[300], 49, 4501.00f, 10.0f, -1.0f);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[12], -1.5f, 1e-3f);
}

TEST_F(GBTFeaturesTest, VWAPDistance_NoTrades) {
    // No real trades → VWAP is undefined → feature should be 0.0
    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[12], 0.0f, 1e-5f);
}

// ===========================================================================
// 12. No NaN in features (edge case: all-zero trades)
// ===========================================================================
TEST_F(GBTFeaturesTest, NoNaN_AllPaddingTrades) {
    // All trades are zero-padded (default) — no real trades in window.
    // Verify every feature is finite (no NaN, no Inf).
    auto features = compute_gbt_features(window_, 0.0f);

    for (int i = 0; i < GBT_FEATURE_DIM; ++i) {
        EXPECT_TRUE(std::isfinite(features[i]))
            << "Feature " << i << " is not finite: " << features[i];
    }
}

TEST_F(GBTFeaturesTest, NoNaN_AllZeroSizes) {
    // Both sides have zero sizes at all levels → epsilon guards must prevent NaN
    std::array<float, 10> zero_sizes = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, zero_sizes, zero_sizes);

    auto features = compute_gbt_features(window_, 0.0f);

    for (int i = 0; i < GBT_FEATURE_DIM; ++i) {
        EXPECT_TRUE(std::isfinite(features[i]))
            << "Feature " << i << " is NaN/Inf with zero-size book";
    }
}

// ===========================================================================
// 13. Epsilon guards — division by zero prevention
// ===========================================================================
TEST_F(GBTFeaturesTest, EpsilonGuard_BookImbalance_BothSidesEmpty) {
    // Both sides empty: sum_bid=0, sum_ask=0
    // imbalance = (0 - 0) / (0 + 0 + 1e-8) = 0.0
    std::array<float, 10> zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, zeros, zeros);

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[0], 0.0f, 1e-5f);
    EXPECT_TRUE(std::isfinite(features[0]));
}

TEST_F(GBTFeaturesTest, EpsilonGuard_BookDepthRatio5_AskSideEmpty) {
    // Ask side all zeros at top 5 → denominator = 0 + 1e-8
    std::array<float, 10> bid_sizes = {10, 5, 3, 2, 1, 0, 0, 0, 0, 0};
    std::array<float, 10> ask_zeros = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, bid_sizes, ask_zeros);

    auto features = compute_gbt_features(window_, 0.0f);

    // 21 / 1e-8 = very large but finite, not Inf or NaN
    EXPECT_TRUE(std::isfinite(features[2]));
    EXPECT_GT(features[2], 0.0f);
}

TEST_F(GBTFeaturesTest, EpsilonGuard_TopLevelSizeRatio_AskZero) {
    // ask_size[0] = 0 → ratio = bid_size[0] / (0 + 1e-8)
    std::array<float, 10> bid_sizes = {20, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::array<float, 10> ask_sizes = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, bid_sizes, ask_sizes);

    auto features = compute_gbt_features(window_, 0.0f);

    EXPECT_TRUE(std::isfinite(features[3]));
    EXPECT_GT(features[3], 0.0f);
}

// ===========================================================================
// 14. Time features: sin/cos encoding
// ===========================================================================
TEST_F(GBTFeaturesTest, TimeSinCos_Noon) {
    // time_of_day = 12.0 → frac = 12/24 = 0.5
    // sin(2π × 0.5) = sin(π) ≈ 0.0
    // cos(2π × 0.5) = cos(π) = -1.0
    window_[599].time_of_day = 12.0f;

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[13], std::sin(TWO_PI_VAL * 0.5f), 1e-5f);
    EXPECT_NEAR(features[14], std::cos(TWO_PI_VAL * 0.5f), 1e-5f);
}

TEST_F(GBTFeaturesTest, TimeSinCos_MarketOpen) {
    // time_of_day = 9.5 → frac = 9.5/24
    float frac = 9.5f / 24.0f;
    window_[599].time_of_day = 9.5f;

    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[13], std::sin(TWO_PI_VAL * frac), 1e-5f);
    EXPECT_NEAR(features[14], std::cos(TWO_PI_VAL * frac), 1e-5f);
}

// ===========================================================================
// 15. Position state
// ===========================================================================
TEST_F(GBTFeaturesTest, PositionState_Flat) {
    auto features = compute_gbt_features(window_, 0.0f);
    EXPECT_NEAR(features[15], 0.0f, 1e-5f);
}

TEST_F(GBTFeaturesTest, PositionState_Long) {
    auto features = compute_gbt_features(window_, 1.0f);
    EXPECT_NEAR(features[15], 1.0f, 1e-5f);
}

TEST_F(GBTFeaturesTest, PositionState_Short) {
    auto features = compute_gbt_features(window_, -1.0f);
    EXPECT_NEAR(features[15], -1.0f, 1e-5f);
}

// ===========================================================================
// 16. Output array has exactly 16 elements
// ===========================================================================
TEST_F(GBTFeaturesTest, OutputArraySize) {
    auto features = compute_gbt_features(window_, 0.0f);

    // std::array<float, GBT_FEATURE_DIM> — compile-time size check
    static_assert(std::tuple_size_v<decltype(features)> == 16,
                  "compute_gbt_features must return exactly 16 features");
}

// ===========================================================================
// 17. Window size enforcement — must be exactly W=600
// ===========================================================================
TEST_F(GBTFeaturesTest, WindowSizeTooSmall_Throws) {
    std::vector<BookSnapshot> small_window(599);  // W-1
    EXPECT_THROW(compute_gbt_features(small_window, 0.0f), std::invalid_argument);
}

TEST_F(GBTFeaturesTest, WindowSizeTooLarge_Throws) {
    std::vector<BookSnapshot> large_window(601);  // W+1
    EXPECT_THROW(compute_gbt_features(large_window, 0.0f), std::invalid_argument);
}

// ===========================================================================
// 18. Feature index ordering matches spec
// ===========================================================================
TEST_F(GBTFeaturesTest, FeatureIndexOrdering) {
    // Verify that features are in the documented order:
    // [0] book_imbalance
    // [1] spread_ticks
    // [2] book_depth_ratio_5
    // [3] top_level_size_ratio
    // [4] mid_return_1s
    // [5] mid_return_5s
    // [6] mid_return_30s
    // [7] mid_return_60s
    // [8] trade_imbalance_1s
    // [9] trade_imbalance_5s
    // [10] trade_arrival_rate_5s
    // [11] large_trade_flag
    // [12] vwap_distance
    // [13] time_sin
    // [14] time_cos
    // [15] position_state

    // Set up a window with distinctive values that let us verify each index.
    // Use known book sizes for features 0, 2, 3
    std::array<float, 10> bid_sizes = {10, 5, 3, 2, 1, 0, 0, 0, 0, 0};
    std::array<float, 10> ask_sizes = {5, 3, 2, 1, 1, 0, 0, 0, 0, 0};
    window_[599] = make_snapshot_with_book(4500.125f, 0.50f, 15.0f, bid_sizes, ask_sizes);

    // Set mid prices for return features
    window_[0].mid_price   = 4500.00f;   // t-599
    window_[299].mid_price = 4500.25f;   // t-300
    window_[549].mid_price = 4500.50f;   // t-50
    window_[589].mid_price = 4500.75f;   // t-10
    window_[599].mid_price = 4501.00f;   // t

    auto features = compute_gbt_features(window_, -1.0f);

    // [1] spread_ticks = 0.50 / 0.25 = 2.0
    EXPECT_NEAR(features[1], 2.0f, 1e-4f);

    // [4] mid_return_1s = (4501 - 4500.75) / 0.25 = 1.0
    EXPECT_NEAR(features[4], 1.0f, 1e-4f);

    // [5] mid_return_5s = (4501 - 4500.50) / 0.25 = 2.0
    EXPECT_NEAR(features[5], 2.0f, 1e-4f);

    // [6] mid_return_30s = (4501 - 4500.25) / 0.25 = 3.0
    EXPECT_NEAR(features[6], 3.0f, 1e-4f);

    // [7] mid_return_60s = (4501 - 4500.00) / 0.25 = 4.0
    EXPECT_NEAR(features[7], 4.0f, 1e-4f);

    // [15] position_state = -1.0
    EXPECT_NEAR(features[15], -1.0f, 1e-5f);
}

// ===========================================================================
// 19. Book features use RAW sizes (not log-transformed)
// ===========================================================================
TEST_F(GBTFeaturesTest, BookFeatures_UseRawSizes) {
    // book_imbalance uses raw sizes, NOT log1p.
    // If it used log1p, the imbalance would be different.
    // bid_sizes all = 100, ask_sizes all = 1
    // raw: imbalance = (1000-10)/(1000+10+eps) = 990/1010 ≈ 0.98
    // log1p: log1p(100)*10=46.15, log1p(1)*10=6.93 → 39.22/53.08 ≈ 0.74
    std::array<float, 10> bid_sizes = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    std::array<float, 10> ask_sizes = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    window_[599] = make_snapshot_with_book(4500.125f, 0.25f, 10.0f, bid_sizes, ask_sizes);

    auto features = compute_gbt_features(window_, 0.0f);

    float sum_bid = 1000.0f;
    float sum_ask = 10.0f;
    float expected_raw = (sum_bid - sum_ask) / (sum_bid + sum_ask + EPSILON);

    EXPECT_NEAR(features[0], expected_raw, 1e-4f);
}

// ===========================================================================
// 20. All features finite with realistic data
// ===========================================================================
TEST_F(GBTFeaturesTest, AllFeaturesFinite_RealisticWindow) {
    // Build a window with varying mid prices, some trades, different book depths
    for (int t = 0; t < 600; ++t) {
        float mid = 4500.0f + 0.25f * (t % 20);
        window_[t] = make_snapshot(mid, 0.25f, 9.5f + static_cast<float>(t) / 36000.0f);

        // Add a trade every 30 snapshots
        if (t % 30 == 0 && t > 0) {
            set_trade(window_[t], 49, mid, static_cast<float>(t % 10 + 1),
                      (t % 2 == 0) ? 1.0f : -1.0f);
        }
    }

    auto features = compute_gbt_features(window_, 1.0f);

    for (int i = 0; i < GBT_FEATURE_DIM; ++i) {
        EXPECT_TRUE(std::isfinite(features[i]))
            << "Feature " << i << " is not finite in realistic window";
    }
}
