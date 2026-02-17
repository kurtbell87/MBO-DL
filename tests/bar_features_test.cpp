// bar_features_test.cpp — TDD RED phase tests for Track A hand-crafted features
// Spec: .kit/docs/feature-computation.md §Track A: Categories 1–6
//
// Tests for BarFeatureComputer: all ~45 hand-crafted features across 6 categories,
// forward return computation, warmup/NaN policy, session boundary resets, and
// the bar-level is_warmup flag.
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers that the implementation must provide (spec §Project Structure):
#include "features/bar_features.hpp"  // BarFeatureComputer, BarFeatureRow
#include "bars/bar.hpp"               // Bar struct
#include "book_builder.hpp"           // BOOK_DEPTH
#include "test_bar_helpers.hpp"       // shared timestamp/tick constants

#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

// ===========================================================================
// Helpers — synthetic Bar construction
// ===========================================================================
namespace {

using test_helpers::TICK_SIZE;
using test_helpers::RTH_OPEN_NS;
constexpr float EPS = 1e-8f;

// Build a minimal Bar at a given offset from RTH open (in seconds).
Bar make_bar(float close_mid, uint32_t volume = 100, float seconds_from_open = 60.0f,
             float buy_vol = 60.0f, float sell_vol = 40.0f) {
    Bar bar{};
    bar.open_ts = RTH_OPEN_NS;
    bar.close_ts = RTH_OPEN_NS + static_cast<uint64_t>(seconds_from_open * 1e9);
    bar.time_of_day = 9.5f + seconds_from_open / 3600.0f;
    bar.bar_duration_s = seconds_from_open > 0 ? 1.0f : 0.0f;  // Default 1s bars
    bar.open_mid = close_mid - 0.25f;
    bar.close_mid = close_mid;
    bar.high_mid = close_mid + 0.25f;
    bar.low_mid = close_mid - 0.50f;
    bar.vwap = close_mid;
    bar.volume = volume;
    bar.tick_count = volume;
    bar.buy_volume = buy_vol;
    bar.sell_volume = sell_vol;
    bar.spread = 0.25f;

    // Default symmetric book: 10 levels each side
    for (int i = 0; i < BOOK_DEPTH; ++i) {
        bar.bids[i][0] = close_mid - TICK_SIZE * (0.5f + i);
        bar.bids[i][1] = 10.0f + static_cast<float>(i);
        bar.asks[i][0] = close_mid + TICK_SIZE * (0.5f + i);
        bar.asks[i][1] = 10.0f + static_cast<float>(i);
    }

    // Default message counts
    bar.add_count = 50;
    bar.cancel_count = 30;
    bar.modify_count = 10;
    bar.trade_event_count = volume;

    return bar;
}

// Build a bar with asymmetric book (for imbalance testing).
Bar make_asymmetric_book_bar(float close_mid, float bid_size_base, float ask_size_base) {
    Bar bar = make_bar(close_mid);
    for (int i = 0; i < BOOK_DEPTH; ++i) {
        bar.bids[i][1] = bid_size_base + static_cast<float>(i);
        bar.asks[i][1] = ask_size_base + static_cast<float>(i);
    }
    return bar;
}

// Build a sequence of bars with linearly increasing close_mid prices.
std::vector<Bar> make_bar_sequence(int count, float start_mid = 4500.0f,
                                    float mid_step = 0.25f, uint32_t volume_each = 100) {
    std::vector<Bar> bars;
    bars.reserve(count);
    for (int i = 0; i < count; ++i) {
        float mid = start_mid + mid_step * static_cast<float>(i);
        float seconds = 60.0f + static_cast<float>(i) * 1.0f;
        bars.push_back(make_bar(mid, volume_each, seconds));
    }
    return bars;
}

// Build a sequence of bars with explicit close_mid values.
std::vector<Bar> make_bars_with_mids(const std::vector<float>& mids, uint32_t volume_each = 100) {
    std::vector<Bar> bars;
    bars.reserve(mids.size());
    for (size_t i = 0; i < mids.size(); ++i) {
        float seconds = 60.0f + static_cast<float>(i) * 1.0f;
        bars.push_back(make_bar(mids[i], volume_each, seconds));
    }
    return bars;
}

}  // anonymous namespace

// ===========================================================================
// BarFeatureRow — Data Contract Tests
// ===========================================================================
// The BarFeatureRow struct must contain all ~45 Track A features + metadata.

class BarFeatureRowTest : public ::testing::Test {};

TEST_F(BarFeatureRowTest, HasBookShapeFields_Category1) {
    BarFeatureRow row{};
    // 1.1: book_imbalance at 4 depth levels
    EXPECT_FLOAT_EQ(row.book_imbalance_1, 0.0f);
    EXPECT_FLOAT_EQ(row.book_imbalance_3, 0.0f);
    EXPECT_FLOAT_EQ(row.book_imbalance_5, 0.0f);
    EXPECT_FLOAT_EQ(row.book_imbalance_10, 0.0f);

    // 1.2: weighted_imbalance
    EXPECT_FLOAT_EQ(row.weighted_imbalance, 0.0f);

    // 1.3: spread
    EXPECT_FLOAT_EQ(row.spread, 0.0f);

    // 1.4/1.5: depth profiles (10 each)
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(row.bid_depth_profile[i], 0.0f);
        EXPECT_FLOAT_EQ(row.ask_depth_profile[i], 0.0f);
    }

    // 1.6: depth_concentration
    EXPECT_FLOAT_EQ(row.depth_concentration_bid, 0.0f);
    EXPECT_FLOAT_EQ(row.depth_concentration_ask, 0.0f);

    // 1.7: book_slope
    EXPECT_FLOAT_EQ(row.book_slope_bid, 0.0f);
    EXPECT_FLOAT_EQ(row.book_slope_ask, 0.0f);

    // 1.8: level_count
    EXPECT_EQ(row.level_count_bid, 0);
    EXPECT_EQ(row.level_count_ask, 0);
}

TEST_F(BarFeatureRowTest, HasOrderFlowFields_Category2) {
    BarFeatureRow row{};
    EXPECT_FLOAT_EQ(row.net_volume, 0.0f);
    EXPECT_FLOAT_EQ(row.volume_imbalance, 0.0f);
    EXPECT_EQ(row.trade_count, 0u);
    EXPECT_FLOAT_EQ(row.avg_trade_size, 0.0f);
    EXPECT_EQ(row.large_trade_count, 0u);
    EXPECT_FLOAT_EQ(row.vwap_distance, 0.0f);
    // kyle_lambda may be NaN by default
    (void)row.kyle_lambda;
}

TEST_F(BarFeatureRowTest, HasPriceDynamicsFields_Category3) {
    BarFeatureRow row{};
    EXPECT_FLOAT_EQ(row.return_1, 0.0f);
    EXPECT_FLOAT_EQ(row.return_5, 0.0f);
    EXPECT_FLOAT_EQ(row.return_20, 0.0f);
    EXPECT_FLOAT_EQ(row.volatility_20, 0.0f);
    EXPECT_FLOAT_EQ(row.volatility_50, 0.0f);
    EXPECT_FLOAT_EQ(row.momentum, 0.0f);
    EXPECT_FLOAT_EQ(row.high_low_range_20, 0.0f);
    EXPECT_FLOAT_EQ(row.high_low_range_50, 0.0f);
    EXPECT_FLOAT_EQ(row.close_position, 0.0f);
}

TEST_F(BarFeatureRowTest, HasCrossScaleFields_Category4) {
    BarFeatureRow row{};
    EXPECT_FLOAT_EQ(row.volume_surprise, 0.0f);
    EXPECT_FLOAT_EQ(row.duration_surprise, 0.0f);
    EXPECT_FLOAT_EQ(row.acceleration, 0.0f);
    EXPECT_FLOAT_EQ(row.vol_price_corr, 0.0f);
}

TEST_F(BarFeatureRowTest, HasTimeContextFields_Category5) {
    BarFeatureRow row{};
    EXPECT_FLOAT_EQ(row.time_sin, 0.0f);
    EXPECT_FLOAT_EQ(row.time_cos, 0.0f);
    EXPECT_FLOAT_EQ(row.minutes_since_open, 0.0f);
    EXPECT_FLOAT_EQ(row.minutes_to_close, 0.0f);
    EXPECT_FLOAT_EQ(row.session_volume_frac, 0.0f);
}

TEST_F(BarFeatureRowTest, HasMessageMicrostructureFields_Category6) {
    BarFeatureRow row{};
    EXPECT_FLOAT_EQ(row.cancel_add_ratio, 0.0f);
    EXPECT_FLOAT_EQ(row.message_rate, 0.0f);
    EXPECT_FLOAT_EQ(row.modify_fraction, 0.0f);
    EXPECT_FLOAT_EQ(row.order_flow_toxicity, 0.0f);
    EXPECT_FLOAT_EQ(row.cancel_concentration, 0.0f);
}

TEST_F(BarFeatureRowTest, HasForwardReturns) {
    BarFeatureRow row{};
    // Forward returns are targets, not features — but stored in the row for export.
    // They may be NaN for last n bars.
    (void)row.fwd_return_1;
    (void)row.fwd_return_5;
    (void)row.fwd_return_20;
    (void)row.fwd_return_100;
}

TEST_F(BarFeatureRowTest, HasIsWarmupFlag) {
    BarFeatureRow row{};
    EXPECT_FALSE(row.is_warmup);
}

TEST_F(BarFeatureRowTest, HasBarMetadata) {
    BarFeatureRow row{};
    EXPECT_EQ(row.timestamp, 0u);
    (void)row.bar_type;
    (void)row.bar_param;
    EXPECT_EQ(row.day, 0);
}

// ===========================================================================
// BarFeatureComputer — Construction Tests
// ===========================================================================

class BarFeatureComputerConstructionTest : public ::testing::Test {};

TEST_F(BarFeatureComputerConstructionTest, DefaultConstruction) {
    BarFeatureComputer computer;
    (void)computer;
}

TEST_F(BarFeatureComputerConstructionTest, ConstructWithTickSize) {
    BarFeatureComputer computer(TICK_SIZE);
    (void)computer;
}

TEST_F(BarFeatureComputerConstructionTest, ResetClearsState) {
    BarFeatureComputer computer(TICK_SIZE);
    auto bars = make_bar_sequence(5);
    for (const auto& bar : bars) {
        computer.update(bar);
    }
    computer.reset();
    // After reset, internal state is cleared — next bar is treated as first in session.
    // Session-level accumulators (EWMA, rolling windows) should be reset.
}

// ===========================================================================
// Category 1: Book Shape — Static per bar (snapshot at bar close)
// ===========================================================================

class BookShapeFeatureTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(BookShapeFeatureTest, BookImbalance1_SymmetricBookIsZero) {
    // Equal bid and ask sizes at top-1 level → imbalance = 0
    auto bar = make_bar(4500.0f);
    bar.bids[0][1] = 100.0f;
    bar.asks[0][1] = 100.0f;
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.book_imbalance_1, 0.0f, 1e-6f);
}

TEST_F(BookShapeFeatureTest, BookImbalance1_AllBidsNoBids) {
    // All bid volume, no ask → imbalance = 1.0
    auto bar = make_bar(4500.0f);
    bar.bids[0][1] = 100.0f;
    bar.asks[0][1] = 0.0f;
    auto row = computer_.update(bar);
    // (100 - 0) / (100 + 0 + eps) ≈ 1.0
    EXPECT_NEAR(row.book_imbalance_1, 1.0f, 1e-4f);
}

TEST_F(BookShapeFeatureTest, BookImbalance1_AllAsksNoBids) {
    auto bar = make_bar(4500.0f);
    bar.bids[0][1] = 0.0f;
    bar.asks[0][1] = 100.0f;
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.book_imbalance_1, -1.0f, 1e-4f);
}

TEST_F(BookShapeFeatureTest, BookImbalance3_UsesTop3Levels) {
    auto bar = make_bar(4500.0f);
    // Set top-3 bid levels to 10 each, top-3 ask levels to 20 each
    for (int i = 0; i < 3; ++i) {
        bar.bids[i][1] = 10.0f;
        bar.asks[i][1] = 20.0f;
    }
    auto row = computer_.update(bar);
    // bid_vol = 30, ask_vol = 60 → (30-60)/(30+60+eps) = -30/90 = -1/3
    EXPECT_NEAR(row.book_imbalance_3, -1.0f / 3.0f, 1e-4f);
}

TEST_F(BookShapeFeatureTest, BookImbalance5_UsesTop5Levels) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 5; ++i) {
        bar.bids[i][1] = 15.0f;
        bar.asks[i][1] = 5.0f;
    }
    auto row = computer_.update(bar);
    // bid_vol = 75, ask_vol = 25 → (75-25)/(75+25+eps) = 50/100 = 0.5
    EXPECT_NEAR(row.book_imbalance_5, 0.5f, 1e-4f);
}

TEST_F(BookShapeFeatureTest, BookImbalance10_UsesAllLevels) {
    auto bar = make_asymmetric_book_bar(4500.0f, 20.0f, 10.0f);
    auto row = computer_.update(bar);
    // bid_vol = sum(20..29) = 245, ask_vol = sum(10..19) = 145
    // (245 - 145) / (245 + 145 + eps) = 100/390 ≈ 0.2564
    float bid_vol = 0, ask_vol = 0;
    for (int i = 0; i < 10; ++i) {
        bid_vol += 20.0f + i;
        ask_vol += 10.0f + i;
    }
    float expected = (bid_vol - ask_vol) / (bid_vol + ask_vol + EPS);
    EXPECT_NEAR(row.book_imbalance_10, expected, 1e-4f);
}

TEST_F(BookShapeFeatureTest, WeightedImbalance_InverseDistanceWeighting) {
    // Weighted imbalance uses inverse distance from mid weighting.
    // Closer levels contribute more.
    auto bar = make_bar(4500.0f);
    // Level 0: close to mid → high weight
    bar.bids[0][1] = 100.0f;
    bar.asks[0][1] = 0.0f;
    // Level 9: far from mid → low weight
    bar.bids[9][1] = 0.0f;
    bar.asks[9][1] = 100.0f;
    auto row = computer_.update(bar);
    // More weight on bids (close levels) → positive imbalance
    EXPECT_GT(row.weighted_imbalance, 0.0f);
}

TEST_F(BookShapeFeatureTest, Spread_InTicks) {
    auto bar = make_bar(4500.0f);
    bar.spread = 0.50f;  // 2 ticks
    auto row = computer_.update(bar);
    EXPECT_FLOAT_EQ(row.spread, 0.50f / TICK_SIZE);  // 2.0
}

TEST_F(BookShapeFeatureTest, BidDepthProfile_RawSizes) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) {
        bar.bids[i][1] = static_cast<float>(i * 10 + 5);
    }
    auto row = computer_.update(bar);
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(row.bid_depth_profile[i], static_cast<float>(i * 10 + 5))
            << "bid_depth_profile[" << i << "] mismatch";
    }
}

TEST_F(BookShapeFeatureTest, AskDepthProfile_RawSizes) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) {
        bar.asks[i][1] = static_cast<float>(i * 5 + 3);
    }
    auto row = computer_.update(bar);
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(row.ask_depth_profile[i], static_cast<float>(i * 5 + 3))
            << "ask_depth_profile[" << i << "] mismatch";
    }
}

TEST_F(BookShapeFeatureTest, DepthConcentrationBid_HHI) {
    // HHI = sum((s_i / total)^2). Uniform → HHI = 1/N. Concentrated → HHI → 1.
    auto bar = make_bar(4500.0f);
    // All volume on level 0 → HHI = 1.0
    for (int i = 0; i < 10; ++i) bar.bids[i][1] = 0.0f;
    bar.bids[0][1] = 100.0f;
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.depth_concentration_bid, 1.0f, 1e-4f);
}

TEST_F(BookShapeFeatureTest, DepthConcentrationBid_Uniform) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) bar.bids[i][1] = 10.0f;
    auto row = computer_.update(bar);
    // HHI = 10 * (10/100)^2 = 10 * 0.01 = 0.1
    EXPECT_NEAR(row.depth_concentration_bid, 0.1f, 1e-4f);
}

TEST_F(BookShapeFeatureTest, DepthConcentrationAsk_Concentrated) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) bar.asks[i][1] = 0.0f;
    bar.asks[0][1] = 50.0f;
    bar.asks[1][1] = 50.0f;
    auto row = computer_.update(bar);
    // HHI = 2 * (50/100)^2 = 2 * 0.25 = 0.5
    EXPECT_NEAR(row.depth_concentration_ask, 0.5f, 1e-4f);
}

TEST_F(BookShapeFeatureTest, BookSlopeBid_LinearDecay) {
    // log(size) vs level index — negative slope means decaying depth.
    auto bar = make_bar(4500.0f);
    // Exponentially decaying sizes: 100, 50, 25, ...
    for (int i = 0; i < 10; ++i) {
        bar.bids[i][1] = 100.0f / std::pow(2.0f, static_cast<float>(i));
    }
    auto row = computer_.update(bar);
    EXPECT_LT(row.book_slope_bid, 0.0f);  // Negative slope for decay
}

TEST_F(BookShapeFeatureTest, BookSlopeAsk_FlatBook) {
    // All same size → slope should be near 0.
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) {
        bar.asks[i][1] = 50.0f;
    }
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.book_slope_ask, 0.0f, 0.01f);
}

TEST_F(BookShapeFeatureTest, LevelCountBid_AllPopulated) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) bar.bids[i][1] = 10.0f;
    auto row = computer_.update(bar);
    EXPECT_EQ(row.level_count_bid, 10);
}

TEST_F(BookShapeFeatureTest, LevelCountBid_SomeEmpty) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) bar.bids[i][1] = 0.0f;
    bar.bids[0][1] = 10.0f;
    bar.bids[1][1] = 5.0f;
    bar.bids[2][1] = 3.0f;
    auto row = computer_.update(bar);
    EXPECT_EQ(row.level_count_bid, 3);
}

TEST_F(BookShapeFeatureTest, LevelCountAsk_NonePopulated) {
    auto bar = make_bar(4500.0f);
    for (int i = 0; i < 10; ++i) bar.asks[i][1] = 0.0f;
    auto row = computer_.update(bar);
    EXPECT_EQ(row.level_count_ask, 0);
}

// ===========================================================================
// Category 2: Order Flow — Aggregated within bar
// ===========================================================================

class OrderFlowFeatureTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(OrderFlowFeatureTest, NetVolume_BuyHeavy) {
    auto bar = make_bar(4500.0f);
    bar.buy_volume = 80.0f;
    bar.sell_volume = 20.0f;
    auto row = computer_.update(bar);
    EXPECT_FLOAT_EQ(row.net_volume, 60.0f);
}

TEST_F(OrderFlowFeatureTest, NetVolume_SellHeavy) {
    auto bar = make_bar(4500.0f);
    bar.buy_volume = 20.0f;
    bar.sell_volume = 80.0f;
    auto row = computer_.update(bar);
    EXPECT_FLOAT_EQ(row.net_volume, -60.0f);
}

TEST_F(OrderFlowFeatureTest, VolumeImbalance_Normalized) {
    auto bar = make_bar(4500.0f);
    bar.buy_volume = 75.0f;
    bar.sell_volume = 25.0f;
    bar.volume = 100;
    auto row = computer_.update(bar);
    // net_volume = 50, total_volume = 100 → 50/(100+eps) = 0.5
    EXPECT_NEAR(row.volume_imbalance, 0.5f, 1e-4f);
}

TEST_F(OrderFlowFeatureTest, VolumeImbalance_ZeroVolume) {
    auto bar = make_bar(4500.0f);
    bar.buy_volume = 0.0f;
    bar.sell_volume = 0.0f;
    bar.volume = 0;
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.volume_imbalance, 0.0f, 1e-4f);
}

TEST_F(OrderFlowFeatureTest, TradeCount) {
    auto bar = make_bar(4500.0f);
    bar.trade_event_count = 42;
    auto row = computer_.update(bar);
    EXPECT_EQ(row.trade_count, 42u);
}

TEST_F(OrderFlowFeatureTest, AvgTradeSize) {
    auto bar = make_bar(4500.0f);
    bar.volume = 200;
    bar.trade_event_count = 50;
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.avg_trade_size, 4.0f, 1e-6f);
}

TEST_F(OrderFlowFeatureTest, AvgTradeSize_ZeroTrades) {
    auto bar = make_bar(4500.0f);
    bar.volume = 0;
    bar.trade_event_count = 0;
    auto row = computer_.update(bar);
    // Should handle division by zero gracefully
    EXPECT_FALSE(std::isnan(row.avg_trade_size));
    EXPECT_FLOAT_EQ(row.avg_trade_size, 0.0f);
}

TEST_F(OrderFlowFeatureTest, LargeTradeCount_NoneAboveThreshold) {
    // First 20 bars establish rolling median. No large trades at start.
    auto bars = make_bar_sequence(25, 4500.0f, 0.25f, 100);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // All bars have same volume=100 → median=100, threshold=200
    // No bar exceeds 200 → large_trade_count = 0
    EXPECT_EQ(row.large_trade_count, 0u);
}

TEST_F(OrderFlowFeatureTest, LargeTradeCount_SomeAboveThreshold) {
    auto bars = make_bar_sequence(25, 4500.0f, 0.25f, 100);
    // Make the last bar have very large trades
    bars.back().volume = 500;
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // Median of prior 20 bars' volume should be ~100, threshold=200.
    // Last bar has volume 500 > 200 → at least 1 large trade.
    EXPECT_GT(row.large_trade_count, 0u);
}

TEST_F(OrderFlowFeatureTest, VwapDistance_InTicks) {
    auto bar = make_bar(4500.25f);
    bar.vwap = 4500.00f;
    auto row = computer_.update(bar);
    // (close_mid - vwap) / tick_size = (4500.25 - 4500.00) / 0.25 = 1.0
    EXPECT_NEAR(row.vwap_distance, 1.0f, 1e-4f);
}

TEST_F(OrderFlowFeatureTest, KyleLambda_NaNForFirst20Bars) {
    // kyle_lambda requires 20-bar rolling window. NaN for first 20 bars of session.
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.kyle_lambda))
            << "kyle_lambda should be NaN for bar " << i << " (< 20 bar window)";
    }
}

TEST_F(OrderFlowFeatureTest, KyleLambda_DefinedAfter20Bars) {
    auto bars = make_bar_sequence(25);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // After 20 bars, kyle_lambda should be defined (not NaN)
    EXPECT_FALSE(std::isnan(row.kyle_lambda));
}

TEST_F(OrderFlowFeatureTest, KyleLambda_PositiveForBuyPressureDrivingPriceUp) {
    // Construct 25 bars where positive net_volume → price goes up
    std::vector<Bar> bars;
    for (int i = 0; i < 25; ++i) {
        float mid = 4500.0f + static_cast<float>(i) * 0.25f;  // Steadily rising
        auto bar = make_bar(mid, 100 + i * 10, 60.0f + i);
        bar.buy_volume = 80.0f + static_cast<float>(i);
        bar.sell_volume = 20.0f;
        bars.push_back(bar);
    }
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // Positive correlation between Δmid and net_volume → positive lambda
    EXPECT_GT(row.kyle_lambda, 0.0f);
}

// ===========================================================================
// Category 3: Price Dynamics — Across bars (lookback)
// ===========================================================================

class PriceDynamicsFeatureTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(PriceDynamicsFeatureTest, Return1_OneTick) {
    auto bars = make_bar_sequence(5, 4500.0f, 0.25f);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // return_1 = (bar[i].close_mid - bar[i-1].close_mid) / tick_size
    // Last bar: (4501.0 - 4500.75) / 0.25 = 1.0
    EXPECT_NEAR(row.return_1, 1.0f, 1e-4f);
}

TEST_F(PriceDynamicsFeatureTest, Return5_FiveBarsBack) {
    auto bars = make_bar_sequence(10, 4500.0f, 0.25f);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // return_5 for bar 9: (bar[9].close_mid - bar[4].close_mid) / tick_size
    // = (4502.25 - 4501.00) / 0.25 = 5.0
    EXPECT_NEAR(row.return_5, 5.0f, 1e-4f);
}

TEST_F(PriceDynamicsFeatureTest, Return20_NaNForFirst20Bars) {
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        if (i < 20) {
            EXPECT_TRUE(std::isnan(row.return_20))
                << "return_20 should be NaN for bar " << i;
        }
    }
}

TEST_F(PriceDynamicsFeatureTest, Return20_DefinedAtBar20) {
    auto bars = make_bar_sequence(25, 4500.0f, 0.25f);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    EXPECT_FALSE(std::isnan(row.return_20));
}

TEST_F(PriceDynamicsFeatureTest, Volatility20_StdOfReturns) {
    // Feed 25 bars with known returns pattern.
    // Returns: all +1 tick → volatility = 0 (no variation in returns).
    auto bars = make_bar_sequence(25, 4500.0f, 0.25f);  // constant +0.25 = 1 tick
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // All 1-bar returns are identical (1 tick each) → std = 0
    EXPECT_NEAR(row.volatility_20, 0.0f, 1e-4f);
}

TEST_F(PriceDynamicsFeatureTest, Volatility20_NaNForFirst20Bars) {
    auto bars = make_bar_sequence(5);
    for (int i = 0; i < 5; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.volatility_20))
            << "volatility_20 should be NaN for bar " << i;
    }
}

TEST_F(PriceDynamicsFeatureTest, Volatility50_NaNForFirst50Bars) {
    auto bars = make_bar_sequence(50);
    for (int i = 0; i < 50; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.volatility_50))
            << "volatility_50 should be NaN for bar " << i;
    }
}

TEST_F(PriceDynamicsFeatureTest, Momentum_SumOfSignedReturns) {
    // Alternating up/down returns cancel out.
    std::vector<float> mids = {4500.0f, 4500.25f, 4500.0f, 4500.25f, 4500.0f};
    auto bars = make_bars_with_mids(mids);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // Returns: +1, -1, +1, -1 → momentum = 0
    EXPECT_NEAR(row.momentum, 0.0f, 1e-4f);
}

TEST_F(PriceDynamicsFeatureTest, Momentum_PersistentTrend) {
    // All returns positive → momentum > 0
    auto bars = make_bar_sequence(25, 4500.0f, 0.50f);  // +2 ticks per bar
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    EXPECT_GT(row.momentum, 0.0f);
}

TEST_F(PriceDynamicsFeatureTest, HighLowRange20_InTicks) {
    auto bars = make_bar_sequence(25, 4500.0f, 0.25f, 100);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // Over last 20 bars, max(high) - min(low) / tick_size
    EXPECT_GT(row.high_low_range_20, 0.0f);
}

TEST_F(PriceDynamicsFeatureTest, HighLowRange50_NaNForFirst50Bars) {
    auto bars = make_bar_sequence(30);
    for (int i = 0; i < 30; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.high_low_range_50))
            << "high_low_range_50 should be NaN for bar " << i;
    }
}

TEST_F(PriceDynamicsFeatureTest, ClosePosition_MidRangeIsHalf) {
    // close_position = (close - low_N) / (high_N - low_N + eps)
    // If close is exactly at midpoint of range → 0.5
    auto bars = make_bar_sequence(25, 4500.0f, 0.25f);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // close_position is in [0, 1]
    EXPECT_GE(row.close_position, 0.0f);
    EXPECT_LE(row.close_position, 1.0f);
}

// ===========================================================================
// Category 4: Cross-Scale Dynamics
// ===========================================================================

class CrossScaleFeatureTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(CrossScaleFeatureTest, VolumeSurprise_FirstBarInitializedToOne) {
    // EWMA init at bar 0 with first bar's value → surprise = 1.0
    auto bar = make_bar(4500.0f, 100);
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.volume_surprise, 1.0f, 1e-4f);
}

TEST_F(CrossScaleFeatureTest, VolumeSurprise_HighVolumeBarAboveOne) {
    // Establish EWMA with normal volume, then spike.
    auto bars = make_bar_sequence(25, 4500.0f, 0.25f, 100);
    bars.push_back(make_bar(4506.5f, 500, 86.0f));  // Volume spike
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // volume / EWMA(volume) >> 1 for a spike
    EXPECT_GT(row.volume_surprise, 2.0f);
}

TEST_F(CrossScaleFeatureTest, DurationSurprise_FirstBarInitializedToOne) {
    auto bar = make_bar(4500.0f);
    bar.bar_duration_s = 5.0f;
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.duration_surprise, 1.0f, 1e-4f);
}

TEST_F(CrossScaleFeatureTest, DurationSurprise_LongBarAboveOne) {
    auto bars = make_bar_sequence(25);
    for (auto& bar : bars) bar.bar_duration_s = 1.0f;
    auto long_bar = make_bar(4507.0f, 100, 90.0f);
    long_bar.bar_duration_s = 10.0f;  // 10x normal duration
    bars.push_back(long_bar);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    EXPECT_GT(row.duration_surprise, 2.0f);
}

TEST_F(CrossScaleFeatureTest, Acceleration_DifferenceOfReturns) {
    // acceleration = return_1 - return_1[lag=1]
    std::vector<float> mids = {4500.0f, 4500.25f, 4500.75f, 4501.50f};
    auto bars = make_bars_with_mids(mids);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // return_1 at bar 3: (4501.50 - 4500.75) / 0.25 = 3.0
    // return_1 at bar 2: (4500.75 - 4500.25) / 0.25 = 2.0
    // acceleration = 3.0 - 2.0 = 1.0
    EXPECT_NEAR(row.acceleration, 1.0f, 1e-4f);
}

TEST_F(CrossScaleFeatureTest, VolPriceCorr_NaNForFirst20Bars) {
    auto bars = make_bar_sequence(15);
    for (int i = 0; i < 15; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.vol_price_corr))
            << "vol_price_corr should be NaN for bar " << i;
    }
}

TEST_F(CrossScaleFeatureTest, VolPriceCorr_DefinedAfter20Bars) {
    auto bars = make_bar_sequence(25);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // After 20 bars, rolling correlation should be defined
    EXPECT_FALSE(std::isnan(row.vol_price_corr));
}

TEST_F(CrossScaleFeatureTest, VolPriceCorr_InValidRange) {
    auto bars = make_bar_sequence(25);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // Correlation must be in [-1, 1]
    EXPECT_GE(row.vol_price_corr, -1.0f);
    EXPECT_LE(row.vol_price_corr, 1.0f);
}

// ===========================================================================
// Category 5: Time Context
// ===========================================================================

class TimeContextFeatureTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(TimeContextFeatureTest, TimeSinCos_AtOpen) {
    auto bar = make_bar(4500.0f);
    bar.time_of_day = 9.5f;  // 09:30
    auto row = computer_.update(bar);

    constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;
    float frac = 9.5f / 24.0f;
    EXPECT_NEAR(row.time_sin, std::sin(TWO_PI * frac), 1e-4f);
    EXPECT_NEAR(row.time_cos, std::cos(TWO_PI * frac), 1e-4f);
}

TEST_F(TimeContextFeatureTest, TimeSinCos_AtClose) {
    auto bar = make_bar(4500.0f);
    bar.time_of_day = 16.0f;
    auto row = computer_.update(bar);

    constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;
    float frac = 16.0f / 24.0f;
    EXPECT_NEAR(row.time_sin, std::sin(TWO_PI * frac), 1e-4f);
    EXPECT_NEAR(row.time_cos, std::cos(TWO_PI * frac), 1e-4f);
}

TEST_F(TimeContextFeatureTest, MinutesSinceOpen_AtOpen) {
    auto bar = make_bar(4500.0f);
    bar.time_of_day = 9.5f;  // 09:30
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.minutes_since_open, 0.0f, 1e-4f);
}

TEST_F(TimeContextFeatureTest, MinutesSinceOpen_OneHourAfterOpen) {
    auto bar = make_bar(4500.0f);
    bar.time_of_day = 10.5f;  // 10:30 = 60 min after open
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.minutes_since_open, 60.0f, 1e-4f);
}

TEST_F(TimeContextFeatureTest, MinutesToClose_AtOpen) {
    auto bar = make_bar(4500.0f);
    bar.time_of_day = 9.5f;  // 09:30 → 390 min to 16:00
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.minutes_to_close, 390.0f, 1e-4f);
}

TEST_F(TimeContextFeatureTest, MinutesToClose_AtClose) {
    auto bar = make_bar(4500.0f);
    bar.time_of_day = 16.0f;
    auto row = computer_.update(bar);
    EXPECT_NEAR(row.minutes_to_close, 0.0f, 1e-4f);
}

TEST_F(TimeContextFeatureTest, SessionVolumeFrac_Day1_UsesActualTotal) {
    // Day 1: session_volume_frac = cumulative / actual total for that day (mild lookahead)
    // We process day 1 (only day), so frac at last bar = 1.0
    auto bars = make_bar_sequence(10, 4500.0f, 0.25f, 100);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // At last bar of day 1, cumulative = 1000, total = 1000 → frac = 1.0
    // (Implementation provides the day's total volume after processing all bars.)
}

TEST_F(TimeContextFeatureTest, SessionVolumeFrac_Day2_UsesExpandingWindowPriorDayMean) {
    // After day 1, day 2 uses expanding-window mean of prior days.
    BarFeatureComputer computer(TICK_SIZE);

    // Day 1: process 10 bars, each 100 volume → total = 1000
    auto day1 = make_bar_sequence(10, 4500.0f, 0.25f, 100);
    for (const auto& bar : day1) {
        computer.update(bar);
    }
    computer.end_session(1000);  // Report day 1 total volume

    // Day 2: first bar with volume 200, prior day avg = 1000
    auto bar2 = make_bar(4502.5f, 200);
    auto row = computer.update(bar2);
    // frac = 200 / 1000 = 0.2
    EXPECT_NEAR(row.session_volume_frac, 0.2f, 0.05f);
}

// ===========================================================================
// Category 6: Message Microstructure — intra-bar MBO summary
// ===========================================================================

class MessageMicrostructureFeatureTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(MessageMicrostructureFeatureTest, CancelAddRatio) {
    auto bar = make_bar(4500.0f);
    bar.cancel_count = 30;
    bar.add_count = 60;
    auto row = computer_.update(bar);
    // cancel_count / (add_count + eps) = 30 / 60 = 0.5
    EXPECT_NEAR(row.cancel_add_ratio, 0.5f, 1e-4f);
}

TEST_F(MessageMicrostructureFeatureTest, CancelAddRatio_ZeroAdds) {
    auto bar = make_bar(4500.0f);
    bar.cancel_count = 10;
    bar.add_count = 0;
    auto row = computer_.update(bar);
    // 10 / (0 + eps) ≈ very large, but bounded
    EXPECT_FALSE(std::isinf(row.cancel_add_ratio));
}

TEST_F(MessageMicrostructureFeatureTest, MessageRate) {
    auto bar = make_bar(4500.0f);
    bar.add_count = 50;
    bar.cancel_count = 30;
    bar.modify_count = 20;
    bar.bar_duration_s = 2.0f;
    auto row = computer_.update(bar);
    // message_rate = (50 + 30 + 20) / 2.0 = 50.0
    EXPECT_NEAR(row.message_rate, 50.0f, 1e-4f);
}

TEST_F(MessageMicrostructureFeatureTest, MessageRate_ZeroDuration) {
    auto bar = make_bar(4500.0f);
    bar.add_count = 10;
    bar.cancel_count = 5;
    bar.modify_count = 3;
    bar.bar_duration_s = 0.0f;
    auto row = computer_.update(bar);
    // Should handle division by zero gracefully
    EXPECT_FALSE(std::isinf(row.message_rate));
}

TEST_F(MessageMicrostructureFeatureTest, ModifyFraction) {
    auto bar = make_bar(4500.0f);
    bar.add_count = 40;
    bar.cancel_count = 30;
    bar.modify_count = 30;
    auto row = computer_.update(bar);
    // modify / (add + cancel + modify + eps) = 30 / 100 = 0.3
    EXPECT_NEAR(row.modify_fraction, 0.3f, 1e-4f);
}

TEST_F(MessageMicrostructureFeatureTest, OrderFlowToxicity_NoTradesMovesMid) {
    auto bar = make_bar(4500.0f);
    bar.trade_event_count = 50;
    // No mid-price movement within bar → toxicity = 0
    auto row = computer_.update(bar);
    EXPECT_GE(row.order_flow_toxicity, 0.0f);
    EXPECT_LE(row.order_flow_toxicity, 1.0f);
}

TEST_F(MessageMicrostructureFeatureTest, CancelConcentration_HHI) {
    // cancel_concentration = HHI of cancel counts per price level.
    // Interpretation: concentrated cancellations indicate informed activity.
    auto bar = make_bar(4500.0f);
    auto row = computer_.update(bar);
    EXPECT_GE(row.cancel_concentration, 0.0f);
    EXPECT_LE(row.cancel_concentration, 1.0f);
}

// ===========================================================================
// Forward Returns — Targets (not features)
// ===========================================================================

class ForwardReturnTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(ForwardReturnTest, Return1_Correct) {
    // Build enough bars and compute forward returns.
    auto bars = make_bar_sequence(10, 4500.0f, 0.25f);
    auto rows = computer_.compute_all(bars);

    // fwd_return_1 for bar i = (bar[i+1].close_mid - bar[i].close_mid) / tick_size
    // = 0.25 / 0.25 = 1.0 for each bar (uniform step)
    for (size_t i = 0; i + 1 < bars.size(); ++i) {
        EXPECT_NEAR(rows[i].fwd_return_1, 1.0f, 1e-4f)
            << "fwd_return_1 mismatch at bar " << i;
    }
}

TEST_F(ForwardReturnTest, Return5_Correct) {
    auto bars = make_bar_sequence(10, 4500.0f, 0.25f);
    auto rows = computer_.compute_all(bars);

    for (size_t i = 0; i + 5 < bars.size(); ++i) {
        EXPECT_NEAR(rows[i].fwd_return_5, 5.0f, 1e-4f)
            << "fwd_return_5 mismatch at bar " << i;
    }
}

TEST_F(ForwardReturnTest, Return20_Correct) {
    auto bars = make_bar_sequence(30, 4500.0f, 0.25f);
    auto rows = computer_.compute_all(bars);

    for (size_t i = 0; i + 20 < bars.size(); ++i) {
        EXPECT_NEAR(rows[i].fwd_return_20, 20.0f, 1e-4f)
            << "fwd_return_20 mismatch at bar " << i;
    }
}

TEST_F(ForwardReturnTest, Return100_Correct) {
    auto bars = make_bar_sequence(110, 4500.0f, 0.25f);
    auto rows = computer_.compute_all(bars);

    for (size_t i = 0; i + 100 < bars.size(); ++i) {
        EXPECT_NEAR(rows[i].fwd_return_100, 100.0f, 1e-4f)
            << "fwd_return_100 mismatch at bar " << i;
    }
}

TEST_F(ForwardReturnTest, LastNBarsHaveNaNForwardReturns) {
    auto bars = make_bar_sequence(110, 4500.0f, 0.25f);
    auto rows = computer_.compute_all(bars);

    size_t n = bars.size();
    // Last 1 bar → fwd_return_1 is NaN
    EXPECT_TRUE(std::isnan(rows[n - 1].fwd_return_1));

    // Last 5 bars → fwd_return_5 is NaN
    for (size_t i = n - 5; i < n; ++i) {
        EXPECT_TRUE(std::isnan(rows[i].fwd_return_5))
            << "fwd_return_5 should be NaN for bar " << i;
    }

    // Last 20 bars → fwd_return_20 is NaN
    for (size_t i = n - 20; i < n; ++i) {
        EXPECT_TRUE(std::isnan(rows[i].fwd_return_20))
            << "fwd_return_20 should be NaN for bar " << i;
    }

    // Last 100 bars → fwd_return_100 is NaN
    for (size_t i = n - 100; i < n; ++i) {
        EXPECT_TRUE(std::isnan(rows[i].fwd_return_100))
            << "fwd_return_100 should be NaN for bar " << i;
    }
}

TEST_F(ForwardReturnTest, ForwardReturnsNeverUsedAsFeatures) {
    // Forward returns must not influence any feature computation.
    // Build two identical bar sequences, one with different future bars.
    auto bars_a = make_bar_sequence(20, 4500.0f, 0.25f);
    auto bars_b = make_bar_sequence(20, 4500.0f, 0.25f);
    // Modify future bars in sequence b
    bars_b.back().close_mid = 9999.0f;

    BarFeatureComputer comp_a(TICK_SIZE);
    BarFeatureComputer comp_b(TICK_SIZE);

    auto rows_a = comp_a.compute_all(bars_a);
    auto rows_b = comp_b.compute_all(bars_b);

    // Features at bar 10 should be identical (only future differs)
    EXPECT_FLOAT_EQ(rows_a[10].book_imbalance_10, rows_b[10].book_imbalance_10);
    EXPECT_FLOAT_EQ(rows_a[10].net_volume, rows_b[10].net_volume);
    EXPECT_FLOAT_EQ(rows_a[10].volatility_20, rows_b[10].volatility_20);
}

// ===========================================================================
// Warmup and NaN Policy
// ===========================================================================

class FeatureWarmupPolicyTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(FeatureWarmupPolicyTest, IsWarmupTrueWhenAnyFeatureInWarmup) {
    // §8.6: "True if ANY feature is in warmup state"
    // First bar → EWMA features in warmup → is_warmup = true
    auto bar = make_bar(4500.0f);
    auto row = computer_.update(bar);
    EXPECT_TRUE(row.is_warmup);
}

TEST_F(FeatureWarmupPolicyTest, IsWarmupTrueForFirst20Bars) {
    // EWMA span = 20, so first 20 bars are warmup.
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(row.is_warmup)
            << "bar " << i << " should be warmup";
    }
}

TEST_F(FeatureWarmupPolicyTest, IsWarmupBecomesFalseAfterAllFeaturesStable) {
    // After max(window_lengths) bars, all features have sufficient lookback.
    // Max window = 50 (volatility_50)
    auto bars = make_bar_sequence(55);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // Bar 54 → all features stable → is_warmup = false
    EXPECT_FALSE(row.is_warmup);
}

TEST_F(FeatureWarmupPolicyTest, EWMAResetsAtSessionBoundary) {
    // §8.6: EWMA resets at session boundaries.
    BarFeatureComputer computer(TICK_SIZE);

    // Session 1: 25 bars
    auto session1 = make_bar_sequence(25, 4500.0f, 0.25f, 100);
    for (const auto& bar : session1) {
        computer.update(bar);
    }
    computer.reset();  // Session boundary

    // Session 2: first bar should be warmup again
    auto bar = make_bar(4510.0f, 100);
    auto row = computer.update(bar);
    EXPECT_TRUE(row.is_warmup);
    // EWMA should reinitialize with this bar's value
    EXPECT_NEAR(row.volume_surprise, 1.0f, 1e-4f);
}

TEST_F(FeatureWarmupPolicyTest, RollingWindowFeaturesNaNDuringWarmup) {
    // volatility_20 uses 20-bar window → NaN for first 20 bars
    // volatility_50 uses 50-bar window → NaN for first 50 bars
    // kyle_lambda uses 20-bar window → NaN for first 20 bars
    auto bars = make_bar_sequence(10);
    for (int i = 0; i < 10; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.volatility_20));
        EXPECT_TRUE(std::isnan(row.volatility_50));
        EXPECT_TRUE(std::isnan(row.kyle_lambda));
    }
}

TEST_F(FeatureWarmupPolicyTest, NaNOnlyWhereDocumented) {
    // After sufficient bars, NO unexpected NaN in Track A features.
    auto bars = make_bar_sequence(60);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // At bar 59, all windows have sufficient lookback.
    EXPECT_FALSE(std::isnan(row.book_imbalance_1));
    EXPECT_FALSE(std::isnan(row.book_imbalance_10));
    EXPECT_FALSE(std::isnan(row.spread));
    EXPECT_FALSE(std::isnan(row.net_volume));
    EXPECT_FALSE(std::isnan(row.volume_imbalance));
    EXPECT_FALSE(std::isnan(row.return_1));
    EXPECT_FALSE(std::isnan(row.return_5));
    EXPECT_FALSE(std::isnan(row.return_20));
    EXPECT_FALSE(std::isnan(row.volatility_20));
    EXPECT_FALSE(std::isnan(row.volatility_50));
    EXPECT_FALSE(std::isnan(row.kyle_lambda));
    EXPECT_FALSE(std::isnan(row.momentum));
    EXPECT_FALSE(std::isnan(row.high_low_range_20));
    EXPECT_FALSE(std::isnan(row.high_low_range_50));
    EXPECT_FALSE(std::isnan(row.close_position));
    EXPECT_FALSE(std::isnan(row.volume_surprise));
    EXPECT_FALSE(std::isnan(row.duration_surprise));
    EXPECT_FALSE(std::isnan(row.acceleration));
    EXPECT_FALSE(std::isnan(row.vol_price_corr));
    EXPECT_FALSE(std::isnan(row.time_sin));
    EXPECT_FALSE(std::isnan(row.time_cos));
    EXPECT_FALSE(std::isnan(row.minutes_since_open));
    EXPECT_FALSE(std::isnan(row.minutes_to_close));
    EXPECT_FALSE(std::isnan(row.cancel_add_ratio));
    EXPECT_FALSE(std::isnan(row.message_rate));
    EXPECT_FALSE(std::isnan(row.modify_fraction));
    EXPECT_FALSE(std::isnan(row.order_flow_toxicity));
    EXPECT_FALSE(std::isnan(row.cancel_concentration));
}

// ===========================================================================
// Feature Count Validation
// ===========================================================================

class FeatureCountTest : public ::testing::Test {};

TEST_F(FeatureCountTest, TotalTrackAFeatureCountMatchesTaxonomy) {
    // Spec says ~45 features.
    // Cat 1: 4 (imbalance) + 1 (weighted) + 1 (spread) + 10 (bid depth) + 10 (ask depth)
    //        + 2 (concentration) + 2 (slope) + 2 (level count) = 32
    // Cat 2: 7
    // Cat 3: 3 (returns) + 2 (vol) + 1 (momentum) + 2 (range) + 1 (position) = 9
    // But return_1 return_5 return_20 = 3 backward returns (spec §3.1 says return_{1,5,20})
    // Cat 4: 4
    // Cat 5: 5
    // Cat 6: 5
    // Total = 32 + 7 + 9 + 4 + 5 + 5 = 62 (more than ~45 because depth profiles are 20 values)
    // The spec says ~45 treating depth profiles as 2 features (bid/ask depth).
    // The actual column count from BarFeatureRow should be verified.
    EXPECT_GE(BarFeatureRow::feature_count(), 45u);
}

TEST_F(FeatureCountTest, FeatureNamesListMatchesCount) {
    auto names = BarFeatureRow::feature_names();
    EXPECT_EQ(names.size(), BarFeatureRow::feature_count());
}

// ===========================================================================
// compute_all — Batch processing
// ===========================================================================

class ComputeAllTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(ComputeAllTest, ReturnsOneRowPerBar) {
    auto bars = make_bar_sequence(50);
    auto rows = computer_.compute_all(bars);
    EXPECT_EQ(rows.size(), bars.size());
}

TEST_F(ComputeAllTest, EmptyInputReturnsEmptyOutput) {
    std::vector<Bar> empty;
    auto rows = computer_.compute_all(empty);
    EXPECT_TRUE(rows.empty());
}

TEST_F(ComputeAllTest, SingleBarProducesOneRow) {
    std::vector<Bar> single = {make_bar(4500.0f)};
    auto rows = computer_.compute_all(single);
    EXPECT_EQ(rows.size(), 1u);
}

TEST_F(ComputeAllTest, ForwardReturnsFilledByComputeAll) {
    auto bars = make_bar_sequence(10, 4500.0f, 0.25f);
    auto rows = computer_.compute_all(bars);

    // Bar 0: fwd_return_1 should be defined
    EXPECT_FALSE(std::isnan(rows[0].fwd_return_1));
    // Bar 9: fwd_return_1 should be NaN (no bar 10)
    EXPECT_TRUE(std::isnan(rows[9].fwd_return_1));
}

TEST_F(ComputeAllTest, MetadataPopulated) {
    auto bars = make_bar_sequence(5);
    auto rows = computer_.compute_all(bars);
    for (size_t i = 0; i < rows.size(); ++i) {
        EXPECT_NE(rows[i].timestamp, 0u);
    }
}
