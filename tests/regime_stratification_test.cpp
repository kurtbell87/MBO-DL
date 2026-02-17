// regime_stratification_test.cpp — TDD RED phase tests for Regime Stratification
// Spec: .kit/docs/multi-day-backtest.md §Regime Stratification (§9.6)
//
// Tests the regime stratification engine: volatility quartiles, time-of-day
// sessions, volume regimes, trend classification, and cross-regime stability.
//
// Headers below do not exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

// Header that the implementation must provide:
#include "backtest/regime_stratification.hpp"

// Already-existing headers from prior phases:
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"
#include "test_bar_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

using test_helpers::TICK;
using test_helpers::NS_PER_SEC;
using test_helpers::NS_PER_HOUR;
using test_helpers::RTH_OPEN_NS;
using test_helpers::make_bar;
using test_helpers::make_bar_series;

// Build a TradeRecord with specified net_pnl and entry hour.
TradeRecord make_trade(float net_pnl, float entry_hour, int direction = 1) {
    TradeRecord tr{};
    tr.direction = direction;
    tr.net_pnl = net_pnl;
    tr.gross_pnl = net_pnl + 2.0f;  // assume ~$2 round-trip cost
    tr.entry_bar_idx = 0;
    tr.exit_bar_idx = 1;
    tr.bars_held = 1;
    tr.entry_price = 4500.0f;
    tr.exit_price = (direction > 0)
                        ? 4500.0f + net_pnl / 5.0f  // contract_multiplier=5
                        : 4500.0f - net_pnl / 5.0f;
    return tr;
}

// Build a vector of bars with specified realized volatility.
// Returns bars where close_mid has the given standard deviation over `count` bars.
std::vector<Bar> make_bars_with_vol(float base_mid, float vol_per_bar,
                                     int count, uint32_t vol_per_bar_volume = 50) {
    std::vector<Bar> bars;
    bars.reserve(count);
    float mid = base_mid;
    for (int i = 0; i < count; ++i) {
        // Alternate +/- to create volatility without drift
        float delta = (i % 2 == 0) ? vol_per_bar : -vol_per_bar;
        mid += delta;
        uint64_t ts_offset = static_cast<uint64_t>(i) * NS_PER_SEC;
        bars.push_back(make_bar(mid, vol_per_bar_volume, ts_offset));
    }
    return bars;
}

// Build bars spanning different time-of-day sessions.
std::vector<Bar> make_bars_at_hour(float mid, int count, float hour_of_day,
                                    uint32_t vol = 50) {
    std::vector<Bar> bars;
    bars.reserve(count);
    uint64_t hour_offset_ns = static_cast<uint64_t>((hour_of_day - 9.5f) * 3600.0f)
                              * NS_PER_SEC;
    for (int i = 0; i < count; ++i) {
        uint64_t ts_offset = hour_offset_ns + static_cast<uint64_t>(i) * NS_PER_SEC;
        auto bar = make_bar(mid, vol, ts_offset);
        bar.time_of_day = hour_of_day + static_cast<float>(i) / 3600.0f;
        bars.push_back(bar);
    }
    return bars;
}

}  // namespace

// ===========================================================================
// 1. VolatilityQuartile — realized volatility quartiles
// ===========================================================================
class VolatilityQuartileTest : public ::testing::Test {};

TEST_F(VolatilityQuartileTest, ComputeRealizedVol20Bar) {
    // Spec: "20-bar realized vol, Q1-Q4"
    // compute_realized_vol should use a 20-bar window
    auto bars = make_bars_with_vol(4500.0f, 0.50f, 30);

    float vol = regime::compute_realized_vol(bars, 20);
    EXPECT_GT(vol, 0.0f);
}

TEST_F(VolatilityQuartileTest, QuartilesPartitionDays) {
    // Given a vector of per-day realized vol values, assign quartiles Q1-Q4
    std::vector<float> daily_vols = {
        0.10f, 0.20f, 0.15f, 0.40f, 0.80f, 0.60f, 0.25f, 0.35f,
        0.12f, 0.22f, 0.50f, 0.70f, 0.30f, 0.45f, 0.55f, 0.65f
    };

    auto quartiles = regime::assign_volatility_quartiles(daily_vols);

    EXPECT_EQ(quartiles.size(), daily_vols.size());

    // Each quartile should be 1-4
    for (int q : quartiles) {
        EXPECT_GE(q, 1);
        EXPECT_LE(q, 4);
    }
}

TEST_F(VolatilityQuartileTest, QuartilesAreBalanced) {
    // With 16 values, each quartile should have 4 entries
    std::vector<float> daily_vols;
    for (int i = 1; i <= 16; ++i) {
        daily_vols.push_back(static_cast<float>(i));
    }

    auto quartiles = regime::assign_volatility_quartiles(daily_vols);

    std::map<int, int> counts;
    for (int q : quartiles) counts[q]++;

    EXPECT_EQ(counts[1], 4);
    EXPECT_EQ(counts[2], 4);
    EXPECT_EQ(counts[3], 4);
    EXPECT_EQ(counts[4], 4);
}

TEST_F(VolatilityQuartileTest, Q1IsLowestVolatility) {
    std::vector<float> daily_vols = {0.10f, 0.90f, 0.20f, 0.80f,
                                     0.30f, 0.70f, 0.40f, 0.60f};

    auto quartiles = regime::assign_volatility_quartiles(daily_vols);

    // The lowest vol (0.10) should be in Q1
    // Find index of min vol
    int min_idx = 0;
    for (size_t i = 1; i < daily_vols.size(); ++i) {
        if (daily_vols[i] < daily_vols[min_idx]) min_idx = static_cast<int>(i);
    }
    EXPECT_EQ(quartiles[min_idx], 1);

    // The highest vol (0.90) should be in Q4
    int max_idx = 0;
    for (size_t i = 1; i < daily_vols.size(); ++i) {
        if (daily_vols[i] > daily_vols[max_idx]) max_idx = static_cast<int>(i);
    }
    EXPECT_EQ(quartiles[max_idx], 4);
}

// ===========================================================================
// 2. Time-of-day sessions
// ===========================================================================
class TimeOfDaySessionTest : public ::testing::Test {};

TEST_F(TimeOfDaySessionTest, ClassifyOpen) {
    // Spec: "Open (09:30–10:30)"
    EXPECT_EQ(regime::classify_session(9.5f), regime::Session::OPEN);
    EXPECT_EQ(regime::classify_session(10.0f), regime::Session::OPEN);
    EXPECT_EQ(regime::classify_session(10.49f), regime::Session::OPEN);
}

TEST_F(TimeOfDaySessionTest, ClassifyMid) {
    // Spec: "Mid (10:30–14:00)"
    EXPECT_EQ(regime::classify_session(10.5f), regime::Session::MID);
    EXPECT_EQ(regime::classify_session(12.0f), regime::Session::MID);
    EXPECT_EQ(regime::classify_session(13.99f), regime::Session::MID);
}

TEST_F(TimeOfDaySessionTest, ClassifyClose) {
    // Spec: "Close (14:00–16:00)"
    EXPECT_EQ(regime::classify_session(14.0f), regime::Session::CLOSE);
    EXPECT_EQ(regime::classify_session(15.0f), regime::Session::CLOSE);
    EXPECT_EQ(regime::classify_session(15.99f), regime::Session::CLOSE);
}

TEST_F(TimeOfDaySessionTest, ClassifyBoundaryOpenMid) {
    // Boundary at 10:30 — exactly 10.5 should be MID
    EXPECT_EQ(regime::classify_session(10.5f), regime::Session::MID);
}

TEST_F(TimeOfDaySessionTest, ClassifyBoundaryMidClose) {
    // Boundary at 14:00 — exactly 14.0 should be CLOSE
    EXPECT_EQ(regime::classify_session(14.0f), regime::Session::CLOSE);
}

// ===========================================================================
// 3. Volume regimes — daily total volume quartiles
// ===========================================================================
class VolumeRegimeTest : public ::testing::Test {};

TEST_F(VolumeRegimeTest, AssignVolumeQuartiles) {
    // Spec: "Daily total volume quartiles. High-volume days may capture event-driven moves."
    std::vector<uint64_t> daily_volumes = {
        1000, 2000, 1500, 4000, 8000, 6000, 2500, 3500,
        1200, 2200, 5000, 7000, 3000, 4500, 5500, 6500
    };

    auto quartiles = regime::assign_volume_quartiles(daily_volumes);
    EXPECT_EQ(quartiles.size(), daily_volumes.size());

    for (int q : quartiles) {
        EXPECT_GE(q, 1);
        EXPECT_LE(q, 4);
    }
}

TEST_F(VolumeRegimeTest, HighVolumeIsQ4) {
    std::vector<uint64_t> daily_volumes = {100, 200, 300, 400, 500, 600, 700, 10000};

    auto quartiles = regime::assign_volume_quartiles(daily_volumes);

    // Highest volume (10000) should be in Q4
    EXPECT_EQ(quartiles.back(), 4);
    // Lowest volume (100) should be in Q1
    EXPECT_EQ(quartiles.front(), 1);
}

// ===========================================================================
// 4. Trend vs mean-reversion classification
// ===========================================================================
class TrendClassificationTest : public ::testing::Test {};

TEST_F(TrendClassificationTest, StrongTrend) {
    // Spec: "strong trend (>1%)"
    float otc_return_pct = 1.5f;
    EXPECT_EQ(regime::classify_trend(otc_return_pct), regime::Trend::STRONG_TREND);
}

TEST_F(TrendClassificationTest, RangeBound) {
    // Spec: "range-bound (<0.3%)"
    float otc_return_pct = 0.1f;
    EXPECT_EQ(regime::classify_trend(otc_return_pct), regime::Trend::RANGE_BOUND);
}

TEST_F(TrendClassificationTest, Moderate) {
    // Spec: "moderate" (0.3% to 1.0%)
    float otc_return_pct = 0.5f;
    EXPECT_EQ(regime::classify_trend(otc_return_pct), regime::Trend::MODERATE);
}

TEST_F(TrendClassificationTest, BoundaryStrongTrend) {
    // Exactly 1.0% — should be strong trend (>1% means strictly >)
    // At exactly 1.0%, should be moderate
    EXPECT_EQ(regime::classify_trend(1.0f), regime::Trend::MODERATE);
    EXPECT_EQ(regime::classify_trend(1.01f), regime::Trend::STRONG_TREND);
}

TEST_F(TrendClassificationTest, BoundaryRangeBound) {
    // At exactly 0.3%, should be moderate
    EXPECT_EQ(regime::classify_trend(0.3f), regime::Trend::MODERATE);
    EXPECT_EQ(regime::classify_trend(0.29f), regime::Trend::RANGE_BOUND);
}

TEST_F(TrendClassificationTest, NegativeReturnUsesAbsoluteValue) {
    // Spec: "|OTC return|" — uses absolute value
    EXPECT_EQ(regime::classify_trend(-1.5f), regime::Trend::STRONG_TREND);
    EXPECT_EQ(regime::classify_trend(-0.1f), regime::Trend::RANGE_BOUND);
}

TEST_F(TrendClassificationTest, ComputeOTCReturn) {
    // OTC return = (close - open) / open * 100 for a day's bars
    auto bars = make_bar_series(4500.0f, 4545.0f, 50, 50);  // +1% return

    float otc_return = regime::compute_otc_return(bars);
    EXPECT_NEAR(otc_return, 1.0f, 0.01f);
}

// ===========================================================================
// 5. RegimeResult — per-regime expectancy
// ===========================================================================
class RegimeResultTest : public ::testing::Test {};

TEST_F(RegimeResultTest, DefaultConstruction) {
    RegimeResult result{};
    EXPECT_EQ(result.trade_count, 0);
    EXPECT_FLOAT_EQ(result.expectancy, 0.0f);
    EXPECT_FLOAT_EQ(result.net_pnl, 0.0f);
    EXPECT_FLOAT_EQ(result.win_rate, 0.0f);
}

TEST_F(RegimeResultTest, HasProfitFactor) {
    RegimeResult result{};
    EXPECT_FLOAT_EQ(result.profit_factor, 0.0f);
}

TEST_F(RegimeResultTest, HasSharpe) {
    RegimeResult result{};
    EXPECT_FLOAT_EQ(result.sharpe, 0.0f);
}

// ===========================================================================
// 6. RegimeStratifier — full stratification pipeline
// ===========================================================================
class RegimeStratifierTest : public ::testing::Test {};

TEST_F(RegimeStratifierTest, StratifyByVolatility) {
    // Should produce a map from volatility quartile (1-4) to RegimeResult
    RegimeStratifier stratifier;

    // Simulate 8 days: 4 low-vol, 4 high-vol
    std::vector<float> daily_vols = {0.1f, 0.15f, 0.12f, 0.18f,
                                     0.80f, 0.90f, 0.85f, 0.95f};
    std::vector<std::vector<TradeRecord>> daily_trades(8);

    // Low-vol days: positive trades
    for (int d = 0; d < 4; ++d) {
        daily_trades[d].push_back(make_trade(5.0f, 10.0f));
        daily_trades[d].push_back(make_trade(3.0f, 11.0f));
    }
    // High-vol days: more positive trades
    for (int d = 4; d < 8; ++d) {
        daily_trades[d].push_back(make_trade(10.0f, 10.0f));
        daily_trades[d].push_back(make_trade(8.0f, 11.0f));
    }

    auto vol_strat = stratifier.by_volatility(daily_vols, daily_trades);

    // Should have entries for Q1 and Q4 (at minimum)
    EXPECT_FALSE(vol_strat.empty());
    EXPECT_LE(vol_strat.size(), 4u);

    // Each quartile result should have valid trade counts
    for (const auto& [quartile, result] : vol_strat) {
        EXPECT_GE(quartile, 1);
        EXPECT_LE(quartile, 4);
        EXPECT_GT(result.trade_count, 0);
    }
}

TEST_F(RegimeStratifierTest, StratifyByTimeOfDay) {
    // Should produce a map from Session to RegimeResult
    RegimeStratifier stratifier;

    std::vector<TradeRecord> trades;
    // Trades in OPEN session (9:30-10:30)
    auto t1 = make_trade(5.0f, 9.75f);
    trades.push_back(t1);
    // Trades in MID session (10:30-14:00)
    auto t2 = make_trade(3.0f, 12.0f);
    trades.push_back(t2);
    // Trades in CLOSE session (14:00-16:00)
    auto t3 = make_trade(-2.0f, 15.0f);
    trades.push_back(t3);

    // Need bars to get time_of_day for each trade
    std::vector<Bar> bars;
    bars.push_back(make_bar(4500.0f, 50, 0));  // bar idx 0, hour ~9.5
    auto bar_mid = make_bar(4500.0f, 50,
                            static_cast<uint64_t>(2.5f * 3600.0f) * NS_PER_SEC);
    bar_mid.time_of_day = 12.0f;
    bars.push_back(bar_mid);
    auto bar_close = make_bar(4500.0f, 50,
                               static_cast<uint64_t>(5.5f * 3600.0f) * NS_PER_SEC);
    bar_close.time_of_day = 15.0f;
    bars.push_back(bar_close);

    auto tod_strat = stratifier.by_time_of_day(trades, bars);

    EXPECT_FALSE(tod_strat.empty());
    // Should have at most 3 sessions
    EXPECT_LE(tod_strat.size(), 3u);
}

TEST_F(RegimeStratifierTest, StratifyByVolume) {
    // Should produce a map from volume quartile (1-4) to RegimeResult
    RegimeStratifier stratifier;

    std::vector<uint64_t> daily_volumes = {1000, 2000, 3000, 4000,
                                           5000, 6000, 7000, 8000};
    std::vector<std::vector<TradeRecord>> daily_trades(8);
    for (int d = 0; d < 8; ++d) {
        daily_trades[d].push_back(make_trade(2.0f, 10.0f));
    }

    auto vol_strat = stratifier.by_volume(daily_volumes, daily_trades);

    EXPECT_FALSE(vol_strat.empty());
    for (const auto& [quartile, result] : vol_strat) {
        EXPECT_GE(quartile, 1);
        EXPECT_LE(quartile, 4);
    }
}

TEST_F(RegimeStratifierTest, StratifyByTrend) {
    // Should produce a map from Trend classification to RegimeResult
    RegimeStratifier stratifier;

    std::vector<float> daily_otc_returns = {
        1.5f, 0.1f, 0.5f, -1.2f, 0.2f, 0.7f, -0.05f, 2.0f
    };
    std::vector<std::vector<TradeRecord>> daily_trades(8);
    for (int d = 0; d < 8; ++d) {
        daily_trades[d].push_back(make_trade(3.0f, 10.0f));
    }

    auto trend_strat = stratifier.by_trend(daily_otc_returns, daily_trades);

    EXPECT_FALSE(trend_strat.empty());
    // Should have entries for at least 2 trend types
    EXPECT_GE(trend_strat.size(), 2u);
}

// ===========================================================================
// 7. Cross-regime stability score
// ===========================================================================
class StabilityScoreTest : public ::testing::Test {};

TEST_F(StabilityScoreTest, RobustStability) {
    // Spec: "> 0.5 → robust"
    // All quartiles have similar expectancy
    std::map<int, RegimeResult> vol_strat;
    for (int q = 1; q <= 4; ++q) {
        RegimeResult r{};
        r.expectancy = 2.0f + static_cast<float>(q) * 0.1f;
        r.trade_count = 100;
        vol_strat[q] = r;
    }

    float stability = regime::compute_stability_score(vol_strat);

    // min/max ≈ 2.1/2.4 ≈ 0.875 → robust
    EXPECT_GT(stability, 0.5f);
}

TEST_F(StabilityScoreTest, RegimeDependentStability) {
    // Spec: "0.2–0.5 → regime-dependent"
    std::map<int, RegimeResult> vol_strat;
    RegimeResult r1{}; r1.expectancy = 1.0f; r1.trade_count = 100;
    RegimeResult r2{}; r2.expectancy = 1.5f; r2.trade_count = 100;
    RegimeResult r3{}; r3.expectancy = 2.0f; r3.trade_count = 100;
    RegimeResult r4{}; r4.expectancy = 3.0f; r4.trade_count = 100;
    vol_strat[1] = r1;
    vol_strat[2] = r2;
    vol_strat[3] = r3;
    vol_strat[4] = r4;

    float stability = regime::compute_stability_score(vol_strat);

    // min/max = 1.0/3.0 ≈ 0.33 → regime-dependent
    EXPECT_GE(stability, 0.2f);
    EXPECT_LE(stability, 0.5f);
}

TEST_F(StabilityScoreTest, FragileStability) {
    // Spec: "< 0.2 → fragile"
    std::map<int, RegimeResult> vol_strat;
    RegimeResult r1{}; r1.expectancy = 0.1f; r1.trade_count = 100;
    RegimeResult r2{}; r2.expectancy = 0.5f; r2.trade_count = 100;
    RegimeResult r3{}; r3.expectancy = 2.0f; r3.trade_count = 100;
    RegimeResult r4{}; r4.expectancy = 5.0f; r4.trade_count = 100;
    vol_strat[1] = r1;
    vol_strat[2] = r2;
    vol_strat[3] = r3;
    vol_strat[4] = r4;

    float stability = regime::compute_stability_score(vol_strat);

    // min/max = 0.1/5.0 = 0.02 → fragile
    EXPECT_LT(stability, 0.2f);
}

TEST_F(StabilityScoreTest, StabilityScoreFormula) {
    // Spec: "stability = min(regime_expectancy) / max(regime_expectancy)"
    std::map<int, RegimeResult> vol_strat;
    RegimeResult r1{}; r1.expectancy = 2.0f; r1.trade_count = 10;
    RegimeResult r2{}; r2.expectancy = 4.0f; r2.trade_count = 10;
    vol_strat[1] = r1;
    vol_strat[2] = r2;

    float stability = regime::compute_stability_score(vol_strat);
    EXPECT_NEAR(stability, 0.5f, 0.01f);  // 2.0 / 4.0 = 0.5
}

TEST_F(StabilityScoreTest, StabilityScoreWithNegativeExpectancy) {
    // Edge case: what if some regimes have negative expectancy?
    std::map<int, RegimeResult> vol_strat;
    RegimeResult r1{}; r1.expectancy = -1.0f; r1.trade_count = 100;
    RegimeResult r2{}; r2.expectancy = 2.0f; r2.trade_count = 100;
    vol_strat[1] = r1;
    vol_strat[2] = r2;

    float stability = regime::compute_stability_score(vol_strat);

    // With negative expectancy, stability should be <= 0 (fragile)
    EXPECT_LT(stability, 0.0f);
}

TEST_F(StabilityScoreTest, StabilityScoreWithSingleRegime) {
    // Edge case: only 1 regime — min == max → stability = 1.0
    std::map<int, RegimeResult> vol_strat;
    RegimeResult r1{}; r1.expectancy = 3.0f; r1.trade_count = 100;
    vol_strat[1] = r1;

    float stability = regime::compute_stability_score(vol_strat);
    EXPECT_NEAR(stability, 1.0f, 0.01f);
}

TEST_F(StabilityScoreTest, StabilityScoreWithZeroMaxExpectancy) {
    // Edge case: max expectancy is 0 → avoid division by zero
    std::map<int, RegimeResult> vol_strat;
    RegimeResult r1{}; r1.expectancy = 0.0f; r1.trade_count = 100;
    RegimeResult r2{}; r2.expectancy = 0.0f; r2.trade_count = 100;
    vol_strat[1] = r1;
    vol_strat[2] = r2;

    float stability = regime::compute_stability_score(vol_strat);

    // Should not crash (NaN or Inf)
    EXPECT_TRUE(std::isfinite(stability));
}

TEST_F(StabilityScoreTest, StabilityClassification) {
    // Verify the classification function
    EXPECT_EQ(regime::classify_stability(0.6f), "robust");
    EXPECT_EQ(regime::classify_stability(0.35f), "regime-dependent");
    EXPECT_EQ(regime::classify_stability(0.15f), "fragile");
}

// ===========================================================================
// 8. Q4 fragility warning
// ===========================================================================

TEST_F(RegimeStratifierTest, Q4FragilityDetection) {
    // Spec: "If expectancy concentrates in Q4, strategy is fragile"
    RegimeStratifier stratifier;

    std::map<int, RegimeResult> vol_strat;
    // Low expectancy in Q1-Q3, high in Q4
    for (int q = 1; q <= 3; ++q) {
        RegimeResult r{};
        r.expectancy = 0.1f;
        r.trade_count = 100;
        vol_strat[q] = r;
    }
    RegimeResult r4{}; r4.expectancy = 5.0f; r4.trade_count = 100;
    vol_strat[4] = r4;

    bool is_q4_fragile = stratifier.is_q4_concentrated(vol_strat);
    EXPECT_TRUE(is_q4_fragile);
}

TEST_F(RegimeStratifierTest, Q4NotConcentratedWhenBalanced) {
    RegimeStratifier stratifier;

    std::map<int, RegimeResult> vol_strat;
    for (int q = 1; q <= 4; ++q) {
        RegimeResult r{};
        r.expectancy = 2.0f;
        r.trade_count = 100;
        vol_strat[q] = r;
    }

    bool is_q4_fragile = stratifier.is_q4_concentrated(vol_strat);
    EXPECT_FALSE(is_q4_fragile);
}

// ===========================================================================
// 9. Empty / edge case inputs
// ===========================================================================

TEST_F(RegimeStratifierTest, EmptyTradesReturnsEmptyStratification) {
    RegimeStratifier stratifier;

    std::vector<float> daily_vols;
    std::vector<std::vector<TradeRecord>> daily_trades;

    auto vol_strat = stratifier.by_volatility(daily_vols, daily_trades);
    EXPECT_TRUE(vol_strat.empty());
}

TEST_F(RegimeStratifierTest, SingleDayStratification) {
    RegimeStratifier stratifier;

    std::vector<float> daily_vols = {0.5f};
    std::vector<std::vector<TradeRecord>> daily_trades(1);
    daily_trades[0].push_back(make_trade(5.0f, 10.0f));

    auto vol_strat = stratifier.by_volatility(daily_vols, daily_trades);

    // With a single day, all trades go to one quartile
    EXPECT_EQ(vol_strat.size(), 1u);
}
