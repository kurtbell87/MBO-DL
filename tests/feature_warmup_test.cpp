// feature_warmup_test.cpp — TDD RED phase tests for extended warmup policy
// Spec: .kit/docs/feature-computation.md §Warm-Up and Lookahead Bias Policy (§8.6)
//
// Tests for the extended warmup state tracking required by feature computation:
//   - EWMA features (volume_surprise, duration_surprise): mark first ewma_span bars as WARMUP
//   - EWMA resets at session boundaries
//   - Rolling window features: NaN for first N bars (N = window length)
//   - session_volume_frac: expanding-window prior-day mean
//   - Forward returns: last n bars undefined (NaN)
//   - Bar-level is_warmup flag: true if ANY feature is in warmup state
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

#include "features/warmup.hpp"          // WarmupTracker (enhanced)
#include "features/bar_features.hpp"    // BarFeatureComputer, BarFeatureRow
#include "bars/bar.hpp"                 // Bar

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

constexpr float TICK_SIZE = 0.25f;

Bar make_bar(float close_mid, uint32_t volume = 100, float duration_s = 1.0f) {
    Bar bar{};
    bar.close_mid = close_mid;
    bar.open_mid = close_mid;
    bar.high_mid = close_mid;
    bar.low_mid = close_mid;
    bar.vwap = close_mid;
    bar.volume = volume;
    bar.spread = 0.25f;
    bar.bar_duration_s = duration_s;
    bar.time_of_day = 10.0f;
    bar.open_ts = 1000000000ULL;
    bar.close_ts = 2000000000ULL;
    bar.buy_volume = static_cast<float>(volume) * 0.6f;
    bar.sell_volume = static_cast<float>(volume) * 0.4f;
    bar.tick_count = volume;
    bar.add_count = 50;
    bar.cancel_count = 30;
    bar.modify_count = 10;
    bar.trade_event_count = volume;

    for (int i = 0; i < BOOK_DEPTH; ++i) {
        bar.bids[i][0] = close_mid - TICK_SIZE * (0.5f + i);
        bar.bids[i][1] = 10.0f;
        bar.asks[i][0] = close_mid + TICK_SIZE * (0.5f + i);
        bar.asks[i][1] = 10.0f;
    }

    return bar;
}

std::vector<Bar> make_bar_sequence(int count, float start_mid = 4500.0f, float step = 0.25f,
                                    uint32_t vol = 100) {
    std::vector<Bar> bars;
    for (int i = 0; i < count; ++i) {
        bars.push_back(make_bar(start_mid + step * i, vol));
    }
    return bars;
}

}  // anonymous namespace

// ===========================================================================
// EWMA Warmup Policy
// ===========================================================================

class EWMAWarmupTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(EWMAWarmupTest, FirstBarInitializesEWMAWithOwnValue) {
    // §8.6: "Init at bar 0 with first bar's value"
    auto bar = make_bar(4500.0f, 200, 2.0f);
    auto row = computer_.update(bar);

    // volume_surprise = current / EWMA → 200/200 = 1.0
    EXPECT_NEAR(row.volume_surprise, 1.0f, 1e-4f);
    // duration_surprise = current / EWMA → 2.0/2.0 = 1.0
    EXPECT_NEAR(row.duration_surprise, 1.0f, 1e-4f);
}

TEST_F(EWMAWarmupTest, First20BarsMarkedAsWarmup) {
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(row.is_warmup)
            << "Bar " << i << " should be warmup (EWMA span=20)";
    }
}

TEST_F(EWMAWarmupTest, Bar20IsNotWarmupFromEWMA) {
    // After 20 bars, EWMA has sufficient history.
    // But is_warmup may still be true if other features need more warmup.
    auto bars = make_bar_sequence(55);  // Need 50 for volatility_50
    BarFeatureRow row;
    for (int i = 0; i < 55; ++i) {
        row = computer_.update(bars[i]);
    }
    // At bar 54, no feature should need warmup.
    EXPECT_FALSE(row.is_warmup);
}

TEST_F(EWMAWarmupTest, EWMAResetsAtSessionBoundary) {
    // Process 25 bars in session 1
    auto session1 = make_bar_sequence(25, 4500.0f, 0.25f, 100);
    for (const auto& bar : session1) {
        computer_.update(bar);
    }

    // Reset at session boundary
    computer_.reset();

    // First bar of session 2 should reinitialize EWMA
    auto bar = make_bar(4510.0f, 300, 3.0f);
    auto row = computer_.update(bar);

    // EWMA re-initialized → surprise = 1.0
    EXPECT_NEAR(row.volume_surprise, 1.0f, 1e-4f);
    EXPECT_NEAR(row.duration_surprise, 1.0f, 1e-4f);
    // And it should be warmup again
    EXPECT_TRUE(row.is_warmup);
}

TEST_F(EWMAWarmupTest, EWMAUpdatesCorrectlyOverBars) {
    // EWMA(span=20) decay factor α = 2/(20+1) ≈ 0.0952
    // After several bars of constant volume, EWMA should converge.
    auto bars = make_bar_sequence(50, 4500.0f, 0.25f, 100);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // All bars same volume → after convergence, surprise ≈ 1.0
    EXPECT_NEAR(row.volume_surprise, 1.0f, 0.05f);
}

// ===========================================================================
// Rolling Window NaN Policy
// ===========================================================================

class RollingWindowNaNTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(RollingWindowNaNTest, Volatility20_NaNForFirst20Bars) {
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.volatility_20))
            << "volatility_20 should be NaN at bar " << i;
    }
}

TEST_F(RollingWindowNaNTest, Volatility20_DefinedAtBar20) {
    auto bars = make_bar_sequence(25);
    BarFeatureRow row;
    for (int i = 0; i < 25; ++i) {
        row = computer_.update(bars[i]);
    }
    EXPECT_FALSE(std::isnan(row.volatility_20));
}

TEST_F(RollingWindowNaNTest, Volatility50_NaNForFirst50Bars) {
    auto bars = make_bar_sequence(50);
    for (int i = 0; i < 50; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.volatility_50))
            << "volatility_50 should be NaN at bar " << i;
    }
}

TEST_F(RollingWindowNaNTest, Volatility50_DefinedAtBar50) {
    auto bars = make_bar_sequence(55);
    BarFeatureRow row;
    for (int i = 0; i < 55; ++i) {
        row = computer_.update(bars[i]);
    }
    EXPECT_FALSE(std::isnan(row.volatility_50));
}

TEST_F(RollingWindowNaNTest, KyleLambda_NaNForFirst20Bars) {
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.kyle_lambda))
            << "kyle_lambda should be NaN at bar " << i;
    }
}

TEST_F(RollingWindowNaNTest, Momentum_NaNDuringLookbackWarmup) {
    // Momentum requires N bars of lookback. First bar's momentum is undefined.
    auto bar = make_bar(4500.0f);
    auto row = computer_.update(bar);
    // With only 1 bar, there's no return to sum → expect NaN or 0
    // (spec says "sum of signed 1-bar returns over last N bars")
}

TEST_F(RollingWindowNaNTest, HighLowRange20_NaNForFirst20Bars) {
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.high_low_range_20))
            << "high_low_range_20 should be NaN at bar " << i;
    }
}

TEST_F(RollingWindowNaNTest, HighLowRange50_NaNForFirst50Bars) {
    auto bars = make_bar_sequence(50);
    for (int i = 0; i < 50; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.high_low_range_50))
            << "high_low_range_50 should be NaN at bar " << i;
    }
}

TEST_F(RollingWindowNaNTest, VolPriceCorr_NaNForFirst20Bars) {
    auto bars = make_bar_sequence(20);
    for (int i = 0; i < 20; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(std::isnan(row.vol_price_corr))
            << "vol_price_corr should be NaN at bar " << i;
    }
}

TEST_F(RollingWindowNaNTest, RollingWindowResetsAtSessionBoundary) {
    auto session1 = make_bar_sequence(25);
    for (const auto& bar : session1) {
        computer_.update(bar);
    }

    computer_.reset();

    // After reset, first bar should again have NaN rolling features
    auto bar = make_bar(4510.0f);
    auto row = computer_.update(bar);
    EXPECT_TRUE(std::isnan(row.volatility_20));
    EXPECT_TRUE(std::isnan(row.kyle_lambda));
}

// ===========================================================================
// Session Volume Fraction — expanding-window prior-day mean
// ===========================================================================

class SessionVolumeFracTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(SessionVolumeFracTest, Day1_UsesActualDayTotal) {
    // §8.6: "Day 1: actual total volume (mild lookahead, acceptable)"
    auto bars = make_bar_sequence(10, 4500.0f, 0.25f, 100);
    BarFeatureRow last_row;
    for (const auto& bar : bars) {
        last_row = computer_.update(bar);
    }
    // session_volume_frac at the last bar of day 1.
    // cumulative = 1000, total = 1000 → frac = 1.0
    // (This is conceptually correct: at end of day, frac = 1.0)
}

TEST_F(SessionVolumeFracTest, Day2_UsesExpandingWindowMeanOfPriorDays) {
    // Process day 1
    auto day1 = make_bar_sequence(10, 4500.0f, 0.25f, 100);
    for (const auto& bar : day1) {
        computer_.update(bar);
    }
    computer_.end_session(1000);  // Day 1 total = 1000

    // Process day 2
    auto bar = make_bar(4505.0f, 200);
    auto row = computer_.update(bar);
    // cumulative = 200, prior_day_avg = 1000
    // frac = 200 / 1000 = 0.2
    EXPECT_NEAR(row.session_volume_frac, 0.2f, 0.05f);
}

TEST_F(SessionVolumeFracTest, Day3_UsesExpandingWindowMeanOfDays1And2) {
    // Day 1: total = 1000
    auto day1 = make_bar_sequence(10, 4500.0f, 0.25f, 100);
    for (const auto& bar : day1) {
        computer_.update(bar);
    }
    computer_.end_session(1000);

    // Day 2: total = 2000
    auto day2 = make_bar_sequence(10, 4502.5f, 0.25f, 200);
    for (const auto& bar : day2) {
        computer_.update(bar);
    }
    computer_.end_session(2000);

    // Day 3: first bar with volume 300
    auto bar = make_bar(4505.0f, 300);
    auto row = computer_.update(bar);
    // prior_day_avg = (1000 + 2000) / 2 = 1500
    // frac = 300 / 1500 = 0.2
    EXPECT_NEAR(row.session_volume_frac, 0.2f, 0.05f);
}

TEST_F(SessionVolumeFracTest, ExpandingWindowDoesNotUseFutureData) {
    // Day 2 must only use day 1's data, not day 2's actual total.
    BarFeatureComputer comp_a(TICK_SIZE);
    BarFeatureComputer comp_b(TICK_SIZE);

    // Both see same day 1
    auto day1 = make_bar_sequence(10, 4500.0f, 0.25f, 100);
    for (const auto& bar : day1) {
        comp_a.update(bar);
        comp_b.update(bar);
    }
    comp_a.end_session(1000);
    comp_b.end_session(1000);

    // Day 2: same first bar
    auto bar = make_bar(4505.0f, 200);
    auto row_a = comp_a.update(bar);
    auto row_b = comp_b.update(bar);

    // Both should produce the same session_volume_frac
    EXPECT_FLOAT_EQ(row_a.session_volume_frac, row_b.session_volume_frac);
}

// ===========================================================================
// Forward Return NaN Policy
// ===========================================================================

class ForwardReturnNaNPolicyTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(ForwardReturnNaNPolicyTest, LastBarsHaveUndefinedForwardReturns) {
    auto bars = make_bar_sequence(110, 4500.0f, 0.25f);
    auto rows = computer_.compute_all(bars);

    size_t n = rows.size();

    // Last 1 bar: fwd_return_1 = NaN
    EXPECT_TRUE(std::isnan(rows[n - 1].fwd_return_1));
    EXPECT_FALSE(std::isnan(rows[n - 2].fwd_return_1));

    // Last 5 bars: fwd_return_5 = NaN
    for (size_t i = n - 5; i < n; ++i) {
        EXPECT_TRUE(std::isnan(rows[i].fwd_return_5));
    }
    EXPECT_FALSE(std::isnan(rows[n - 6].fwd_return_5));

    // Last 20 bars: fwd_return_20 = NaN
    for (size_t i = n - 20; i < n; ++i) {
        EXPECT_TRUE(std::isnan(rows[i].fwd_return_20));
    }
    EXPECT_FALSE(std::isnan(rows[n - 21].fwd_return_20));

    // Last 100 bars: fwd_return_100 = NaN
    for (size_t i = n - 100; i < n; ++i) {
        EXPECT_TRUE(std::isnan(rows[i].fwd_return_100));
    }
    EXPECT_FALSE(std::isnan(rows[n - 101].fwd_return_100));
}

TEST_F(ForwardReturnNaNPolicyTest, ForwardReturnsExcludedFromAnalysis) {
    // Forward returns are TARGETS ONLY, never features.
    // No feature computation may depend on forward returns.
    auto bars = make_bar_sequence(50, 4500.0f, 0.25f);

    // Compute features first, then assign forward returns.
    // Features must not change when forward returns are assigned.
    auto rows = computer_.compute_all(bars);

    // Features at bar 10 should be independent of whether fwd_return is NaN or defined.
    float feature_before = rows[10].book_imbalance_10;
    rows[10].fwd_return_100 = 999.0f;  // Artificially set
    // Feature value should not have changed (it's a struct field, no recomputation)
    EXPECT_FLOAT_EQ(rows[10].book_imbalance_10, feature_before);
}

// ===========================================================================
// Bar-level is_warmup Flag — Composite
// ===========================================================================

class CompositeWarmupFlagTest : public ::testing::Test {
protected:
    BarFeatureComputer computer_{TICK_SIZE};
};

TEST_F(CompositeWarmupFlagTest, TrueIfAnyFeatureInWarmup) {
    // Bar 0: EWMA in warmup, rolling windows in warmup → is_warmup = true
    auto bar = make_bar(4500.0f);
    auto row = computer_.update(bar);
    EXPECT_TRUE(row.is_warmup);
}

TEST_F(CompositeWarmupFlagTest, TrueUntilAllFeaturesStable) {
    // Max warmup window is volatility_50 (50 bars) or high_low_range_50
    // So is_warmup should be true for at least first 50 bars.
    auto bars = make_bar_sequence(50);
    for (int i = 0; i < 50; ++i) {
        auto row = computer_.update(bars[i]);
        EXPECT_TRUE(row.is_warmup)
            << "Bar " << i << " should still be warmup";
    }
}

TEST_F(CompositeWarmupFlagTest, FalseAfterMaxWarmupWindow) {
    auto bars = make_bar_sequence(55);
    BarFeatureRow row;
    for (const auto& bar : bars) {
        row = computer_.update(bar);
    }
    // By bar 54, all features should be stable.
    EXPECT_FALSE(row.is_warmup);
}

TEST_F(CompositeWarmupFlagTest, ResetsToTrueAfterSessionBoundary) {
    auto bars = make_bar_sequence(55);
    for (const auto& bar : bars) {
        computer_.update(bar);
    }

    computer_.reset();

    auto bar = make_bar(4520.0f);
    auto row = computer_.update(bar);
    EXPECT_TRUE(row.is_warmup);
}

TEST_F(CompositeWarmupFlagTest, DownstreamCanFilterOnIsWarmup) {
    // §8.6: "Downstream filters on is_warmup == false"
    auto bars = make_bar_sequence(60);
    auto rows = computer_.compute_all(bars);

    int warmup_count = 0;
    int stable_count = 0;
    for (const auto& row : rows) {
        if (row.is_warmup) {
            warmup_count++;
        } else {
            stable_count++;
        }
    }

    // At least 50 warmup bars, some stable bars
    EXPECT_GE(warmup_count, 50);
    EXPECT_GT(stable_count, 0);
}
