// bar_comparison_test.cpp — TDD RED phase tests for Bar Type Comparison + Power Analysis
// Spec: .kit/docs/feature-analysis.md §Bar Type Comparison (§8.5), §Power Analysis (§8.7)
//
// Tests for:
//   - Jarque-Bera normality test on 1-bar returns
//   - ARCH LM heteroskedasticity test
//   - ACF of |return_1| at lags 1, 5, 10
//   - Ljung-Box autocorrelation test at lags 1, 5, 10
//   - AR R² for return_h from last 10 returns
//   - Sum of excess MI across Track A features
//   - CV of daily bar counts
//   - Holm-Bonferroni within each metric family
//   - Power analysis (minimum sample sizes, per-stratum power, detectable effect size)
//   - Validation gates
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers the implementation must provide:
#include "analysis/bar_comparison.hpp"       // BarComparisonAnalyzer, BarComparisonResult
#include "analysis/statistical_tests.hpp"    // JarqueBera, ARCH_LM, ACF, LjungBox, AR_R2
#include "analysis/power_analysis.hpp"       // PowerAnalyzer, PowerResult
#include "analysis/multiple_comparison.hpp"  // holm_bonferroni_correct
#include "features/bar_features.hpp"         // BarFeatureRow

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

constexpr float PI = 3.14159265358979323846f;

// Build normally distributed returns (deterministic approximation).
std::vector<float> make_normal_returns(int n) {
    std::vector<float> v(n);
    // Box-Muller-like deterministic: pair (i, i+1) → approximately normal
    for (int i = 0; i < n; ++i) {
        float u1 = (static_cast<float>(i % 997) + 0.5f) / 997.0f;
        float u2 = (static_cast<float>((i * 7 + 3) % 991) + 0.5f) / 991.0f;
        v[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * PI * u2);
    }
    return v;
}

// Build returns with heavy tails (non-normal).
std::vector<float> make_heavy_tail_returns(int n) {
    auto v = make_normal_returns(n);
    // Add occasional large values to create heavy tails
    for (int i = 0; i < n; i += 10) {
        v[i] *= 5.0f;
    }
    return v;
}

// Build returns with ARCH effects (volatility clustering).
std::vector<float> make_arch_returns(int n) {
    std::vector<float> v(n);
    float sigma = 1.0f;
    for (int i = 0; i < n; ++i) {
        float u1 = (static_cast<float>(i % 997) + 0.5f) / 997.0f;
        float u2 = (static_cast<float>((i * 7 + 3) % 991) + 0.5f) / 991.0f;
        float eps = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * PI * u2);
        v[i] = sigma * eps;
        // ARCH(1): σ²(t+1) = 0.1 + 0.8 * r²(t)
        sigma = std::sqrt(0.1f + 0.8f * v[i] * v[i]);
    }
    return v;
}

// Build returns with autocorrelation.
std::vector<float> make_autocorrelated_returns(int n) {
    std::vector<float> v(n, 0.0f);
    for (int i = 1; i < n; ++i) {
        float noise = std::sin(static_cast<float>(i * 137 + 42) * 0.01f);
        v[i] = 0.5f * v[i - 1] + noise;  // AR(1) with φ = 0.5
    }
    return v;
}

// Build IID returns (no autocorrelation).
// Uses a proper LCG to avoid the repeating-cycle autocorrelation in make_normal_returns.
std::vector<float> make_iid_returns(int n) {
    std::vector<float> v(n);
    // Numerical Recipes LCG: period 2^32, no short-cycle autocorrelation
    uint32_t state = 12345u;
    auto next_uniform = [&]() -> float {
        state = state * 1664525u + 1013904223u;
        return (static_cast<float>(state >> 8) + 0.5f) / 16777216.0f; // (0, 1)
    };
    for (int i = 0; i < n; ++i) {
        float u1 = next_uniform();
        float u2 = next_uniform();
        v[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * PI * u2);
    }
    return v;
}

// Build daily bar counts for CV computation.
std::vector<int> make_stable_daily_counts(int n_days, int bars_per_day = 1000) {
    std::vector<int> counts(n_days, bars_per_day);
    // Add small variation
    for (int i = 0; i < n_days; ++i) {
        counts[i] += (i % 5) * 10;
    }
    return counts;
}

std::vector<int> make_unstable_daily_counts(int n_days) {
    std::vector<int> counts(n_days);
    for (int i = 0; i < n_days; ++i) {
        counts[i] = 100 + (i % 3) * 500;  // High variation: 100, 600, 1100, ...
    }
    return counts;
}

}  // anonymous namespace

// ===========================================================================
// Jarque-Bera Normality Test
// ===========================================================================

class JarqueBeraTest : public ::testing::Test {};

TEST_F(JarqueBeraTest, NormalDataHasLowJBStatistic) {
    auto returns = make_normal_returns(5000);
    auto result = jarque_bera_test(returns);
    // Normal data → JB stat should be small
    EXPECT_LT(result.statistic, 10.0f);
}

TEST_F(JarqueBeraTest, NormalDataHasHighPValue) {
    auto returns = make_normal_returns(5000);
    auto result = jarque_bera_test(returns);
    // Cannot reject normality → p > 0.05
    EXPECT_GT(result.p_value, 0.01f);
}

TEST_F(JarqueBeraTest, HeavyTailDataHasHighJBStatistic) {
    auto returns = make_heavy_tail_returns(5000);
    auto result = jarque_bera_test(returns);
    // Heavy tails → JB stat large → reject normality
    EXPECT_GT(result.statistic, 10.0f);
}

TEST_F(JarqueBeraTest, HeavyTailDataHasLowPValue) {
    auto returns = make_heavy_tail_returns(5000);
    auto result = jarque_bera_test(returns);
    EXPECT_LT(result.p_value, 0.05f);
}

TEST_F(JarqueBeraTest, StatisticIsNonNegative) {
    auto returns = make_normal_returns(500);
    auto result = jarque_bera_test(returns);
    EXPECT_GE(result.statistic, 0.0f);
}

TEST_F(JarqueBeraTest, PValueBetweenZeroAndOne) {
    auto returns = make_normal_returns(500);
    auto result = jarque_bera_test(returns);
    EXPECT_GE(result.p_value, 0.0f);
    EXPECT_LE(result.p_value, 1.0f);
}

TEST_F(JarqueBeraTest, SmallSampleHandled) {
    auto returns = make_normal_returns(10);
    auto result = jarque_bera_test(returns);
    EXPECT_FALSE(std::isnan(result.statistic));
}

// ===========================================================================
// ARCH LM Heteroskedasticity Test
// ===========================================================================

class ARCHLMTest : public ::testing::Test {};

TEST_F(ARCHLMTest, VolatilityClusteringDetected) {
    auto returns = make_arch_returns(5000);
    auto result = arch_lm_test(returns);
    // ARCH returns → significant heteroskedasticity
    EXPECT_GT(result.statistic, 5.0f);
    EXPECT_LT(result.p_value, 0.05f);
}

TEST_F(ARCHLMTest, IIDReturnsNoHeteroskedasticity) {
    auto returns = make_iid_returns(5000);
    auto result = arch_lm_test(returns);
    // IID → no ARCH effects
    EXPECT_GT(result.p_value, 0.01f);
}

TEST_F(ARCHLMTest, StatisticIsNonNegative) {
    auto returns = make_arch_returns(1000);
    auto result = arch_lm_test(returns);
    EXPECT_GE(result.statistic, 0.0f);
}

TEST_F(ARCHLMTest, PValueInValidRange) {
    auto returns = make_arch_returns(1000);
    auto result = arch_lm_test(returns);
    EXPECT_GE(result.p_value, 0.0f);
    EXPECT_LE(result.p_value, 1.0f);
}

// ===========================================================================
// ACF of |return_1| (Volatility Clustering Measure)
// ===========================================================================

class ACFTest : public ::testing::Test {};

TEST_F(ACFTest, ComputesAtLags1_5_10) {
    // Spec: "ACF of |return_1| at lags 1, 5, 10"
    auto returns = make_arch_returns(2000);
    std::vector<float> abs_returns(returns.size());
    for (size_t i = 0; i < returns.size(); ++i) abs_returns[i] = std::abs(returns[i]);

    auto acf_values = compute_acf(abs_returns, {1, 5, 10});
    EXPECT_EQ(acf_values.size(), 3u);
}

TEST_F(ACFTest, ACFAtLag0IsOne) {
    auto returns = make_normal_returns(1000);
    auto acf_values = compute_acf(returns, {0});
    EXPECT_NEAR(acf_values[0], 1.0f, 1e-4f);
}

TEST_F(ACFTest, ACFInValidRange) {
    auto returns = make_arch_returns(2000);
    std::vector<float> abs_returns(returns.size());
    for (size_t i = 0; i < returns.size(); ++i) abs_returns[i] = std::abs(returns[i]);

    auto acf_values = compute_acf(abs_returns, {1, 5, 10});
    for (float acf : acf_values) {
        EXPECT_GE(acf, -1.0f);
        EXPECT_LE(acf, 1.0f);
    }
}

TEST_F(ACFTest, ARCHReturnsHavePositiveAbsACF) {
    // |returns| of ARCH process should show positive autocorrelation (volatility clustering).
    auto returns = make_arch_returns(5000);
    std::vector<float> abs_returns(returns.size());
    for (size_t i = 0; i < returns.size(); ++i) abs_returns[i] = std::abs(returns[i]);

    auto acf_values = compute_acf(abs_returns, {1, 5, 10});
    EXPECT_GT(acf_values[0], 0.0f) << "ACF(|r|, lag=1) should be positive for ARCH returns";
}

TEST_F(ACFTest, IIDReturnsHaveNearZeroACF) {
    auto returns = make_iid_returns(5000);
    auto acf_values = compute_acf(returns, {1, 5, 10});
    for (float acf : acf_values) {
        EXPECT_NEAR(acf, 0.0f, 0.1f);
    }
}

// ===========================================================================
// Ljung-Box Autocorrelation Test
// ===========================================================================

class LjungBoxTest : public ::testing::Test {};

TEST_F(LjungBoxTest, ComputesAtLags1_5_10) {
    // Spec: "Ljung-Box at lags 1, 5, 10"
    auto returns = make_autocorrelated_returns(2000);
    auto results = ljung_box_test(returns, {1, 5, 10});
    EXPECT_EQ(results.size(), 3u);
}

TEST_F(LjungBoxTest, AutocorrelatedReturnsRejected) {
    auto returns = make_autocorrelated_returns(5000);
    auto results = ljung_box_test(returns, {1, 5, 10});
    // At lag 1, strong AR(1) should be detected
    EXPECT_LT(results[0].p_value, 0.05f);
}

TEST_F(LjungBoxTest, IIDReturnsNotRejected) {
    auto returns = make_iid_returns(5000);
    auto results = ljung_box_test(returns, {1, 5, 10});
    // IID returns → cannot reject independence at any lag
    for (const auto& r : results) {
        EXPECT_GT(r.p_value, 0.01f);
    }
}

TEST_F(LjungBoxTest, StatisticsAreNonNegative) {
    auto returns = make_normal_returns(1000);
    auto results = ljung_box_test(returns, {1, 5, 10});
    for (const auto& r : results) {
        EXPECT_GE(r.statistic, 0.0f);
    }
}

TEST_F(LjungBoxTest, PValuesInValidRange) {
    auto returns = make_normal_returns(1000);
    auto results = ljung_box_test(returns, {1, 5, 10});
    for (const auto& r : results) {
        EXPECT_GE(r.p_value, 0.0f);
        EXPECT_LE(r.p_value, 1.0f);
    }
}

// ===========================================================================
// AR R² — Temporal Predictability
// ===========================================================================

class ARR2Test : public ::testing::Test {};

TEST_F(ARR2Test, HighR2ForPredictableReturns) {
    // Spec: "AR R² for return_h from last 10 returns"
    auto returns = make_autocorrelated_returns(5000);
    auto r2 = compute_ar_r2(returns, 10);
    EXPECT_GT(r2, 0.1f) << "AR(10) R² should be positive for autocorrelated returns";
}

TEST_F(ARR2Test, LowR2ForIIDReturns) {
    auto returns = make_iid_returns(5000);
    auto r2 = compute_ar_r2(returns, 10);
    EXPECT_LT(r2, 0.1f) << "AR(10) R² should be near 0 for IID returns";
}

TEST_F(ARR2Test, R2BetweenZeroAndOne) {
    auto returns = make_autocorrelated_returns(1000);
    auto r2 = compute_ar_r2(returns, 10);
    EXPECT_GE(r2, 0.0f);
    EXPECT_LE(r2, 1.0f);
}

TEST_F(ARR2Test, UsesLast10Returns) {
    // Verify the AR model uses 10 lags
    auto returns = make_autocorrelated_returns(1000);
    auto r2_10 = compute_ar_r2(returns, 10);
    auto r2_1 = compute_ar_r2(returns, 1);
    // AR(10) should capture at least as much variance as AR(1)
    EXPECT_GE(r2_10, r2_1 - 0.01f);
}

TEST_F(ARR2Test, InsufficientDataHandled) {
    // Fewer than 10 data points → cannot fit AR(10)
    std::vector<float> returns(5, 1.0f);
    auto r2 = compute_ar_r2(returns, 10);
    EXPECT_TRUE(std::isnan(r2) || r2 == 0.0f);
}

// ===========================================================================
// CV of Daily Bar Counts
// ===========================================================================

class BarCountCVTest : public ::testing::Test {};

TEST_F(BarCountCVTest, StableCountsLowCV) {
    // CV = std / mean. Stable counts → low CV.
    auto counts = make_stable_daily_counts(100);
    float cv = compute_bar_count_cv(counts);
    EXPECT_LT(cv, 0.1f);
}

TEST_F(BarCountCVTest, UnstableCountsHighCV) {
    auto counts = make_unstable_daily_counts(100);
    float cv = compute_bar_count_cv(counts);
    EXPECT_GT(cv, 0.3f);
}

TEST_F(BarCountCVTest, CVIsNonNegative) {
    auto counts = make_stable_daily_counts(50);
    float cv = compute_bar_count_cv(counts);
    EXPECT_GE(cv, 0.0f);
}

TEST_F(BarCountCVTest, ConstantCountsHaveZeroCV) {
    std::vector<int> counts(50, 1000);
    float cv = compute_bar_count_cv(counts);
    EXPECT_NEAR(cv, 0.0f, 1e-6f);
}

TEST_F(BarCountCVTest, SingleDayNotNaN) {
    std::vector<int> counts = {1000};
    float cv = compute_bar_count_cv(counts);
    EXPECT_FALSE(std::isnan(cv));
}

// ===========================================================================
// Bar Type Comparison — Full Analysis
// ===========================================================================

class BarComparisonFullTest : public ::testing::Test {};

TEST_F(BarComparisonFullTest, AnalyzesMultipleBarTypes) {
    // Spec: "For each bar type configuration (~10 configs)"
    BarComparisonAnalyzer analyzer;

    // Build mock data per bar type
    std::vector<BarTypeData> bar_types;
    for (const auto& name : {"tick_100", "tick_200", "vol_500", "vol_1000",
                              "dollar_1M", "time_60s", "time_300s"}) {
        BarTypeData data;
        data.name = name;
        data.returns = make_normal_returns(5000);
        data.daily_bar_counts = make_stable_daily_counts(100);
        data.excess_mi_sum = 0.5f;
        bar_types.push_back(data);
    }

    auto results = analyzer.compare_bar_types(bar_types);
    EXPECT_EQ(results.size(), 7u);
}

TEST_F(BarComparisonFullTest, EachResultContainsAllMetrics) {
    BarComparisonAnalyzer analyzer;

    BarTypeData data;
    data.name = "tick_100";
    data.returns = make_normal_returns(5000);
    data.daily_bar_counts = make_stable_daily_counts(100);
    data.excess_mi_sum = 0.5f;

    auto results = analyzer.compare_bar_types({data});
    ASSERT_EQ(results.size(), 1u);
    const auto& r = results[0];

    // All metrics should be populated
    EXPECT_FALSE(std::isnan(r.jarque_bera_stat));
    EXPECT_FALSE(std::isnan(r.jarque_bera_p));
    EXPECT_FALSE(std::isnan(r.arch_lm_stat));
    EXPECT_FALSE(std::isnan(r.arch_lm_p));
    EXPECT_EQ(r.acf_values.size(), 3u);  // lags 1, 5, 10
    EXPECT_EQ(r.ljung_box_results.size(), 3u);
    EXPECT_FALSE(std::isnan(r.ar_r2));
    EXPECT_FALSE(std::isnan(r.excess_mi_sum));
    EXPECT_FALSE(std::isnan(r.bar_count_cv));
}

TEST_F(BarComparisonFullTest, HolmBonferroniAppliedWithinMetricFamily) {
    // Spec: "Apply Holm-Bonferroni within each metric family (10 tests per metric)"
    BarComparisonAnalyzer analyzer;

    std::vector<BarTypeData> bar_types;
    for (int i = 0; i < 10; ++i) {
        BarTypeData data;
        data.name = "type_" + std::to_string(i);
        data.returns = make_normal_returns(5000);
        data.daily_bar_counts = make_stable_daily_counts(100);
        data.excess_mi_sum = static_cast<float>(i) * 0.1f;
        bar_types.push_back(data);
    }

    auto results = analyzer.compare_bar_types(bar_types);

    // Corrected p-values should be present and >= raw p-values
    for (const auto& r : results) {
        EXPECT_GE(r.jarque_bera_corrected_p, r.jarque_bera_p - 1e-6f);
        EXPECT_GE(r.arch_lm_corrected_p, r.arch_lm_p - 1e-6f);
    }
}

TEST_F(BarComparisonFullTest, ResultsContainCorrectedPValues) {
    BarComparisonAnalyzer analyzer;

    std::vector<BarTypeData> bar_types;
    for (int i = 0; i < 5; ++i) {
        BarTypeData data;
        data.name = "type_" + std::to_string(i);
        data.returns = make_normal_returns(3000);
        data.daily_bar_counts = make_stable_daily_counts(50);
        data.excess_mi_sum = 0.3f;
        bar_types.push_back(data);
    }

    auto results = analyzer.compare_bar_types(bar_types);

    for (const auto& r : results) {
        // Corrected p-values should be defined (not NaN)
        EXPECT_FALSE(std::isnan(r.jarque_bera_corrected_p));
        EXPECT_FALSE(std::isnan(r.arch_lm_corrected_p));
        // Corrected p-values capped at 1.0
        EXPECT_LE(r.jarque_bera_corrected_p, 1.0f);
        EXPECT_LE(r.arch_lm_corrected_p, 1.0f);
    }
}

// ===========================================================================
// Power Analysis
// ===========================================================================

class PowerAnalysisTest : public ::testing::Test {};

TEST_F(PowerAnalysisTest, MinSampleSizeForSpearmanR005) {
    // Spec: "Spearman r=0.05 at α=0.05, power=0.80: n ≈ 2,500 bars"
    PowerAnalyzer analyzer;
    auto n = analyzer.min_sample_size_spearman(0.05f, 0.05f, 0.80f);
    EXPECT_NEAR(n, 2500, 500) << "Min n for r=0.05, α=0.05, power=0.80 should be ~2500";
}

TEST_F(PowerAnalysisTest, LargerEffectRequiresFewerSamples) {
    PowerAnalyzer analyzer;
    auto n_small = analyzer.min_sample_size_spearman(0.05f, 0.05f, 0.80f);
    auto n_large = analyzer.min_sample_size_spearman(0.10f, 0.05f, 0.80f);
    EXPECT_GT(n_small, n_large);
}

TEST_F(PowerAnalysisTest, HigherPowerRequiresMoreSamples) {
    PowerAnalyzer analyzer;
    auto n_80 = analyzer.min_sample_size_spearman(0.05f, 0.05f, 0.80f);
    auto n_90 = analyzer.min_sample_size_spearman(0.05f, 0.05f, 0.90f);
    EXPECT_LT(n_80, n_90);
}

TEST_F(PowerAnalysisTest, DetectableEffectSize) {
    // Spec: "Per-stratum: 5,000–20,000 → detectable r ≈ 0.03–0.04"
    PowerAnalyzer analyzer;
    auto r = analyzer.detectable_effect_size(10000, 0.05f, 0.80f);
    EXPECT_GT(r, 0.01f);
    EXPECT_LT(r, 0.10f);
}

TEST_F(PowerAnalysisTest, MoreSamplesDetectsSmallEffect) {
    PowerAnalyzer analyzer;
    auto r_small_n = analyzer.detectable_effect_size(1000, 0.05f, 0.80f);
    auto r_large_n = analyzer.detectable_effect_size(50000, 0.05f, 0.80f);
    EXPECT_GT(r_small_n, r_large_n);
}

TEST_F(PowerAnalysisTest, PowerForGivenSampleSize) {
    // Compute power given n, effect size, and alpha.
    PowerAnalyzer analyzer;
    auto power = analyzer.compute_power(10000, 0.05f, 0.05f);
    EXPECT_GT(power, 0.70f);
    EXPECT_LE(power, 1.0f);
}

TEST_F(PowerAnalysisTest, PerStratumPowerReported) {
    // Spec: "Report per-stratum power alongside results"
    PowerAnalyzer analyzer;

    std::vector<StratumInfo> strata = {
        {"tick_100_h1", 10000},
        {"tick_100_h5", 8000},
        {"vol_500_h1", 15000},
        {"vol_500_h20", 12000},
    };

    auto power_results = analyzer.per_stratum_power(strata, 0.05f, 0.80f);
    EXPECT_EQ(power_results.size(), strata.size());

    for (const auto& pr : power_results) {
        EXPECT_FALSE(std::isnan(pr.detectable_r));
        EXPECT_FALSE(std::isnan(pr.power_at_r005));
        EXPECT_GT(pr.detectable_r, 0.0f);
    }
}

TEST_F(PowerAnalysisTest, PerStratumPowerWithSmallSample) {
    PowerAnalyzer analyzer;

    std::vector<StratumInfo> strata = {
        {"small_stratum", 100},  // Very small — low power
    };

    auto power_results = analyzer.per_stratum_power(strata, 0.05f, 0.80f);
    ASSERT_EQ(power_results.size(), 1u);
    // Small sample → large detectable effect size
    EXPECT_GT(power_results[0].detectable_r, 0.15f);
}

TEST_F(PowerAnalysisTest, AdequacyCheckForFullDataset) {
    // Spec: "Full dataset: 50,000–250,000 bars → adequate for small effects"
    PowerAnalyzer analyzer;
    auto r = analyzer.detectable_effect_size(50000, 0.05f, 0.80f);
    EXPECT_LT(r, 0.02f) << "50k samples should detect effects < 0.02";
}

// ===========================================================================
// Validation Gates — End-to-End
// ===========================================================================

class ValidationGateTest : public ::testing::Test {};

TEST_F(ValidationGateTest, WarmupExclusionEnforced) {
    // Spec: "Assert: MI analysis runs on is_warmup == false bars only"
    // This is tested in feature_mi_test.cpp — here we verify the gate check.
    BarComparisonAnalyzer analyzer;

    BarTypeData data;
    data.name = "tick_100";
    data.returns = make_normal_returns(5000);
    data.daily_bar_counts = make_stable_daily_counts(100);
    data.excess_mi_sum = 0.5f;

    auto results = analyzer.compare_bar_types({data});
    // All results should have warmup_excluded = true
    for (const auto& r : results) {
        EXPECT_TRUE(r.warmup_excluded);
    }
}

TEST_F(ValidationGateTest, PerStratumPowerReportedForAllStratifications) {
    // Spec: "Assert: Per-stratum power analysis reported for all stratifications"
    PowerAnalyzer analyzer;

    // 7 bar types × 4 horizons = 28 strata
    std::vector<StratumInfo> strata;
    for (const auto& bt : {"tick_100", "tick_200", "vol_500", "vol_1000",
                             "dollar_1M", "time_60s", "time_300s"}) {
        for (int h : {1, 5, 20, 100}) {
            StratumInfo s;
            s.name = std::string(bt) + "_h" + std::to_string(h);
            s.sample_size = 5000 + h * 100;
            strata.push_back(s);
        }
    }

    auto power_results = analyzer.per_stratum_power(strata, 0.05f, 0.80f);
    EXPECT_EQ(power_results.size(), 28u);
}

TEST_F(ValidationGateTest, AllTablesIncludeRawAndCorrectedPValues) {
    // Spec: "Assert: All tables include raw p-value, corrected p-value, and correction flag"
    BarComparisonAnalyzer analyzer;

    std::vector<BarTypeData> bar_types;
    for (int i = 0; i < 5; ++i) {
        BarTypeData data;
        data.name = "type_" + std::to_string(i);
        data.returns = make_normal_returns(3000);
        data.daily_bar_counts = make_stable_daily_counts(50);
        data.excess_mi_sum = 0.3f;
        bar_types.push_back(data);
    }

    auto results = analyzer.compare_bar_types(bar_types);

    for (const auto& r : results) {
        // Raw p-values
        EXPECT_FALSE(std::isnan(r.jarque_bera_p));
        EXPECT_FALSE(std::isnan(r.arch_lm_p));

        // Corrected p-values
        EXPECT_FALSE(std::isnan(r.jarque_bera_corrected_p));
        EXPECT_FALSE(std::isnan(r.arch_lm_corrected_p));

        // Correction flags
        // (survives_correction is a bool, always defined)
        (void)r.jarque_bera_survives;
        (void)r.arch_lm_survives;
    }
}

// ===========================================================================
// Aggregate MI Sum
// ===========================================================================

class AggregateMISumTest : public ::testing::Test {};

TEST_F(AggregateMISumTest, SumOfExcessMIAcrossFeatures) {
    // Spec: "Sum of excess MI across Track A"
    std::vector<float> excess_mis = {0.1f, 0.05f, 0.0f, -0.01f, 0.2f};
    float total = compute_aggregate_mi(excess_mis);
    // Sum of positive excess MI = 0.1 + 0.05 + 0.2 = 0.35
    // Or could be sum of all = 0.34
    EXPECT_GT(total, 0.0f);
}

TEST_F(AggregateMISumTest, AllZeroExcessMI) {
    std::vector<float> excess_mis = {0.0f, 0.0f, 0.0f};
    float total = compute_aggregate_mi(excess_mis);
    EXPECT_NEAR(total, 0.0f, 1e-6f);
}

TEST_F(AggregateMISumTest, HigherAggregateForMorePredictiveBarType) {
    std::vector<float> good_mis = {0.2f, 0.15f, 0.1f, 0.08f, 0.05f};
    std::vector<float> bad_mis = {0.01f, 0.005f, 0.0f, 0.0f, 0.0f};

    float good_total = compute_aggregate_mi(good_mis);
    float bad_total = compute_aggregate_mi(bad_mis);
    EXPECT_GT(good_total, bad_total);
}

// ===========================================================================
// Edge Cases and Error Handling
// ===========================================================================

class BarComparisonEdgeCaseTest : public ::testing::Test {};

TEST_F(BarComparisonEdgeCaseTest, EmptyReturnsHandled) {
    std::vector<float> empty;
    auto jb = jarque_bera_test(empty);
    EXPECT_TRUE(std::isnan(jb.statistic) || jb.statistic == 0.0f);
}

TEST_F(BarComparisonEdgeCaseTest, SingleReturnHandled) {
    std::vector<float> single = {1.0f};
    auto jb = jarque_bera_test(single);
    EXPECT_FALSE(std::isinf(jb.statistic));
}

TEST_F(BarComparisonEdgeCaseTest, ConstantReturnsHandled) {
    std::vector<float> constant(1000, 5.0f);
    auto jb = jarque_bera_test(constant);
    // Constant returns have zero variance → handle gracefully
    EXPECT_FALSE(std::isinf(jb.statistic));
}

TEST_F(BarComparisonEdgeCaseTest, NaNReturnsFiltered) {
    auto returns = make_normal_returns(1000);
    returns[100] = std::numeric_limits<float>::quiet_NaN();
    returns[200] = std::numeric_limits<float>::quiet_NaN();

    auto jb = jarque_bera_test(returns);
    // Should handle NaN gracefully (filter or error, not crash)
    EXPECT_FALSE(std::isnan(jb.p_value));
}
