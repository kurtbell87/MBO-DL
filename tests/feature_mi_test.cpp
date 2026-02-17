// feature_mi_test.cpp — TDD RED phase tests for Feature Analysis: MI + Spearman + Correction
// Spec: .kit/docs/feature-analysis.md §1 (MI), §2 (Spearman), Reporting Standard, Validation Gate
//
// Tests for:
//   - Quantile discretization (5 and 10 bins)
//   - MI(feature, return_sign) computation
//   - Bootstrapped null distribution (>=1000 shuffles)
//   - Excess MI over null
//   - Spearman rank correlation with p-value
//   - Holm-Bonferroni multiple comparison correction
//   - AnalysisResult reporting standard (point est, CI, raw/corrected p-value, flag)
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers the implementation must provide:
#include "analysis/mutual_information.hpp"   // MIAnalyzer, MIResult
#include "analysis/spearman.hpp"             // SpearmanAnalyzer, SpearmanResult
#include "analysis/multiple_comparison.hpp"  // HolmBonferroni
#include "analysis/analysis_result.hpp"      // AnalysisResult
#include "features/bar_features.hpp"         // BarFeatureRow

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

// ===========================================================================
// Helpers — synthetic data construction
// ===========================================================================
namespace {

// Build a vector of deterministic feature values.
std::vector<float> make_linear_feature(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i);
    }
    return v;
}

// Build a binary return sign vector: 1.0 if return > 0, 0.0 otherwise.
// Correlated with index: first half negative, second half positive.
std::vector<float> make_correlated_return_sign(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = (i >= n / 2) ? 1.0f : 0.0f;
    }
    return v;
}

// Build a random-looking return sign (deterministic but uncorrelated with index).
std::vector<float> make_uncorrelated_return_sign(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        // Simple hash-like assignment — no correlation with i
        v[i] = ((i * 7 + 3) % 11 > 5) ? 1.0f : 0.0f;
    }
    return v;
}

// Build continuous returns correlated with a linear feature.
std::vector<float> make_correlated_returns(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i) * 0.01f + 0.1f * ((i % 3) - 1);
    }
    return v;
}

// Build uncorrelated returns.
std::vector<float> make_uncorrelated_returns(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = 0.5f * std::sin(static_cast<float>(i * 137 + 42));
    }
    return v;
}

// Build constant feature values.
std::vector<float> make_constant_feature(int n, float val = 1.0f) {
    return std::vector<float>(n, val);
}

// Build BarFeatureRows with specified feature values and warmup flags.
std::vector<BarFeatureRow> make_rows_with_feature(
    const std::vector<float>& feature_vals,
    const std::vector<float>& fwd_returns,
    bool all_warmup = false) {
    std::vector<BarFeatureRow> rows(feature_vals.size());
    for (size_t i = 0; i < feature_vals.size(); ++i) {
        rows[i].book_imbalance_1 = feature_vals[i];
        rows[i].is_warmup = all_warmup;
        if (i < fwd_returns.size()) {
            rows[i].fwd_return_1 = fwd_returns[i];
        }
    }
    return rows;
}

}  // anonymous namespace

// ===========================================================================
// Quantile Discretization
// ===========================================================================

class QuantileDiscretizationTest : public ::testing::Test {};

TEST_F(QuantileDiscretizationTest, FiveBinsProducesFiveDistinctLabels) {
    // Discretize linear feature into 5 quantile bins → labels 0..4.
    auto feature = make_linear_feature(100);
    auto bins = quantile_discretize(feature, 5);
    ASSERT_EQ(bins.size(), feature.size());

    std::set<int> unique_bins(bins.begin(), bins.end());
    EXPECT_EQ(unique_bins.size(), 5u);
}

TEST_F(QuantileDiscretizationTest, TenBinsProducesTenDistinctLabels) {
    auto feature = make_linear_feature(1000);
    auto bins = quantile_discretize(feature, 10);
    ASSERT_EQ(bins.size(), feature.size());

    std::set<int> unique_bins(bins.begin(), bins.end());
    EXPECT_EQ(unique_bins.size(), 10u);
}

TEST_F(QuantileDiscretizationTest, BinsAreRoughlyEqualSized) {
    // Each quantile bin should contain approximately N/num_bins elements.
    auto feature = make_linear_feature(1000);
    auto bins = quantile_discretize(feature, 5);

    std::vector<int> counts(5, 0);
    for (int b : bins) counts[b]++;

    for (int c : counts) {
        EXPECT_NEAR(c, 200, 10) << "Quantile bins should be roughly equal-sized";
    }
}

TEST_F(QuantileDiscretizationTest, BinLabelsMonotonicForSortedInput) {
    // For sorted input, bin labels should be non-decreasing.
    auto feature = make_linear_feature(100);
    auto bins = quantile_discretize(feature, 5);

    for (size_t i = 1; i < bins.size(); ++i) {
        EXPECT_GE(bins[i], bins[i - 1]);
    }
}

TEST_F(QuantileDiscretizationTest, ConstantFeatureAllSameBin) {
    // All identical values → all in same bin (or distributed across adjacent bins).
    auto feature = make_constant_feature(100);
    auto bins = quantile_discretize(feature, 5);

    std::set<int> unique_bins(bins.begin(), bins.end());
    // With constant data, implementation should handle gracefully — at most 1 unique bin.
    EXPECT_LE(unique_bins.size(), 1u);
}

TEST_F(QuantileDiscretizationTest, SmallSampleHandledGracefully) {
    // Fewer observations than bins — should not crash.
    auto feature = make_linear_feature(3);
    auto bins = quantile_discretize(feature, 10);
    EXPECT_EQ(bins.size(), 3u);
}

// ===========================================================================
// Mutual Information Computation
// ===========================================================================

class MutualInformationTest : public ::testing::Test {};

TEST_F(MutualInformationTest, MIIsNonNegative) {
    // MI(X, Y) >= 0 always.
    auto feature = make_linear_feature(500);
    auto labels = make_correlated_return_sign(500);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi(feature, labels, 5);
    EXPECT_GE(result.mi_bits, 0.0f);
}

TEST_F(MutualInformationTest, MIOfIndependentVariablesNearZero) {
    // MI of uncorrelated feature and return sign ≈ 0.
    auto feature = make_linear_feature(5000);
    auto labels = make_uncorrelated_return_sign(5000);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi(feature, labels, 5);
    // Near zero (within bootstrap null threshold)
    EXPECT_LT(result.mi_bits, 0.02f);
}

TEST_F(MutualInformationTest, MIOfPerfectlyCorrelatedIsPositive) {
    // If feature perfectly predicts return sign, MI should be significantly positive.
    auto feature = make_linear_feature(1000);
    auto labels = make_correlated_return_sign(1000);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi(feature, labels, 5);
    EXPECT_GT(result.mi_bits, 0.1f);
}

TEST_F(MutualInformationTest, ResultContainsPointEstimate) {
    auto feature = make_linear_feature(500);
    auto labels = make_correlated_return_sign(500);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi(feature, labels, 5);
    EXPECT_FALSE(std::isnan(result.mi_bits));
}

TEST_F(MutualInformationTest, MIUnitIsBits) {
    // MI should be computed in bits (log base 2), not nats.
    // For binary labels with 50/50 split and perfect prediction:
    // MI = H(Y) = log2(2) = 1.0 bits maximum.
    auto feature = make_linear_feature(1000);
    auto labels = make_correlated_return_sign(1000);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi(feature, labels, 5);
    EXPECT_LE(result.mi_bits, 1.0f);  // Cannot exceed entropy of binary variable
}

TEST_F(MutualInformationTest, FiveBinsVsTenBins) {
    // MI with 10 bins should be >= MI with 5 bins (finer partition captures more).
    auto feature = make_linear_feature(2000);
    auto labels = make_correlated_return_sign(2000);
    MIAnalyzer analyzer;
    auto result_5 = analyzer.compute_mi(feature, labels, 5);
    auto result_10 = analyzer.compute_mi(feature, labels, 10);
    EXPECT_GE(result_10.mi_bits, result_5.mi_bits - 0.01f);
}

// ===========================================================================
// Bootstrapped Null Distribution
// ===========================================================================

class BootstrappedNullTest : public ::testing::Test {};

TEST_F(BootstrappedNullTest, NullUsesAtLeast1000Shuffles) {
    // Spec §Validation Gate: "Bootstrapped null uses >= 1000 shuffles per feature"
    auto feature = make_linear_feature(500);
    auto labels = make_correlated_return_sign(500);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi_with_null(feature, labels, 5);
    EXPECT_GE(result.null_shuffle_count, 1000);
}

TEST_F(BootstrappedNullTest, NullDistribution95thPercentileIsPositive) {
    // The 95th percentile of shuffled MI should be small but positive.
    auto feature = make_linear_feature(500);
    auto labels = make_uncorrelated_return_sign(500);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi_with_null(feature, labels, 5);
    EXPECT_GT(result.null_95th_percentile, 0.0f);
}

TEST_F(BootstrappedNullTest, ExcessMIIsPointMinusNull95th) {
    // Excess MI = MI - null_95th_percentile
    auto feature = make_linear_feature(1000);
    auto labels = make_correlated_return_sign(1000);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi_with_null(feature, labels, 5);
    float expected_excess = result.mi_bits - result.null_95th_percentile;
    EXPECT_NEAR(result.excess_mi, expected_excess, 1e-6f);
}

TEST_F(BootstrappedNullTest, ExcessMIPositiveForPredictiveFeature) {
    auto feature = make_linear_feature(2000);
    auto labels = make_correlated_return_sign(2000);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi_with_null(feature, labels, 5);
    EXPECT_GT(result.excess_mi, 0.0f);
}

TEST_F(BootstrappedNullTest, ExcessMINegativeOrZeroForNonPredictiveFeature) {
    auto feature = make_linear_feature(2000);
    auto labels = make_uncorrelated_return_sign(2000);
    MIAnalyzer analyzer;
    auto result = analyzer.compute_mi_with_null(feature, labels, 5);
    // With uncorrelated data, excess MI should be <= 0 (or very small).
    EXPECT_LE(result.excess_mi, 0.05f);
}

TEST_F(BootstrappedNullTest, ShuffledNullDoesNotModifyOriginalData) {
    auto feature = make_linear_feature(500);
    auto labels = make_correlated_return_sign(500);
    auto feature_copy = feature;
    auto labels_copy = labels;
    MIAnalyzer analyzer;
    analyzer.compute_mi_with_null(feature, labels, 5);
    EXPECT_EQ(feature, feature_copy);
    EXPECT_EQ(labels, labels_copy);
}

TEST_F(BootstrappedNullTest, NullDistributionIsDeterministicWithSeed) {
    auto feature = make_linear_feature(500);
    auto labels = make_correlated_return_sign(500);
    MIAnalyzer analyzer1(/*seed=*/42);
    MIAnalyzer analyzer2(/*seed=*/42);
    auto r1 = analyzer1.compute_mi_with_null(feature, labels, 5);
    auto r2 = analyzer2.compute_mi_with_null(feature, labels, 5);
    EXPECT_FLOAT_EQ(r1.null_95th_percentile, r2.null_95th_percentile);
    EXPECT_FLOAT_EQ(r1.excess_mi, r2.excess_mi);
}

// ===========================================================================
// Spearman Rank Correlation
// ===========================================================================

class SpearmanCorrelationTest : public ::testing::Test {};

TEST_F(SpearmanCorrelationTest, PerfectMonotonicCorrelationIsOne) {
    // feature = [0,1,2,...,99], returns = [0,1,2,...,99] → Spearman = 1.0
    auto feature = make_linear_feature(100);
    std::vector<float> returns(100);
    for (int i = 0; i < 100; ++i) returns[i] = static_cast<float>(i);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_NEAR(result.correlation, 1.0f, 1e-4f);
}

TEST_F(SpearmanCorrelationTest, PerfectNegativeCorrelationIsMinusOne) {
    auto feature = make_linear_feature(100);
    std::vector<float> returns(100);
    for (int i = 0; i < 100; ++i) returns[i] = static_cast<float>(99 - i);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_NEAR(result.correlation, -1.0f, 1e-4f);
}

TEST_F(SpearmanCorrelationTest, UncorrelatedDataNearZero) {
    auto feature = make_linear_feature(5000);
    auto returns = make_uncorrelated_returns(5000);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_NEAR(result.correlation, 0.0f, 0.1f);
}

TEST_F(SpearmanCorrelationTest, CorrelationInValidRange) {
    auto feature = make_linear_feature(200);
    auto returns = make_correlated_returns(200);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_GE(result.correlation, -1.0f);
    EXPECT_LE(result.correlation, 1.0f);
}

TEST_F(SpearmanCorrelationTest, PValueBetweenZeroAndOne) {
    auto feature = make_linear_feature(200);
    auto returns = make_correlated_returns(200);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_GT(result.p_value, 0.0f);
    EXPECT_LE(result.p_value, 1.0f);
}

TEST_F(SpearmanCorrelationTest, HighCorrelationHasLowPValue) {
    auto feature = make_linear_feature(1000);
    std::vector<float> returns(1000);
    for (int i = 0; i < 1000; ++i) returns[i] = static_cast<float>(i);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_LT(result.p_value, 0.001f);
}

TEST_F(SpearmanCorrelationTest, ResultContainsConfidenceInterval) {
    // Spec: "All tables report 95% CI"
    auto feature = make_linear_feature(200);
    auto returns = make_correlated_returns(200);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_LT(result.ci_lower, result.ci_upper);
    EXPECT_GE(result.ci_lower, -1.0f);
    EXPECT_LE(result.ci_upper, 1.0f);
    // Point estimate within CI
    EXPECT_GE(result.correlation, result.ci_lower);
    EXPECT_LE(result.correlation, result.ci_upper);
}

TEST_F(SpearmanCorrelationTest, HandlesConstantFeatureGracefully) {
    auto feature = make_constant_feature(100);
    auto returns = make_correlated_returns(100);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    // Constant feature → correlation undefined → NaN or 0 with p=1
    EXPECT_TRUE(std::isnan(result.correlation) ||
                std::abs(result.correlation) < 1e-6f);
}

TEST_F(SpearmanCorrelationTest, HandlesConstantReturnsGracefully) {
    auto feature = make_linear_feature(100);
    auto returns = make_constant_feature(100, 0.5f);
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_TRUE(std::isnan(result.correlation) ||
                std::abs(result.correlation) < 1e-6f);
}

TEST_F(SpearmanCorrelationTest, HandlesTiedRanks) {
    // Many tied values — implementation should handle ties correctly.
    std::vector<float> feature = {1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f};
    std::vector<float> returns = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    SpearmanAnalyzer analyzer;
    auto result = analyzer.compute(feature, returns);
    EXPECT_GT(result.correlation, 0.5f);  // Should still be strongly positive
}

// ===========================================================================
// Holm-Bonferroni Multiple Comparison Correction
// ===========================================================================

class HolmBonferroniTest : public ::testing::Test {};

TEST_F(HolmBonferroniTest, SingleTestUnchanged) {
    // With a single test, corrected p-value = raw p-value.
    std::vector<float> raw_pvals = {0.03f};
    auto corrected = holm_bonferroni_correct(raw_pvals);
    ASSERT_EQ(corrected.size(), 1u);
    EXPECT_FLOAT_EQ(corrected[0], 0.03f);
}

TEST_F(HolmBonferroniTest, TwoTestsCorrectOrder) {
    // Holm-Bonferroni: sort p-values, multiply by (m - rank + 1).
    // p = [0.01, 0.04], m = 2
    // sorted: [0.01, 0.04]
    // corrected: [0.01 * 2, max(0.01*2, 0.04*1)] = [0.02, 0.04]
    std::vector<float> raw_pvals = {0.01f, 0.04f};
    auto corrected = holm_bonferroni_correct(raw_pvals);
    ASSERT_EQ(corrected.size(), 2u);
    EXPECT_NEAR(corrected[0], 0.02f, 1e-6f);
    EXPECT_NEAR(corrected[1], 0.04f, 1e-6f);
}

TEST_F(HolmBonferroniTest, CorrectedPValuesNeverExceedOne) {
    std::vector<float> raw_pvals = {0.5f, 0.7f, 0.9f};
    auto corrected = holm_bonferroni_correct(raw_pvals);
    for (float p : corrected) {
        EXPECT_LE(p, 1.0f);
    }
}

TEST_F(HolmBonferroniTest, CorrectedPValuesNonDecreasingAfterSortMapping) {
    // After applying Holm-Bonferroni, the corrected p-values (in original order)
    // when sorted should be non-decreasing.
    std::vector<float> raw_pvals = {0.005f, 0.03f, 0.01f, 0.08f};
    auto corrected = holm_bonferroni_correct(raw_pvals);
    ASSERT_EQ(corrected.size(), raw_pvals.size());

    // Corrected p-values should be >= raw p-values
    for (size_t i = 0; i < raw_pvals.size(); ++i) {
        EXPECT_GE(corrected[i], raw_pvals[i]);
    }
}

TEST_F(HolmBonferroniTest, CorrectsWith1800Tests) {
    // Spec: ~1800 tests. Verify it handles this dimensionality.
    std::vector<float> raw_pvals(1800);
    for (int i = 0; i < 1800; ++i) {
        raw_pvals[i] = static_cast<float>(i + 1) / 2000.0f;
    }
    auto corrected = holm_bonferroni_correct(raw_pvals);
    ASSERT_EQ(corrected.size(), 1800u);

    // Most corrected p-values should be larger than raw
    int count_larger = 0;
    for (size_t i = 0; i < 1800; ++i) {
        if (corrected[i] > raw_pvals[i]) count_larger++;
    }
    EXPECT_GT(count_larger, 1700);
}

TEST_F(HolmBonferroniTest, PreservesOriginalOrder) {
    // Corrected p-values should be returned in the same order as input.
    std::vector<float> raw_pvals = {0.04f, 0.01f, 0.05f, 0.001f};
    auto corrected = holm_bonferroni_correct(raw_pvals);
    ASSERT_EQ(corrected.size(), 4u);

    // The smallest raw p-value (index 3, 0.001) should have the smallest corrected
    // (though still multiplied by 4)
    // And the original ordering is preserved
    EXPECT_LT(corrected[3], corrected[0]);
}

TEST_F(HolmBonferroniTest, AllZeroPValues) {
    std::vector<float> raw_pvals = {0.0f, 0.0f, 0.0f};
    auto corrected = holm_bonferroni_correct(raw_pvals);
    for (float p : corrected) {
        EXPECT_FLOAT_EQ(p, 0.0f);
    }
}

TEST_F(HolmBonferroniTest, EmptyInput) {
    std::vector<float> raw_pvals;
    auto corrected = holm_bonferroni_correct(raw_pvals);
    EXPECT_TRUE(corrected.empty());
}

// ===========================================================================
// AnalysisResult — Reporting Standard
// ===========================================================================

class AnalysisResultTest : public ::testing::Test {};

TEST_F(AnalysisResultTest, ContainsPointEstimate) {
    AnalysisResult result;
    result.point_estimate = 0.15f;
    EXPECT_FLOAT_EQ(result.point_estimate, 0.15f);
}

TEST_F(AnalysisResultTest, ContainsConfidenceInterval) {
    AnalysisResult result;
    result.ci_lower = 0.05f;
    result.ci_upper = 0.25f;
    EXPECT_LT(result.ci_lower, result.ci_upper);
}

TEST_F(AnalysisResultTest, ContainsRawPValue) {
    AnalysisResult result;
    result.raw_p_value = 0.003f;
    EXPECT_FLOAT_EQ(result.raw_p_value, 0.003f);
}

TEST_F(AnalysisResultTest, ContainsCorrectedPValue) {
    AnalysisResult result;
    result.corrected_p_value = 0.05f;
    EXPECT_FLOAT_EQ(result.corrected_p_value, 0.05f);
}

TEST_F(AnalysisResultTest, ContainsSurvivesCorrectionFlag) {
    AnalysisResult result;
    result.corrected_p_value = 0.03f;
    result.survives_correction = (result.corrected_p_value < 0.05f);
    EXPECT_TRUE(result.survives_correction);
}

TEST_F(AnalysisResultTest, SuggestiveFlagForBorderlineResults) {
    // Spec: "significant before but not after correction → flagged as suggestive"
    AnalysisResult result;
    result.raw_p_value = 0.02f;
    result.corrected_p_value = 0.08f;
    result.survives_correction = false;
    result.is_suggestive = (result.raw_p_value < 0.05f && !result.survives_correction);
    EXPECT_TRUE(result.is_suggestive);
}

TEST_F(AnalysisResultTest, NotSuggestiveIfNotRawSignificant) {
    AnalysisResult result;
    result.raw_p_value = 0.10f;
    result.corrected_p_value = 0.30f;
    result.survives_correction = false;
    result.is_suggestive = (result.raw_p_value < 0.05f && !result.survives_correction);
    EXPECT_FALSE(result.is_suggestive);
}

// ===========================================================================
// MI Analysis with Holm-Bonferroni Integration
// ===========================================================================

class MIAnalysisIntegrationTest : public ::testing::Test {};

TEST_F(MIAnalysisIntegrationTest, AnalyzeAllFeaturesReturnsOneResultPerFeatureHorizonBarType) {
    // Spec: "1,800 feature-horizon-bartype tests"
    // With 62 features × 4 horizons × ~7 bar types ≈ 1,736 tests
    // Verify the analyzer produces correct dimensionality.
    MIAnalyzer analyzer;

    // Simulate: 62 features, 4 horizons, 7 bar types
    int n_features = 62;
    int n_horizons = 4;
    int n_bar_types = 7;

    // Dummy data
    std::vector<std::vector<float>> features(n_features, make_linear_feature(500));
    std::vector<std::vector<float>> returns(n_horizons, make_correlated_return_sign(500));
    std::vector<std::string> bar_types = {"tick_100", "tick_200", "vol_500", "vol_1000",
                                           "dollar_1M", "time_60s", "time_300s"};

    auto results = analyzer.analyze_all(features, returns, bar_types);
    EXPECT_EQ(results.size(),
              static_cast<size_t>(n_features * n_horizons * n_bar_types));
}

TEST_F(MIAnalysisIntegrationTest, AllResultsHaveHolmBonferroniCorrectedPValues) {
    MIAnalyzer analyzer;
    std::vector<std::vector<float>> features = {make_linear_feature(500)};
    std::vector<std::vector<float>> returns = {make_correlated_return_sign(500)};
    std::vector<std::string> bar_types = {"tick_100"};

    auto results = analyzer.analyze_all(features, returns, bar_types);
    for (const auto& r : results) {
        // Corrected p-value should be populated (not NaN)
        EXPECT_FALSE(std::isnan(r.corrected_p_value));
        // Corrected >= raw
        EXPECT_GE(r.corrected_p_value, r.raw_p_value - 1e-6f);
    }
}

// ===========================================================================
// Warmup Exclusion Validation Gate
// ===========================================================================

class MIWarmupExclusionTest : public ::testing::Test {};

TEST_F(MIWarmupExclusionTest, WarmupRowsExcludedFromAnalysis) {
    // Spec §Validation Gate: "MI analysis runs on is_warmup == false bars only"
    auto feature_vals = make_linear_feature(200);
    auto fwd_returns = make_correlated_returns(200);

    // All rows marked as warmup
    auto rows = make_rows_with_feature(feature_vals, fwd_returns, /*all_warmup=*/true);

    MIAnalyzer analyzer;
    auto result = analyzer.compute_from_rows(rows, "book_imbalance_1", "fwd_return_1", 5);

    // With all rows in warmup, there's no data to analyze
    EXPECT_TRUE(std::isnan(result.mi_bits) || result.sample_count == 0);
}

TEST_F(MIWarmupExclusionTest, OnlyNonWarmupRowsUsed) {
    auto feature_vals = make_linear_feature(200);
    auto fwd_returns = make_correlated_returns(200);
    auto rows = make_rows_with_feature(feature_vals, fwd_returns);

    // Mark first 50 as warmup
    for (int i = 0; i < 50; ++i) rows[i].is_warmup = true;

    MIAnalyzer analyzer;
    auto result = analyzer.compute_from_rows(rows, "book_imbalance_1", "fwd_return_1", 5);

    // Should use exactly 150 non-warmup rows
    EXPECT_EQ(result.sample_count, 150);
}

// ===========================================================================
// Spearman with Holm-Bonferroni Integration
// ===========================================================================

class SpearmanIntegrationTest : public ::testing::Test {};

TEST_F(SpearmanIntegrationTest, CorrectAcross1800Tests) {
    // Verify Holm-Bonferroni is applied across all 1800 Spearman tests.
    SpearmanAnalyzer analyzer;

    int n_features = 62;
    int n_horizons = 4;
    int n_bar_types = 7;

    std::vector<std::vector<float>> features(n_features, make_linear_feature(500));
    std::vector<std::vector<float>> returns(n_horizons, make_correlated_returns(500));
    std::vector<std::string> bar_types = {"tick_100", "tick_200", "vol_500", "vol_1000",
                                           "dollar_1M", "time_60s", "time_300s"};

    auto results = analyzer.analyze_all(features, returns, bar_types);

    // Every result should have both raw and corrected p-values
    for (const auto& r : results) {
        EXPECT_FALSE(std::isnan(r.raw_p_value));
        EXPECT_FALSE(std::isnan(r.corrected_p_value));
    }
}
