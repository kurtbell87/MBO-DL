// conditional_returns_test.cpp — TDD RED phase tests for Conditional Returns + Decay Analysis
// Spec: .kit/docs/feature-analysis.md §4 (Conditional Returns), §5 (Decay Analysis)
//
// Tests for:
//   - Quintile bucketing of feature values
//   - Mean return per quintile
//   - Monotonicity measure: Q5 mean - Q1 mean
//   - T-statistic for Q5 vs Q1
//   - Decay curves: correlation at horizons n = 1, 2, 5, 10, 20, 50, 100
//   - Decay classification: short-horizon signal vs regime indicator
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers the implementation must provide:
#include "analysis/conditional_returns.hpp"  // ConditionalReturnAnalyzer, QuintileResult
#include "analysis/decay_analysis.hpp"       // DecayAnalyzer, DecayCurve
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

// Build feature values linearly spaced in [0, 1].
std::vector<float> make_linear_feature(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i) / static_cast<float>(n - 1);
    }
    return v;
}

// Build returns positively correlated with feature (monotonic relationship).
std::vector<float> make_monotonic_returns(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        // returns increase with feature value → monotonic Q1 < Q5
        v[i] = 0.01f * static_cast<float>(i) / static_cast<float>(n - 1)
             + 0.005f * std::sin(static_cast<float>(i));
    }
    return v;
}

// Build returns with no monotonic relationship to feature.
std::vector<float> make_nonmonotonic_returns(int n) {
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = 0.5f * std::sin(static_cast<float>(i * 137 + 42) * 0.01f);
    }
    return v;
}

// Build constant returns.
std::vector<float> make_constant_returns(int n, float val = 0.0f) {
    return std::vector<float>(n, val);
}

// Build BarFeatureRows with specified feature and multi-horizon forward returns.
std::vector<BarFeatureRow> make_decay_dataset(int n) {
    std::vector<BarFeatureRow> rows(n);
    for (int i = 0; i < n; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(n - 1);
        rows[i].is_warmup = false;

        // Predictive feature
        rows[i].book_imbalance_1 = std::sin(frac * 6.28f);

        // Returns at various horizons — correlation decays with horizon
        float signal = rows[i].book_imbalance_1;
        float noise_scale = 0.3f;
        float noise = noise_scale * std::sin(static_cast<float>(i * 31 + 7));

        // Short horizon: strong signal
        rows[i].fwd_return_1 = signal * 1.0f + noise;
        rows[i].fwd_return_5 = signal * 0.7f + noise * 1.2f;
        rows[i].fwd_return_20 = signal * 0.3f + noise * 1.5f;
        rows[i].fwd_return_100 = signal * 0.05f + noise * 2.0f;
    }
    return rows;
}

}  // anonymous namespace

// ===========================================================================
// Quintile Bucketing
// ===========================================================================

class QuintileBucketingTest : public ::testing::Test {};

TEST_F(QuintileBucketingTest, ProducesFiveQuintiles) {
    auto feature = make_linear_feature(1000);
    ConditionalReturnAnalyzer analyzer;
    auto quintiles = analyzer.compute_quintiles(feature);
    EXPECT_EQ(quintiles.size(), 5u);
}

TEST_F(QuintileBucketingTest, QuintilesRoughlyEqualSized) {
    auto feature = make_linear_feature(1000);
    ConditionalReturnAnalyzer analyzer;
    auto quintiles = analyzer.compute_quintiles(feature);

    for (const auto& q : quintiles) {
        EXPECT_NEAR(q.count, 200, 10)
            << "Quintile " << q.quintile_index << " should have ~200 elements";
    }
}

TEST_F(QuintileBucketingTest, Q1HasLowestFeatureValues) {
    auto feature = make_linear_feature(1000);
    ConditionalReturnAnalyzer analyzer;
    auto quintiles = analyzer.compute_quintiles(feature);

    EXPECT_LT(quintiles[0].mean_feature_value, quintiles[4].mean_feature_value);
}

TEST_F(QuintileBucketingTest, Q5HasHighestFeatureValues) {
    auto feature = make_linear_feature(1000);
    ConditionalReturnAnalyzer analyzer;
    auto quintiles = analyzer.compute_quintiles(feature);

    // Q5 (index 4) should have the highest mean feature value
    for (int i = 0; i < 4; ++i) {
        EXPECT_LT(quintiles[i].mean_feature_value, quintiles[4].mean_feature_value);
    }
}

TEST_F(QuintileBucketingTest, AllSamplesAssigned) {
    auto feature = make_linear_feature(1000);
    ConditionalReturnAnalyzer analyzer;
    auto quintiles = analyzer.compute_quintiles(feature);

    int total = 0;
    for (const auto& q : quintiles) total += q.count;
    EXPECT_EQ(total, 1000);
}

// ===========================================================================
// Mean Return Per Quintile
// ===========================================================================

class MeanReturnPerQuintileTest : public ::testing::Test {};

TEST_F(MeanReturnPerQuintileTest, MonotonicReturnsShowIncreasingMeans) {
    // Feature linearly related to returns → quintile means should be monotonically increasing.
    auto feature = make_linear_feature(5000);
    auto returns = make_monotonic_returns(5000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);

    // Q1 mean < Q2 mean < ... < Q5 mean (approximately)
    for (size_t i = 1; i < result.quintile_means.size(); ++i) {
        EXPECT_GE(result.quintile_means[i], result.quintile_means[i - 1] - 0.01f)
            << "Quintile means should be approximately non-decreasing";
    }
}

TEST_F(MeanReturnPerQuintileTest, FiveMeansReported) {
    auto feature = make_linear_feature(1000);
    auto returns = make_monotonic_returns(1000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);
    EXPECT_EQ(result.quintile_means.size(), 5u);
}

TEST_F(MeanReturnPerQuintileTest, MeansAreNotNaN) {
    auto feature = make_linear_feature(1000);
    auto returns = make_monotonic_returns(1000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);

    for (float m : result.quintile_means) {
        EXPECT_FALSE(std::isnan(m));
    }
}

TEST_F(MeanReturnPerQuintileTest, ConstantReturnsAllQuintileMeansEqual) {
    auto feature = make_linear_feature(1000);
    auto returns = make_constant_returns(1000, 0.5f);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);

    for (float m : result.quintile_means) {
        EXPECT_NEAR(m, 0.5f, 1e-4f);
    }
}

// ===========================================================================
// Monotonicity Measure: Q5 - Q1
// ===========================================================================

class MonotonicityTest : public ::testing::Test {};

TEST_F(MonotonicityTest, PositiveForPositiveRelationship) {
    // Spec: "Report monotonicity: Q5 mean - Q1 mean"
    auto feature = make_linear_feature(5000);
    auto returns = make_monotonic_returns(5000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);

    EXPECT_GT(result.q5_minus_q1, 0.0f)
        << "Q5 - Q1 should be positive for positive feature-return relationship";
}

TEST_F(MonotonicityTest, NegativeForInverseRelationship) {
    auto feature = make_linear_feature(5000);
    // Reverse returns → inverse relationship
    std::vector<float> returns(5000);
    for (int i = 0; i < 5000; ++i) {
        returns[i] = -0.01f * static_cast<float>(i) / 5000.0f;
    }
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);
    EXPECT_LT(result.q5_minus_q1, 0.0f);
}

TEST_F(MonotonicityTest, NearZeroForNoRelationship) {
    auto feature = make_linear_feature(5000);
    auto returns = make_nonmonotonic_returns(5000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);
    EXPECT_NEAR(result.q5_minus_q1, 0.0f, 0.05f);
}

TEST_F(MonotonicityTest, ZeroForConstantReturns) {
    auto feature = make_linear_feature(1000);
    auto returns = make_constant_returns(1000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);
    EXPECT_NEAR(result.q5_minus_q1, 0.0f, 1e-6f);
}

// ===========================================================================
// T-Statistic (Q5 vs Q1)
// ===========================================================================

class TStatisticTest : public ::testing::Test {};

TEST_F(TStatisticTest, LargeTStatForStrongSignal) {
    // Spec: "t-statistic" for Q5 vs Q1.
    auto feature = make_linear_feature(10000);
    auto returns = make_monotonic_returns(10000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);

    // Strong monotonic relationship → large |t-stat|
    EXPECT_GT(std::abs(result.t_statistic), 2.0f);
}

TEST_F(TStatisticTest, SmallTStatForNoSignal) {
    auto feature = make_linear_feature(5000);
    auto returns = make_nonmonotonic_returns(5000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);

    // No relationship → |t-stat| near 0
    EXPECT_LT(std::abs(result.t_statistic), 3.0f);
}

TEST_F(TStatisticTest, TStatIsFinite) {
    auto feature = make_linear_feature(1000);
    auto returns = make_monotonic_returns(1000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);
    EXPECT_FALSE(std::isnan(result.t_statistic));
    EXPECT_FALSE(std::isinf(result.t_statistic));
}

TEST_F(TStatisticTest, PValueReported) {
    auto feature = make_linear_feature(1000);
    auto returns = make_monotonic_returns(1000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);
    EXPECT_GT(result.t_p_value, 0.0f);
    EXPECT_LE(result.t_p_value, 1.0f);
}

TEST_F(TStatisticTest, ZeroVarianceReturnsHandled) {
    // If Q1 and Q5 returns have zero variance, t-stat should be NaN or 0.
    auto feature = make_linear_feature(1000);
    auto returns = make_constant_returns(1000);
    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze(feature, returns);
    // With constant returns, Q5 mean = Q1 mean = 0, variance = 0
    // t-stat should be 0 or NaN (not inf)
    EXPECT_FALSE(std::isinf(result.t_statistic));
}

// ===========================================================================
// Conditional Returns — Warmup Exclusion
// ===========================================================================

class ConditionalReturnsWarmupTest : public ::testing::Test {};

TEST_F(ConditionalReturnsWarmupTest, WarmupRowsExcluded) {
    auto feature = make_linear_feature(200);
    auto returns = make_monotonic_returns(200);

    // Build rows, mark first 50 as warmup
    std::vector<BarFeatureRow> rows(200);
    for (int i = 0; i < 200; ++i) {
        rows[i].book_imbalance_1 = feature[i];
        rows[i].fwd_return_1 = returns[i];
        rows[i].is_warmup = (i < 50);
    }

    ConditionalReturnAnalyzer analyzer;
    auto result = analyzer.analyze_from_rows(rows, "book_imbalance_1", "fwd_return_1");
    EXPECT_EQ(result.sample_count, 150);
}

// ===========================================================================
// Decay Analysis
// ===========================================================================

class DecayAnalysisTest : public ::testing::Test {};

TEST_F(DecayAnalysisTest, ComputesCorrelationAtMultipleHorizons) {
    // Spec: "Compute correlation with return_n for n = 1, 2, 5, 10, 20, 50, 100 bars"
    auto rows = make_decay_dataset(5000);
    DecayAnalyzer analyzer;
    auto curve = analyzer.compute_decay(rows, "book_imbalance_1");

    // Should have correlations at 7 horizons
    EXPECT_EQ(curve.horizons.size(), 7u);

    std::vector<int> expected_horizons = {1, 2, 5, 10, 20, 50, 100};
    EXPECT_EQ(curve.horizons, expected_horizons);
}

TEST_F(DecayAnalysisTest, CorrelationsAtAllHorizonsAreDefined) {
    auto rows = make_decay_dataset(5000);
    DecayAnalyzer analyzer;
    auto curve = analyzer.compute_decay(rows, "book_imbalance_1");

    for (size_t i = 0; i < curve.correlations.size(); ++i) {
        EXPECT_FALSE(std::isnan(curve.correlations[i]))
            << "Correlation at horizon " << curve.horizons[i] << " should not be NaN";
    }
}

TEST_F(DecayAnalysisTest, CorrelationsInValidRange) {
    auto rows = make_decay_dataset(5000);
    DecayAnalyzer analyzer;
    auto curve = analyzer.compute_decay(rows, "book_imbalance_1");

    for (size_t i = 0; i < curve.correlations.size(); ++i) {
        EXPECT_GE(curve.correlations[i], -1.0f);
        EXPECT_LE(curve.correlations[i], 1.0f);
    }
}

TEST_F(DecayAnalysisTest, ShortHorizonStrongerThanLongHorizon) {
    // For our synthetic data, the signal decays with horizon.
    auto rows = make_decay_dataset(5000);
    DecayAnalyzer analyzer;
    auto curve = analyzer.compute_decay(rows, "book_imbalance_1");

    // |corr at horizon 1| > |corr at horizon 100|
    EXPECT_GT(std::abs(curve.correlations[0]), std::abs(curve.correlations.back()));
}

TEST_F(DecayAnalysisTest, DecayCurveIsMonotonicallyDecreasing) {
    // For a typical short-horizon signal, |correlation| should roughly decrease.
    auto rows = make_decay_dataset(10000);
    DecayAnalyzer analyzer;
    auto curve = analyzer.compute_decay(rows, "book_imbalance_1");

    // At least the first and last should show decay
    EXPECT_GT(std::abs(curve.correlations[0]), std::abs(curve.correlations.back()) - 0.05f);
}

// ===========================================================================
// Decay Classification
// ===========================================================================

class DecayClassificationTest : public ::testing::Test {};

TEST_F(DecayClassificationTest, SharpDecayClassifiedAsShortHorizonSignal) {
    // Spec: "Sharp decay → short-horizon signal"
    DecayCurve curve;
    curve.horizons = {1, 2, 5, 10, 20, 50, 100};
    curve.correlations = {0.8f, 0.6f, 0.3f, 0.1f, 0.02f, 0.01f, 0.005f};

    DecayAnalyzer analyzer;
    auto classification = analyzer.classify_decay(curve);
    EXPECT_EQ(classification, DecayType::SHORT_HORIZON_SIGNAL);
}

TEST_F(DecayClassificationTest, SlowDecayClassifiedAsRegimeIndicator) {
    // Spec: "Slow decay → regime indicator"
    DecayCurve curve;
    curve.horizons = {1, 2, 5, 10, 20, 50, 100};
    curve.correlations = {0.5f, 0.48f, 0.45f, 0.42f, 0.38f, 0.30f, 0.25f};

    DecayAnalyzer analyzer;
    auto classification = analyzer.classify_decay(curve);
    EXPECT_EQ(classification, DecayType::REGIME_INDICATOR);
}

TEST_F(DecayClassificationTest, NoSignalClassifiedCorrectly) {
    DecayCurve curve;
    curve.horizons = {1, 2, 5, 10, 20, 50, 100};
    curve.correlations = {0.01f, 0.005f, -0.002f, 0.003f, -0.001f, 0.002f, -0.003f};

    DecayAnalyzer analyzer;
    auto classification = analyzer.classify_decay(curve);
    EXPECT_EQ(classification, DecayType::NO_SIGNAL);
}

TEST_F(DecayClassificationTest, ClassificationHandlesNegativeCorrelations) {
    // Negative correlation that decays in absolute value
    DecayCurve curve;
    curve.horizons = {1, 2, 5, 10, 20, 50, 100};
    curve.correlations = {-0.7f, -0.5f, -0.25f, -0.1f, -0.03f, -0.01f, -0.005f};

    DecayAnalyzer analyzer;
    auto classification = analyzer.classify_decay(curve);
    // Sharp decay in |corr| → short-horizon signal
    EXPECT_EQ(classification, DecayType::SHORT_HORIZON_SIGNAL);
}

// ===========================================================================
// Decay Analysis with Feature Name Resolution
// ===========================================================================

class DecayFeatureResolutionTest : public ::testing::Test {};

TEST_F(DecayFeatureResolutionTest, ResolvesValidFeatureName) {
    auto rows = make_decay_dataset(1000);
    DecayAnalyzer analyzer;

    // Should work for any Track A feature name
    auto curve = analyzer.compute_decay(rows, "book_imbalance_1");
    EXPECT_FALSE(curve.correlations.empty());
}

TEST_F(DecayFeatureResolutionTest, InvalidFeatureNameThrows) {
    auto rows = make_decay_dataset(1000);
    DecayAnalyzer analyzer;
    EXPECT_THROW(analyzer.compute_decay(rows, "nonexistent_feature"), std::invalid_argument);
}

TEST_F(DecayFeatureResolutionTest, WorksForAllTrackAFeatures) {
    auto rows = make_decay_dataset(1000);
    DecayAnalyzer analyzer;

    auto names = BarFeatureRow::feature_names();
    for (const auto& name : names) {
        auto curve = analyzer.compute_decay(rows, name);
        EXPECT_EQ(curve.horizons.size(), 7u)
            << "Decay curve missing horizons for feature: " << name;
    }
}

// ===========================================================================
// Decay — Only Predictive Features Analyzed
// ===========================================================================

class DecayPredictiveFilterTest : public ::testing::Test {};

TEST_F(DecayPredictiveFilterTest, OnlyPredictiveFeaturesGetDecayCurves) {
    // Spec: "For each predictive feature: Compute correlation..."
    // Non-predictive features (no significant MI/correlation) should be skipped.
    auto rows = make_decay_dataset(5000);
    DecayAnalyzer analyzer;

    // Given a list of "predictive" feature names from prior analysis
    std::vector<std::string> predictive = {"book_imbalance_1", "net_volume"};
    auto curves = analyzer.compute_decay_for_predictive(rows, predictive);

    EXPECT_EQ(curves.size(), predictive.size());
}

// ===========================================================================
// Decay — Warmup Exclusion
// ===========================================================================

class DecayWarmupExclusionTest : public ::testing::Test {};

TEST_F(DecayWarmupExclusionTest, WarmupRowsExcluded) {
    auto rows = make_decay_dataset(1000);
    for (int i = 0; i < 100; ++i) rows[i].is_warmup = true;

    DecayAnalyzer analyzer;
    auto curve = analyzer.compute_decay(rows, "book_imbalance_1");

    // Should only use 900 non-warmup rows
    EXPECT_EQ(curve.sample_count, 900);
}
