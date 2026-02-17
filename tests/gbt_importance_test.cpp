// gbt_importance_test.cpp — TDD RED phase tests for GBT Feature Importance with Stability Selection
// Spec: .kit/docs/feature-analysis.md §3 (GBT Feature Importance)
//
// Tests for:
//   - XGBoost regressor: all Track A features → return_n
//   - 5-fold expanding-window time-series CV (no shuffling, no future leakage)
//   - Stability selection: 20 runs with different seeds and 80% subsamples
//   - Top-20 features appearing in >60% of runs
//   - Expanding-window CV validation (no lookahead)
//
// No implementation files exist yet — all tests must fail to compile or fail at runtime.

#include <gtest/gtest.h>

// Headers the implementation must provide:
#include "analysis/gbt_importance.hpp"   // GBTImportanceAnalyzer, StabilityResult, CVFold
#include "features/bar_features.hpp"     // BarFeatureRow

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>
#include <vector>

// ===========================================================================
// Helpers — synthetic data for GBT importance
// ===========================================================================
namespace {

// Build BarFeatureRows with deterministic features and returns.
// Uses a simple pattern where some features are correlated with forward returns.
std::vector<BarFeatureRow> make_gbt_dataset(int n) {
    std::vector<BarFeatureRow> rows(n);
    for (int i = 0; i < n; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(n);

        rows[i].timestamp = static_cast<uint64_t>(i) * 1000000000ULL;
        rows[i].is_warmup = false;
        rows[i].day = 20220103 + i / 1000;

        // Predictive features (should rank high)
        rows[i].book_imbalance_1 = std::sin(frac * 6.28f) * 0.5f;
        rows[i].net_volume = 100.0f * frac - 50.0f;
        rows[i].kyle_lambda = 0.001f * frac;
        rows[i].volume_surprise = 1.0f + 0.5f * std::sin(frac * 3.14f);

        // Noise features (should rank low)
        rows[i].time_sin = std::sin(static_cast<float>(i * 137 + 42) * 0.01f);
        rows[i].time_cos = std::cos(static_cast<float>(i * 97 + 13) * 0.01f);
        rows[i].spread = 1.0f;  // Constant

        // Returns: partially predictable from book_imbalance_1 and net_volume
        rows[i].fwd_return_1 = rows[i].book_imbalance_1 * 2.0f
                              + rows[i].net_volume * 0.01f
                              + 0.3f * std::sin(static_cast<float>(i * 31));
        rows[i].fwd_return_5 = rows[i].book_imbalance_1 * 1.5f
                              + 0.5f * std::sin(static_cast<float>(i * 17));
        rows[i].fwd_return_20 = rows[i].net_volume * 0.005f
                               + 0.8f * std::sin(static_cast<float>(i * 7));
        rows[i].fwd_return_100 = 0.2f * rows[i].kyle_lambda
                                + std::sin(static_cast<float>(i * 3));
    }
    return rows;
}

// Build a small dataset for CV fold validation.
std::vector<BarFeatureRow> make_sequential_dataset(int n) {
    auto rows = make_gbt_dataset(n);
    // Set sequential timestamps to verify temporal ordering is preserved
    for (int i = 0; i < n; ++i) {
        rows[i].timestamp = 1641219000000000000ULL + static_cast<uint64_t>(i) * 1000000000ULL;
    }
    return rows;
}

}  // anonymous namespace

// ===========================================================================
// GBTImportanceAnalyzer — Construction
// ===========================================================================

class GBTImportanceConstructionTest : public ::testing::Test {};

TEST_F(GBTImportanceConstructionTest, DefaultConstruction) {
    GBTImportanceAnalyzer analyzer;
    (void)analyzer;
}

TEST_F(GBTImportanceConstructionTest, ConstructWithConfig) {
    GBTImportanceConfig config;
    config.n_stability_runs = 20;
    config.subsample_fraction = 0.8f;
    config.n_cv_folds = 5;
    config.top_k = 20;
    config.stability_threshold = 0.6f;
    GBTImportanceAnalyzer analyzer(config);
    (void)analyzer;
}

// ===========================================================================
// Expanding-Window Time-Series Cross-Validation
// ===========================================================================

class ExpandingWindowCVTest : public ::testing::Test {};

TEST_F(ExpandingWindowCVTest, FiveFoldsProduced) {
    // Spec: "5-fold expanding-window time-series CV"
    GBTImportanceAnalyzer analyzer;
    auto rows = make_sequential_dataset(5000);
    auto folds = analyzer.generate_cv_folds(rows, 5);
    EXPECT_EQ(folds.size(), 5u);
}

TEST_F(ExpandingWindowCVTest, TrainSetExpandsAcrossFolds) {
    // Expanding window: fold k has train = [0, split_k), test = [split_k, split_{k+1})
    // Each successive fold's train set is larger.
    GBTImportanceAnalyzer analyzer;
    auto rows = make_sequential_dataset(5000);
    auto folds = analyzer.generate_cv_folds(rows, 5);

    for (size_t i = 1; i < folds.size(); ++i) {
        EXPECT_GT(folds[i].train_end, folds[i - 1].train_end)
            << "Train set should expand across folds";
    }
}

TEST_F(ExpandingWindowCVTest, TestSetIsAfterTrainSet) {
    // No future leakage: test indices always after train indices.
    GBTImportanceAnalyzer analyzer;
    auto rows = make_sequential_dataset(5000);
    auto folds = analyzer.generate_cv_folds(rows, 5);

    for (const auto& fold : folds) {
        EXPECT_GE(fold.test_begin, fold.train_end)
            << "Test set must start at or after train end (no future leakage)";
    }
}

TEST_F(ExpandingWindowCVTest, NoShuffling) {
    // Spec §Validation Gate: "Time-series CV uses expanding window (no shuffling, no future leakage)"
    // Verify timestamps in train are always < timestamps in test.
    GBTImportanceAnalyzer analyzer;
    auto rows = make_sequential_dataset(5000);
    auto folds = analyzer.generate_cv_folds(rows, 5);

    for (const auto& fold : folds) {
        // Last train timestamp < first test timestamp
        if (fold.train_end > 0 && fold.test_begin < rows.size()) {
            EXPECT_LT(rows[fold.train_end - 1].timestamp,
                       rows[fold.test_begin].timestamp)
                << "Train timestamps must precede test timestamps";
        }
    }
}

TEST_F(ExpandingWindowCVTest, AllDataCoveredByFolds) {
    // Union of all test sets should cover most of the data.
    GBTImportanceAnalyzer analyzer;
    auto rows = make_sequential_dataset(5000);
    auto folds = analyzer.generate_cv_folds(rows, 5);

    size_t total_test_samples = 0;
    for (const auto& fold : folds) {
        total_test_samples += (fold.test_end - fold.test_begin);
    }
    // At least 80% of data should be in test sets (first fold has no test for early data)
    EXPECT_GT(total_test_samples, 3000u);
}

TEST_F(ExpandingWindowCVTest, FoldTrainStartsAtZero) {
    // Expanding window: train always starts from the beginning.
    GBTImportanceAnalyzer analyzer;
    auto rows = make_sequential_dataset(5000);
    auto folds = analyzer.generate_cv_folds(rows, 5);

    for (const auto& fold : folds) {
        EXPECT_EQ(fold.train_begin, 0u)
            << "Expanding window train should always start at index 0";
    }
}

// ===========================================================================
// Stability Selection
// ===========================================================================

class StabilitySelectionTest : public ::testing::Test {};

TEST_F(StabilitySelectionTest, Uses20IndependentRuns) {
    // Spec §Validation Gate: "GBT stability selection uses 20 independent runs"
    GBTImportanceConfig config;
    config.n_stability_runs = 20;
    config.subsample_fraction = 0.8f;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_gbt_dataset(2000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");
    EXPECT_EQ(result.n_runs, 20);
}

TEST_F(StabilitySelectionTest, Each80PercentSubsample) {
    // Each run uses 80% of the data.
    GBTImportanceConfig config;
    config.n_stability_runs = 5;  // fewer for speed in test
    config.subsample_fraction = 0.8f;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_gbt_dataset(1000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    for (const auto& run : result.per_run_details) {
        int expected_samples = static_cast<int>(1000 * 0.8f);
        EXPECT_NEAR(run.n_samples_used, expected_samples, 50)
            << "Each run should use ~80% of data";
    }
}

TEST_F(StabilitySelectionTest, DifferentSeedsPerRun) {
    // Each run should use a different random seed.
    GBTImportanceConfig config;
    config.n_stability_runs = 20;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_gbt_dataset(2000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    std::set<int> seeds;
    for (const auto& run : result.per_run_details) {
        seeds.insert(run.seed);
    }
    EXPECT_EQ(seeds.size(), 20u) << "All 20 runs must use distinct seeds";
}

TEST_F(StabilitySelectionTest, ReportsTop20PerRun) {
    GBTImportanceConfig config;
    config.n_stability_runs = 5;
    config.top_k = 20;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_gbt_dataset(2000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    for (const auto& run : result.per_run_details) {
        EXPECT_LE(run.top_features.size(), 20u);
        EXPECT_GT(run.top_features.size(), 0u);
    }
}

TEST_F(StabilitySelectionTest, StableFeatureAppearsIn60PercentOfRuns) {
    // Spec: "features appearing in top-20 in >60% of runs"
    GBTImportanceConfig config;
    config.n_stability_runs = 20;
    config.stability_threshold = 0.6f;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_gbt_dataset(5000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    // Stable features should have selection_frequency > 0.6
    for (const auto& feat : result.stable_features) {
        EXPECT_GT(feat.selection_frequency, 0.6f)
            << "Stable feature " << feat.name << " should appear in >60% of runs";
    }
}

TEST_F(StabilitySelectionTest, SelectionFrequencyBetweenZeroAndOne) {
    GBTImportanceConfig config;
    config.n_stability_runs = 20;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_gbt_dataset(2000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    for (const auto& [name, freq] : result.all_feature_frequencies) {
        EXPECT_GE(freq, 0.0f) << name << " frequency out of range";
        EXPECT_LE(freq, 1.0f) << name << " frequency out of range";
    }
}

TEST_F(StabilitySelectionTest, ImportanceValuesNonNegative) {
    GBTImportanceConfig config;
    config.n_stability_runs = 5;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_gbt_dataset(2000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    for (const auto& run : result.per_run_details) {
        for (const auto& [name, importance] : run.importances) {
            EXPECT_GE(importance, 0.0f) << "Feature " << name << " importance must be >= 0";
        }
    }
}

TEST_F(StabilitySelectionTest, ResultsReproducibleWithSeed) {
    GBTImportanceConfig config;
    config.n_stability_runs = 5;
    config.master_seed = 42;
    GBTImportanceAnalyzer analyzer1(config);
    GBTImportanceAnalyzer analyzer2(config);

    auto rows = make_gbt_dataset(1000);
    auto r1 = analyzer1.run_stability_selection(rows, "fwd_return_1");
    auto r2 = analyzer2.run_stability_selection(rows, "fwd_return_1");

    // Same seed → same stable features
    EXPECT_EQ(r1.stable_features.size(), r2.stable_features.size());
}

// ===========================================================================
// GBT with Expanding-Window CV Integration
// ===========================================================================

class GBTCVIntegrationTest : public ::testing::Test {};

TEST_F(GBTCVIntegrationTest, TrainsOnExpandingWindow) {
    // Verify each stability run uses expanding-window CV internally.
    GBTImportanceConfig config;
    config.n_stability_runs = 3;
    config.n_cv_folds = 5;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_sequential_dataset(5000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    // Each run should report 5 folds
    for (const auto& run : result.per_run_details) {
        EXPECT_EQ(run.n_cv_folds, 5);
    }
}

TEST_F(GBTCVIntegrationTest, CVScoresReported) {
    GBTImportanceConfig config;
    config.n_stability_runs = 3;
    config.n_cv_folds = 5;
    GBTImportanceAnalyzer analyzer(config);

    auto rows = make_sequential_dataset(5000);
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    // Each run should have a CV score (e.g., mean RMSE across folds)
    for (const auto& run : result.per_run_details) {
        EXPECT_FALSE(std::isnan(run.mean_cv_score));
    }
}

// ===========================================================================
// Warmup Exclusion in GBT
// ===========================================================================

class GBTWarmupExclusionTest : public ::testing::Test {};

TEST_F(GBTWarmupExclusionTest, WarmupRowsExcludedFromTraining) {
    GBTImportanceAnalyzer analyzer;
    auto rows = make_gbt_dataset(2000);

    // Mark first 200 as warmup
    for (int i = 0; i < 200; ++i) rows[i].is_warmup = true;

    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");

    // Total samples across runs should be based on 1800 non-warmup rows
    for (const auto& run : result.per_run_details) {
        EXPECT_LE(run.n_samples_used, 1800)
            << "Warmup rows should be excluded from training";
    }
}

TEST_F(GBTWarmupExclusionTest, AllWarmupReturnsEmptyResult) {
    GBTImportanceAnalyzer analyzer;
    auto rows = make_gbt_dataset(100);
    for (auto& row : rows) row.is_warmup = true;

    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");
    EXPECT_TRUE(result.stable_features.empty());
}

// ===========================================================================
// Multiple Return Horizons
// ===========================================================================

class GBTMultiHorizonTest : public ::testing::Test {};

TEST_F(GBTMultiHorizonTest, AnalyzesAllFourHorizons) {
    // Spec: return_n for n = 1, 5, 20, 100
    GBTImportanceAnalyzer analyzer;
    auto rows = make_gbt_dataset(3000);

    std::vector<std::string> horizons = {
        "fwd_return_1", "fwd_return_5", "fwd_return_20", "fwd_return_100"
    };

    for (const auto& horizon : horizons) {
        auto result = analyzer.run_stability_selection(rows, horizon);
        EXPECT_GT(result.n_runs, 0) << "Should produce results for horizon: " << horizon;
    }
}

TEST_F(GBTMultiHorizonTest, DifferentHorizonsMayProduceDifferentRankings) {
    GBTImportanceAnalyzer analyzer;
    auto rows = make_gbt_dataset(3000);

    auto result_1 = analyzer.run_stability_selection(rows, "fwd_return_1");
    auto result_100 = analyzer.run_stability_selection(rows, "fwd_return_100");

    // Different horizons should not necessarily produce identical rankings
    // (This is a weak test — just verifies both produce valid output)
    EXPECT_FALSE(result_1.stable_features.empty() && result_100.stable_features.empty());
}

// ===========================================================================
// NaN Return Handling
// ===========================================================================

class GBTNaNHandlingTest : public ::testing::Test {};

TEST_F(GBTNaNHandlingTest, NaNForwardReturnsExcluded) {
    GBTImportanceAnalyzer analyzer;
    auto rows = make_gbt_dataset(500);

    // Last 100 bars have NaN fwd_return_100
    for (int i = 400; i < 500; ++i) {
        rows[i].fwd_return_100 = std::numeric_limits<float>::quiet_NaN();
    }

    auto result = analyzer.run_stability_selection(rows, "fwd_return_100");
    // Should still work with the 400 valid rows
    EXPECT_GT(result.n_runs, 0);
    for (const auto& run : result.per_run_details) {
        EXPECT_LE(run.n_samples_used, 400);
    }
}

TEST_F(GBTNaNHandlingTest, NaNFeatureValuesHandled) {
    GBTImportanceAnalyzer analyzer;
    auto rows = make_gbt_dataset(500);

    // Some rows have NaN kyle_lambda
    for (int i = 0; i < 50; ++i) {
        rows[i].kyle_lambda = std::numeric_limits<float>::quiet_NaN();
    }

    // Should not crash — NaN features handled by XGBoost or excluded
    auto result = analyzer.run_stability_selection(rows, "fwd_return_1");
    EXPECT_GT(result.n_runs, 0);
}
