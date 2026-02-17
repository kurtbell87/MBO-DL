// oracle_comparison_test.cpp — TDD RED phase tests for Oracle Comparison (§5.3)
// Spec: .kit/docs/multi-day-backtest.md §Oracle Comparison
//
// Tests the oracle comparison framework: label distribution, label stability,
// label-return correlation, conditional entropy, and expectancy comparison
// between FIRST_TO_HIT and TRIPLE_BARRIER labeling methods.
//
// Headers below do not exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

// Header that the implementation must provide:
#include "backtest/oracle_comparison.hpp"

// Already-existing headers from prior phases:
#include "backtest/execution_costs.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"
#include "test_bar_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

using test_helpers::TICK;
using test_helpers::NS_PER_SEC;
using test_helpers::NS_PER_HOUR;
using test_helpers::make_bar;
using test_helpers::make_bar_series;
using test_helpers::make_bar_path;

// Make a BacktestResult with specified label counts and trades.
BacktestResult make_result_with_labels(const std::map<int, int>& label_counts,
                                        int wins, int losses, float net_pnl) {
    BacktestResult result{};
    result.label_counts = label_counts;
    result.winning_trades = wins;
    result.losing_trades = losses;
    result.total_trades = wins + losses;
    result.net_pnl = net_pnl;
    if (result.total_trades > 0) {
        result.win_rate = static_cast<float>(wins) / static_cast<float>(result.total_trades);
        result.expectancy = net_pnl / static_cast<float>(result.total_trades);
    }
    return result;
}

}  // namespace

// ===========================================================================
// 1. OracleComparison — construction
// ===========================================================================
class OracleComparisonTest : public ::testing::Test {
protected:
    BacktestResult fth_result;
    BacktestResult tb_result;

    void SetUp() override {
        // First-to-hit: more trades, lower expectancy
        fth_result = make_result_with_labels(
            {{1, 50}, {-1, 30}},  // 50 LONG, 30 SHORT
            45, 35, 100.0f
        );

        // Triple barrier: fewer trades, higher expectancy
        tb_result = make_result_with_labels(
            {{1, 40}, {-1, 25}},  // 40 LONG, 25 SHORT
            35, 30, 120.0f
        );
    }
};

TEST_F(OracleComparisonTest, ConstructsWithTwoResults) {
    OracleComparison comparison(fth_result, tb_result);
    (void)comparison;  // compiles and doesn't throw
}

// ===========================================================================
// 2. Label distribution — class frequencies
// ===========================================================================

TEST_F(OracleComparisonTest, LabelDistributionFirstToHit) {
    // Spec: "Label distribution (class frequencies)"
    OracleComparison comparison(fth_result, tb_result);
    auto dist = comparison.label_distribution_fth();

    // Should have entries for LONG (+1) and SHORT (-1)
    EXPECT_GT(dist.size(), 0u);
    EXPECT_TRUE(dist.count(1) > 0);   // LONG
    EXPECT_TRUE(dist.count(-1) > 0);  // SHORT
}

TEST_F(OracleComparisonTest, LabelDistributionTripleBarrier) {
    OracleComparison comparison(fth_result, tb_result);
    auto dist = comparison.label_distribution_tb();

    EXPECT_GT(dist.size(), 0u);
    EXPECT_TRUE(dist.count(1) > 0);
    EXPECT_TRUE(dist.count(-1) > 0);
}

TEST_F(OracleComparisonTest, LabelDistributionSumsToTotalTrades) {
    OracleComparison comparison(fth_result, tb_result);

    auto dist_fth = comparison.label_distribution_fth();
    int total_fth = 0;
    for (const auto& [label, count] : dist_fth) {
        total_fth += count;
    }
    EXPECT_EQ(total_fth, fth_result.total_trades);

    auto dist_tb = comparison.label_distribution_tb();
    int total_tb = 0;
    for (const auto& [label, count] : dist_tb) {
        total_tb += count;
    }
    EXPECT_EQ(total_tb, tb_result.total_trades);
}

// ===========================================================================
// 3. Label stability — consecutive label agreement rate
// ===========================================================================

TEST_F(OracleComparisonTest, LabelStabilityFromSequence) {
    // Spec: "Label stability (consecutive label agreement rate)"
    // Given a sequence of labels, compute fraction where label[i] == label[i-1]
    std::vector<int> labels = {1, 1, 1, -1, -1, 1, 1, 1, -1, 1};

    float stability = oracle_comparison::label_stability(labels);

    // Agreements: (1,1),(1,1),(1,-1)=no,(-1,-1),(−1,1)=no,(1,1),(1,1),(1,-1)=no,(-1,1)=no
    // Agreements: positions 1,2,4,6,7 = 5 agreements out of 9 pairs
    EXPECT_NEAR(stability, 5.0f / 9.0f, 0.02f);
}

TEST_F(OracleComparisonTest, LabelStabilityPerfectAgreement) {
    std::vector<int> labels = {1, 1, 1, 1, 1};
    float stability = oracle_comparison::label_stability(labels);
    EXPECT_NEAR(stability, 1.0f, 0.01f);
}

TEST_F(OracleComparisonTest, LabelStabilityAlternating) {
    std::vector<int> labels = {1, -1, 1, -1, 1};
    float stability = oracle_comparison::label_stability(labels);
    EXPECT_NEAR(stability, 0.0f, 0.01f);
}

TEST_F(OracleComparisonTest, LabelStabilityEmptySequence) {
    std::vector<int> labels;
    float stability = oracle_comparison::label_stability(labels);
    EXPECT_FLOAT_EQ(stability, 0.0f);
}

TEST_F(OracleComparisonTest, LabelStabilitySingleElement) {
    std::vector<int> labels = {1};
    float stability = oracle_comparison::label_stability(labels);
    // With 1 element, no pairs to compare
    EXPECT_FLOAT_EQ(stability, 0.0f);
}

// ===========================================================================
// 4. Label-return correlation
// ===========================================================================

TEST_F(OracleComparisonTest, LabelReturnCorrelation) {
    // Spec: "Label-return correlation"
    // Correlation between oracle label (+1/-1) and actual return after the label
    std::vector<int> labels = {1, -1, 1, 1, -1};
    std::vector<float> returns = {0.5f, -0.3f, 0.2f, 0.8f, -0.6f};

    float corr = oracle_comparison::label_return_correlation(labels, returns);

    // Perfect positive correlation: labels match return signs → corr should be > 0
    EXPECT_GT(corr, 0.0f);
    EXPECT_LE(corr, 1.0f);
}

TEST_F(OracleComparisonTest, LabelReturnCorrelationNegative) {
    // Labels are opposite of returns → negative correlation
    std::vector<int> labels = {1, 1, 1, 1};
    std::vector<float> returns = {-0.5f, -0.3f, -0.2f, -0.8f};

    float corr = oracle_comparison::label_return_correlation(labels, returns);
    EXPECT_LT(corr, 0.0f);
}

TEST_F(OracleComparisonTest, LabelReturnCorrelationEmptyInput) {
    std::vector<int> labels;
    std::vector<float> returns;

    float corr = oracle_comparison::label_return_correlation(labels, returns);
    EXPECT_FLOAT_EQ(corr, 0.0f);
}

TEST_F(OracleComparisonTest, LabelReturnCorrelationMismatchedSizes) {
    // Should handle gracefully or use min of sizes
    std::vector<int> labels = {1, -1, 1};
    std::vector<float> returns = {0.5f, -0.3f};

    // Should not crash
    float corr = oracle_comparison::label_return_correlation(labels, returns);
    EXPECT_TRUE(std::isfinite(corr));
}

// ===========================================================================
// 5. Conditional entropy of labels given time-of-day
// ===========================================================================

TEST_F(OracleComparisonTest, ConditionalEntropyComputation) {
    // Spec: "Conditional entropy of labels given time-of-day"
    // H(Label | TimeOfDay)
    // If labels are the same regardless of time → low entropy
    // If labels vary by time → higher entropy
    std::vector<int> labels = {1, 1, -1, -1, 1, 1, -1, -1};
    std::vector<float> times_of_day = {
        9.5f, 9.75f,   // OPEN session
        12.0f, 12.5f,  // MID session
        14.0f, 14.5f,  // CLOSE session
        15.0f, 15.5f   // CLOSE session
    };

    float cond_entropy = oracle_comparison::conditional_entropy(labels, times_of_day);

    // Should be non-negative (entropy is always >= 0)
    EXPECT_GE(cond_entropy, 0.0f);
}

TEST_F(OracleComparisonTest, ConditionalEntropyZeroWhenDeterministic) {
    // If time-of-day perfectly predicts label, conditional entropy = 0
    std::vector<int> labels = {1, 1, 1, -1, -1, -1};
    std::vector<float> times_of_day = {
        9.5f, 9.75f, 10.0f,     // OPEN → always LONG
        14.0f, 14.5f, 15.0f     // CLOSE → always SHORT
    };

    float cond_entropy = oracle_comparison::conditional_entropy(labels, times_of_day);
    EXPECT_NEAR(cond_entropy, 0.0f, 0.01f);
}

TEST_F(OracleComparisonTest, ConditionalEntropyHighWhenRandom) {
    // If labels are uniformly random within each session, high conditional entropy
    std::vector<int> labels = {1, -1, 1, -1, 1, -1, 1, -1};
    std::vector<float> times_of_day = {
        9.5f, 9.75f, 10.0f, 10.25f,       // OPEN: mixed 50/50
        14.0f, 14.5f, 15.0f, 15.5f         // CLOSE: mixed 50/50
    };

    float cond_entropy = oracle_comparison::conditional_entropy(labels, times_of_day);
    // Should be close to max entropy (log2(2) = 1.0 for binary labels)
    EXPECT_GT(cond_entropy, 0.5f);
}

// ===========================================================================
// 6. Expectancy comparison
// ===========================================================================

TEST_F(OracleComparisonTest, ExpectancyComparison) {
    // Spec: "Expectancy after costs"
    OracleComparison comparison(fth_result, tb_result);

    float fth_exp = comparison.expectancy_fth();
    float tb_exp = comparison.expectancy_tb();

    EXPECT_FLOAT_EQ(fth_exp, fth_result.expectancy);
    EXPECT_FLOAT_EQ(tb_exp, tb_result.expectancy);
}

TEST_F(OracleComparisonTest, PreferHigherExpectancyTimesTradeCount) {
    // Spec: "If both pass, prefer higher expectancy × trade_count"
    OracleComparison comparison(fth_result, tb_result);

    float fth_score = comparison.score_fth();
    float tb_score = comparison.score_tb();

    // score = expectancy × trade_count
    float expected_fth = fth_result.expectancy * static_cast<float>(fth_result.total_trades);
    float expected_tb = tb_result.expectancy * static_cast<float>(tb_result.total_trades);

    EXPECT_NEAR(fth_score, expected_fth, 0.01f);
    EXPECT_NEAR(tb_score, expected_tb, 0.01f);
}

TEST_F(OracleComparisonTest, PreferredMethodReturnsHigherScore) {
    OracleComparison comparison(fth_result, tb_result);

    auto preferred = comparison.preferred_method();

    float fth_score = fth_result.expectancy * static_cast<float>(fth_result.total_trades);
    float tb_score = tb_result.expectancy * static_cast<float>(tb_result.total_trades);

    if (fth_score > tb_score) {
        EXPECT_EQ(preferred, OracleConfig::LabelMethod::FIRST_TO_HIT);
    } else {
        EXPECT_EQ(preferred, OracleConfig::LabelMethod::TRIPLE_BARRIER);
    }
}

// ===========================================================================
// 7. Regime dependence comparison
// ===========================================================================

TEST_F(OracleComparisonTest, RegimeDependenceComparison) {
    // Spec: "Regime dependence"
    OracleComparison comparison(fth_result, tb_result);

    // Should be able to compare stability scores between the two methods
    float fth_stability = 0.6f;  // robust
    float tb_stability = 0.3f;   // regime-dependent

    auto dep_comparison = comparison.regime_dependence(fth_stability, tb_stability);

    EXPECT_FALSE(dep_comparison.empty());
    // The comparison should indicate which method is more regime-dependent
}

// ===========================================================================
// 8. ComparisonReport — structured output
// ===========================================================================

TEST_F(OracleComparisonTest, GenerateReport) {
    OracleComparison comparison(fth_result, tb_result);

    auto report = comparison.generate_report();

    // Report should have key comparison fields
    EXPECT_GT(report.fth_trade_count, 0);
    EXPECT_GT(report.tb_trade_count, 0);
    EXPECT_TRUE(std::isfinite(report.fth_expectancy));
    EXPECT_TRUE(std::isfinite(report.tb_expectancy));
    EXPECT_TRUE(std::isfinite(report.fth_win_rate));
    EXPECT_TRUE(std::isfinite(report.tb_win_rate));
}

TEST_F(OracleComparisonTest, ReportHasLabelCorrelation) {
    // Need to compute label-return correlation for both methods
    OracleComparison comparison(fth_result, tb_result);
    auto report = comparison.generate_report();

    EXPECT_TRUE(std::isfinite(report.fth_label_return_corr));
    EXPECT_TRUE(std::isfinite(report.tb_label_return_corr));
}

// ===========================================================================
// 9. Both methods on identical bars (validation gate)
// ===========================================================================

TEST_F(OracleComparisonTest, RunBothMethodsOnSameBars) {
    // Spec: "Assert: Both labeling methods (first-to-hit, triple barrier) run
    //         on identical bar sequences"
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    ExecutionCosts costs{};

    OracleConfig cfg_fth{};
    cfg_fth.target_ticks = 4;
    cfg_fth.stop_ticks = 2;
    cfg_fth.volume_horizon = 200;
    cfg_fth.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;

    OracleConfig cfg_tb = cfg_fth;
    cfg_tb.label_method = OracleConfig::LabelMethod::TRIPLE_BARRIER;

    auto result_pair = oracle_comparison::run_both(bars, cfg_fth, cfg_tb, costs);

    // Both should have run on the same bar count
    EXPECT_GE(result_pair.first.total_trades, 0);
    EXPECT_GE(result_pair.second.total_trades, 0);
}

TEST_F(OracleComparisonTest, RunBothReturnsValidResults) {
    auto bars = make_bar_series(4500.0f, 4502.0f, 50, 50);
    ExecutionCosts costs{};

    OracleConfig cfg_fth{};
    cfg_fth.target_ticks = 4;
    cfg_fth.stop_ticks = 2;
    cfg_fth.volume_horizon = 200;
    cfg_fth.label_method = OracleConfig::LabelMethod::FIRST_TO_HIT;

    OracleConfig cfg_tb = cfg_fth;
    cfg_tb.label_method = OracleConfig::LabelMethod::TRIPLE_BARRIER;

    auto result_pair = oracle_comparison::run_both(bars, cfg_fth, cfg_tb, costs);

    // Both results should have consistent internal state
    EXPECT_EQ(result_pair.first.total_trades,
              result_pair.first.winning_trades + result_pair.first.losing_trades);
    EXPECT_EQ(result_pair.second.total_trades,
              result_pair.second.winning_trades + result_pair.second.losing_trades);
}

// ===========================================================================
// 10. Edge cases
// ===========================================================================

TEST_F(OracleComparisonTest, EmptyResultsComparison) {
    BacktestResult empty1{};
    BacktestResult empty2{};

    OracleComparison comparison(empty1, empty2);
    auto report = comparison.generate_report();

    EXPECT_EQ(report.fth_trade_count, 0);
    EXPECT_EQ(report.tb_trade_count, 0);
}

TEST_F(OracleComparisonTest, OneEmptyResultComparison) {
    BacktestResult empty{};

    OracleComparison comparison(fth_result, empty);
    auto report = comparison.generate_report();

    EXPECT_GT(report.fth_trade_count, 0);
    EXPECT_EQ(report.tb_trade_count, 0);
}
