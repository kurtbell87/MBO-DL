// backtest_criteria_test.cpp — TDD RED phase tests for Success Criteria & Go/No-Go
// Spec: .kit/docs/multi-day-backtest.md §Success Criteria (§9.4), §Oracle Failure Diagnosis (§9.5)
//
// Tests the go/no-go assessment framework: threshold checks for expectancy,
// profit factor, win rate, OOS net PnL, max drawdown, and trade count.
// Also tests oracle failure diagnosis and Pareto frontier computation.
//
// Headers below do not exist yet — all tests must fail to compile.

#include <gtest/gtest.h>

// Header that the implementation must provide:
#include "backtest/success_criteria.hpp"

// Already-existing headers from prior phases:
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"

#include <cmath>
#include <string>
#include <vector>

// ===========================================================================
// Helpers
// ===========================================================================
namespace {

// Build a BacktestResult with specified aggregate metrics.
BacktestResult make_result(float expectancy, float profit_factor, float win_rate,
                           float net_pnl, float max_drawdown, int total_trades,
                           int trading_days = 10) {
    BacktestResult result{};
    result.expectancy = expectancy;
    result.profit_factor = profit_factor;
    result.win_rate = win_rate;
    result.net_pnl = net_pnl;
    result.max_drawdown = max_drawdown;
    result.total_trades = total_trades;
    result.winning_trades = static_cast<int>(win_rate * total_trades);
    result.losing_trades = total_trades - result.winning_trades;
    result.trades_per_day = static_cast<float>(total_trades)
                            / static_cast<float>(trading_days);
    return result;
}

}  // namespace

// ===========================================================================
// 1. SuccessCriteria — threshold constants from §9.4
// ===========================================================================
class SuccessCriteriaTest : public ::testing::Test {};

TEST_F(SuccessCriteriaTest, DefaultExpectancyThreshold) {
    // Spec: "Net expectancy > $0.50 per trade"
    SuccessCriteria criteria{};
    EXPECT_FLOAT_EQ(criteria.min_expectancy, 0.50f);
}

TEST_F(SuccessCriteriaTest, DefaultProfitFactorThreshold) {
    // Spec: "Profit factor > 1.3"
    SuccessCriteria criteria{};
    EXPECT_FLOAT_EQ(criteria.min_profit_factor, 1.3f);
}

TEST_F(SuccessCriteriaTest, DefaultWinRateThreshold) {
    // Spec: "Win rate > 45%"
    SuccessCriteria criteria{};
    EXPECT_FLOAT_EQ(criteria.min_win_rate, 0.45f);
}

TEST_F(SuccessCriteriaTest, DefaultMaxDrawdownMultiple) {
    // Spec: "Max drawdown < 50 × expectancy"
    SuccessCriteria criteria{};
    EXPECT_FLOAT_EQ(criteria.max_drawdown_multiple, 50.0f);
}

TEST_F(SuccessCriteriaTest, DefaultMinTradesPerDay) {
    // Spec: "Trade count > 10 per day avg"
    SuccessCriteria criteria{};
    EXPECT_FLOAT_EQ(criteria.min_trades_per_day, 10.0f);
}

// ===========================================================================
// 2. Go/No-Go evaluation — all pass
// ===========================================================================
class GoNoGoTest : public ::testing::Test {
protected:
    SuccessCriteria criteria;

    void SetUp() override {
        criteria = SuccessCriteria{};
    }
};

TEST_F(GoNoGoTest, AllCriteriaPass) {
    // Build a result that passes all thresholds
    auto result = make_result(
        1.50f,    // expectancy > $0.50 ✓
        2.0f,     // profit_factor > 1.3 ✓
        0.55f,    // win_rate > 45% ✓
        1500.0f,  // net_pnl > 0 (will check OOS separately)
        50.0f,    // max_drawdown < 50 × 1.50 = 75 ✓
        150,      // total_trades
        10        // 150/10 = 15 trades/day > 10 ✓
    );

    auto assessment = criteria.evaluate(result);
    EXPECT_TRUE(assessment.passed);
}

TEST_F(GoNoGoTest, GoDecision) {
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_EQ(assessment.decision, "GO");
}

// ===========================================================================
// 3. Individual criterion failures
// ===========================================================================

TEST_F(GoNoGoTest, FailsOnLowExpectancy) {
    // Spec: "Net expectancy > $0.50 per trade"
    auto result = make_result(0.30f, 2.0f, 0.55f, 300.0f, 10.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
    EXPECT_FALSE(assessment.expectancy_passed);
}

TEST_F(GoNoGoTest, FailsOnLowProfitFactor) {
    // Spec: "Profit factor > 1.3"
    auto result = make_result(1.50f, 1.1f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
    EXPECT_FALSE(assessment.profit_factor_passed);
}

TEST_F(GoNoGoTest, FailsOnLowWinRate) {
    // Spec: "Win rate > 45%"
    auto result = make_result(1.50f, 2.0f, 0.40f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
    EXPECT_FALSE(assessment.win_rate_passed);
}

TEST_F(GoNoGoTest, FailsOnHighDrawdown) {
    // Spec: "Max drawdown < 50 × expectancy"
    // expectancy = 1.50, max allowed drawdown = 50 × 1.50 = 75
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 100.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
    EXPECT_FALSE(assessment.drawdown_passed);
}

TEST_F(GoNoGoTest, FailsOnLowTradeCount) {
    // Spec: "Trade count > 10 per day avg"
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 50, 10);
    // 50/10 = 5 trades/day < 10
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
    EXPECT_FALSE(assessment.trade_count_passed);
}

// ===========================================================================
// 4. OOS net PnL check
// ===========================================================================

TEST_F(GoNoGoTest, OOSNetPnlPositive) {
    // Spec: "OOS net PnL > 0 — In-sample could be overfit"
    auto is_result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto oos_result = make_result(0.80f, 1.5f, 0.50f, 200.0f, 30.0f, 80, 10);

    auto assessment = criteria.evaluate_with_oos(is_result, oos_result);
    EXPECT_TRUE(assessment.oos_pnl_passed);
}

TEST_F(GoNoGoTest, OOSNetPnlNegativeFails) {
    auto is_result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto oos_result = make_result(-0.50f, 0.8f, 0.40f, -100.0f, 80.0f, 80, 10);

    auto assessment = criteria.evaluate_with_oos(is_result, oos_result);
    EXPECT_FALSE(assessment.oos_pnl_passed);
}

TEST_F(GoNoGoTest, FullEvaluationWithOOS) {
    auto is_result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto oos_result = make_result(0.80f, 1.5f, 0.50f, 200.0f, 30.0f, 120, 10);

    auto assessment = criteria.evaluate_with_oos(is_result, oos_result);

    // All criteria should pass
    EXPECT_TRUE(assessment.passed);
    EXPECT_TRUE(assessment.oos_pnl_passed);
}

// ===========================================================================
// 5. Assessment — detailed breakdown
// ===========================================================================

TEST_F(GoNoGoTest, AssessmentHasAllFields) {
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);

    // Should have individual pass/fail for each criterion
    EXPECT_TRUE(assessment.expectancy_passed);
    EXPECT_TRUE(assessment.profit_factor_passed);
    EXPECT_TRUE(assessment.win_rate_passed);
    EXPECT_TRUE(assessment.drawdown_passed);
    EXPECT_TRUE(assessment.trade_count_passed);
}

TEST_F(GoNoGoTest, AssessmentHasActualValues) {
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);

    EXPECT_FLOAT_EQ(assessment.actual_expectancy, 1.50f);
    EXPECT_FLOAT_EQ(assessment.actual_profit_factor, 2.0f);
    EXPECT_FLOAT_EQ(assessment.actual_win_rate, 0.55f);
    EXPECT_FLOAT_EQ(assessment.actual_max_drawdown, 50.0f);
    EXPECT_NEAR(assessment.actual_trades_per_day, 15.0f, 0.1f);
}

TEST_F(GoNoGoTest, AssessmentHasDecision) {
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_EQ(assessment.decision, "GO");

    auto bad_result = make_result(0.10f, 0.8f, 0.30f, -100.0f, 200.0f, 30, 10);
    auto bad_assessment = criteria.evaluate(bad_result);
    EXPECT_EQ(bad_assessment.decision, "NO-GO");
}

// ===========================================================================
// 6. Boundary conditions
// ===========================================================================

TEST_F(GoNoGoTest, ExpectancyExactlyAtThreshold) {
    // Exactly 0.50 — spec says "> $0.50", so exactly 0.50 should fail
    auto result = make_result(0.50f, 2.0f, 0.55f, 500.0f, 20.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.expectancy_passed);
}

TEST_F(GoNoGoTest, ProfitFactorExactlyAtThreshold) {
    // Exactly 1.3 — spec says "> 1.3", so exactly 1.3 should fail
    auto result = make_result(1.50f, 1.3f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.profit_factor_passed);
}

TEST_F(GoNoGoTest, WinRateExactlyAtThreshold) {
    // Exactly 45% — spec says "> 45%", so exactly 0.45 should fail
    auto result = make_result(1.50f, 2.0f, 0.45f, 1500.0f, 50.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.win_rate_passed);
}

TEST_F(GoNoGoTest, TradesPerDayExactlyAtThreshold) {
    // Exactly 10 — spec says "> 10", so exactly 10 should fail
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 100, 10);
    // 100/10 = 10.0 trades/day
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.trade_count_passed);
}

TEST_F(GoNoGoTest, DrawdownExactlyAtThreshold) {
    // max_drawdown exactly = 50 × expectancy — spec says "< 50 × expectancy"
    auto result = make_result(1.00f, 2.0f, 0.55f, 1000.0f, 50.0f, 150, 10);
    // max_drawdown = 50 = 50 × 1.00 — exactly at threshold, should fail
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.drawdown_passed);
}

// ===========================================================================
// 7. Edge cases
// ===========================================================================

TEST_F(GoNoGoTest, ZeroTradesAlwaysFails) {
    auto result = make_result(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
}

TEST_F(GoNoGoTest, NegativeExpectancyFails) {
    auto result = make_result(-1.0f, 0.5f, 0.30f, -500.0f, 200.0f, 150, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
    EXPECT_FALSE(assessment.expectancy_passed);
}

TEST_F(GoNoGoTest, MultipleCriteriaCanFailSimultaneously) {
    auto result = make_result(0.10f, 0.5f, 0.20f, -500.0f, 500.0f, 30, 10);
    auto assessment = criteria.evaluate(result);
    EXPECT_FALSE(assessment.passed);
    EXPECT_FALSE(assessment.expectancy_passed);
    EXPECT_FALSE(assessment.profit_factor_passed);
    EXPECT_FALSE(assessment.win_rate_passed);
    EXPECT_FALSE(assessment.trade_count_passed);
}

// ===========================================================================
// 8. Oracle failure diagnosis (§9.5)
// ===========================================================================
class OracleFailureDiagnosisTest : public ::testing::Test {};

TEST_F(OracleFailureDiagnosisTest, DiagnoseHighCosts) {
    // Spec: "Costs too high for scale → try larger targets (20, 40 ticks)"
    auto result = make_result(-0.20f, 0.9f, 0.50f, -200.0f, 100.0f, 150, 10);

    auto diagnosis = oracle_diagnosis::diagnose(result);

    // Should suggest larger targets if gross PnL is positive but net PnL is negative
    EXPECT_FALSE(diagnosis.recommendations.empty());
}

TEST_F(OracleFailureDiagnosisTest, DiagnoseNoisyMicrostructure) {
    // Spec: "MES microstructure too noisy → filter: only label when spread < 2 ticks"
    auto result = make_result(-0.50f, 0.7f, 0.35f, -500.0f, 200.0f, 150, 10);
    result.safety_cap_fraction = 0.05f;  // high safety cap rate

    auto diagnosis = oracle_diagnosis::diagnose(result);
    EXPECT_FALSE(diagnosis.recommendations.empty());
}

TEST_F(OracleFailureDiagnosisTest, DiagnoseNaiveThreshold) {
    // Spec: "Oracle threshold logic too naive → proceed to feature discovery on returns"
    auto result = make_result(-1.0f, 0.3f, 0.25f, -1000.0f, 500.0f, 100, 10);

    auto diagnosis = oracle_diagnosis::diagnose(result);
    EXPECT_FALSE(diagnosis.recommendations.empty());
}

TEST_F(OracleFailureDiagnosisTest, DiagnosisSuggestsContinueToPhase4) {
    // Spec: "Feature discovery may reveal better labels → continue to Phase 4 anyway"
    auto result = make_result(-0.80f, 0.6f, 0.30f, -800.0f, 300.0f, 100, 10);

    auto diagnosis = oracle_diagnosis::diagnose(result);

    // Should always suggest continuing to Phase 4
    EXPECT_TRUE(diagnosis.continue_to_phase4);
}

// ===========================================================================
// 9. Pareto frontier (if oracle fails default config)
// ===========================================================================
class ParetoFrontierTest : public ::testing::Test {};

TEST_F(ParetoFrontierTest, ComputeParetoFrontier) {
    // Spec: "Pareto frontier of (expectancy, trade_count, max_drawdown) across parameter sweep"
    struct SweepResult {
        float expectancy;
        int trade_count;
        float max_drawdown;
        OracleConfig config;
    };

    std::vector<SweepResult> sweep_results;
    sweep_results.push_back({1.0f, 100, 30.0f, OracleConfig{}});
    sweep_results.push_back({2.0f, 50, 60.0f, OracleConfig{}});
    sweep_results.push_back({0.5f, 200, 20.0f, OracleConfig{}});
    sweep_results.push_back({1.5f, 80, 25.0f, OracleConfig{}});
    sweep_results.push_back({0.8f, 150, 100.0f, OracleConfig{}});  // dominated

    auto frontier = oracle_diagnosis::pareto_frontier(sweep_results);

    // Pareto frontier should be non-empty
    EXPECT_FALSE(frontier.empty());

    // Frontier should be smaller than or equal to the input
    EXPECT_LE(frontier.size(), sweep_results.size());

    // Dominated points should not be on the frontier
    // The point (0.8, 150, 100) is dominated by (1.5, 80, 25) in expectancy and drawdown
    // (though trade_count is higher, drawdown is much worse)
    // Exact frontier depends on dominance definition, but it should have at least 2 points
    EXPECT_GE(frontier.size(), 2u);
}

TEST_F(ParetoFrontierTest, SinglePointIsOnFrontier) {
    struct SweepResult {
        float expectancy;
        int trade_count;
        float max_drawdown;
        OracleConfig config;
    };

    std::vector<SweepResult> sweep_results;
    sweep_results.push_back({1.0f, 100, 30.0f, OracleConfig{}});

    auto frontier = oracle_diagnosis::pareto_frontier(sweep_results);
    EXPECT_EQ(frontier.size(), 1u);
}

TEST_F(ParetoFrontierTest, EmptyInputReturnsEmptyFrontier) {
    struct SweepResult {
        float expectancy;
        int trade_count;
        float max_drawdown;
        OracleConfig config;
    };

    std::vector<SweepResult> sweep_results;
    auto frontier = oracle_diagnosis::pareto_frontier(sweep_results);
    EXPECT_TRUE(frontier.empty());
}

// ===========================================================================
// 10. Safety cap validation gate
// ===========================================================================

TEST_F(GoNoGoTest, SafetyCapTriggerRateBelow1Percent) {
    // Spec: "Assert: Safety cap trigger rate < 1% during RTH"
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    result.safety_cap_fraction = 0.005f;  // 0.5% < 1%

    EXPECT_TRUE(criteria.safety_cap_ok(result));
}

TEST_F(GoNoGoTest, SafetyCapTriggerRateAbove1PercentWarning) {
    // Spec: "if not, volume_horizon needs recalibration"
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    result.safety_cap_fraction = 0.02f;  // 2% > 1%

    EXPECT_FALSE(criteria.safety_cap_ok(result));
}

TEST_F(GoNoGoTest, SafetyCapExactlyAtOnePercent) {
    // Exactly 1% — spec says "< 1%", so 1% should fail
    auto result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    result.safety_cap_fraction = 0.01f;

    EXPECT_FALSE(criteria.safety_cap_ok(result));
}

// ===========================================================================
// 11. At-least-one labeling method must pass
// ===========================================================================

TEST_F(GoNoGoTest, AtLeastOneLabelingMethodPasses) {
    // Spec: "At least one labeling method must pass."
    auto fth_result = make_result(1.50f, 2.0f, 0.55f, 1500.0f, 50.0f, 150, 10);
    auto tb_result = make_result(0.10f, 0.8f, 0.30f, -100.0f, 200.0f, 30, 10);

    auto fth_assessment = criteria.evaluate(fth_result);
    auto tb_assessment = criteria.evaluate(tb_result);

    bool any_passed = fth_assessment.passed || tb_assessment.passed;
    EXPECT_TRUE(any_passed);
}

TEST_F(GoNoGoTest, BothLabelingMethodsFail) {
    auto fth_result = make_result(0.10f, 0.8f, 0.30f, -100.0f, 200.0f, 30, 10);
    auto tb_result = make_result(0.20f, 0.9f, 0.35f, -50.0f, 150.0f, 40, 10);

    auto fth_assessment = criteria.evaluate(fth_result);
    auto tb_assessment = criteria.evaluate(tb_result);

    bool any_passed = fth_assessment.passed || tb_assessment.passed;
    EXPECT_FALSE(any_passed);
}
