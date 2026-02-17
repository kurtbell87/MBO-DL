#pragma once

#include "backtest/oracle_replay.hpp"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Assessment — result of a go/no-go evaluation
// ---------------------------------------------------------------------------
struct Assessment {
    bool passed = false;
    bool expectancy_passed = false;
    bool profit_factor_passed = false;
    bool win_rate_passed = false;
    bool drawdown_passed = false;
    bool trade_count_passed = false;
    bool oos_pnl_passed = false;

    float actual_expectancy = 0.0f;
    float actual_profit_factor = 0.0f;
    float actual_win_rate = 0.0f;
    float actual_max_drawdown = 0.0f;
    float actual_trades_per_day = 0.0f;

    std::string decision = "NO-GO";
};

// ---------------------------------------------------------------------------
// SuccessCriteria — threshold-based go/no-go framework (§9.4)
// ---------------------------------------------------------------------------
struct SuccessCriteria {
    float min_expectancy = 0.50f;
    float min_profit_factor = 1.3f;
    float min_win_rate = 0.45f;
    float max_drawdown_multiple = 50.0f;
    float min_trades_per_day = 10.0f;
    float max_safety_cap_fraction = 0.01f;

    Assessment evaluate(const BacktestResult& result) const {
        Assessment a{};
        a.actual_expectancy = result.expectancy;
        a.actual_profit_factor = result.profit_factor;
        a.actual_win_rate = result.win_rate;
        a.actual_max_drawdown = result.max_drawdown;
        a.actual_trades_per_day = result.trades_per_day;

        a.expectancy_passed = result.expectancy > min_expectancy;
        a.profit_factor_passed = result.profit_factor > min_profit_factor;
        a.win_rate_passed = result.win_rate > min_win_rate;

        float max_allowed_dd = max_drawdown_multiple * result.expectancy;
        a.drawdown_passed = result.max_drawdown < max_allowed_dd;

        a.trade_count_passed = result.trades_per_day > min_trades_per_day;

        a.passed = a.expectancy_passed && a.profit_factor_passed &&
                   a.win_rate_passed && a.drawdown_passed && a.trade_count_passed;
        a.decision = a.passed ? "GO" : "NO-GO";

        return a;
    }

    Assessment evaluate_with_oos(const BacktestResult& is_result,
                                  const BacktestResult& oos_result) const {
        auto a = evaluate(is_result);
        a.oos_pnl_passed = oos_result.net_pnl > 0.0f;
        a.passed = a.passed && a.oos_pnl_passed;
        a.decision = a.passed ? "GO" : "NO-GO";
        return a;
    }

    bool safety_cap_ok(const BacktestResult& result) const {
        return result.safety_cap_fraction < max_safety_cap_fraction;
    }
};

// ---------------------------------------------------------------------------
// OracleDiagnosis — diagnosis output
// ---------------------------------------------------------------------------
struct OracleDiagnosis {
    std::vector<std::string> recommendations;
    bool continue_to_phase4 = true;
};

// ---------------------------------------------------------------------------
// oracle_diagnosis namespace — failure diagnosis & Pareto frontier (§9.5)
// ---------------------------------------------------------------------------
namespace oracle_diagnosis {

inline OracleDiagnosis diagnose(const BacktestResult& result) {
    OracleDiagnosis diag{};
    diag.continue_to_phase4 = true;

    // Costs too high: gross PnL positive but net PnL negative
    if (result.gross_pnl > 0.0f && result.net_pnl < 0.0f) {
        diag.recommendations.push_back(
            "Costs too high for scale — try larger targets (20, 40 ticks)");
    }

    // Safety cap rate high — noisy microstructure
    if (result.safety_cap_fraction > 0.01f) {
        diag.recommendations.push_back(
            "MES microstructure too noisy — filter: only label when spread < 2 ticks");
    }

    // Very bad results — naive threshold
    if (result.expectancy < -0.5f && result.win_rate < 0.30f) {
        diag.recommendations.push_back(
            "Oracle threshold logic too naive — proceed to feature discovery on returns");
    }

    // General poor performance
    if (result.net_pnl < 0.0f && diag.recommendations.empty()) {
        diag.recommendations.push_back(
            "Negative net PnL — feature discovery may reveal better labels");
    }

    // Always suggest continuing to Phase 4
    diag.recommendations.push_back(
        "Continue to Phase 4 — feature discovery may improve results");

    return diag;
}

// Pareto frontier computation — templated to work with any SweepResult-like struct
template <typename T>
bool dominates(const T& a, const T& b) {
    return (a.expectancy >= b.expectancy &&
            a.trade_count >= b.trade_count &&
            a.max_drawdown <= b.max_drawdown) &&
           (a.expectancy > b.expectancy ||
            a.trade_count > b.trade_count ||
            a.max_drawdown < b.max_drawdown);
}

template <typename T>
std::vector<T> pareto_frontier(const std::vector<T>& results) {
    std::vector<T> frontier;
    for (size_t i = 0; i < results.size(); ++i) {
        bool is_dominated = false;
        for (size_t j = 0; j < results.size(); ++j) {
            if (i != j && dominates(results[j], results[i])) {
                is_dominated = true;
                break;
            }
        }
        if (!is_dominated) {
            frontier.push_back(results[i]);
        }
    }
    return frontier;
}

}  // namespace oracle_diagnosis
