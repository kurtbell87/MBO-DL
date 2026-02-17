#pragma once

#include "backtest/execution_costs.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// ComparisonReport — structured comparison output
// ---------------------------------------------------------------------------
struct ComparisonReport {
    int fth_trade_count = 0;
    int tb_trade_count = 0;
    float fth_expectancy = 0.0f;
    float tb_expectancy = 0.0f;
    float fth_win_rate = 0.0f;
    float tb_win_rate = 0.0f;
    float fth_label_return_corr = 0.0f;
    float tb_label_return_corr = 0.0f;
};

// ---------------------------------------------------------------------------
// oracle_comparison namespace — free functions
// ---------------------------------------------------------------------------
namespace oracle_comparison {

inline float label_stability(const std::vector<int>& labels) {
    if (labels.size() < 2) return 0.0f;
    int agreements = 0;
    for (size_t i = 1; i < labels.size(); ++i) {
        if (labels[i] == labels[i - 1]) agreements++;
    }
    return static_cast<float>(agreements) / static_cast<float>(labels.size() - 1);
}

inline float label_return_correlation(const std::vector<int>& labels,
                                       const std::vector<float>& returns) {
    size_t n = std::min(labels.size(), returns.size());
    if (n == 0) return 0.0f;

    float sum_l = 0.0f, sum_r = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_l += static_cast<float>(labels[i]);
        sum_r += returns[i];
    }
    float mean_l = sum_l / static_cast<float>(n);
    float mean_r = sum_r / static_cast<float>(n);

    float cov = 0.0f, var_l = 0.0f, var_r = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float dl = static_cast<float>(labels[i]) - mean_l;
        float dr = returns[i] - mean_r;
        cov += dl * dr;
        var_l += dl * dl;
        var_r += dr * dr;
    }

    float denom = std::sqrt(var_l * var_r);
    if (denom < 1e-10f) {
        // Degenerate case: one variable has zero variance.
        // Use sign-agreement heuristic: if all labels share one sign,
        // report direction based on mean return alignment.
        if (var_l < 1e-10f && mean_l != 0.0f && var_r > 1e-10f) {
            return (mean_l * mean_r > 0.0f) ? 1.0f : -1.0f;
        }
        return 0.0f;
    }
    return cov / denom;
}

// Conditional entropy H(Label | TimeOfDay)
// Bucket time_of_day into sessions: OPEN [9.5,10.5), MID [10.5,14.0), CLOSE [14.0,16.0)
inline float conditional_entropy(const std::vector<int>& labels,
                                  const std::vector<float>& times_of_day) {
    size_t n = std::min(labels.size(), times_of_day.size());
    if (n == 0) return 0.0f;

    // Bucket: 0=OPEN, 1=MID, 2=CLOSE
    auto bucket = [](float tod) -> int {
        if (tod < 10.5f) return 0;
        if (tod < 14.0f) return 1;
        return 2;
    };

    // Count (bucket, label) pairs and bucket totals
    std::map<int, std::map<int, int>> bucket_label_counts;
    std::map<int, int> bucket_totals;

    for (size_t i = 0; i < n; ++i) {
        int b = bucket(times_of_day[i]);
        bucket_label_counts[b][labels[i]]++;
        bucket_totals[b]++;
    }

    // H(Label | Bucket) = sum over buckets of P(bucket) * H(Label | bucket)
    float total = static_cast<float>(n);
    float cond_h = 0.0f;

    for (const auto& [b, label_counts] : bucket_label_counts) {
        float bucket_total = static_cast<float>(bucket_totals[b]);
        float p_bucket = bucket_total / total;

        float h_given_bucket = 0.0f;
        for (const auto& [label, count] : label_counts) {
            float p_label = static_cast<float>(count) / bucket_total;
            if (p_label > 0.0f) {
                h_given_bucket -= p_label * std::log2(p_label);
            }
        }
        cond_h += p_bucket * h_given_bucket;
    }

    return cond_h;
}

// Run both labeling methods on identical bars
inline std::pair<BacktestResult, BacktestResult>
run_both(const std::vector<Bar>& bars,
         const OracleConfig& cfg_fth,
         const OracleConfig& cfg_tb,
         const ExecutionCosts& costs) {
    OracleReplay replay_fth(cfg_fth, costs);
    OracleReplay replay_tb(cfg_tb, costs);

    auto result_fth = replay_fth.run(bars);
    auto result_tb = replay_tb.run(bars);

    return {result_fth, result_tb};
}

}  // namespace oracle_comparison

// ---------------------------------------------------------------------------
// OracleComparison — compare FTH vs TB results
// ---------------------------------------------------------------------------
class OracleComparison {
public:
    OracleComparison(const BacktestResult& fth, const BacktestResult& tb)
        : fth_(fth), tb_(tb) {}

    std::map<int, int> label_distribution_fth() const {
        return fth_.label_counts;
    }

    std::map<int, int> label_distribution_tb() const {
        return tb_.label_counts;
    }

    float expectancy_fth() const { return fth_.expectancy; }
    float expectancy_tb()  const { return tb_.expectancy; }

    float score_fth() const {
        return fth_.expectancy * static_cast<float>(fth_.total_trades);
    }

    float score_tb() const {
        return tb_.expectancy * static_cast<float>(tb_.total_trades);
    }

    OracleConfig::LabelMethod preferred_method() const {
        if (score_fth() > score_tb()) {
            return OracleConfig::LabelMethod::FIRST_TO_HIT;
        }
        return OracleConfig::LabelMethod::TRIPLE_BARRIER;
    }

    std::string regime_dependence(float fth_stability, float tb_stability) const {
        if (fth_stability > tb_stability) {
            return "FIRST_TO_HIT is more robust across regimes";
        } else if (tb_stability > fth_stability) {
            return "TRIPLE_BARRIER is more robust across regimes";
        }
        return "Both methods have similar regime dependence";
    }

    ComparisonReport generate_report() const {
        ComparisonReport report{};
        report.fth_trade_count = fth_.total_trades;
        report.tb_trade_count = tb_.total_trades;
        report.fth_expectancy = fth_.expectancy;
        report.tb_expectancy = tb_.expectancy;
        report.fth_win_rate = fth_.win_rate;
        report.tb_win_rate = tb_.win_rate;

        // Label-return correlation: computed from trades
        report.fth_label_return_corr = compute_trade_corr(fth_);
        report.tb_label_return_corr = compute_trade_corr(tb_);

        return report;
    }

private:
    BacktestResult fth_;
    BacktestResult tb_;

    static float compute_trade_corr(const BacktestResult& result) {
        if (result.trades.empty()) return 0.0f;
        std::vector<int> labels;
        std::vector<float> returns;
        labels.reserve(result.trades.size());
        returns.reserve(result.trades.size());
        for (const auto& t : result.trades) {
            labels.push_back(t.direction);
            returns.push_back(t.net_pnl);
        }
        return oracle_comparison::label_return_correlation(labels, returns);
    }
};
