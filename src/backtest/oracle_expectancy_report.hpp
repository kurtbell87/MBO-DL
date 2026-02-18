#pragma once

#include "backtest/oracle_replay.hpp"
#include "backtest/multi_day_runner.hpp"
#include "backtest/trade_record.hpp"

#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// OracleExpectancyReport — aggregated oracle expectancy across days/quarters
// ---------------------------------------------------------------------------
struct OracleExpectancyReport {
    int days_processed = 0;
    int days_skipped = 0;
    BacktestResult first_to_hit;
    BacktestResult triple_barrier;
    std::map<std::string, BacktestResult> fth_per_quarter;
    std::map<std::string, BacktestResult> tb_per_quarter;
};

// ---------------------------------------------------------------------------
// oracle_expectancy namespace — aggregation and serialization
// ---------------------------------------------------------------------------
namespace oracle_expectancy {

// Remove quarters with no trades from a per-quarter result map.
inline void remove_empty_quarters(std::map<std::string, BacktestResult>& quarters) {
    for (auto it = quarters.begin(); it != quarters.end();) {
        if (it->second.total_trades == 0 && it->second.trades.empty()) {
            it = quarters.erase(it);
        } else {
            ++it;
        }
    }
}

// Quarter assignment based on MES 2022 contract boundaries.
// Q1: 20220103–20220318 (MESH2), Q2: 20220319–20220617 (MESM2),
// Q3: 20220618–20220916 (MESU2), Q4: 20220917–20221230 (MESZ2).
inline std::string date_to_quarter(int date) {
    if (date <= 20220318) return "Q1";
    if (date <= 20220617) return "Q2";
    if (date <= 20220916) return "Q3";
    return "Q4";
}

// Aggregate a single DayResult into a BacktestResult accumulator.
inline void accumulate(BacktestResult& agg, const DayResult& day) {
    agg.total_trades += day.result.total_trades;
    agg.winning_trades += day.result.winning_trades;
    agg.losing_trades += day.result.losing_trades;
    agg.net_pnl += day.result.net_pnl;
    agg.gross_pnl += day.result.gross_pnl;

    // Collect individual trades
    agg.trades.insert(agg.trades.end(),
                      day.result.trades.begin(), day.result.trades.end());

    // Accumulate exit reason counts
    for (const auto& [reason, count] : day.result.exit_reason_counts) {
        agg.exit_reason_counts[reason] += count;
    }

    // Track daily PnL
    agg.daily_pnl.push_back(day.result.net_pnl);
}

// Recompute derived metrics — delegates to backtest_util::recompute_derived.
inline void recompute_derived(BacktestResult& agg, int active_days) {
    backtest_util::recompute_derived(agg, active_days);
}

// Aggregate day results into an OracleExpectancyReport with per-quarter splitting.
inline OracleExpectancyReport aggregate_day_results(
    const std::vector<DayResult>& fth_results,
    const std::vector<DayResult>& tb_results,
    const std::vector<int>& dates) {

    OracleExpectancyReport report{};

    // Track active days per quarter for trades_per_day recomputation
    std::map<std::string, int> fth_quarter_days;
    std::map<std::string, int> tb_quarter_days;

    // Process FTH results
    for (size_t i = 0; i < fth_results.size(); ++i) {
        const auto& day = fth_results[i];
        if (day.skipped) {
            ++report.days_skipped;
            continue;
        }
        ++report.days_processed;

        accumulate(report.first_to_hit, day);

        std::string quarter = date_to_quarter(day.date);
        accumulate(report.fth_per_quarter[quarter], day);
        fth_quarter_days[quarter]++;
    }

    // Process TB results (days_processed/skipped already counted from FTH)
    // Reset counters — we only count once from whichever side the test expects.
    // But looking at the tests, days_processed/skipped are counted from either
    // side equivalently (they have same dates). We counted from FTH above.
    for (size_t i = 0; i < tb_results.size(); ++i) {
        const auto& day = tb_results[i];
        if (day.skipped) continue;

        accumulate(report.triple_barrier, day);

        std::string quarter = date_to_quarter(day.date);
        accumulate(report.tb_per_quarter[quarter], day);
        tb_quarter_days[quarter]++;
    }

    // Recompute derived metrics for overall aggregates
    recompute_derived(report.first_to_hit, report.days_processed);
    recompute_derived(report.triple_barrier, report.days_processed);

    // Recompute derived metrics for per-quarter aggregates
    for (auto& [q, result] : report.fth_per_quarter) {
        recompute_derived(result, fth_quarter_days[q]);
    }
    for (auto& [q, result] : report.tb_per_quarter) {
        recompute_derived(result, tb_quarter_days[q]);
    }

    // Remove empty quarters (from skipped-only days)
    remove_empty_quarters(report.fth_per_quarter);
    remove_empty_quarters(report.tb_per_quarter);

    return report;
}

// ---------------------------------------------------------------------------
// JSON serialization helpers
// ---------------------------------------------------------------------------
namespace detail {

inline void write_backtest_result_json(std::ostringstream& ss,
                                        const BacktestResult& r) {
    ss << "{";
    ss << "\"total_trades\":" << r.total_trades;
    ss << ",\"winning_trades\":" << r.winning_trades;
    ss << ",\"losing_trades\":" << r.losing_trades;
    ss << ",\"win_rate\":" << r.win_rate;
    ss << ",\"gross_pnl\":" << r.gross_pnl;
    ss << ",\"net_pnl\":" << r.net_pnl;
    ss << ",\"expectancy\":" << r.expectancy;
    ss << ",\"profit_factor\":" << r.profit_factor;
    ss << ",\"sharpe\":" << r.sharpe;
    ss << ",\"max_drawdown\":" << r.max_drawdown;
    ss << ",\"trades_per_day\":" << r.trades_per_day;
    ss << ",\"avg_bars_held\":" << r.avg_bars_held;
    ss << ",\"avg_duration_s\":" << r.avg_duration_s;
    ss << ",\"hold_fraction\":" << r.hold_fraction;

    // Exit reasons
    ss << ",\"exit_reasons\":{";
    // Always emit all six exit reasons
    ss << "\"target\":" << (r.exit_reason_counts.count(exit_reason::TARGET) ? r.exit_reason_counts.at(exit_reason::TARGET) : 0);
    ss << ",\"stop\":" << (r.exit_reason_counts.count(exit_reason::STOP) ? r.exit_reason_counts.at(exit_reason::STOP) : 0);
    ss << ",\"take_profit\":" << (r.exit_reason_counts.count(exit_reason::TAKE_PROFIT) ? r.exit_reason_counts.at(exit_reason::TAKE_PROFIT) : 0);
    ss << ",\"expiry\":" << (r.exit_reason_counts.count(exit_reason::EXPIRY) ? r.exit_reason_counts.at(exit_reason::EXPIRY) : 0);
    ss << ",\"session_end\":" << (r.exit_reason_counts.count(exit_reason::SESSION_END) ? r.exit_reason_counts.at(exit_reason::SESSION_END) : 0);
    ss << ",\"safety_cap\":" << (r.exit_reason_counts.count(exit_reason::SAFETY_CAP) ? r.exit_reason_counts.at(exit_reason::SAFETY_CAP) : 0);
    ss << "}";

    ss << "}";
}

}  // namespace detail

// Serialize an OracleExpectancyReport to JSON.
inline std::string to_json(const OracleExpectancyReport& report) {
    std::ostringstream ss;

    ss << "{";

    // Config block (spec defaults)
    ss << "\"config\":{";
    ss << "\"bar_type\":\"time_5s\"";
    ss << ",\"target_ticks\":10";
    ss << ",\"stop_ticks\":5";
    ss << ",\"take_profit_ticks\":20";
    ss << ",\"volume_horizon\":500";
    ss << ",\"max_time_horizon_s\":300";
    ss << "}";

    // Costs block
    ss << ",\"costs\":{";
    ss << "\"commission_per_side\":0.62";
    ss << ",\"fixed_spread_ticks\":1";
    ss << ",\"slippage_ticks\":0";
    ss << ",\"contract_multiplier\":5";
    ss << ",\"tick_size\":0.25";
    ss << "}";

    // Days processed/skipped
    ss << ",\"days_processed\":" << report.days_processed;
    ss << ",\"days_skipped\":" << report.days_skipped;

    // First-to-hit results
    ss << ",\"first_to_hit\":";
    detail::write_backtest_result_json(ss, report.first_to_hit);

    // Triple barrier results
    ss << ",\"triple_barrier\":";
    detail::write_backtest_result_json(ss, report.triple_barrier);

    // Per-quarter results
    ss << ",\"per_quarter\":{";
    bool first_quarter = true;
    // Gather all quarter keys from both maps
    std::map<std::string, bool> all_quarters;
    for (const auto& [q, _] : report.fth_per_quarter) all_quarters[q] = true;
    for (const auto& [q, _] : report.tb_per_quarter) all_quarters[q] = true;

    for (const auto& [q, _] : all_quarters) {
        if (!first_quarter) ss << ",";
        first_quarter = false;

        ss << "\"" << q << "\":{";

        ss << "\"first_to_hit\":";
        if (report.fth_per_quarter.count(q)) {
            detail::write_backtest_result_json(ss, report.fth_per_quarter.at(q));
        } else {
            BacktestResult empty{};
            detail::write_backtest_result_json(ss, empty);
        }

        ss << ",\"triple_barrier\":";
        if (report.tb_per_quarter.count(q)) {
            detail::write_backtest_result_json(ss, report.tb_per_quarter.at(q));
        } else {
            BacktestResult empty{};
            detail::write_backtest_result_json(ss, empty);
        }

        ss << "}";
    }
    ss << "}";

    ss << "}";

    return ss.str();
}

}  // namespace oracle_expectancy
