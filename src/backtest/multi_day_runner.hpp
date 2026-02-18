#pragma once

#include "backtest/execution_costs.hpp"
#include "backtest/oracle_replay.hpp"
#include "backtest/rollover.hpp"
#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// BacktestConfig — configuration for multi-day runner
// ---------------------------------------------------------------------------
struct BacktestConfig {
    std::string bar_type;
    double bar_param = 0.0;
    OracleConfig oracle;
    ExecutionCosts costs;
};

// ---------------------------------------------------------------------------
// DaySchedule — in-sample / out-of-sample date ranges
// ---------------------------------------------------------------------------
struct DaySchedule {
    int in_sample_start = 0;
    int in_sample_end = 0;
    int oos_start = 0;
    int oos_end = 0;

    bool is_in_sample(int date) const {
        return date >= in_sample_start && date <= in_sample_end;
    }

    bool is_oos(int date) const {
        return date >= oos_start && date <= oos_end;
    }
};

// ---------------------------------------------------------------------------
// DayResult — result of a single day's backtest
// ---------------------------------------------------------------------------
struct DayResult {
    int date = 0;
    BacktestResult result;
    bool skipped = false;
    std::string skip_reason;
    std::string contract_symbol;
    uint32_t instrument_id = 0;
    int bar_count = 0;
};

// ---------------------------------------------------------------------------
// SplitResults — IS / OOS split
// ---------------------------------------------------------------------------
struct SplitResults {
    std::vector<DayResult> in_sample;
    std::vector<DayResult> oos;
};

// ---------------------------------------------------------------------------
// MultiDayRunner — runs backtest across multiple days
// ---------------------------------------------------------------------------
class MultiDayRunner {
public:
    MultiDayRunner(const BacktestConfig& config,
                   const RolloverCalendar& calendar,
                   const DaySchedule& schedule)
        : config_(config), calendar_(calendar), schedule_(schedule) {}

    DayResult run_day(int date, const std::vector<Bar>& bars) {
        DayResult day{};
        day.date = date;
        day.bar_count = static_cast<int>(bars.size());

        // Check if date is excluded (rollover window)
        if (calendar_.is_excluded(date)) {
            day.skipped = true;
            day.skip_reason = "Excluded: near rollover";
            return day;
        }

        // Look up contract for this date
        auto contract = calendar_.get_contract_for_date(date);
        if (contract.has_value()) {
            day.contract_symbol = contract->symbol;
            day.instrument_id = contract->instrument_id;
        }

        // Run oracle replay
        OracleReplay replay(config_.oracle, config_.costs);
        day.result = replay.run(bars);

        return day;
    }

    BacktestResult aggregate(const std::vector<DayResult>& day_results) {
        BacktestResult agg{};

        int active_days = 0;

        for (const auto& day : day_results) {
            if (day.skipped) continue;
            ++active_days;

            agg.total_trades += day.result.total_trades;
            agg.winning_trades += day.result.winning_trades;
            agg.losing_trades += day.result.losing_trades;
            agg.net_pnl += day.result.net_pnl;
            agg.gross_pnl += day.result.gross_pnl;

            // Collect all trades
            agg.trades.insert(agg.trades.end(),
                              day.result.trades.begin(), day.result.trades.end());

            // Track daily PnL
            agg.daily_pnl.push_back(day.result.net_pnl);

            // Accumulate safety cap counts
            agg.safety_cap_triggered_count += day.result.safety_cap_triggered_count;
        }

        // Recompute aggregate metrics (win_rate, expectancy, profit_factor,
        // trades_per_day, max_drawdown, sharpe)
        backtest_util::recompute_derived(agg, active_days);

        // Safety cap fraction (not covered by recompute_derived)
        if (agg.total_trades > 0) {
            agg.safety_cap_fraction = static_cast<float>(agg.safety_cap_triggered_count)
                                       / static_cast<float>(agg.total_trades);
        }

        return agg;
    }

    SplitResults split_results(const std::vector<DayResult>& day_results) {
        SplitResults split{};
        for (const auto& day : day_results) {
            if (schedule_.is_in_sample(day.date)) {
                split.in_sample.push_back(day);
            } else if (schedule_.is_oos(day.date)) {
                split.oos.push_back(day);
            }
        }
        return split;
    }

private:
    BacktestConfig config_;
    RolloverCalendar calendar_;
    DaySchedule schedule_;
};
