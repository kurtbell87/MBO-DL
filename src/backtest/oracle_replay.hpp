#pragma once

#include "backtest/execution_costs.hpp"
#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

// ---------------------------------------------------------------------------
// OracleConfig — configuration for the oracle replay engine
// ---------------------------------------------------------------------------
struct OracleConfig {
    enum class LabelMethod { FIRST_TO_HIT, TRIPLE_BARRIER };

    uint32_t volume_horizon = 50000;
    uint32_t max_time_horizon_s = 3600;
    int target_ticks = 10;
    int stop_ticks = 5;
    int take_profit_ticks = 20;
    float tick_size = 0.25f;
    LabelMethod label_method = LabelMethod::FIRST_TO_HIT;
};

// ---------------------------------------------------------------------------
// BacktestResult — aggregate statistics from an oracle replay
// ---------------------------------------------------------------------------
struct BacktestResult {
    std::vector<TradeRecord> trades;
    int total_trades = 0;
    int winning_trades = 0;
    int losing_trades = 0;
    float win_rate = 0.0f;
    float gross_pnl = 0.0f;
    float net_pnl = 0.0f;
    float profit_factor = 0.0f;
    float expectancy = 0.0f;
    float max_drawdown = 0.0f;
    float sharpe = 0.0f;
    float hold_fraction = 0.0f;
    float avg_bars_held = 0.0f;
    float avg_duration_s = 0.0f;
    float trades_per_day = 0.0f;
    int safety_cap_triggered_count = 0;
    float safety_cap_fraction = 0.0f;
    std::map<int, float> pnl_by_hour;
    std::map<int, int> label_counts;
    std::map<int, int> exit_reason_counts;
    std::vector<float> daily_pnl;
};

// ---------------------------------------------------------------------------
// BacktestResult utilities — shared by OracleReplay and MultiDayRunner
// ---------------------------------------------------------------------------
namespace backtest_util {

inline void compute_max_drawdown(BacktestResult& result) {
    if (result.trades.empty()) return;
    float peak = 0.0f;
    float equity = 0.0f;
    float max_dd = 0.0f;
    for (const auto& trade : result.trades) {
        equity += trade.net_pnl;
        if (equity > peak) peak = equity;
        float dd = peak - equity;
        if (dd > max_dd) max_dd = dd;
    }
    result.max_drawdown = max_dd;
}

inline void compute_sharpe(BacktestResult& result) {
    if (result.trades.size() < 2) return;
    float n = static_cast<float>(result.trades.size());
    float mean = result.net_pnl / n;
    float sum_sq = 0.0f;
    for (const auto& trade : result.trades) {
        float diff = trade.net_pnl - mean;
        sum_sq += diff * diff;
    }
    float variance = sum_sq / (n - 1.0f);
    float stddev = std::sqrt(variance);
    if (stddev > 0.0f) {
        result.sharpe = mean / stddev;
    }
}

// Recompute derived metrics (win_rate, expectancy, profit_factor, trades_per_day,
// max_drawdown, sharpe) on a BacktestResult whose trades have already been accumulated.
inline void recompute_derived(BacktestResult& agg, int active_days) {
    if (agg.total_trades > 0) {
        agg.win_rate = static_cast<float>(agg.winning_trades)
                       / static_cast<float>(agg.total_trades);
        agg.expectancy = agg.net_pnl / static_cast<float>(agg.total_trades);
    }

    // Profit factor from individual trades
    float gross_wins = 0.0f;
    float gross_losses = 0.0f;
    for (const auto& trade : agg.trades) {
        if (trade.gross_pnl > 0.0f) {
            gross_wins += trade.gross_pnl;
        } else {
            gross_losses += std::abs(trade.gross_pnl);
        }
    }
    if (gross_losses > 0.0f) {
        agg.profit_factor = gross_wins / gross_losses;
    }

    if (active_days > 0) {
        agg.trades_per_day = static_cast<float>(agg.total_trades)
                              / static_cast<float>(active_days);
    }

    compute_max_drawdown(agg);
    compute_sharpe(agg);
}

}  // namespace backtest_util

// ---------------------------------------------------------------------------
// OracleReplay — oracle replay engine
// ---------------------------------------------------------------------------
class OracleReplay {
public:
    OracleReplay(const OracleConfig& cfg, const ExecutionCosts& costs)
        : cfg_(cfg), costs_(costs) {}

    BacktestResult run(const std::vector<Bar>& bars) {
        BacktestResult result{};

        if (bars.size() <= 1) {
            return result;
        }

        int n = static_cast<int>(bars.size());
        int hold_bars = 0;  // bars where oracle says HOLD

        for (int i = 1; i < n - 1; ++i) {
            // Look ahead from bar i to determine oracle label
            int label = compute_label(bars, i);

            if (label == 0) {
                ++hold_bars;
                continue;
            }

            // Record label count
            result.label_counts[label]++;

            // Enter trade at bar i
            TradeRecord trade{};
            trade.entry_bar_idx = i;
            trade.entry_price = bars[i].close_mid;
            trade.entry_ts = bars[i].close_ts;
            trade.direction = label;

            // Find exit
            find_exit(bars, i, label, trade);

            // Compute PnL
            float price_diff = trade.exit_price - trade.entry_price;
            trade.gross_pnl = price_diff * static_cast<float>(trade.direction)
                              * costs_.contract_multiplier;

            // Compute spread in ticks for cost calculation
            float entry_spread_ticks = bars[trade.entry_bar_idx].spread / cfg_.tick_size;
            float exit_spread_ticks = bars[trade.exit_bar_idx].spread / cfg_.tick_size;
            float rt_cost = costs_.round_trip_cost(entry_spread_ticks, exit_spread_ticks);
            trade.net_pnl = trade.gross_pnl - rt_cost;

            // Bars held and duration
            trade.bars_held = trade.exit_bar_idx - trade.entry_bar_idx;
            trade.duration_s = static_cast<float>(trade.exit_ts - trade.entry_ts) / 1.0e9f;

            result.trades.push_back(trade);

            // Skip to after exit bar (no overlapping positions)
            i = trade.exit_bar_idx;
        }

        // Compute aggregate statistics
        compute_aggregates(result, bars, n, hold_bars);

        return result;
    }

private:
    OracleConfig cfg_;
    ExecutionCosts costs_;

    // Compute oracle label for bar at index `idx` by looking ahead.
    // Returns +1 (LONG), -1 (SHORT), or 0 (HOLD).
    int compute_label(const std::vector<Bar>& bars, int idx) {
        float entry_mid = bars[idx].close_mid;
        float target_dist = static_cast<float>(cfg_.target_ticks) * cfg_.tick_size;
        float stop_dist = static_cast<float>(cfg_.stop_ticks) * cfg_.tick_size;

        if (cfg_.label_method == OracleConfig::LabelMethod::FIRST_TO_HIT) {
            return first_to_hit_label(bars, idx, entry_mid, target_dist, stop_dist);
        } else {
            return triple_barrier_label(bars, idx, entry_mid, target_dist, stop_dist);
        }
    }

    int first_to_hit_label(const std::vector<Bar>& bars, int idx,
                           float entry_mid, float target_dist, float stop_dist) {
        int n = static_cast<int>(bars.size());
        uint32_t cum_volume = 0;

        // Scan forward within volume horizon to find which barrier is hit first.
        // For LONG: does price hit +target before -stop?
        // For SHORT: does price hit -target before +stop?
        int long_target_idx = n;
        int long_stop_idx = n;
        int short_target_idx = n;
        int short_stop_idx = n;
        int window_end = idx;

        for (int j = idx + 1; j < n; ++j) {
            cum_volume += bars[j].volume;
            float diff = bars[j].close_mid - entry_mid;
            window_end = j;

            if (diff >= target_dist && long_target_idx == n)
                long_target_idx = j;
            if (-diff >= stop_dist && long_stop_idx == n)
                long_stop_idx = j;
            if (-diff >= target_dist && short_target_idx == n)
                short_target_idx = j;
            if (diff >= stop_dist && short_stop_idx == n)
                short_stop_idx = j;

            if (cum_volume >= cfg_.volume_horizon) break;
        }

        // LONG is viable if target hit before stop (or stop not hit)
        bool long_viable = (long_target_idx < long_stop_idx);
        bool short_viable = (short_target_idx < short_stop_idx);

        if (long_viable && !short_viable) return 1;
        if (short_viable && !long_viable) return -1;
        if (long_viable && short_viable) {
            return (long_target_idx <= short_target_idx) ? 1 : -1;
        }

        // Neither target hit within volume horizon — use next-bar direction
        if (idx + 1 < n) {
            float next_diff = bars[idx + 1].close_mid - entry_mid;
            if (next_diff > 0.0f) return 1;
            if (next_diff < 0.0f) return -1;
        }

        return 0;  // HOLD
    }

    int triple_barrier_label(const std::vector<Bar>& bars, int idx,
                             float entry_mid, float target_dist, float stop_dist) {
        int n = static_cast<int>(bars.size());
        uint32_t cum_volume = 0;
        float min_return_dist = 2.0f * cfg_.tick_size;  // min_return_ticks=2

        // Scan forward, checking upper (+target) and lower (-stop) barriers
        for (int j = idx + 1; j < n; ++j) {
            cum_volume += bars[j].volume;
            float diff = bars[j].close_mid - entry_mid;

            float elapsed_s = static_cast<float>(bars[j].close_ts - bars[idx].close_ts) / 1.0e9f;
            if (elapsed_s >= static_cast<float>(cfg_.max_time_horizon_s)) {
                // Safety cap / time expiry — check return
                if (std::abs(diff) >= min_return_dist) {
                    return (diff > 0) ? 1 : -1;
                }
                return 0;
            }

            // Upper barrier: price up by target
            if (diff >= target_dist) {
                return 1;  // LONG
            }
            // Lower barrier: price down by stop (using stop for lower barrier)
            if (-diff >= stop_dist) {
                return -1;  // SHORT
            }

            if (cum_volume >= cfg_.volume_horizon) {
                // Volume expiry — check return magnitude
                if (std::abs(diff) >= min_return_dist) {
                    return (diff > 0) ? 1 : -1;
                }
                return 0;  // HOLD
            }
        }

        // Reached end of bars (session end) — check return
        if (idx + 1 < n) {
            float final_diff = bars[n - 1].close_mid - entry_mid;
            if (std::abs(final_diff) >= min_return_dist) {
                return (final_diff > 0) ? 1 : -1;
            }
        }
        return 0;
    }

    void find_exit(const std::vector<Bar>& bars, int entry_idx, int direction,
                   TradeRecord& trade) {
        int n = static_cast<int>(bars.size());
        float entry_mid = bars[entry_idx].close_mid;
        float target_dist = static_cast<float>(cfg_.target_ticks) * cfg_.tick_size;
        float stop_dist = static_cast<float>(cfg_.stop_ticks) * cfg_.tick_size;
        float tp_dist = static_cast<float>(cfg_.take_profit_ticks) * cfg_.tick_size;

        for (int j = entry_idx + 1; j < n; ++j) {
            float diff = bars[j].close_mid - entry_mid;
            float directional_diff = diff * static_cast<float>(direction);

            // Take profit (check before target since tp_dist >= target_dist)
            if (directional_diff >= tp_dist) {
                trade.exit_bar_idx = j;
                trade.exit_price = bars[j].close_mid;
                trade.exit_ts = bars[j].close_ts;
                trade.exit_reason = exit_reason::TAKE_PROFIT;
                return;
            }

            // Target hit
            if (directional_diff >= target_dist) {
                trade.exit_bar_idx = j;
                trade.exit_price = bars[j].close_mid;
                trade.exit_ts = bars[j].close_ts;
                trade.exit_reason = exit_reason::TARGET;
                return;
            }

            // Stop hit (directional_diff going negative beyond stop)
            if (directional_diff <= -stop_dist) {
                trade.exit_bar_idx = j;
                trade.exit_price = bars[j].close_mid;
                trade.exit_ts = bars[j].close_ts;
                trade.exit_reason = exit_reason::STOP;
                return;
            }
        }

        // End of bars → session end (force close)
        trade.exit_bar_idx = n - 1;
        trade.exit_price = bars[n - 1].close_mid;
        trade.exit_ts = bars[n - 1].close_ts;
        trade.exit_reason = exit_reason::SESSION_END;
    }

    void compute_aggregates(BacktestResult& result, const std::vector<Bar>& bars,
                            int total_bars, int hold_bars) {
        result.total_trades = static_cast<int>(result.trades.size());

        float sum_gross_pnl = 0.0f;
        float sum_net_pnl = 0.0f;
        float gross_wins = 0.0f;
        float gross_losses = 0.0f;
        float sum_bars_held = 0.0f;
        float sum_duration = 0.0f;
        int bars_in_trade = 0;

        for (auto& trade : result.trades) {
            sum_gross_pnl += trade.gross_pnl;
            sum_net_pnl += trade.net_pnl;
            sum_bars_held += static_cast<float>(trade.bars_held);
            sum_duration += trade.duration_s;
            bars_in_trade += trade.bars_held;

            if (trade.net_pnl > 0.0f) {
                result.winning_trades++;
                gross_wins += trade.gross_pnl;
            } else {
                result.losing_trades++;
                gross_losses += std::abs(trade.gross_pnl);
            }

            // Exit reason counts
            result.exit_reason_counts[trade.exit_reason]++;

            // Safety cap count
            if (trade.exit_reason == exit_reason::SAFETY_CAP) {
                result.safety_cap_triggered_count++;
            }

            // PnL by hour
            int entry_hour = static_cast<int>(bars[trade.entry_bar_idx].time_of_day);
            result.pnl_by_hour[entry_hour] += trade.net_pnl;
        }

        result.gross_pnl = sum_gross_pnl;
        result.net_pnl = sum_net_pnl;

        if (result.total_trades > 0) {
            result.win_rate = static_cast<float>(result.winning_trades)
                              / static_cast<float>(result.total_trades);
            result.expectancy = result.net_pnl / static_cast<float>(result.total_trades);
            result.avg_bars_held = sum_bars_held / static_cast<float>(result.total_trades);
            result.avg_duration_s = sum_duration / static_cast<float>(result.total_trades);
            result.safety_cap_fraction = static_cast<float>(result.safety_cap_triggered_count)
                                         / static_cast<float>(result.total_trades);
        }

        if (gross_losses > 0.0f) {
            result.profit_factor = gross_wins / gross_losses;
        }

        // Hold fraction: fraction of labeling opportunities that resulted in HOLD
        int total_opportunities = result.total_trades + hold_bars;
        if (total_opportunities > 0) {
            result.hold_fraction = static_cast<float>(hold_bars)
                                   / static_cast<float>(total_opportunities);
        }

        // Max drawdown
        backtest_util::compute_max_drawdown(result);

        // Sharpe ratio
        backtest_util::compute_sharpe(result);
    }
};
