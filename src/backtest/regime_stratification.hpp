#pragma once

#include "backtest/trade_record.hpp"
#include "bars/bar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// RegimeResult — per-regime aggregation (defined at global scope since tests use it directly)
// ---------------------------------------------------------------------------
struct RegimeResult {
    int trade_count = 0;
    float expectancy = 0.0f;
    float net_pnl = 0.0f;
    float win_rate = 0.0f;
    float profit_factor = 0.0f;
    float sharpe = 0.0f;
};

namespace regime {

// ---------------------------------------------------------------------------
// Session — time-of-day session classification
// ---------------------------------------------------------------------------
enum class Session { OPEN, MID, CLOSE };

inline Session classify_session(float hour_of_day) {
    if (hour_of_day < 10.5f) return Session::OPEN;
    if (hour_of_day < 14.0f) return Session::MID;
    return Session::CLOSE;
}

// ---------------------------------------------------------------------------
// Trend — daily trend classification
// ---------------------------------------------------------------------------
enum class Trend { RANGE_BOUND, MODERATE, STRONG_TREND };

inline Trend classify_trend(float otc_return_pct) {
    float abs_return = std::abs(otc_return_pct);
    if (abs_return > 1.0f) return Trend::STRONG_TREND;
    if (abs_return < 0.3f) return Trend::RANGE_BOUND;
    return Trend::MODERATE;
}

// Compute OTC return from a day's bars: (close - open) / open * 100
inline float compute_otc_return(const std::vector<Bar>& bars) {
    if (bars.empty()) return 0.0f;
    float open = bars.front().open_mid;
    float close = bars.back().close_mid;
    if (open == 0.0f) return 0.0f;
    return (close - open) / open * 100.0f;
}

// ---------------------------------------------------------------------------
// Realized volatility
// ---------------------------------------------------------------------------
inline float compute_realized_vol(const std::vector<Bar>& bars, int window) {
    if (static_cast<int>(bars.size()) < window || window < 2) return 0.0f;

    int start = static_cast<int>(bars.size()) - window;
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int n = 0;
    for (int i = start + 1; i < static_cast<int>(bars.size()); ++i) {
        float ret = bars[i].close_mid - bars[i - 1].close_mid;
        sum += ret;
        sum_sq += ret * ret;
        ++n;
    }
    if (n < 2) return 0.0f;
    float mean = sum / static_cast<float>(n);
    float var = (sum_sq / static_cast<float>(n)) - mean * mean;
    return std::sqrt(std::abs(var));
}

// ---------------------------------------------------------------------------
// Quartile assignment — generic over value type
// ---------------------------------------------------------------------------
template <typename T>
std::vector<int> assign_quartiles(const std::vector<T>& values) {
    size_t n = values.size();
    if (n == 0) return {};

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](size_t a, size_t b) { return values[a] < values[b]; });

    std::vector<int> quartiles(n);
    for (size_t rank = 0; rank < n; ++rank) {
        int q = static_cast<int>((rank * 4) / n) + 1;
        if (q > 4) q = 4;
        quartiles[indices[rank]] = q;
    }
    return quartiles;
}

inline std::vector<int> assign_volatility_quartiles(const std::vector<float>& values) {
    return assign_quartiles(values);
}

inline std::vector<int> assign_volume_quartiles(const std::vector<uint64_t>& values) {
    return assign_quartiles(values);
}

// ---------------------------------------------------------------------------
// Stability score
// ---------------------------------------------------------------------------
template <typename Key>
float compute_stability_score(const std::map<Key, RegimeResult>& strat) {
    if (strat.empty()) return 0.0f;
    if (strat.size() == 1) return 1.0f;

    float min_exp = strat.begin()->second.expectancy;
    float max_exp = strat.begin()->second.expectancy;
    for (const auto& [key, result] : strat) {
        if (result.expectancy < min_exp) min_exp = result.expectancy;
        if (result.expectancy > max_exp) max_exp = result.expectancy;
    }

    if (max_exp == 0.0f) return 0.0f;
    return min_exp / max_exp;
}

inline std::string classify_stability(float score) {
    if (score > 0.5f) return "robust";
    if (score >= 0.2f) return "regime-dependent";
    return "fragile";
}

}  // namespace regime

// ---------------------------------------------------------------------------
// RegimeStratifier — stratification engine
// ---------------------------------------------------------------------------
class RegimeStratifier {
public:
    std::map<int, RegimeResult> by_volatility(
            const std::vector<float>& daily_vols,
            const std::vector<std::vector<TradeRecord>>& daily_trades) {
        if (daily_vols.empty()) return {};

        auto quartiles = regime::assign_volatility_quartiles(daily_vols);
        return aggregate_by_key(quartiles, daily_trades);
    }

    std::map<regime::Session, RegimeResult> by_time_of_day(
            const std::vector<TradeRecord>& trades,
            const std::vector<Bar>& bars) {
        std::map<regime::Session, std::vector<TradeRecord>> bucketed;
        for (const auto& trade : trades) {
            int bar_idx = trade.entry_bar_idx;
            float tod = 9.5f;
            if (bar_idx >= 0 && bar_idx < static_cast<int>(bars.size())) {
                tod = bars[bar_idx].time_of_day;
            }
            auto session = regime::classify_session(tod);
            bucketed[session].push_back(trade);
        }

        std::map<regime::Session, RegimeResult> result;
        for (const auto& [session, strades] : bucketed) {
            result[session] = compute_regime_result(strades);
        }
        return result;
    }

    std::map<int, RegimeResult> by_volume(
            const std::vector<uint64_t>& daily_volumes,
            const std::vector<std::vector<TradeRecord>>& daily_trades) {
        if (daily_volumes.empty()) return {};

        auto quartiles = regime::assign_volume_quartiles(daily_volumes);
        return aggregate_by_key(quartiles, daily_trades);
    }

    std::map<regime::Trend, RegimeResult> by_trend(
            const std::vector<float>& daily_otc_returns,
            const std::vector<std::vector<TradeRecord>>& daily_trades) {
        std::map<regime::Trend, std::vector<TradeRecord>> bucketed;
        size_t n = std::min(daily_otc_returns.size(), daily_trades.size());
        for (size_t i = 0; i < n; ++i) {
            auto trend = regime::classify_trend(daily_otc_returns[i]);
            for (const auto& t : daily_trades[i]) {
                bucketed[trend].push_back(t);
            }
        }

        std::map<regime::Trend, RegimeResult> result;
        for (const auto& [trend, trades] : bucketed) {
            result[trend] = compute_regime_result(trades);
        }
        return result;
    }

    bool is_q4_concentrated(const std::map<int, RegimeResult>& vol_strat) const {
        if (vol_strat.find(4) == vol_strat.end()) return false;

        float q4_exp = vol_strat.at(4).expectancy;
        float total_exp = 0.0f;
        for (const auto& [q, r] : vol_strat) {
            total_exp += r.expectancy;
        }

        if (total_exp <= 0.0f) return false;
        return (q4_exp / total_exp) > 0.5f;
    }

private:
    std::map<int, RegimeResult> aggregate_by_key(
            const std::vector<int>& keys,
            const std::vector<std::vector<TradeRecord>>& daily_trades) {
        std::map<int, std::vector<TradeRecord>> bucketed;
        size_t n = std::min(keys.size(), daily_trades.size());
        for (size_t i = 0; i < n; ++i) {
            for (const auto& t : daily_trades[i]) {
                bucketed[keys[i]].push_back(t);
            }
        }

        std::map<int, RegimeResult> result;
        for (const auto& [key, trades] : bucketed) {
            result[key] = compute_regime_result(trades);
        }
        return result;
    }

    static RegimeResult compute_regime_result(const std::vector<TradeRecord>& trades) {
        RegimeResult r{};
        r.trade_count = static_cast<int>(trades.size());
        if (r.trade_count == 0) return r;

        float sum_net = 0.0f;
        float gross_wins = 0.0f;
        float gross_losses = 0.0f;
        int wins = 0;

        for (const auto& t : trades) {
            sum_net += t.net_pnl;
            if (t.net_pnl > 0.0f) {
                wins++;
                gross_wins += t.gross_pnl;
            } else {
                gross_losses += std::abs(t.gross_pnl);
            }
        }

        r.net_pnl = sum_net;
        r.expectancy = sum_net / static_cast<float>(r.trade_count);
        r.win_rate = static_cast<float>(wins) / static_cast<float>(r.trade_count);

        if (gross_losses > 0.0f) {
            r.profit_factor = gross_wins / gross_losses;
        }

        if (r.trade_count >= 2) {
            float mean = r.expectancy;
            float sum_sq = 0.0f;
            for (const auto& t : trades) {
                float d = t.net_pnl - mean;
                sum_sq += d * d;
            }
            float var = sum_sq / static_cast<float>(r.trade_count - 1);
            float std_dev = std::sqrt(var);
            if (std_dev > 0.0f) r.sharpe = mean / std_dev;
        }

        return r;
    }
};
