#pragma once

#include "book_builder.hpp"
#include "feature_encoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int GBT_FEATURE_DIM = 16;

// Lookback windows in snapshots (100ms each)
constexpr int LOOKBACK_1S  = 10;   // 1 second  = 10 snapshots
constexpr int LOOKBACK_5S  = 50;   // 5 seconds = 50 snapshots
constexpr int LOOKBACK_30S = 300;  // 30 seconds = 300 snapshots

// ---------------------------------------------------------------------------
// Trade — a deduplicated trade with its originating snapshot index
// ---------------------------------------------------------------------------
struct Trade {
    float price;
    float size;
    float side;
    int snapshot_idx;
};

// ---------------------------------------------------------------------------
// deduplicate_trades — collect unique trades across a W-snapshot window
//
// Compares each snapshot's trade buffer with the previous snapshot's buffer
// position-by-position. A trade at position j in snapshot s is new if it
// differs from position j in snapshot s-1. Zero-padded entries (size == 0)
// are filtered out.
// ---------------------------------------------------------------------------
inline std::vector<Trade> deduplicate_trades(
    const std::vector<BookSnapshot>& window)
{
    std::vector<Trade> unique_trades;

    for (int s = 0; s < W; ++s) {
        for (int j = 0; j < TRADE_BUF_LEN; ++j) {
            float tp = window[s].trades[j][0];
            float ts_size = window[s].trades[j][1];
            float ts_side = window[s].trades[j][2];

            if (ts_size == 0.0f) continue;

            bool is_new = true;
            if (s > 0) {
                float prev_p = window[s - 1].trades[j][0];
                float prev_sz = window[s - 1].trades[j][1];
                float prev_sd = window[s - 1].trades[j][2];
                if (tp == prev_p && ts_size == prev_sz && ts_side == prev_sd) {
                    is_new = false;
                }
            }

            if (is_new) {
                unique_trades.push_back({tp, ts_size, ts_side, s});
            }
        }
    }

    return unique_trades;
}

// ---------------------------------------------------------------------------
// compute_gbt_features — 16 hand-crafted features from a W=600 snapshot window
//
// Feature index ordering (from spec):
//   [0]  book_imbalance
//   [1]  spread_ticks
//   [2]  book_depth_ratio_5
//   [3]  top_level_size_ratio
//   [4]  mid_return_1s       (lookback 10 snapshots)
//   [5]  mid_return_5s       (lookback 50 snapshots)
//   [6]  mid_return_30s      (lookback 300 snapshots)
//   [7]  mid_return_60s      (lookback 599 snapshots)
//   [8]  trade_imbalance_1s
//   [9]  trade_imbalance_5s
//   [10] trade_arrival_rate_5s
//   [11] large_trade_flag
//   [12] vwap_distance
//   [13] time_sin
//   [14] time_cos
//   [15] position_state
// ---------------------------------------------------------------------------
inline std::array<float, GBT_FEATURE_DIM> compute_gbt_features(
    const std::vector<BookSnapshot>& window,
    float position_state)
{
    if (static_cast<int>(window.size()) != W) {
        throw std::invalid_argument("compute_gbt_features requires exactly W=" +
                                    std::to_string(W) + " snapshots, got " +
                                    std::to_string(window.size()));
    }

    constexpr float EPS = 1e-8f;

    const int t = W - 1; // current snapshot index (599)
    const auto& snap = window[t];

    std::array<float, GBT_FEATURE_DIM> f{};

    // -----------------------------------------------------------------------
    // [0] book_imbalance = (sum_bid - sum_ask) / (sum_bid + sum_ask + eps)
    // Uses RAW sizes (not log-transformed), all 10 levels
    // -----------------------------------------------------------------------
    float sum_bid = 0.0f, sum_ask = 0.0f;
    for (int i = 0; i < BOOK_DEPTH; ++i) {
        sum_bid += snap.bids[i][1];
        sum_ask += snap.asks[i][1];
    }
    f[0] = (sum_bid - sum_ask) / (sum_bid + sum_ask + EPS);

    // -----------------------------------------------------------------------
    // [1] spread_ticks = spread / tick_size
    // -----------------------------------------------------------------------
    f[1] = snap.spread / TICK_SIZE;

    // -----------------------------------------------------------------------
    // [2] book_depth_ratio_5 = sum(bid_size[0:5]) / (sum(ask_size[0:5]) + eps)
    // -----------------------------------------------------------------------
    float sum_bid5 = 0.0f, sum_ask5 = 0.0f;
    for (int i = 0; i < 5; ++i) {
        sum_bid5 += snap.bids[i][1];
        sum_ask5 += snap.asks[i][1];
    }
    f[2] = sum_bid5 / (sum_ask5 + EPS);

    // -----------------------------------------------------------------------
    // [3] top_level_size_ratio = bid_size[0] / (ask_size[0] + eps)
    // -----------------------------------------------------------------------
    f[3] = snap.bids[0][1] / (snap.asks[0][1] + EPS);

    // -----------------------------------------------------------------------
    // [4-7] mid returns in ticks
    //   1s  = 10 snapshots lookback
    //   5s  = 50 snapshots lookback
    //   30s = 300 snapshots lookback
    //   60s = 599 snapshots lookback (t - 0 = full window)
    // -----------------------------------------------------------------------
    f[4] = (snap.mid_price - window[t - LOOKBACK_1S].mid_price)  / TICK_SIZE;
    f[5] = (snap.mid_price - window[t - LOOKBACK_5S].mid_price)  / TICK_SIZE;
    f[6] = (snap.mid_price - window[t - LOOKBACK_30S].mid_price) / TICK_SIZE;
    f[7] = (snap.mid_price - window[0].mid_price)                / TICK_SIZE;

    // -----------------------------------------------------------------------
    // Trade-based features: deduplicate, then compute imbalance/arrival/VWAP
    // -----------------------------------------------------------------------
    auto unique_trades = deduplicate_trades(window);

    // -----------------------------------------------------------------------
    // [8] trade_imbalance_1s = sum(size * side) for trades in [t-LOOKBACK_1S : t+1]
    // [9] trade_imbalance_5s = sum(size * side) for trades in [t-LOOKBACK_5S : t+1]
    // [10] trade_arrival_rate_5s = count(trades in [t-LOOKBACK_5S : t+1]) / 5.0
    // -----------------------------------------------------------------------
    float imbalance_1s = 0.0f;
    float imbalance_5s = 0.0f;
    int count_5s = 0;
    for (const auto& tr : unique_trades) {
        if (tr.snapshot_idx >= t - LOOKBACK_5S && tr.snapshot_idx <= t) {
            imbalance_5s += tr.size * tr.side;
            count_5s++;
            if (tr.snapshot_idx >= t - LOOKBACK_1S) {
                imbalance_1s += tr.size * tr.side;
            }
        }
    }
    f[8] = imbalance_1s;
    f[9] = imbalance_5s;
    f[10] = static_cast<float>(count_5s) / 5.0f;

    // -----------------------------------------------------------------------
    // [11] large_trade_flag
    //   1.0 if any trade in 1s window has size >= 2 * median(all trade sizes in window)
    //   0.0 otherwise. If no trades exist, flag = 0.0.
    // -----------------------------------------------------------------------
    if (!unique_trades.empty()) {
        std::vector<float> all_sizes;
        all_sizes.reserve(unique_trades.size());
        for (const auto& tr : unique_trades) {
            all_sizes.push_back(tr.size);
        }
        std::sort(all_sizes.begin(), all_sizes.end());
        float median = all_sizes[all_sizes.size() / 2];
        float threshold = 2.0f * median;

        bool found_large = false;
        for (const auto& tr : unique_trades) {
            if (tr.snapshot_idx >= t - LOOKBACK_1S && tr.snapshot_idx <= t) {
                if (tr.size >= threshold) {
                    found_large = true;
                    break;
                }
            }
        }
        f[11] = found_large ? 1.0f : 0.0f;
    } else {
        f[11] = 0.0f;
    }

    // -----------------------------------------------------------------------
    // [12] vwap_distance = (mid_price[t] - VWAP) / tick_size
    //   VWAP = sum(price * size) / sum(size) over all unique trades in window
    //   If no trades, feature = 0.0
    // -----------------------------------------------------------------------
    if (!unique_trades.empty()) {
        double sum_pv = 0.0;
        double sum_sz = 0.0;
        for (const auto& tr : unique_trades) {
            sum_pv += static_cast<double>(tr.price) * static_cast<double>(tr.size);
            sum_sz += static_cast<double>(tr.size);
        }
        if (sum_sz > 0.0) {
            float vwap = static_cast<float>(sum_pv / sum_sz);
            f[12] = (snap.mid_price - vwap) / TICK_SIZE;
        } else {
            f[12] = 0.0f;
        }
    } else {
        f[12] = 0.0f;
    }

    // -----------------------------------------------------------------------
    // [13] time_sin, [14] time_cos
    // -----------------------------------------------------------------------
    float frac = snap.time_of_day / 24.0f;
    f[13] = std::sin(TWO_PI * frac);
    f[14] = std::cos(TWO_PI * frac);

    // -----------------------------------------------------------------------
    // [15] position_state
    // -----------------------------------------------------------------------
    f[15] = position_state;

    return f;
}
