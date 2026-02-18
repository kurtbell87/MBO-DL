#pragma once

#include "bars/bar.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct TripleBarrierConfig {
    int target_ticks = 10;
    int stop_ticks = 5;
    uint32_t volume_horizon = 500;
    int min_return_ticks = 2;
    uint32_t max_time_horizon_s = 300;
    float tick_size = 0.25f;
};

struct TripleBarrierResult {
    int label = 0;               // -1 (short), 0 (hold), +1 (long)
    std::string exit_type;       // "target", "stop", "expiry", "timeout"
    int bars_held = 0;
};

// Position-independent triple barrier label for a single bar.
// At bar `idx`, asks: "if we entered long here, would target or stop hit first?"
// Scans forward accumulating volume until volume_horizon or max_time_horizon_s.
inline TripleBarrierResult compute_tb_label(const std::vector<Bar>& bars, int idx,
                                             const TripleBarrierConfig& cfg) {
    int n = static_cast<int>(bars.size());
    float entry_mid = bars[idx].close_mid;
    float target_dist = static_cast<float>(cfg.target_ticks) * cfg.tick_size;
    float stop_dist = static_cast<float>(cfg.stop_ticks) * cfg.tick_size;
    float min_return_dist = static_cast<float>(cfg.min_return_ticks) * cfg.tick_size;

    uint32_t cum_volume = 0;

    for (int j = idx + 1; j < n; ++j) {
        cum_volume += bars[j].volume;
        float diff = bars[j].close_mid - entry_mid;
        int held = j - idx;

        float elapsed_s = static_cast<float>(bars[j].close_ts - bars[idx].close_ts) / 1.0e9f;

        // Time cap (hard safety limit)
        if (elapsed_s >= static_cast<float>(cfg.max_time_horizon_s)) {
            if (std::abs(diff) >= min_return_dist) {
                return {(diff > 0.0f) ? 1 : -1, "timeout", held};
            }
            return {0, "timeout", held};
        }

        // Upper barrier: target hit
        if (diff >= target_dist) {
            return {1, "target", held};
        }
        // Lower barrier: stop hit
        if (-diff >= stop_dist) {
            return {-1, "stop", held};
        }

        // Volume expiry
        if (cum_volume >= cfg.volume_horizon) {
            if (std::abs(diff) >= min_return_dist) {
                return {(diff > 0.0f) ? 1 : -1, "expiry", held};
            }
            return {0, "expiry", held};
        }
    }

    // Ran out of bars (end of day)
    int held = (n > idx + 1) ? (n - 1 - idx) : 0;
    if (n > idx + 1) {
        float final_diff = bars[n - 1].close_mid - entry_mid;
        if (std::abs(final_diff) >= min_return_dist) {
            return {(final_diff > 0.0f) ? 1 : -1, "expiry", held};
        }
    }
    return {0, "expiry", held};
}
