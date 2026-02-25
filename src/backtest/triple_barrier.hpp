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
    bool bidirectional = true;
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

struct BidirectionalTBResult {
    int label = 0;               // -1 (short), 0 (hold), +1 (long)
    std::string exit_type;       // "long_target", "short_target", "both", "neither", etc.
    int bars_held = 0;
    bool long_triggered = false;
    bool short_triggered = false;
    bool both_triggered = false;
};

namespace detail {

// Check if a V-reversal completes: after one race triggered, does the opposite
// race's target get reached without continuation past target or re-stop?
// dir: +1.0f to check for long target, -1.0f to check for short target.
inline bool v_reversal_target_reached(
    const std::vector<Bar>& bars, int idx, int scan_end,
    float entry_mid, float target_dist, float stop_dist,
    float max_time_s, float dir) {

    int n = static_cast<int>(bars.size());

    // Phase 1: find the target bar
    int target_bar = -1;
    for (int j = scan_end + 1; j < n; ++j) {
        float elapsed_s = static_cast<float>(bars[j].close_ts - bars[idx].close_ts) / 1.0e9f;
        if (elapsed_s >= max_time_s) break;
        float diff = bars[j].close_mid - entry_mid;
        if (dir * diff >= target_dist) { target_bar = j; break; }
    }
    if (target_bar < 0) return false;

    // Phase 2: validate no continuation past target or re-stop
    for (int j = target_bar + 1; j < n; ++j) {
        float elapsed_s = static_cast<float>(bars[j].close_ts - bars[idx].close_ts) / 1.0e9f;
        if (elapsed_s >= max_time_s) break;
        float diff = bars[j].close_mid - entry_mid;
        if (dir * diff > target_dist) return false;    // continuation past target
        if (-dir * diff >= stop_dist) return false;     // re-stop
    }
    return true;
}

}  // namespace detail

// Bidirectional triple barrier label for a single bar.
// Runs two independent races:
//   Long race:  does price hit +target_dist before -stop_dist?
//   Short race: does price hit -target_dist before +stop_dist?
// Label = +1 if only long wins, -1 if only short wins, 0 if both or neither.
// V-reversal override: when one race triggers and the other was stopped,
// check if the stopped race's target is also reached without continuation
// past target or re-stop — if so, override to "both" (label=0).
// When bidirectional=false, falls back to compute_tb_label behavior.
inline BidirectionalTBResult compute_bidirectional_tb_label(
    const std::vector<Bar>& bars, int idx, const TripleBarrierConfig& cfg) {

    // Non-bidirectional: delegate to original function for backward compatibility
    if (!cfg.bidirectional) {
        auto old = compute_tb_label(bars, idx, cfg);
        BidirectionalTBResult result{};
        result.label = old.label;
        result.exit_type = old.exit_type;
        result.bars_held = old.bars_held;
        return result;
    }

    int n = static_cast<int>(bars.size());
    float entry_mid = bars[idx].close_mid;
    float target_dist = static_cast<float>(cfg.target_ticks) * cfg.tick_size;
    float stop_dist = static_cast<float>(cfg.stop_ticks) * cfg.tick_size;
    float max_time_s = static_cast<float>(cfg.max_time_horizon_s);

    // Track each race independently
    bool long_resolved = false;
    bool long_triggered = false;
    bool short_resolved = false;
    bool short_triggered = false;

    uint32_t cum_volume = 0;
    int held = 0;
    int scan_end = idx;

    for (int j = idx + 1; j < n; ++j) {
        cum_volume += bars[j].volume;
        float diff = bars[j].close_mid - entry_mid;
        held = j - idx;
        scan_end = j;

        float elapsed_s = static_cast<float>(bars[j].close_ts - bars[idx].close_ts) / 1.0e9f;

        // Time cap — stop scanning
        if (elapsed_s >= max_time_s) {
            break;
        }

        // Long race: target at +target_dist, stop at -stop_dist
        if (!long_resolved) {
            if (diff >= target_dist) {
                long_triggered = true;
                long_resolved = true;
            } else if (-diff >= stop_dist) {
                long_resolved = true;  // stopped, not triggered
            }
        }

        // Short race: target at -target_dist, stop at +stop_dist
        if (!short_resolved) {
            if (-diff >= target_dist) {
                short_triggered = true;
                short_resolved = true;
            } else if (diff >= stop_dist) {
                short_resolved = true;  // stopped, not triggered
            }
        }

        // Both races resolved → done
        if (long_resolved && short_resolved) {
            break;
        }

        // Volume expiry — stop scanning
        if (cum_volume >= cfg.volume_horizon) {
            break;
        }
    }

    // V-reversal "both" override: when one race triggered and the other was
    // stopped, check if the stopped race's target is also reached.
    if (long_triggered && !short_triggered && short_resolved) {
        if (detail::v_reversal_target_reached(bars, idx, scan_end, entry_mid,
                                               target_dist, stop_dist, max_time_s, -1.0f))
            short_triggered = true;
    } else if (short_triggered && !long_triggered && long_resolved) {
        if (detail::v_reversal_target_reached(bars, idx, scan_end, entry_mid,
                                               target_dist, stop_dist, max_time_s, 1.0f))
            long_triggered = true;
    }

    // Build result
    BidirectionalTBResult result{};
    result.bars_held = held;
    result.long_triggered = long_triggered;
    result.short_triggered = short_triggered;
    result.both_triggered = long_triggered && short_triggered;

    if (long_triggered && short_triggered) {
        result.label = 0;
        result.exit_type = "both";
    } else if (long_triggered) {
        result.label = 1;
        result.exit_type = "long_target";
    } else if (short_triggered) {
        result.label = -1;
        result.exit_type = "short_target";
    } else {
        result.label = 0;
        result.exit_type = "neither";
    }

    return result;
}
