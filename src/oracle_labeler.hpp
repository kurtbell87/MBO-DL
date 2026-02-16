#pragma once

#include "book_builder.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// oracle_label — generate ground-truth action label using future price data
//
// Returns: 0=HOLD, 1=ENTER_LONG, 2=ENTER_SHORT, 3=EXIT
// Action 4 (REVERSE) is never generated.
// ---------------------------------------------------------------------------
inline int oracle_label(
    const std::vector<BookSnapshot>& snapshots,
    int t,
    int position_state,       // -1, 0, +1
    float entry_price,        // NaN when flat
    int horizon = 100,
    int target_ticks = 10,
    int stop_ticks = 5,
    int take_profit_ticks = 20,
    float tick_size = 0.25f)
{
    // Preconditions
    if (t + horizon > static_cast<int>(snapshots.size()))
        throw std::invalid_argument("t + horizon must not exceed snapshots.size()");
    if ((position_state == 0) != std::isnan(entry_price))
        throw std::invalid_argument("flat requires NaN entry_price; in-position requires valid entry_price");

    float ref_mid = snapshots[t].mid_price;
    int scan_end = static_cast<int>(snapshots.size()) - 1;

    // Small epsilon for float comparison (fraction of a tick)
    constexpr float EPS = 0.1f;

    if (position_state == 0) {
        // Flat: scan forward to see if price hits target or stop first
        float target_thresh = static_cast<float>(target_ticks) - EPS;
        float stop_thresh = static_cast<float>(stop_ticks) - EPS;

        bool long_viable = true;
        bool short_viable = true;

        for (int k = t + 1; k <= scan_end; ++k) {
            float delta_ticks = (snapshots[k].mid_price - ref_mid) / tick_size;

            // Check long stop: adverse move >= stop_ticks
            if (long_viable && delta_ticks <= -stop_thresh) {
                long_viable = false;
            }
            // Check long target: favorable move >= target_ticks
            if (long_viable && delta_ticks >= target_thresh) {
                return 1; // ENTER_LONG
            }

            // Check short stop: adverse move >= stop_ticks
            if (short_viable && delta_ticks >= stop_thresh) {
                short_viable = false;
            }
            // Check short target: favorable move >= target_ticks
            if (short_viable && delta_ticks <= -target_thresh) {
                return 2; // ENTER_SHORT
            }

            if (!long_viable && !short_viable) break;
        }

        return 0; // HOLD
    } else {
        // In position (long or short): check take-profit or stop-loss
        float tp_thresh = static_cast<float>(take_profit_ticks) - EPS;
        float sl_thresh = static_cast<float>(stop_ticks) - EPS;

        for (int k = t + 1; k <= scan_end; ++k) {
            float pnl_ticks = static_cast<float>(position_state) *
                              (snapshots[k].mid_price - entry_price) / tick_size;

            if (pnl_ticks >= tp_thresh) {
                return 3; // EXIT (take profit)
            }
            if (pnl_ticks <= -sl_thresh) {
                return 3; // EXIT (stop loss)
            }
        }

        return 0; // HOLD
    }
}
