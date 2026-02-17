#pragma once

#include <cstdint>

namespace exit_reason {
    constexpr int TARGET      = 0;
    constexpr int STOP        = 1;
    constexpr int TAKE_PROFIT = 2;
    constexpr int EXPIRY      = 3;
    constexpr int SESSION_END = 4;
    constexpr int SAFETY_CAP  = 5;
}  // namespace exit_reason

struct TradeRecord {
    uint64_t entry_ts = 0;
    uint64_t exit_ts = 0;
    float entry_price = 0.0f;
    float exit_price = 0.0f;
    int direction = 0;       // +1 = LONG, -1 = SHORT
    float gross_pnl = 0.0f;
    float net_pnl = 0.0f;
    int entry_bar_idx = 0;
    int exit_bar_idx = 0;
    int bars_held = 0;
    float duration_s = 0.0f;
    int exit_reason = 0;
};
