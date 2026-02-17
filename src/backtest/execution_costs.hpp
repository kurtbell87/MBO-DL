#pragma once

#include <cstdint>

struct ExecutionCosts {
    enum class SpreadModel { FIXED, EMPIRICAL };

    float commission_per_side = 0.62f;
    SpreadModel spread_model = SpreadModel::FIXED;
    int fixed_spread_ticks = 1;
    int slippage_ticks = 0;
    float contract_multiplier = 5.0f;
    float tick_size = 0.25f;
    float tick_value = 1.25f;  // contract_multiplier * tick_size

    // Per-side cost: commission + half-spread cost + slippage cost.
    // In FIXED mode, actual_spread_ticks is ignored; fixed_spread_ticks is used.
    // In EMPIRICAL mode, actual_spread_ticks is used instead.
    float per_side_cost(float actual_spread_ticks = 0.0f) const {
        float spread_ticks_used = (spread_model == SpreadModel::FIXED)
            ? static_cast<float>(fixed_spread_ticks)
            : actual_spread_ticks;
        float half_spread_cost = (spread_ticks_used / 2.0f) * tick_value;
        float slippage_cost = static_cast<float>(slippage_ticks) * tick_value;
        return commission_per_side + half_spread_cost + slippage_cost;
    }

    // Round-trip cost: entry per-side cost + exit per-side cost.
    float round_trip_cost(float entry_spread_ticks, float exit_spread_ticks) const {
        return per_side_cost(entry_spread_ticks) + per_side_cost(exit_spread_ticks);
    }
};
