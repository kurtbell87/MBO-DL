#pragma once

#include <cstdint>

struct TripleBarrierConfig {
    int target_ticks = 10;
    int stop_ticks = 5;
    uint32_t volume_horizon = 500;
    int min_return_ticks = 2;
    uint32_t max_time_horizon_s = 300;
};
