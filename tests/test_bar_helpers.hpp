#pragma once

// Shared test helpers for constructing synthetic Bar sequences.
// Used by oracle_replay_test.cpp and triple_barrier_test.cpp.

#include "bars/bar.hpp"

#include <cstdint>
#include <vector>

namespace test_helpers {

constexpr float TICK = 0.25f;
constexpr float TICK_SIZE = TICK;  // alias used by some test files
constexpr uint64_t NS_PER_SEC = 1'000'000'000ULL;
constexpr uint64_t NS_PER_MIN = 60ULL * NS_PER_SEC;
constexpr uint64_t NS_PER_HOUR = 3600ULL * NS_PER_SEC;
constexpr uint64_t MIDNIGHT_ET_NS = 1641186000ULL * NS_PER_SEC;  // 2022-01-03 00:00 ET
constexpr uint64_t RTH_OPEN_NS = MIDNIGHT_ET_NS + 9ULL * NS_PER_HOUR + 30ULL * NS_PER_MIN;
constexpr uint64_t RTH_CLOSE_NS = MIDNIGHT_ET_NS + 16ULL * NS_PER_HOUR;

// Build a Bar with specified mid_price, volume, and timestamp offset.
inline Bar make_bar(float mid_price, uint32_t volume, uint64_t ts_offset_ns,
                    float spread_ticks = 1.0f) {
    Bar bar{};
    bar.open_mid = mid_price;
    bar.close_mid = mid_price;
    bar.high_mid = mid_price;
    bar.low_mid = mid_price;
    bar.vwap = mid_price;
    bar.volume = volume;
    bar.tick_count = volume;
    bar.open_ts = RTH_OPEN_NS + ts_offset_ns;
    bar.close_ts = RTH_OPEN_NS + ts_offset_ns + 1 * NS_PER_SEC;
    bar.time_of_day = 9.5f + static_cast<float>(ts_offset_ns) / static_cast<float>(NS_PER_HOUR);
    bar.bar_duration_s = 1.0f;
    bar.spread = spread_ticks * TICK;

    float half_spread = (spread_ticks * TICK) / 2.0f;
    bar.bids[0][0] = mid_price - half_spread;
    bar.bids[0][1] = 10.0f;
    bar.asks[0][0] = mid_price + half_spread;
    bar.asks[0][1] = 10.0f;

    return bar;
}

// Build a series of bars with linearly moving mid_price.
// Each bar has `vol_per_bar` volume and bars are 1 second apart.
inline std::vector<Bar> make_bar_series(float start_mid, float end_mid, int count,
                                         uint32_t vol_per_bar = 50) {
    std::vector<Bar> bars;
    bars.reserve(count);
    for (int i = 0; i < count; ++i) {
        float frac = (count > 1) ? static_cast<float>(i) / (count - 1) : 0.0f;
        float mid = start_mid + frac * (end_mid - start_mid);
        uint64_t ts_offset = static_cast<uint64_t>(i) * NS_PER_SEC;
        bars.push_back(make_bar(mid, vol_per_bar, ts_offset));
    }
    return bars;
}

// Build bars following an explicit price path (start + per-bar deltas).
inline std::vector<Bar> make_bar_path(float start_mid, const std::vector<float>& deltas,
                                       uint32_t vol_per_bar = 50) {
    int count = static_cast<int>(deltas.size()) + 1;
    std::vector<Bar> bars;
    bars.reserve(count);
    float mid = start_mid;
    for (int i = 0; i < count; ++i) {
        uint64_t ts_offset = static_cast<uint64_t>(i) * NS_PER_SEC;
        bars.push_back(make_bar(mid, vol_per_bar, ts_offset));
        if (i < static_cast<int>(deltas.size())) {
            mid += deltas[i];
        }
    }
    return bars;
}

}  // namespace test_helpers
