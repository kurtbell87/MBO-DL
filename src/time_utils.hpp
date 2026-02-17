#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// Time constants and utilities for ET (Eastern Time) nanosecond timestamps
// ---------------------------------------------------------------------------
namespace time_utils {

constexpr uint64_t NS_PER_SEC         = 1'000'000'000ULL;
constexpr uint64_t NS_PER_HOUR        = 3600ULL * NS_PER_SEC;
constexpr uint64_t NS_PER_DAY         = 24ULL * NS_PER_HOUR;
// 2022-01-03 00:00:00 ET in UTC nanoseconds (reference epoch)
constexpr uint64_t REF_MIDNIGHT_ET_NS = 1641186000ULL * NS_PER_SEC;

inline uint64_t midnight_et_ns(uint64_t ts) {
    int64_t diff = static_cast<int64_t>(ts) - static_cast<int64_t>(REF_MIDNIGHT_ET_NS);
    int64_t day_offset = diff / static_cast<int64_t>(NS_PER_DAY);
    if (diff < 0 && diff % static_cast<int64_t>(NS_PER_DAY) != 0) day_offset -= 1;
    return REF_MIDNIGHT_ET_NS + static_cast<uint64_t>(day_offset) * NS_PER_DAY;
}

inline uint64_t rth_open_ns(uint64_t ts) {
    return midnight_et_ns(ts) + 9ULL * NS_PER_HOUR + 30ULL * 60 * NS_PER_SEC;
}

inline uint64_t rth_close_ns(uint64_t ts) {
    return midnight_et_ns(ts) + 16ULL * NS_PER_HOUR;
}

inline float compute_time_of_day(uint64_t ts) {
    uint64_t midnight = midnight_et_ns(ts);
    double ns_since_midnight = static_cast<double>(ts - midnight);
    return static_cast<float>(ns_since_midnight / static_cast<double>(NS_PER_HOUR));
}

}  // namespace time_utils
