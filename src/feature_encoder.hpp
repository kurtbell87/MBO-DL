#pragma once

#include "book_builder.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int L = 10;          // Book depth (levels per side)
constexpr int T = 50;          // Trade buffer length
constexpr int W = 600;         // Window length (snapshots)
constexpr float TICK_SIZE = 0.25f;

// Feature index ranges — contiguous, non-overlapping
constexpr int BID_PRICE_BEGIN      = 0;
constexpr int BID_PRICE_END        = 10;   // L
constexpr int BID_SIZE_BEGIN       = 10;
constexpr int BID_SIZE_END         = 20;   // 2*L
constexpr int ASK_PRICE_BEGIN      = 20;
constexpr int ASK_PRICE_END        = 30;   // 3*L
constexpr int ASK_SIZE_BEGIN       = 30;
constexpr int ASK_SIZE_END         = 40;   // 4*L
constexpr int TRADE_PRICE_BEGIN    = 40;
constexpr int TRADE_PRICE_END      = 90;   // 4*L + T
constexpr int TRADE_SIZE_BEGIN     = 90;
constexpr int TRADE_SIZE_END       = 140;  // 4*L + 2*T
constexpr int TRADE_AGGRESSOR_BEGIN = 140;
constexpr int TRADE_AGGRESSOR_END  = 190;  // 4*L + 3*T
constexpr int SPREAD_TICKS_IDX    = 190;
constexpr int TIME_SIN_IDX        = 191;
constexpr int TIME_COS_IDX        = 192;
constexpr int POSITION_STATE_IDX  = 193;
constexpr int FEATURE_DIM         = 194;

inline constexpr float TWO_PI = 2.0f * 3.14159265358979323846f;

// ---------------------------------------------------------------------------
// encode_snapshot — transform a single BookSnapshot into a 194-dim feature vector
// Sizes are raw log1p (NOT z-scored). Z-scoring happens in encode_window.
// ---------------------------------------------------------------------------
inline std::array<float, FEATURE_DIM> encode_snapshot(const BookSnapshot& snap,
                                                       float position_state) {
    std::array<float, FEATURE_DIM> f{};

    // Bid levels: price deltas and log1p sizes
    for (int i = 0; i < L; ++i) {
        f[BID_PRICE_BEGIN + i] = (snap.bids[i][0] - snap.mid_price) / TICK_SIZE;
        f[BID_SIZE_BEGIN  + i] = std::log1p(snap.bids[i][1]);
    }

    // Ask levels: price deltas and log1p sizes
    for (int i = 0; i < L; ++i) {
        f[ASK_PRICE_BEGIN + i] = (snap.asks[i][0] - snap.mid_price) / TICK_SIZE;
        f[ASK_SIZE_BEGIN  + i] = std::log1p(snap.asks[i][1]);
    }

    // Trades: price deltas, log1p sizes, and aggressor side
    for (int i = 0; i < T; ++i) {
        f[TRADE_PRICE_BEGIN     + i] = (snap.trades[i][0] - snap.mid_price) / TICK_SIZE;
        f[TRADE_SIZE_BEGIN      + i] = std::log1p(snap.trades[i][1]);
        f[TRADE_AGGRESSOR_BEGIN + i] = snap.trades[i][2];
    }

    // Spread in ticks
    f[SPREAD_TICKS_IDX] = snap.spread / TICK_SIZE;

    // Time encoding: sin/cos of fractional hour
    float frac = snap.time_of_day / 24.0f;
    f[TIME_SIN_IDX] = std::sin(TWO_PI * frac);
    f[TIME_COS_IDX] = std::cos(TWO_PI * frac);

    // Position state
    f[POSITION_STATE_IDX] = position_state;

    return f;
}

// ---------------------------------------------------------------------------
// encode_window — encode W snapshots with z-scored size features
// ---------------------------------------------------------------------------
inline std::vector<std::array<float, FEATURE_DIM>> encode_window(
    const std::vector<BookSnapshot>& window, float position_state) {

    if (static_cast<int>(window.size()) != W) {
        throw std::invalid_argument("encode_window requires exactly W=" +
                                    std::to_string(W) + " snapshots, got " +
                                    std::to_string(window.size()));
    }

    // Step 1: encode each snapshot (sizes are raw log1p)
    std::vector<std::array<float, FEATURE_DIM>> encoded;
    encoded.reserve(W);
    for (int t = 0; t < W; ++t) {
        encoded.push_back(encode_snapshot(window[t], position_state));
    }

    // Step 2: z-score size features across the window
    // A single (mean, std) is computed across ALL size features across ALL snapshots.

    constexpr float EPSILON = 1e-8f;
    constexpr int SIZE_FEATURES_PER_SNAP = (BID_SIZE_END - BID_SIZE_BEGIN) +
                                            (ASK_SIZE_END - ASK_SIZE_BEGIN) +
                                            (TRADE_SIZE_END - TRADE_SIZE_BEGIN);
    constexpr int TOTAL_SIZE_VALUES = W * SIZE_FEATURES_PER_SNAP;

    // Helper: apply a function to every size-feature slot across the window
    constexpr std::pair<int,int> SIZE_RANGES[] = {
        {BID_SIZE_BEGIN, BID_SIZE_END},
        {ASK_SIZE_BEGIN, ASK_SIZE_END},
        {TRADE_SIZE_BEGIN, TRADE_SIZE_END},
    };

    auto for_each_size_feature = [&](auto fn) {
        for (int t = 0; t < W; ++t)
            for (auto [lo, hi] : SIZE_RANGES)
                for (int i = lo; i < hi; ++i)
                    fn(encoded[t][i]);
    };

    double sum = 0.0;
    for_each_size_feature([&](float v) { sum += v; });
    double mean = sum / TOTAL_SIZE_VALUES;

    double var_sum = 0.0;
    for_each_size_feature([&](float v) {
        double d = v - mean;
        var_sum += d * d;
    });
    double std_dev = std::sqrt(var_sum / TOTAL_SIZE_VALUES);

    float inv_std = 1.0f / static_cast<float>(std_dev + EPSILON);
    float fmean = static_cast<float>(mean);
    for_each_size_feature([&](float& v) { v = (v - fmean) * inv_std; });

    return encoded;
}
