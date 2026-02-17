#pragma once

#include "book_builder.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int L = 10;          // Book depth (levels per side)
constexpr int T = 50;          // Trade buffer length
constexpr int W = 600;         // Window length (snapshots)
constexpr float TICK_SIZE = 0.25f;

// Feature index ranges — contiguous, non-overlapping.
// Each range begins where the previous one ends.
constexpr int BID_PRICE_BEGIN       = 0;
constexpr int BID_PRICE_END         = BID_PRICE_BEGIN + L;
constexpr int BID_SIZE_BEGIN        = BID_PRICE_END;
constexpr int BID_SIZE_END          = BID_SIZE_BEGIN + L;
constexpr int ASK_PRICE_BEGIN       = BID_SIZE_END;
constexpr int ASK_PRICE_END         = ASK_PRICE_BEGIN + L;
constexpr int ASK_SIZE_BEGIN        = ASK_PRICE_END;
constexpr int ASK_SIZE_END          = ASK_SIZE_BEGIN + L;
constexpr int TRADE_PRICE_BEGIN     = ASK_SIZE_END;
constexpr int TRADE_PRICE_END       = TRADE_PRICE_BEGIN + T;
constexpr int TRADE_SIZE_BEGIN      = TRADE_PRICE_END;
constexpr int TRADE_SIZE_END        = TRADE_SIZE_BEGIN + T;
constexpr int TRADE_AGGRESSOR_BEGIN = TRADE_SIZE_END;
constexpr int TRADE_AGGRESSOR_END   = TRADE_AGGRESSOR_BEGIN + T;
constexpr int SPREAD_TICKS_IDX     = TRADE_AGGRESSOR_END;
constexpr int TIME_SIN_IDX         = SPREAD_TICKS_IDX + 1;
constexpr int TIME_COS_IDX         = TIME_SIN_IDX + 1;
constexpr int POSITION_STATE_IDX   = TIME_COS_IDX + 1;
constexpr int FEATURE_DIM          = POSITION_STATE_IDX + 1;

inline constexpr float TWO_PI = 2.0f * std::numbers::pi_v<float>;

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
