// feature_encoder_test.cpp — TDD RED phase tests for FeatureEncoder
// Spec: .kit/docs/feature-encoder.md
//
// Tests the encode_snapshot() and encode_window() functions that transform
// BookSnapshot structs into 194-dimensional feature vectors.

#include <gtest/gtest.h>
#include "feature_encoder.hpp"
#include "book_builder.hpp"  // BookSnapshot struct

#include <array>
#include <cmath>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers — synthetic BookSnapshot construction
// ---------------------------------------------------------------------------
namespace {

constexpr float PI = 3.14159265358979323846f;

// Build a BookSnapshot with known values for deterministic testing.
// Bid prices descend from base_bid, ask prices ascend from base_ask,
// each separated by tick_size (0.25).
BookSnapshot make_snapshot(
    float base_bid = 4500.00f,
    float base_ask = 4500.25f,
    float time_of_day = 9.5f,   // 09:30 ET
    int num_bid_levels = 10,
    int num_ask_levels = 10,
    int num_trades = 50
) {
    BookSnapshot snap{};
    snap.mid_price = (base_bid + base_ask) / 2.0f;
    snap.spread = base_ask - base_bid;
    snap.time_of_day = time_of_day;
    snap.timestamp = 0;

    // Fill bid levels: descending from base_bid by 0.25 each
    for (int i = 0; i < num_bid_levels && i < 10; ++i) {
        snap.bids[i][0] = base_bid - i * 0.25f;  // price
        snap.bids[i][1] = static_cast<float>(10 + i);  // size: 10, 11, 12, ...
    }

    // Fill ask levels: ascending from base_ask by 0.25 each
    for (int i = 0; i < num_ask_levels && i < 10; ++i) {
        snap.asks[i][0] = base_ask + i * 0.25f;  // price
        snap.asks[i][1] = static_cast<float>(10 + i);  // size: 10, 11, 12, ...
    }

    // Fill trades: alternating buy/sell aggressor
    int start = 50 - num_trades;
    for (int i = 0; i < num_trades && i < 50; ++i) {
        int idx = start + i;
        snap.trades[idx][0] = base_bid + 0.125f;  // trade price near mid
        snap.trades[idx][1] = static_cast<float>(5 + i);  // trade size
        snap.trades[idx][2] = (i % 2 == 0) ? 1.0f : -1.0f;  // B or A
    }

    return snap;
}

// Build a window of W snapshots with slight variations for z-score testing
std::vector<BookSnapshot> make_window(int w = 600) {
    std::vector<BookSnapshot> window;
    window.reserve(w);
    for (int i = 0; i < w; ++i) {
        // Vary sizes slightly across the window for meaningful z-score stats
        BookSnapshot snap = make_snapshot();
        // Modify bid sizes to create variance
        for (int j = 0; j < 10; ++j) {
            snap.bids[j][1] = static_cast<float>(10 + j + (i % 20));
            snap.asks[j][1] = static_cast<float>(10 + j + ((i + 5) % 20));
        }
        // Modify trade sizes similarly
        for (int j = 0; j < 50; ++j) {
            snap.trades[j][1] = static_cast<float>(5 + j + (i % 15));
        }
        // Vary time_of_day slightly across window
        snap.time_of_day = 9.5f + static_cast<float>(i) * 0.1f / 3600.0f;
        window.push_back(snap);
    }
    return window;
}

// Build a window where all sizes are identical (for epsilon floor test)
std::vector<BookSnapshot> make_uniform_window(float uniform_size = 10.0f, int w = 600) {
    std::vector<BookSnapshot> window;
    window.reserve(w);
    for (int i = 0; i < w; ++i) {
        BookSnapshot snap = make_snapshot();
        for (int j = 0; j < 10; ++j) {
            snap.bids[j][1] = uniform_size;
            snap.asks[j][1] = uniform_size;
        }
        for (int j = 0; j < 50; ++j) {
            snap.trades[j][1] = uniform_size;
        }
        snap.time_of_day = 9.5f + static_cast<float>(i) * 0.1f / 3600.0f;
        window.push_back(snap);
    }
    return window;
}

}  // anonymous namespace

// ===========================================================================
// 1. Feature dimension is 194 and named index constants are correct
// ===========================================================================
TEST(FeatureEncoderConstantsTest, FeatureDimIs194) {
    EXPECT_EQ(FEATURE_DIM, 194);
}

TEST(FeatureEncoderConstantsTest, BookDepthConstants) {
    EXPECT_EQ(L, 10);
    EXPECT_EQ(T, 50);
    EXPECT_EQ(W, 600);
    EXPECT_FLOAT_EQ(TICK_SIZE, 0.25f);
}

// ===========================================================================
// 13. Index constants contiguous — no gaps or overlaps
// ===========================================================================
TEST(FeatureEncoderConstantsTest, IndexRangesContiguousAndNonOverlapping) {
    // Verify each END equals the next BEGIN
    EXPECT_EQ(BID_PRICE_BEGIN, 0);
    EXPECT_EQ(BID_PRICE_END, BID_SIZE_BEGIN);      // 10
    EXPECT_EQ(BID_SIZE_END, ASK_PRICE_BEGIN);       // 20
    EXPECT_EQ(ASK_PRICE_END, ASK_SIZE_BEGIN);       // 30
    EXPECT_EQ(ASK_SIZE_END, TRADE_PRICE_BEGIN);     // 40
    EXPECT_EQ(TRADE_PRICE_END, TRADE_SIZE_BEGIN);   // 90
    EXPECT_EQ(TRADE_SIZE_END, TRADE_AGGRESSOR_BEGIN); // 140
    EXPECT_EQ(TRADE_AGGRESSOR_END, SPREAD_TICKS_IDX); // 190
    EXPECT_EQ(SPREAD_TICKS_IDX + 1, TIME_SIN_IDX);   // 191
    EXPECT_EQ(TIME_SIN_IDX + 1, TIME_COS_IDX);        // 192
    EXPECT_EQ(TIME_COS_IDX + 1, POSITION_STATE_IDX);  // 193
    EXPECT_EQ(POSITION_STATE_IDX + 1, FEATURE_DIM);   // 194
}

TEST(FeatureEncoderConstantsTest, IndexRangeSizesMatchSpec) {
    EXPECT_EQ(BID_PRICE_END - BID_PRICE_BEGIN, 10);       // L=10 bid price deltas
    EXPECT_EQ(BID_SIZE_END - BID_SIZE_BEGIN, 10);          // L=10 bid sizes
    EXPECT_EQ(ASK_PRICE_END - ASK_PRICE_BEGIN, 10);       // L=10 ask price deltas
    EXPECT_EQ(ASK_SIZE_END - ASK_SIZE_BEGIN, 10);          // L=10 ask sizes
    EXPECT_EQ(TRADE_PRICE_END - TRADE_PRICE_BEGIN, 50);   // T=50 trade price deltas
    EXPECT_EQ(TRADE_SIZE_END - TRADE_SIZE_BEGIN, 50);      // T=50 trade sizes
    EXPECT_EQ(TRADE_AGGRESSOR_END - TRADE_AGGRESSOR_BEGIN, 50); // T=50 aggressors
}

// ===========================================================================
// 2. Bid price deltas: (bid_price[i] - mid_price) / tick_size
// ===========================================================================
TEST(EncodeSnapshotTest, BidPriceDeltas) {
    // Known snapshot: bid[0]=4500.00, mid=4500.125, tick=0.25
    // delta[0] = (4500.00 - 4500.125) / 0.25 = -0.5
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f);
    // mid = (4500.00 + 4500.25) / 2 = 4500.125

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 10; ++i) {
        float expected_price = 4500.00f - i * 0.25f;
        float expected_delta = (expected_price - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[BID_PRICE_BEGIN + i], expected_delta, 0.01f)
            << "Bid price delta at level " << i;
    }
}

// ===========================================================================
// 3. Ask price deltas: (ask_price[i] - mid_price) / tick_size
// ===========================================================================
TEST(EncodeSnapshotTest, AskPriceDeltas) {
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f);
    // mid = 4500.125

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 10; ++i) {
        float expected_price = 4500.25f + i * 0.25f;
        float expected_delta = (expected_price - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[ASK_PRICE_BEGIN + i], expected_delta, 0.01f)
            << "Ask price delta at level " << i;
    }
}

// ===========================================================================
// 4. Trade price deltas: (trade_price[j] - mid_price) / tick_size
// ===========================================================================
TEST(EncodeSnapshotTest, TradePriceDeltas) {
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 9.5f, 10, 10, 50);
    // All trades have price = 4500.125 (base_bid + 0.125)
    // mid = 4500.125
    // delta = (4500.125 - 4500.125) / 0.25 = 0.0

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 50; ++i) {
        float expected_delta = (snap.trades[i][0] - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[TRADE_PRICE_BEGIN + i], expected_delta, 0.01f)
            << "Trade price delta at index " << i;
    }
}

// ===========================================================================
// Per-snapshot size encoding: raw log1p (NOT z-scored)
// ===========================================================================
TEST(EncodeSnapshotTest, BidSizesAreRawLog1p) {
    BookSnapshot snap = make_snapshot();

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 10; ++i) {
        float expected = std::log1p(snap.bids[i][1]);
        EXPECT_NEAR(features[BID_SIZE_BEGIN + i], expected, 1e-5f)
            << "Bid size at level " << i << " should be log1p(size)";
    }
}

TEST(EncodeSnapshotTest, AskSizesAreRawLog1p) {
    BookSnapshot snap = make_snapshot();

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 10; ++i) {
        float expected = std::log1p(snap.asks[i][1]);
        EXPECT_NEAR(features[ASK_SIZE_BEGIN + i], expected, 1e-5f)
            << "Ask size at level " << i << " should be log1p(size)";
    }
}

TEST(EncodeSnapshotTest, TradeSizesAreRawLog1p) {
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 9.5f, 10, 10, 50);

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 50; ++i) {
        float expected = std::log1p(snap.trades[i][1]);
        EXPECT_NEAR(features[TRADE_SIZE_BEGIN + i], expected, 1e-5f)
            << "Trade size at index " << i << " should be log1p(size)";
    }
}

// ===========================================================================
// 5. Spread in ticks: spread / tick_size
// ===========================================================================
TEST(EncodeSnapshotTest, SpreadInTicks) {
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f);
    // spread = 0.25, spread_ticks = 0.25 / 0.25 = 1.0

    auto features = encode_snapshot(snap, 0.0f);

    EXPECT_NEAR(features[SPREAD_TICKS_IDX], 1.0f, 0.01f);
}

TEST(EncodeSnapshotTest, SpreadInTicksWideSpread) {
    BookSnapshot snap{};
    snap.bids[0][0] = 4500.00f;
    snap.bids[0][1] = 10.0f;
    snap.asks[0][0] = 4501.00f;
    snap.asks[0][1] = 10.0f;
    snap.mid_price = 4500.50f;
    snap.spread = 1.00f;  // 4 ticks
    snap.time_of_day = 9.5f;

    auto features = encode_snapshot(snap, 0.0f);

    // spread_ticks = 1.00 / 0.25 = 4.0
    EXPECT_NEAR(features[SPREAD_TICKS_IDX], 4.0f, 0.01f);
}

// ===========================================================================
// 6. Time encoding: sin/cos of fractional hour
// ===========================================================================
TEST(EncodeSnapshotTest, TimeEncodingAtNoon) {
    // 12:00 ET = 12.0 fractional hours
    // sin(2π × 12 / 24) = sin(π) ≈ 0.0
    // cos(2π × 12 / 24) = cos(π) = -1.0
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 12.0f);

    auto features = encode_snapshot(snap, 0.0f);

    EXPECT_NEAR(features[TIME_SIN_IDX], 0.0f, 1e-5f);
    EXPECT_NEAR(features[TIME_COS_IDX], -1.0f, 1e-5f);
}

TEST(EncodeSnapshotTest, TimeEncodingAtMidnight) {
    // 0.0 hours (midnight)
    // sin(2π × 0 / 24) = sin(0) = 0.0
    // cos(2π × 0 / 24) = cos(0) = 1.0
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 0.0f);

    auto features = encode_snapshot(snap, 0.0f);

    EXPECT_NEAR(features[TIME_SIN_IDX], 0.0f, 1e-5f);
    EXPECT_NEAR(features[TIME_COS_IDX], 1.0f, 1e-5f);
}

TEST(EncodeSnapshotTest, TimeEncodingAt6AM) {
    // 6.0 hours
    // sin(2π × 6 / 24) = sin(π/2) = 1.0
    // cos(2π × 6 / 24) = cos(π/2) ≈ 0.0
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 6.0f);

    auto features = encode_snapshot(snap, 0.0f);

    EXPECT_NEAR(features[TIME_SIN_IDX], 1.0f, 1e-5f);
    EXPECT_NEAR(features[TIME_COS_IDX], 0.0f, 1e-5f);
}

TEST(EncodeSnapshotTest, TimeEncodingAt930ET) {
    // 9.5 hours (RTH open)
    // sin(2π × 9.5 / 24) = sin(2π × 0.39583...)
    // cos(2π × 9.5 / 24) = cos(2π × 0.39583...)
    float frac = 9.5f / 24.0f;
    float expected_sin = std::sin(2.0f * PI * frac);
    float expected_cos = std::cos(2.0f * PI * frac);

    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 9.5f);
    auto features = encode_snapshot(snap, 0.0f);

    EXPECT_NEAR(features[TIME_SIN_IDX], expected_sin, 1e-5f);
    EXPECT_NEAR(features[TIME_COS_IDX], expected_cos, 1e-5f);
}

// ===========================================================================
// 7. Position state injection: -1.0, 0.0, +1.0
// ===========================================================================
TEST(EncodeSnapshotTest, PositionStateLong) {
    BookSnapshot snap = make_snapshot();
    auto features = encode_snapshot(snap, 1.0f);
    EXPECT_FLOAT_EQ(features[POSITION_STATE_IDX], 1.0f);
}

TEST(EncodeSnapshotTest, PositionStateFlat) {
    BookSnapshot snap = make_snapshot();
    auto features = encode_snapshot(snap, 0.0f);
    EXPECT_FLOAT_EQ(features[POSITION_STATE_IDX], 0.0f);
}

TEST(EncodeSnapshotTest, PositionStateShort) {
    BookSnapshot snap = make_snapshot();
    auto features = encode_snapshot(snap, -1.0f);
    EXPECT_FLOAT_EQ(features[POSITION_STATE_IDX], -1.0f);
}

// ===========================================================================
// 8. Aggressor side encoding: B → +1.0, A → -1.0, zero-padded → 0.0
// ===========================================================================
TEST(EncodeSnapshotTest, AggressorSideEncoding) {
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 9.5f, 10, 10, 5);
    // 5 trades at positions [45..49], alternating B/A
    // trades[45][2] = 1.0 (B), trades[46][2] = -1.0 (A), etc.

    auto features = encode_snapshot(snap, 0.0f);

    // Zero-padded slots (0..44) → aggressor feature should be 0.0
    for (int i = 0; i < 45; ++i) {
        EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + i], 0.0f)
            << "Zero-padded trade at index " << i << " should have aggressor = 0.0";
    }

    // Real trades (45..49): alternating +1.0 / -1.0
    EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + 45], 1.0f);   // B
    EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + 46], -1.0f);  // A
    EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + 47], 1.0f);   // B
    EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + 48], -1.0f);  // A
    EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + 49], 1.0f);   // B
}

TEST(EncodeSnapshotTest, AggressorAllBuyAggressor) {
    BookSnapshot snap{};
    snap.mid_price = 4500.125f;
    snap.spread = 0.25f;
    snap.time_of_day = 9.5f;
    snap.bids[0][0] = 4500.00f; snap.bids[0][1] = 10.0f;
    snap.asks[0][0] = 4500.25f; snap.asks[0][1] = 10.0f;

    // Fill all 50 trades as buy aggressor
    for (int i = 0; i < 50; ++i) {
        snap.trades[i][0] = 4500.25f;
        snap.trades[i][1] = 1.0f;
        snap.trades[i][2] = 1.0f;  // B = buy aggressor
    }

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + i], 1.0f)
            << "Buy aggressor at trade " << i << " should be +1.0";
    }
}

TEST(EncodeSnapshotTest, AggressorAllSellAggressor) {
    BookSnapshot snap{};
    snap.mid_price = 4500.125f;
    snap.spread = 0.25f;
    snap.time_of_day = 9.5f;
    snap.bids[0][0] = 4500.00f; snap.bids[0][1] = 10.0f;
    snap.asks[0][0] = 4500.25f; snap.asks[0][1] = 10.0f;

    // Fill all 50 trades as sell aggressor
    for (int i = 0; i < 50; ++i) {
        snap.trades[i][0] = 4500.00f;
        snap.trades[i][1] = 1.0f;
        snap.trades[i][2] = -1.0f;  // A = sell aggressor
    }

    auto features = encode_snapshot(snap, 0.0f);

    for (int i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + i], -1.0f)
            << "Sell aggressor at trade " << i << " should be -1.0";
    }
}

// ===========================================================================
// 14. Round-trip consistency — encode produces exactly 194 features
// ===========================================================================
TEST(EncodeSnapshotTest, OutputIsExactly194Elements) {
    BookSnapshot snap = make_snapshot();
    auto features = encode_snapshot(snap, 0.0f);

    // std::array<float, FEATURE_DIM> has compile-time size = 194
    EXPECT_EQ(features.size(), 194u);
}

// ===========================================================================
// 11. Zero-padded levels — fewer than 10 levels produce valid features
// ===========================================================================
TEST(EncodeSnapshotTest, ZeroPaddedBidLevels) {
    // Only 3 bid levels populated, rest are zero-padded
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 9.5f, 3, 10, 50);
    // bids[3..9] have price=0.0, size=0.0

    auto features = encode_snapshot(snap, 0.0f);

    // Real bid levels [0..2] should have valid deltas
    for (int i = 0; i < 3; ++i) {
        float expected_price = 4500.00f - i * 0.25f;
        float expected_delta = (expected_price - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[BID_PRICE_BEGIN + i], expected_delta, 0.01f);
    }

    // Zero-padded levels [3..9]: price=0.0, delta = (0 - mid) / tick
    // This will be a large negative number — that's acceptable per spec
    for (int i = 3; i < 10; ++i) {
        float expected_delta = (0.0f - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[BID_PRICE_BEGIN + i], expected_delta, 0.01f);
        // Size: log1p(0) = 0
        EXPECT_NEAR(features[BID_SIZE_BEGIN + i], 0.0f, 1e-5f);
    }

    // No NaN or Inf in the output
    for (int i = 0; i < FEATURE_DIM; ++i) {
        EXPECT_FALSE(std::isnan(features[i])) << "NaN at feature index " << i;
        EXPECT_FALSE(std::isinf(features[i])) << "Inf at feature index " << i;
    }
}

TEST(EncodeSnapshotTest, ZeroPaddedAskLevels) {
    // Only 2 ask levels populated
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 9.5f, 10, 2, 50);

    auto features = encode_snapshot(snap, 0.0f);

    // Real ask levels
    for (int i = 0; i < 2; ++i) {
        float expected_price = 4500.25f + i * 0.25f;
        float expected_delta = (expected_price - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[ASK_PRICE_BEGIN + i], expected_delta, 0.01f);
    }

    // Zero-padded ask levels
    for (int i = 2; i < 10; ++i) {
        float expected_delta = (0.0f - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[ASK_PRICE_BEGIN + i], expected_delta, 0.01f);
        EXPECT_NEAR(features[ASK_SIZE_BEGIN + i], 0.0f, 1e-5f);
    }

    // No NaN or Inf
    for (int i = 0; i < FEATURE_DIM; ++i) {
        EXPECT_FALSE(std::isnan(features[i])) << "NaN at feature index " << i;
        EXPECT_FALSE(std::isinf(features[i])) << "Inf at feature index " << i;
    }
}

// ===========================================================================
// 12. Zero-padded trades — fewer than 50 trades
// ===========================================================================
TEST(EncodeSnapshotTest, ZeroPaddedTrades) {
    // Only 5 trades (at positions [45..49])
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 9.5f, 10, 10, 5);

    auto features = encode_snapshot(snap, 0.0f);

    // Zero-padded trade prices: delta = (0 - mid) / tick (large negative)
    for (int i = 0; i < 45; ++i) {
        float expected_delta = (0.0f - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[TRADE_PRICE_BEGIN + i], expected_delta, 0.01f)
            << "Zero-padded trade price delta at index " << i;
        // Size: log1p(0) = 0
        EXPECT_NEAR(features[TRADE_SIZE_BEGIN + i], 0.0f, 1e-5f)
            << "Zero-padded trade size at index " << i;
        // Aggressor: 0.0
        EXPECT_FLOAT_EQ(features[TRADE_AGGRESSOR_BEGIN + i], 0.0f)
            << "Zero-padded trade aggressor at index " << i;
    }

    // Real trades [45..49] should have valid features
    for (int i = 45; i < 50; ++i) {
        float expected_delta = (snap.trades[i][0] - snap.mid_price) / TICK_SIZE;
        EXPECT_NEAR(features[TRADE_PRICE_BEGIN + i], expected_delta, 0.01f);
    }

    // No NaN or Inf
    for (int i = 0; i < FEATURE_DIM; ++i) {
        EXPECT_FALSE(std::isnan(features[i])) << "NaN at feature index " << i;
        EXPECT_FALSE(std::isinf(features[i])) << "Inf at feature index " << i;
    }
}

// ===========================================================================
// 9. Window encoding — z-scored sizes
// ===========================================================================
TEST(EncodeWindowTest, ZScoredSizesHaveMeanNearZeroStdNearOne) {
    auto window = make_window(W);
    auto encoded = encode_window(window, 0.0f);

    ASSERT_EQ(encoded.size(), static_cast<size_t>(W));

    // Collect ALL z-scored size features across the window:
    // bid_sizes [10:20], ask_sizes [30:40], trade_sizes [90:140]
    std::vector<float> all_zscored_sizes;
    all_zscored_sizes.reserve(W * (L + L + T));  // 600 * 70 = 42,000

    for (int t = 0; t < W; ++t) {
        for (int i = BID_SIZE_BEGIN; i < BID_SIZE_END; ++i) {
            all_zscored_sizes.push_back(encoded[t][i]);
        }
        for (int i = ASK_SIZE_BEGIN; i < ASK_SIZE_END; ++i) {
            all_zscored_sizes.push_back(encoded[t][i]);
        }
        for (int i = TRADE_SIZE_BEGIN; i < TRADE_SIZE_END; ++i) {
            all_zscored_sizes.push_back(encoded[t][i]);
        }
    }

    // Compute mean and std of z-scored values
    double sum = 0.0;
    for (float v : all_zscored_sizes) sum += v;
    double mean = sum / all_zscored_sizes.size();

    double var_sum = 0.0;
    for (float v : all_zscored_sizes) var_sum += (v - mean) * (v - mean);
    double std_dev = std::sqrt(var_sum / all_zscored_sizes.size());

    // After z-scoring, mean should be ≈ 0 and std ≈ 1
    EXPECT_NEAR(mean, 0.0, 0.01) << "Z-scored size mean should be near 0";
    EXPECT_NEAR(std_dev, 1.0, 0.01) << "Z-scored size std should be near 1";
}

TEST(EncodeWindowTest, WindowOutputLength) {
    auto window = make_window(W);
    auto encoded = encode_window(window, 0.0f);

    EXPECT_EQ(encoded.size(), static_cast<size_t>(W));
    for (const auto& vec : encoded) {
        EXPECT_EQ(vec.size(), static_cast<size_t>(FEATURE_DIM));
    }
}

TEST(EncodeWindowTest, NonSizeFeaturesMatchSnapshotEncoding) {
    // Price deltas, spread, time, position, aggressor should be the same
    // whether encoded via encode_snapshot or encode_window (only size
    // features differ due to z-scoring).
    auto window = make_window(W);
    auto encoded = encode_window(window, 1.0f);

    // Check first snapshot's non-size features
    auto single = encode_snapshot(window[0], 1.0f);

    // Bid price deltas (not z-scored, should match)
    for (int i = BID_PRICE_BEGIN; i < BID_PRICE_END; ++i) {
        EXPECT_NEAR(encoded[0][i], single[i], 1e-5f)
            << "Bid price delta at " << i << " should match encode_snapshot";
    }

    // Ask price deltas
    for (int i = ASK_PRICE_BEGIN; i < ASK_PRICE_END; ++i) {
        EXPECT_NEAR(encoded[0][i], single[i], 1e-5f)
            << "Ask price delta at " << i << " should match encode_snapshot";
    }

    // Trade price deltas
    for (int i = TRADE_PRICE_BEGIN; i < TRADE_PRICE_END; ++i) {
        EXPECT_NEAR(encoded[0][i], single[i], 1e-5f)
            << "Trade price delta at " << i << " should match encode_snapshot";
    }

    // Aggressor side
    for (int i = TRADE_AGGRESSOR_BEGIN; i < TRADE_AGGRESSOR_END; ++i) {
        EXPECT_FLOAT_EQ(encoded[0][i], single[i])
            << "Aggressor at " << i << " should match encode_snapshot";
    }

    // Spread, time, position
    EXPECT_NEAR(encoded[0][SPREAD_TICKS_IDX], single[SPREAD_TICKS_IDX], 1e-5f);
    EXPECT_NEAR(encoded[0][TIME_SIN_IDX], single[TIME_SIN_IDX], 1e-5f);
    EXPECT_NEAR(encoded[0][TIME_COS_IDX], single[TIME_COS_IDX], 1e-5f);
    EXPECT_FLOAT_EQ(encoded[0][POSITION_STATE_IDX], single[POSITION_STATE_IDX]);
}

// ===========================================================================
// 10. Epsilon floor — uniform sizes produce no NaN/Inf
// ===========================================================================
TEST(EncodeWindowTest, EpsilonFloorUniformSizes) {
    // All sizes identical → std = 0 → z-score would be 0/0 without epsilon
    auto window = make_uniform_window(10.0f, W);
    auto encoded = encode_window(window, 0.0f);

    ASSERT_EQ(encoded.size(), static_cast<size_t>(W));

    for (int t = 0; t < W; ++t) {
        for (int i = 0; i < FEATURE_DIM; ++i) {
            EXPECT_FALSE(std::isnan(encoded[t][i]))
                << "NaN at snapshot " << t << " feature " << i;
            EXPECT_FALSE(std::isinf(encoded[t][i]))
                << "Inf at snapshot " << t << " feature " << i;
        }
    }

    // With uniform sizes, all log1p values are equal → (val - mean) = 0
    // z-score = 0 / (0 + 1e-8) ≈ 0
    // All z-scored size features should be approximately 0
    for (int t = 0; t < W; ++t) {
        for (int i = BID_SIZE_BEGIN; i < BID_SIZE_END; ++i) {
            EXPECT_NEAR(encoded[t][i], 0.0f, 1e-3f)
                << "Uniform-size z-score should be ~0 at snap " << t << " feat " << i;
        }
        for (int i = ASK_SIZE_BEGIN; i < ASK_SIZE_END; ++i) {
            EXPECT_NEAR(encoded[t][i], 0.0f, 1e-3f)
                << "Uniform-size z-score should be ~0 at snap " << t << " feat " << i;
        }
        for (int i = TRADE_SIZE_BEGIN; i < TRADE_SIZE_END; ++i) {
            EXPECT_NEAR(encoded[t][i], 0.0f, 1e-3f)
                << "Uniform-size z-score should be ~0 at snap " << t << " feat " << i;
        }
    }
}

TEST(EncodeWindowTest, EpsilonFloorAllZeroSizes) {
    // Edge case: all sizes are 0 → log1p(0) = 0 → mean=0, std=0
    auto window = make_uniform_window(0.0f, W);
    auto encoded = encode_window(window, 0.0f);

    ASSERT_EQ(encoded.size(), static_cast<size_t>(W));

    // No NaN or Inf anywhere
    for (int t = 0; t < W; ++t) {
        for (int i = 0; i < FEATURE_DIM; ++i) {
            EXPECT_FALSE(std::isnan(encoded[t][i]))
                << "NaN at snapshot " << t << " feature " << i;
            EXPECT_FALSE(std::isinf(encoded[t][i]))
                << "Inf at snapshot " << t << " feature " << i;
        }
    }
}

// ===========================================================================
// 15. Window size assertion — wrong-sized input
// ===========================================================================
TEST(EncodeWindowTest, RejectsWrongWindowSize) {
    // Pass fewer than W snapshots — should throw or assert
    auto short_window = make_window(100);  // 100 != 600

    EXPECT_ANY_THROW(encode_window(short_window, 0.0f))
        << "encode_window should reject input of size != W=600";
}

TEST(EncodeWindowTest, RejectsEmptyWindow) {
    std::vector<BookSnapshot> empty_window;

    EXPECT_ANY_THROW(encode_window(empty_window, 0.0f))
        << "encode_window should reject empty input";
}

TEST(EncodeWindowTest, RejectsOversizedWindow) {
    auto long_window = make_window(700);  // 700 != 600

    EXPECT_ANY_THROW(encode_window(long_window, 0.0f))
        << "encode_window should reject input of size != W=600";
}

// ===========================================================================
// Additional edge cases
// ===========================================================================
TEST(EncodeSnapshotTest, ZeroSpread) {
    // Degenerate case: spread = 0 (best bid == best ask, crossed/locked market)
    BookSnapshot snap{};
    snap.bids[0][0] = 4500.00f; snap.bids[0][1] = 10.0f;
    snap.asks[0][0] = 4500.00f; snap.asks[0][1] = 10.0f;
    snap.mid_price = 4500.00f;
    snap.spread = 0.0f;
    snap.time_of_day = 9.5f;

    auto features = encode_snapshot(snap, 0.0f);

    EXPECT_FLOAT_EQ(features[SPREAD_TICKS_IDX], 0.0f);
    // No NaN or Inf
    for (int i = 0; i < FEATURE_DIM; ++i) {
        EXPECT_FALSE(std::isnan(features[i])) << "NaN at feature " << i;
        EXPECT_FALSE(std::isinf(features[i])) << "Inf at feature " << i;
    }
}

TEST(EncodeSnapshotTest, LargeSizeValues) {
    // Verify log1p handles large sizes correctly
    BookSnapshot snap = make_snapshot();
    snap.bids[0][1] = 100000.0f;  // 100k contracts
    snap.asks[0][1] = 50000.0f;

    auto features = encode_snapshot(snap, 0.0f);

    EXPECT_NEAR(features[BID_SIZE_BEGIN], std::log1p(100000.0f), 1e-3f);
    EXPECT_NEAR(features[ASK_SIZE_BEGIN], std::log1p(50000.0f), 1e-3f);
}

TEST(EncodeSnapshotTest, CompleteFeaturesAllPopulated) {
    // Full snapshot with all levels and trades — verify every feature is populated
    BookSnapshot snap = make_snapshot(4500.00f, 4500.25f, 10.5f, 10, 10, 50);
    auto features = encode_snapshot(snap, 0.0f);

    // No feature should be exactly NaN or Inf
    for (int i = 0; i < FEATURE_DIM; ++i) {
        EXPECT_FALSE(std::isnan(features[i])) << "NaN at feature " << i;
        EXPECT_FALSE(std::isinf(features[i])) << "Inf at feature " << i;
    }
}

TEST(EncodeWindowTest, PositionStatePropagatedToAllSnapshots) {
    auto window = make_window(W);
    auto encoded = encode_window(window, -1.0f);

    // Every snapshot in the window should have position_state = -1.0
    for (int t = 0; t < W; ++t) {
        EXPECT_FLOAT_EQ(encoded[t][POSITION_STATE_IDX], -1.0f)
            << "Position state at snapshot " << t << " should be -1.0";
    }
}

TEST(EncodeWindowTest, SpreadPreservedAcrossWindow) {
    auto window = make_window(W);
    auto encoded = encode_window(window, 0.0f);

    // All snapshots in make_window have spread = 0.25 → spread_ticks = 1.0
    for (int t = 0; t < W; ++t) {
        EXPECT_NEAR(encoded[t][SPREAD_TICKS_IDX], 1.0f, 0.01f)
            << "Spread in ticks at snapshot " << t;
    }
}
