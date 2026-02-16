# feature_encoder — TDD Spec

## Summary

C++ module that transforms a `BookSnapshot` into a fixed-width 194-dimensional feature vector. Also provides window-level encoding that z-scores size features across an observation window of W=600 snapshots.

## Input

- `BookSnapshot` (from `book_builder.hpp`)
- Position state: float (-1.0, 0.0, +1.0)
- For window encoding: `std::vector<BookSnapshot>` of length W=600

## Output

- Per-snapshot: `std::array<float, 194>` feature vector
- Per-window: `std::vector<std::array<float, 194>>` of length W, with z-scored size features

## Feature Vector Layout

```
Index range   | Field                | Description
[0:10]        | bid_prices_delta     | (bid_price[i] - mid_price) / tick_size
[10:20]       | bid_sizes_norm       | log1p(bid_size[i]), z-scored per window
[20:30]       | ask_prices_delta     | (ask_price[i] - mid_price) / tick_size
[30:40]       | ask_sizes_norm       | log1p(ask_size[i]), z-scored per window
[40:90]       | trade_prices_delta   | (trade_price[j] - mid_price) / tick_size
[90:140]      | trade_sizes_norm     | log1p(trade_size[j]), z-scored per window
[140:190]     | trade_aggressor      | -1.0 (sell) / +1.0 (buy)
[190]         | spread_ticks         | spread / tick_size
[191]         | time_sin             | sin(2π × fractional_hour / 24)
[192]         | time_cos             | cos(2π × fractional_hour / 24)
[193]         | position_state       | -1.0 (short) / 0.0 (flat) / +1.0 (long)
```

**Constants**: `feature_dim = 194`, `L = 10`, `T = 50`, `W = 600`, `tick_size = 0.25f`

## Named Index Constants (REQUIRED)

Export as `constexpr` in `feature_encoder.hpp`:

```cpp
constexpr int FEATURE_DIM = 194;
constexpr int L = 10;
constexpr int T = 50;
constexpr int W = 600;
constexpr float TICK_SIZE = 0.25f;

constexpr int BID_PRICE_BEGIN = 0;
constexpr int BID_PRICE_END = 10;
constexpr int BID_SIZE_BEGIN = 10;
constexpr int BID_SIZE_END = 20;
constexpr int ASK_PRICE_BEGIN = 20;
constexpr int ASK_PRICE_END = 30;
constexpr int ASK_SIZE_BEGIN = 30;
constexpr int ASK_SIZE_END = 40;
constexpr int TRADE_PRICE_BEGIN = 40;
constexpr int TRADE_PRICE_END = 90;
constexpr int TRADE_SIZE_BEGIN = 90;
constexpr int TRADE_SIZE_END = 140;
constexpr int TRADE_AGGRESSOR_BEGIN = 140;
constexpr int TRADE_AGGRESSOR_END = 190;
constexpr int SPREAD_TICKS_IDX = 190;
constexpr int TIME_SIN_IDX = 191;
constexpr int TIME_COS_IDX = 192;
constexpr int POSITION_STATE_IDX = 193;
```

## Normalization Rules

### Price deltas
- `delta = (price - mid_price) / tick_size` — expressed in ticks from mid
- For zero-padded levels (price=0.0): delta will be a large negative number (mid is positive). This is acceptable.

### Size normalization (per-window z-score)
- Raw: `log1p(size)` for each size value
- Collect ALL size values across the window (bid sizes + ask sizes + trade sizes): `W * (L + L + T)` = `600 * 70` = `42,000` values
- Compute `mean` and `std` of the log1p values across the entire window
- Z-score: `(log1p_val - mean) / (std + 1e-8)`
- The epsilon `1e-8` prevents division by zero when all sizes are identical

### Time encoding
- `fractional_hour` = hours since midnight ET (0.0 - 24.0)
- `time_sin = sin(2π × fractional_hour / 24)`
- `time_cos = cos(2π × fractional_hour / 24)`

### Aggressor side
- Bid/'B' = +1.0 (buy aggressor)
- Ask/'A' = -1.0 (sell aggressor)
- Zero-padded trades = 0.0

### Position state
- Raw value injected: -1.0 (short), 0.0 (flat), +1.0 (long)

## API Design

```cpp
// Per-snapshot encoding (without z-scoring — raw log1p for sizes)
std::array<float, FEATURE_DIM> encode_snapshot(
    const BookSnapshot& snap,
    float position_state
);

// Window encoding (with z-scored sizes)
std::vector<std::array<float, FEATURE_DIM>> encode_window(
    const std::vector<BookSnapshot>& snapshots,  // exactly W snapshots
    float position_state
);
```

## Dependencies

- `book_builder.hpp` (for BookSnapshot struct)
- C++20
- GTest for tests

## File Layout

```
src/
  feature_encoder.hpp    # Constants + function declarations
  feature_encoder.cpp    # Implementation
tests/
  feature_encoder_test.cpp  # GTest unit tests
```

## Test Cases

1. **Feature dim is 194** — Verify `FEATURE_DIM == 194` and all index ranges are contiguous and non-overlapping.
2. **Bid price deltas** — Snapshot with known bid prices and mid_price. Verify `features[0:10]` = `(bid_price[i] - mid_price) / 0.25`.
3. **Ask price deltas** — Same for ask prices at `features[20:30]`.
4. **Trade price deltas** — Known trade prices, verify `features[40:90]`.
5. **Spread in ticks** — Known best bid/ask, verify `features[190] = spread / 0.25`.
6. **Time encoding** — Known timestamp (e.g., 12:00 ET = 12.0h). Verify `sin(2π*12/24) = 0.0`, `cos(2π*12/24) = 1.0`.
7. **Position state injection** — Encode with position_state = +1.0, verify `features[193] = 1.0`. Repeat for 0.0 and -1.0.
8. **Aggressor side encoding** — Trades with side 'B' → +1.0, side 'A' → -1.0, zero-padded → 0.0.
9. **Size z-scoring across window** — Window of W=600 snapshots with known sizes. Verify mean ≈ 0 and std ≈ 1 for z-scored size features (within floating point tolerance).
10. **Epsilon floor** — Window where all sizes are identical. Verify no NaN/Inf in output (epsilon prevents div-by-zero).
11. **Zero-padded levels** — Snapshot with < 10 levels. Verify padding produces valid (non-NaN) features.
12. **Zero-padded trades** — Fewer than 50 trades. Verify padding features are well-defined.
13. **Index constants contiguous** — Verify `BID_PRICE_END == BID_SIZE_BEGIN`, etc. — no gaps or overlaps.
14. **Round-trip consistency** — Encode a snapshot, verify total feature count is exactly 194.
15. **Window size assertion** — Pass vector of size != W to `encode_window`, expect error/assertion.
