# gbt_model — TDD Spec

## Summary

C++ XGBoost GBT model using hand-crafted features from the observation window. Uses XGBoost C API for training and prediction. Includes a `gbt_features` module for feature engineering with trade deduplication.

## GBT Feature Engineering (16 features)

All features computed from the observation window ending at timestep `t` (where `t = W - 1`).

### Book features (4)
1. **book_imbalance**: `(sum(bid_sizes) - sum(ask_sizes)) / (sum(bid_sizes) + sum(ask_sizes) + 1e-8)`. Range [-1, 1]. Raw sizes.
2. **spread_ticks**: `spread / tick_size`.
3. **book_depth_ratio_5**: `sum(bid_sizes[0:5]) / (sum(ask_sizes[0:5]) + 1e-8)`.
4. **top_level_size_ratio**: `bid_size[0] / (ask_size[0] + 1e-8)`.

### Price dynamics (4)
5. **mid_return_1s**: `(mid_price[t] - mid_price[t-10]) / tick_size`.
6. **mid_return_5s**: `(mid_price[t] - mid_price[t-50]) / tick_size`.
7. **mid_return_30s**: `(mid_price[t] - mid_price[t-300]) / tick_size`.
8. **mid_return_60s**: `(mid_price[t] - mid_price[t-599]) / tick_size`.

### Trade features (4, DEDUPLICATED per §2.7)
9. **trade_imbalance_1s**: `sum(trade_size × aggressor_side)` for deduplicated trades in `[t-10 : t+1]`.
10. **trade_imbalance_5s**: Same for `[t-50 : t+1]`.
11. **trade_arrival_rate_5s**: `count(dedup trades with size > 0 in [t-50 : t+1]) / 5.0`.
12. **large_trade_flag**: `1.0` if any dedup trade in `[t-10 : t+1]` has `size > 2 × median(all dedup trade sizes in window)`. Else `0.0`.

### VWAP (1)
13. **vwap_distance**: `(mid_price[t] - VWAP) / tick_size`. VWAP from dedup trades in full window.

### Time (2)
14. **time_sin**: `sin(2π × fractional_hour / 24)`.
15. **time_cos**: `cos(2π × fractional_hour / 24)`.

### Position (1)
16. **position_state**: -1.0, 0.0, or +1.0.

## Trade Deduplication

Trades appear in multiple consecutive 100ms snapshots. To compute trade features over `[t_start : t_end]`:
1. Collect all trade arrays from snapshots in range
2. Deduplicate by `(price, size, aggressor_side)` 4-tuple (or sequence number if available)
3. Filter: discard entries where `size == 0` (padding)
4. Compute features on deduplicated set

## XGBoost Configuration

```
objective: multi:softmax
num_class: 5
max_depth: 10
learning_rate: 0.1
n_estimators: 1000
subsample: 1.0
colsample_bytree: 1.0
min_child_weight: 1
seed: 42
```

## API

```cpp
// Feature extraction
constexpr int GBT_FEATURE_DIM = 16;

std::array<float, GBT_FEATURE_DIM> compute_gbt_features(
    const std::vector<BookSnapshot>& window,  // exactly W=600 snapshots
    float position_state
);

// GBT wrapper
class GBTModel {
public:
    GBTModel();
    ~GBTModel();
    void train(const std::vector<std::array<float, GBT_FEATURE_DIM>>& features,
               const std::vector<int>& labels,
               int num_rounds = 1000);
    std::vector<int> predict(const std::vector<std::array<float, GBT_FEATURE_DIM>>& features);
    void save(const std::string& path);
    void load(const std::string& path);
};

struct GBTOverfitResult {
    float accuracy;
    int rounds;
    bool success;
};

GBTOverfitResult overfit_gbt(
    const std::vector<TrainingSample>& samples,
    int max_rounds = 1000,
    float target_accuracy = 0.99f
);
```

## Dependencies

- xgboost >= 2.0 (C API, via CMake FetchContent)
- `book_builder.hpp` (BookSnapshot)
- `feature_encoder.hpp` (constants)
- `trajectory_builder.hpp` (TrainingSample)
- GTest

## File Layout

```
src/
  gbt_features.hpp     # compute_gbt_features declaration + constants
  gbt_features.cpp     # Feature engineering + deduplication
  gbt_model.hpp        # GBTModel class + GBTOverfitResult
  gbt_model.cpp        # XGBoost C API wrapper + overfit function
tests/
  gbt_features_test.cpp  # Feature engineering unit tests
  gbt_model_test.cpp     # Model wrapper tests
```

## Test Cases

### Feature Engineering Tests

1. **Feature dim is 16** — Verify `GBT_FEATURE_DIM == 16`.
2. **Book imbalance** — Synthetic snapshot: bid_sizes = [10,5,...], ask_sizes = [5,3,...]. Verify correct imbalance value.
3. **Spread ticks** — Known spread, verify `spread / 0.25`.
4. **Book depth ratio** — Known top-5 sizes, verify ratio.
5. **Top level size ratio** — Known BBO sizes, verify ratio.
6. **Mid returns** — Window with known mid_prices. Verify 1s, 5s, 30s, 60s returns in ticks.
7. **Trade deduplication** — Overlapping trade buffers across snapshots, verify dedup produces correct count.
8. **Trade imbalance** — Known trades with mixed sides, verify net signed volume.
9. **Trade arrival rate** — Known number of dedup trades in 5s window, verify rate.
10. **Large trade flag** — One trade at 3× median size, verify flag = 1.0. All trades below threshold, verify flag = 0.0.
11. **VWAP distance** — Known trades, verify VWAP computation and distance from mid in ticks.
12. **No NaN in features** — Edge case: empty trades (all padding), verify all features are finite.
13. **Epsilon guards** — Both sides empty (all sizes zero), verify no division by zero.

### GBT Model Tests

14. **Train and predict** — Train on 32 synthetic samples, predict, verify predictions in {0,1,2,3,4}.
15. **Overfit synthetic** — 32 samples with deterministic labels, train 1000 rounds, verify >= 95% accuracy.
16. **Save/load** — Train, save, load, predict. Verify identical predictions.
17. **Deterministic with seed** — Two runs with seed=42, same data → identical predictions.
18. **Prediction speed** — 128 samples predicted in < 10ms.
