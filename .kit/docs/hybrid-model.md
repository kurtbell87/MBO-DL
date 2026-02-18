# hybrid_model — TDD Spec

## Summary

End-to-end CNN+GBT Hybrid model pipeline: C++ data export (with triple barrier labels and structured book), Python CNN encoder training, embedding extraction, XGBoost classification, and 5-fold expanding window cross-validation.

This is the first model that produces actionable trading signals. All research phases are complete — this spec implements the architecture recommended by the R6 synthesis.

## Research Inputs (Locked)

| Decision | Source | Value |
|----------|--------|-------|
| Architecture | R6 synthesis | CNN + GBT Hybrid (OPTION B) |
| Spatial encoder | R3 (CNN R²=0.132, p=0.042 vs Attention) | Conv1d on (20,2) book → 16-dim embedding |
| Message encoder | R2 (Δ_msg < 0) | None — dropped |
| Temporal encoder | R4 chain (0/168+ passes) | None — dropped |
| Bar type | R1 (subordination refuted) | time_5s |
| Labels | R7b (TB: $4.00/trade, PF=3.30) | Triple barrier (target=10, stop=5, vol_horizon=500) |
| Horizons | R6 synthesis | h=1 and h=5 |
| Features | R4 importance + R6 §4.2 | ~20 non-spatial + CNN 16-dim |

## Architecture

```
Per bar:

  Book State (20 levels × 2 channels)       Non-Spatial Features (~20 dims)
  [bids_reversed + asks_ordered]             [order flow, price dynamics,
  (price_offset_from_mid, size)               volatility, time, microstructure]
       │                                           │
       ▼                                           │
  Conv1d Encoder (~2.6k params)                    │
  ────────────────────────────                     │
  Permute → (B, 2, 20)                            │
  → Conv1d(2, 32, k=3, pad=1) → ReLU             │
  → Conv1d(32, 64, k=3, pad=1) → ReLU            │
  → AdaptiveAvgPool1d(1) → (B, 64)               │
  → Linear(64, 16) → (B, 16)                      │
       │                                           │
       └────────── Concatenate ────────────────────┘
                      │
                      ▼
              Combined (B, 36)
                      │
                      ▼
              XGBoost Classifier
                      │
                      ▼
              Triple Barrier Label {−1, 0, +1}
```

### CNN Encoder Detail

| Layer | In | Out | Kernel | Params |
|-------|----|-----|--------|--------|
| Conv1d_1 | 2 | 32 | 3, pad=1 | 2×32×3 + 32 = 224 |
| Conv1d_2 | 32 | 64 | 3, pad=1 | 32×64×3 + 64 = 6,208 |
| AdaptiveAvgPool1d(1) | — | — | — | 0 |
| Linear | 64 | 16 | — | 64×16 + 16 = 1,040 |
| **Total** | | | | **~7.5k** |

### CNN Training Head (removed after Stage 1)

Linear(16, 1) for fwd_return_h regression. Params: 16 + 1 = 17.

Stage 1 trains CNN end-to-end with this regression head. After training, the head is discarded and the 16-dim embedding is frozen.

## Two-Stage Training

### Stage 1: CNN Encoder Training (per horizon)

Train the CNN with a linear regression head on forward return prediction.

```
Input: (B, 2, 20) — book snapshot per bar
Target: fwd_return_h (h=1 or h=5)
Loss: MSE
Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
Batch size: 256
Max epochs: 50
Early stopping: patience=5, monitor=val_loss
Seed: 42
```

Train two separate CNN encoders: one for h=1, one for h=5. Compare R² to determine which horizon the CNN adds value at.

### Stage 2: XGBoost Classification

Freeze the trained CNN encoder. Extract 16-dim embeddings for all bars. Concatenate with non-spatial features. Train XGBoost on triple barrier labels.

```
Input: CNN_embedding (16) + non_spatial_features (~20) = ~36 dims
Target: triple_barrier_label ∈ {−1, 0, +1}
Objective: multi:softmax, num_class=3
Max depth: 6
Learning rate: 0.05
N estimators: 500
Subsample: 0.8
Colsample_bytree: 0.8
Min_child_weight: 10
Reg_alpha: 0.1
Reg_lambda: 1.0
Seed: 42
```

Note: XGBoost hyperparameters are intentionally conservative (lower depth, higher regularization) vs. the overfit-phase GBT spec. This is a generalization model, not an overfit test.

## Data Pipeline

### Phase A: C++ Data Export Extension

Extend `bar_feature_export` to emit triple barrier labels alongside existing outputs.

**New columns in export CSV:**

| Column | Type | Description |
|--------|------|-------------|
| `tb_label` | int | Triple barrier label: −1 (short), 0 (hold), +1 (long) |
| `tb_exit_type` | string | "target", "stop", "expiry", "timeout" |
| `tb_bars_held` | int | Bars from entry to barrier hit |

**Triple barrier parameters** (from R7b oracle expectancy):

```
target_ticks = 10
stop_ticks = 5
volume_horizon = 500
min_return_ticks = 2
max_time_horizon_s = 300
```

The labeling is position-independent: at each bar, ask "if we entered long here, would we hit target or stop first?" This produces a directional label without needing to track position state.

**Implementation**: Add triple barrier scan to the export loop in `bar_feature_export.cpp`. For each bar `i`, scan forward through subsequent bars accumulating volume until `volume_horizon` is reached or `max_time_horizon_s` elapses. Assign label based on which barrier is hit first.

### Phase B: Python Training Pipeline

Python scripts in `scripts/hybrid_model/`:

```
scripts/hybrid_model/
  train_cnn.py          # Stage 1: CNN encoder training
  extract_embeddings.py # Extract 16-dim embeddings from frozen CNN
  train_xgboost.py      # Stage 2: XGBoost on embeddings + features
  evaluate_cv.py        # 5-fold expanding window CV
  config.py             # Shared configuration
  data_loader.py        # Load exported CSV, normalize, split
  metrics.py            # Evaluation metrics (R², accuracy, PnL, Sharpe)
```

## Non-Spatial Feature Set (~20 dimensions)

Selected from the 62 Track A features based on R4 importance analysis and R6 §4.2. Excludes features redundant with the CNN's (20, 2) book input (depth profiles, book imbalances at specific levels, slopes).

| # | Feature | Source Category | Justification |
|---|---------|----------------|---------------|
| 1 | weighted_imbalance | Cat 1: Book Shape | Aggregate book signal (not per-level) |
| 2 | spread | Cat 1: Book Shape | Scalar, not captured by price offsets |
| 3 | net_volume | Cat 2: Order Flow | Non-spatial |
| 4 | volume_imbalance | Cat 2: Order Flow | Non-spatial |
| 5 | trade_count | Cat 2: Order Flow | Activity level |
| 6 | avg_trade_size | Cat 2: Order Flow | Institutional flow proxy |
| 7 | vwap_distance | Cat 2: Order Flow | R4 top-10 at h=20, h=100 |
| 8 | return_1 | Cat 3: Price Dynamics | Current-bar return |
| 9 | return_5 | Cat 3: Price Dynamics | Short momentum |
| 10 | return_20 | Cat 3: Price Dynamics | Medium momentum |
| 11 | volatility_20 | Cat 3: Price Dynamics | Short-term regime |
| 12 | volatility_50 | Cat 3: Price Dynamics | R4 top-10 at h=100 |
| 13 | high_low_range_50 | Cat 3: Price Dynamics | R4 top-10 at h=5, h=20 |
| 14 | close_position | Cat 3: Price Dynamics | R4 top-10 at h=20 |
| 15 | cancel_add_ratio | Cat 6: Microstructure | Retained as scalar |
| 16 | message_rate | Cat 6: Microstructure | Activity proxy |
| 17 | modify_fraction | Cat 6: Microstructure | R4 HC top-10 at h=1 |
| 18 | time_sin | Cat 5: Time | R4 HC top-10 at h=5 |
| 19 | time_cos | Cat 5: Time | R4 HC top-10 at h=1 |
| 20 | minutes_since_open | Cat 5: Time | R4 HC top-10 at h=20, h=100 |

**Total non-spatial dimension**: 20.

### Features explicitly excluded (redundant with CNN)

| Feature Group | Dim | Reason |
|---------------|-----|--------|
| bid_depth_profile_0..9 | 10 | CNN sees raw book levels |
| ask_depth_profile_0..9 | 10 | CNN sees raw book levels |
| book_imbalance_1/3/5/10 | 4 | CNN can learn these from raw levels |
| depth_concentration_bid/ask | 2 | Derived from per-level sizes |
| book_slope_bid/ask | 2 | Derived from per-level sizes |
| level_count_bid/ask | 2 | Implicit in zero-padding of CNN input |

## Normalization

### CNN input normalization (per day)
- **Price offsets**: Already in ticks from mid (via `BookSnapshotExport::from_bar()`)
- **Sizes**: z-score normalize per day. Compute mean and std of all size values (bid + ask) across the day. Apply: `(size - mean) / (std + 1e-8)`.

### Non-spatial feature normalization (per expanding window)
- z-score normalize each feature using training set statistics only.
- No leakage: test fold features are normalized using training fold mean/std.
- Handle NaN: fill with 0.0 after normalization (NaN features are rare, only in first few bars which are excluded by warmup).

## Cross-Validation Protocol

5-fold expanding window, consistent with R2/R3/R4 methodology.

### Fold Structure (19 days)

| Fold | Train Days | Test Days | Train Bars (approx) | Test Bars (approx) |
|------|-----------|-----------|---------------------|-------------------|
| 1 | Days 1-4 | Days 5-7 | ~18k | ~14k |
| 2 | Days 1-7 | Days 8-10 | ~32k | ~14k |
| 3 | Days 1-10 | Days 11-13 | ~46k | ~14k |
| 4 | Days 1-13 | Days 14-16 | ~60k | ~14k |
| 5 | Days 1-16 | Days 17-19 | ~74k | ~14k |

Days are the 19 selected days in `SELECTED_DAYS` from `bar_feature_export.cpp`, mapped to the fold structure.

### Per-Fold Pipeline

```
For each fold k:
  1. Split data into train/test by day boundaries
  2. Normalize features using train-set statistics
  3. Normalize CNN book sizes using train-set size statistics
  4. Train CNN encoder on train data (fwd_return_h regression)
  5. Extract 16-dim embeddings for train + test data
  6. Train XGBoost on train embeddings + features → TB labels
  7. Predict on test data
  8. Compute metrics on test data
```

## Evaluation Metrics

### Per-fold metrics (test set)

| Metric | Description | Target |
|--------|-------------|--------|
| cnn_r2_h1 | CNN regression R² for h=1 | Report (no minimum — this is the unknown) |
| cnn_r2_h5 | CNN regression R² for h=5 | ≥ 0.08 (below R3's 0.132 due to smaller training set in early folds) |
| xgb_accuracy | XGBoost classification accuracy | ≥ 0.38 (above 1/3 random for 3-class) |
| xgb_f1_macro | Macro F1 across 3 classes | Report |
| expectancy | Mean PnL per predicted trade after costs | Report |
| profit_factor | Gross profit / gross loss | Report |
| trade_count | Number of non-HOLD predictions | Report |
| sharpe | Annualized Sharpe ratio of daily PnL | Report |

### Aggregate metrics (across all test folds)

| Metric | Description | Target |
|--------|-------------|--------|
| mean_cnn_r2_h5 | Mean CNN R² at h=5 across folds | ≥ 0.08 |
| mean_xgb_accuracy | Mean XGBoost accuracy | ≥ 0.38 |
| fold_r2_std | Std of CNN R² across folds | Report (lower is better) |
| negative_fold_count | Folds with R² < 0 | 0 (all folds must be non-negative) |
| aggregate_expectancy | Expectancy pooled across all test predictions | ≥ $0.50/trade |
| aggregate_pf | Profit factor pooled across all test predictions | ≥ 1.5 |

### Transaction cost sensitivity

Run the evaluation with 3 cost scenarios:

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

Report expectancy and profit factor under each scenario.

## PnL Computation

For each test-set prediction:
- Prediction = +1 (long) or −1 (short): enter at mid_price
- Prediction = 0 (hold): no trade, PnL = 0
- Realized PnL = direction × (exit_price − entry_price) × $5.00/point − costs
- Exit price: use actual forward price path from the export data. Apply the same triple barrier logic to determine exit (target, stop, or expiry).

This is NOT oracle PnL — the model predicts direction, and the actual realized outcome determines PnL. The oracle expectancy ($4.00/trade) is the ceiling.

## Ablation Comparisons

To validate that the CNN+GBT hybrid outperforms simpler baselines, also train and evaluate:

1. **GBT-only baseline**: XGBoost on all 62 Track A features (no CNN). Same CV protocol. This is the "no spatial encoder" baseline.
2. **CNN-only baseline**: CNN with classification head directly on TB labels (no XGBoost, no non-spatial features). This isolates the CNN contribution.

Report the same metrics for both baselines. The hybrid should outperform both.

## File Layout

### C++ (data export extension)

```
tools/
  bar_feature_export.cpp   # Modified: add TB label columns
src/
  backtest/triple_barrier.hpp  # Already exists — may need label function
```

### Python (training pipeline)

```
scripts/hybrid_model/
  config.py             # Paths, feature lists, hyperparameters
  data_loader.py        # Load CSV, normalize, fold splitting
  cnn_encoder.py        # PyTorch CNN encoder module
  train_cnn.py          # Stage 1 training loop
  extract_embeddings.py # Embedding extraction from frozen CNN
  train_xgboost.py      # Stage 2 XGBoost training
  evaluate_cv.py        # Full 5-fold CV pipeline
  metrics.py            # Evaluation: R², accuracy, PnL, Sharpe
  run_all.py            # End-to-end: export → train → evaluate
```

### Output

```
.kit/results/hybrid-model/
  fold_{k}/
    cnn_h1_metrics.json
    cnn_h5_metrics.json
    xgb_metrics.json
    predictions.csv
    cnn_encoder_h5.pt     # Saved CNN weights
    xgb_model.json        # Saved XGBoost model
  aggregate_metrics.json
  ablation_gbt_only.json
  ablation_cnn_only.json
  cost_sensitivity.json
  analysis.md
```

## Dependencies

### C++ (existing)

- databento-cpp, libtorch, xgboost C API, GTest (all via FetchContent)

### Python (new)

```
torch >= 2.0
xgboost >= 2.0
pandas
numpy
scikit-learn
```

No CUDA required — CNN is ~7.5k params, trains in seconds on CPU. XGBoost also CPU-only.

## Test Cases

### C++ Tests (bar_feature_export extension)

1. **TB label computation** — Synthetic bar sequence with known price path. Verify label = +1 when target hit first, −1 when stop hit first, 0 when expiry with small return.
2. **TB exit type** — Verify exit_type correctly reports "target", "stop", "expiry", "timeout".
3. **TB bars held** — Verify bars_held count matches the number of bars scanned.
4. **TB volume accumulation** — Verify volume_horizon is respected: scan stops when cumulative volume ≥ 500.
5. **TB time cap** — Verify max_time_horizon_s = 300 triggers timeout when volume insufficient.
6. **TB min_return filter** — At expiry, return < min_return_ticks → label = 0.
7. **TB label distribution** — On exported data: verify 3 classes present, no class > 60%.
8. **TB no NaN labels** — All bars with sufficient forward data have a valid label.
9. **Export CSV schema** — Verify new columns present and parseable.
10. **Export backward compatibility** — Existing columns unchanged; old consumers unaffected.

### Python Tests (training pipeline)

11. **CNN output shape** — Input (B=4, 2, 20) → output (4, 16).
12. **CNN forward no NaN** — Random input produces finite output.
13. **CNN gradient flow** — All parameters receive non-zero gradients after backward pass.
14. **CNN regression loss decreases** — 10 epochs on synthetic data, MSE decreases.
15. **CNN deterministic** — Two runs with seed=42 produce identical loss at epoch 5.
16. **Feature selector** — Verify exactly 20 features selected from 62-feature CSV.
17. **Feature normalization** — After z-score: mean ≈ 0, std ≈ 1 (within tolerance).
18. **Fold splitting** — Verify expanding window: fold k train set is strict subset of fold k+1 train set. No test day appears in any earlier train set.
19. **No data leakage** — Test fold features normalized with train-fold statistics only.
20. **XGBoost trains** — Train on synthetic 36-dim features + 3-class labels, verify predictions in {−1, 0, +1}.
21. **XGBoost deterministic** — Two runs with seed=42 produce identical predictions.
22. **PnL computation** — Known prediction + known price path → verify PnL calculation matches hand-computed result.
23. **Cost scenarios** — Verify 3 cost levels produce decreasing expectancy (optimistic > base > pessimistic).
24. **Full pipeline integration** — Run evaluate_cv.py on a small subset (2 folds, 3 days each). Verify output JSON schema, all metrics present, no NaN.
25. **Ablation GBT-only** — Verify GBT-only baseline runs and produces valid metrics.
26. **Ablation CNN-only** — Verify CNN-only baseline runs and produces valid metrics.

## Exit Criteria

- [ ] `bar_feature_export` produces CSV with `tb_label`, `tb_exit_type`, `tb_bars_held` columns
- [ ] Triple barrier labels have 3 classes present, no class > 60% of total
- [ ] CNN encoder forward pass: (B, 2, 20) → (B, 16), no NaN
- [ ] CNN R² at h=5: mean across 5 folds ≥ 0.08
- [ ] CNN R² at h=1: reported (positive or negative — this resolves the R3 open question)
- [ ] XGBoost accuracy on triple barrier: mean across 5 folds ≥ 0.38
- [ ] No fold produces negative CNN R² at h=5 (0 negative folds)
- [ ] Aggregate test-set expectancy (base cost scenario) ≥ $0.50/trade
- [ ] Aggregate profit factor (base cost scenario) ≥ 1.5
- [ ] Transaction cost sensitivity table produced for 3 scenarios
- [ ] Ablation: hybrid outperforms GBT-only on mean R² or mean accuracy
- [ ] Ablation: hybrid outperforms CNN-only on mean accuracy or expectancy
- [ ] Per-fold results saved to `.kit/results/hybrid-model/fold_{k}/`
- [ ] Aggregate results saved to `.kit/results/hybrid-model/aggregate_metrics.json`
- [ ] Analysis document written to `.kit/results/hybrid-model/analysis.md`
- [ ] All existing C++ unit tests still pass (1003+)
- [ ] All Python tests pass
