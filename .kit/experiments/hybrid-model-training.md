# Experiment: CNN+GBT Hybrid Model Training on MES Triple Barrier Labels

## Hypothesis

The CNN+GBT Hybrid architecture — Conv1d spatial encoder on structured (20,2) book input producing a 16-dim embedding, concatenated with 20 non-spatial hand-crafted features, fed to an XGBoost classifier on triple barrier labels — will achieve:

1. **CNN regression R² at h=5 >= 0.08** (mean across 5 expanding-window folds), reproducing at least 60% of R3's R²=0.132, which used identical architecture and data but different training infrastructure.
2. **XGBoost classification accuracy >= 0.38** on 3-class triple barrier labels (above 0.33 random baseline), with the hybrid feature set (CNN embedding + non-spatial features) outperforming both GBT-only (62 features, no CNN) and CNN-only (no XGBoost, no non-spatial features) baselines.
3. **Aggregate per-trade expectancy >= $0.50** under the base transaction cost scenario ($3.74 round-trip), demonstrating that the model's directional accuracy converts R3's regression signal into economically viable trading signals.

These thresholds are conservative relative to the oracle's $4.00/trade expectancy (R7) and R3's R²=0.132, accounting for the gap between oracle knowledge and model prediction, and the difference between regression R² and classification accuracy on discrete labels.

## Independent Variables

1. **Model configuration** (3 levels):
   - **Hybrid** (primary): CNN 16-dim embedding + 20 non-spatial features → XGBoost classifier (36 input dims)
   - **GBT-only** (ablation baseline): All 62 Track A hand-crafted features → XGBoost classifier (no CNN)
   - **CNN-only** (ablation baseline): CNN encoder → Linear(16, 3) classification head (no XGBoost, no non-spatial features)

2. **CNN regression horizon** (2 levels, Stage 1 only):
   - h=1 (1-bar, ~5s forward return) — resolves R3 open question on CNN at h=1
   - h=5 (5-bar, ~25s forward return) — reproduces R3's primary result

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | `.kit/results/hybrid-model/time_5s.csv` (87,970 bars, 19 days) | Single data source from Phase 9A C++ export; eliminates data processing variance |
| Bar type | time_5s | Locked by R1 (subordination refuted) and R6 synthesis |
| Labels | Triple barrier (target=10, stop=5, vol_horizon=500) | Locked by R7 oracle expectancy ($4.00/trade, PF=3.30) |
| CV strategy | 5-fold expanding-window, split by day boundaries | Consistent with R2/R3/R4; no shuffling, no leakage |
| CNN architecture | Conv1d(2→32, k=3) → Conv1d(32→64, k=3) → Pool → Linear(64→16) | From R3 and synthesis architecture_decision.json |
| XGBoost hyperparameters | max_depth=6, lr=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0 | Reasonable defaults for 36-dim, ~70k rows; no tuning to avoid overfitting the CV |
| Random seed | 42 | Single seed (Standard tier; variance assessed across 5 temporal folds instead) |
| CNN optimizer | Adam(lr=1e-3, weight_decay=1e-5), batch_size=256, max_epochs=50, early_stop patience=5 | From R3 training protocol |
| CNN input normalization | Price offsets: raw ticks from mid. Sizes: z-score per fold (train stats only) | From R3 and synthesis spec |
| Feature normalization | z-score per fold (train stats only). NaN → 0.0 after normalization | No leakage from test fold |
| Hardware | CPU only | CNN is ~7.5k params; XGBoost native CPU. No GPU needed |

## Metrics (ALL must be reported)

### Primary

1. **mean_cnn_r2_h5**: Mean out-of-sample R² of CNN regression at h=5, across 5 folds. This directly tests whether R3's spatial signal reproduces in the full pipeline.
2. **aggregate_expectancy_base**: Per-trade expectancy (dollars) pooled across all test-set predictions under the base cost scenario ($3.74 RT). This tests economic viability.

### Secondary

| Metric | Description |
|--------|-------------|
| mean_cnn_r2_h1 | CNN R² at h=1 across 5 folds (resolves R3 open question) |
| mean_xgb_accuracy | XGBoost 3-class accuracy across 5 folds |
| mean_xgb_f1_macro | Macro F1 across 5 folds |
| aggregate_profit_factor | Gross profit / gross loss pooled across all test predictions (base cost) |
| aggregate_sharpe | Annualized Sharpe of daily PnL across test days |
| per_fold_r2_std | Standard deviation of CNN R² at h=5 across folds |
| ablation_delta_accuracy | Hybrid accuracy minus max(GBT-only, CNN-only) accuracy |
| ablation_delta_expectancy | Hybrid expectancy minus max(GBT-only, CNN-only) expectancy |
| cost_sensitivity_table | Expectancy and PF under optimistic ($2.49), base ($3.74), pessimistic ($6.25) costs |
| xgb_top10_features | XGBoost feature importance (gain) top-10 features |
| label_distribution | Per-fold class counts for tb_label {-1, 0, +1} |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| CNN R² at h=5 on train set | > 0.15 (higher than test) | Underfitting — training broken |
| CNN R² at h=5 on test set for fold 4 or 5 | > 0 | If the best-data folds (most training data) produce negative R², architecture or data pipeline is broken |
| XGBoost accuracy > 0.33 | Above random baseline for 3-class | Model is learning nothing |
| No NaN in CNN output | 0 NaN embeddings | Forward pass or normalization bug |
| Fold train/test day boundaries non-overlapping | No day appears in both | Leakage |
| XGBoost accuracy ≤ 0.90 | Below implausible ceiling | Leakage or label leakage through features |

## Baselines

### 1. R3 CNN Result (historical, for reproduction comparison)
- **Source**: R3 book-encoder-bias experiment
- **Value**: CNN mean R²=0.132 ± 0.048 on return_5, 5-fold expanding-window CV on 19 days
- **Note**: R3 used ~12k param CNN and MSE on return_5. This experiment uses ~7.5k param CNN (slightly smaller architecture) and the same return_5 target. Expect slightly lower R² due to architecture difference, hence threshold of 0.08 (60% of R3).

### 2. Oracle Expectancy (ceiling, from R7)
- **Source**: R7 oracle-expectancy
- **Value**: Triple barrier: $4.00/trade, PF=3.30, WR=64.3%, Sharpe=0.362
- **Note**: The oracle knows the true label. The model's expectancy should be strictly lower. Any model expectancy approaching oracle indicates leakage.

### 3. Random Baseline
- **3-class accuracy**: 0.333
- **Expectancy**: Negative (random direction selection loses to transaction costs)
- If the model fails to beat these, the signal is not learnable from these features.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: mean_cnn_r2_h5 >= 0.08 (CNN reproduces >= 60% of R3's signal)
- [ ] **SC-2**: mean_cnn_r2_h1 is reported (any value — resolves R3 open question; no threshold)
- [ ] **SC-3**: mean_xgb_accuracy >= 0.38 (above random by >= 5 percentage points)
- [ ] **SC-4**: No fold produces negative CNN R² at h=5 (0/5 negative folds)
- [ ] **SC-5**: aggregate_expectancy_base >= $0.50/trade (economically viable under base costs)
- [ ] **SC-6**: aggregate_profit_factor_base >= 1.5 (profitable with margin)
- [ ] **SC-7**: Hybrid outperforms GBT-only baseline on mean_xgb_accuracy OR aggregate_expectancy (at least one)
- [ ] **SC-8**: Hybrid outperforms CNN-only baseline on mean_xgb_accuracy OR aggregate_expectancy (at least one)
- [ ] **SC-9**: Cost sensitivity table produced for all 3 scenarios (optimistic, base, pessimistic)
- [ ] **SC-10**: No sanity check failures

## Minimum Viable Experiment

Before running the full 5-fold CV pipeline:

1. **Data loading check**: Load `time_5s.csv`, verify shape (~87,970 rows), verify expected columns exist (40 book columns, feature columns, `fwd_return_1`, `fwd_return_5`, `tb_label`, `tb_exit_type`, day identifier). Print column list and first 5 rows. Print per-day bar counts and label distribution.

2. **Single-fold CNN overfit check**: On fold 1 only (days 1-4 train, days 5-7 test):
   - Train CNN on h=5 for 50 epochs with early stopping.
   - Print train R² (expect > 0.15) and test R² (expect > 0).
   - If train R² < 0.01: pipeline is broken, abort.
   - If test R² < -0.1: severe overfitting, check normalization.

3. **Single-fold XGBoost check**: Using fold 1 CNN embeddings + non-spatial features:
   - Train XGBoost on tb_label.
   - Print accuracy (expect > 0.33).
   - If accuracy < 0.33: feature pipeline is broken, abort.

If all three MVE checks pass, proceed to full protocol.

## Full Protocol

### Step 0: Environment Setup
- Verify Python dependencies: torch, xgboost, pandas/polars, numpy, scikit-learn.
- Set seed=42 globally (torch, numpy, random).

### Step 1: Data Loading and Validation
- Load `.kit/results/hybrid-model/time_5s.csv`.
- Identify the 19 unique days, sorted chronologically.
- Verify label distribution (tb_label in {-1, 0, +1}).
- Identify the 40 book snapshot columns and reshape to (N, 20, 2) for CNN input.
- Identify the 20 non-spatial feature columns (see feature list below).
- Log dataset statistics: rows, days, label counts, feature NaN counts.

### Step 2: Define 5-Fold Expanding-Window Splits
| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | Days 1-4 | Days 5-7 |
| 2 | Days 1-7 | Days 8-10 |
| 3 | Days 1-10 | Days 11-13 |
| 4 | Days 1-13 | Days 14-16 |
| 5 | Days 1-16 | Days 17-19 |

### Step 3: Run MVE (fold 1 only)
Execute the three MVE checks described above. Abort if any fail.

### Step 4: Full 5-Fold Pipeline
For each fold k in [1..5]:

**Stage 1a: CNN Training (h=1)**
- Normalize book sizes using train-fold statistics.
- Train CNN encoder + Linear(16→1) head on fwd_return_1 (MSE loss).
- Adam(lr=1e-3, weight_decay=1e-5), batch_size=256, max_epochs=50, early_stop patience=5 on last 20% of train days as validation.
- Record train R² and test R².

**Stage 1b: CNN Training (h=5)**
- Same as 1a but target = fwd_return_5.
- Record train R² and test R².

**Stage 1c: Select Best Horizon CNN**
- Select the horizon (h=1 or h=5) with higher mean test R² across completed folds so far.
- Use that CNN for Stage 2 embedding extraction. (Expected: h=5 based on R3.)

**Stage 2: Embedding Extraction**
- Freeze the selected CNN encoder. Remove the regression head.
- Extract 16-dim embeddings for all train and test bars.
- Sanity check: no NaN in embeddings.

**Stage 3: Feature Assembly**
- Normalize 20 non-spatial features using train-fold mean/std.
- Concatenate CNN 16-dim embedding + 20 non-spatial features → 36-dim input.

**Stage 4: XGBoost Classification**
- Train XGBoost (multi:softmax, num_class=3) on train: 36 features → tb_label.
- Predict on test set. Record predictions and probabilities.

**Stage 5: Evaluation**
- CNN R² (h=1, h=5) on test.
- XGBoost accuracy, F1_macro on test.
- PnL computation using simplified model:
  - Predict +1 and true label +1 → PnL = +target_ticks * $1.25 - costs
  - Predict +1 and true label -1 → PnL = -stop_ticks * $1.25 - costs
  - Predict -1 and true label -1 → PnL = +target_ticks * $1.25 - costs
  - Predict -1 and true label +1 → PnL = -stop_ticks * $1.25 - costs
  - Predict 0 or true label 0 → PnL = 0 (no trade)
- Compute expectancy, profit factor, trade count under 3 cost scenarios.
- Save fold results.

### Step 5: Ablation — GBT-Only Baseline
For each fold k in [1..5]:
- Use all 62 Track A features (or all available non-book + book features) → XGBoost with same hyperparameters → tb_label.
- Compute same metrics (accuracy, F1, PnL under 3 cost scenarios).

### Step 6: Ablation — CNN-Only Baseline
For each fold k in [1..5]:
- CNN encoder → Linear(16, 3) classification head on tb_label (CrossEntropyLoss).
- Same CNN training protocol (Adam, lr=1e-3, 50 epochs, early stop).
- Compute same metrics.

### Step 7: Aggregate Results
- Pool test predictions across all 5 folds.
- Compute aggregate metrics (mean R², mean accuracy, pooled expectancy, pooled PF, Sharpe).
- Compute per-fold variance metrics (std of R², negative fold count).
- Produce cost sensitivity table.
- Produce XGBoost feature importance (top-10 by gain).
- Write analysis.md and all JSON/CSV deliverables.

### Non-Spatial Feature Set (20 dimensions)

| # | Feature | Category |
|---|---------|----------|
| 1 | weighted_imbalance | Book Shape |
| 2 | spread | Book Shape |
| 3 | net_volume | Order Flow |
| 4 | volume_imbalance | Order Flow |
| 5 | trade_count | Order Flow |
| 6 | avg_trade_size | Order Flow |
| 7 | vwap_distance | Order Flow |
| 8 | return_1 | Price Dynamics |
| 9 | return_5 | Price Dynamics |
| 10 | return_20 | Price Dynamics |
| 11 | volatility_20 | Price Dynamics |
| 12 | volatility_50 | Price Dynamics |
| 13 | high_low_range_50 | Price Dynamics |
| 14 | close_position | Price Dynamics |
| 15 | cancel_add_ratio | Microstructure |
| 16 | message_rate | Microstructure |
| 17 | modify_fraction | Microstructure |
| 18 | time_sin | Time |
| 19 | time_cos | Time |
| 20 | minutes_since_open | Time |

**Note**: The synthesis specified 20 features with slightly different names (e.g., `trade_imbalance` vs `weighted_imbalance`). The RUN agent must map from actual CSV column names. If a specified feature is missing from the CSV, substitute the closest available feature and document the mapping. The exact 20-feature set is secondary to the CNN+GBT integration; no feature was individually critical in R4 importance.

### Transaction Cost Scenarios

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

### PnL Model

```
tick_value = $0.25 * 5 = $1.25 per tick
target_ticks = 10 → win PnL = $12.50 - costs
stop_ticks = 5 → loss PnL = -$6.25 - costs

If model predicts correct sign (prediction matches tb_label sign):
  PnL = +target_ticks * $1.25 - RT_cost

If model predicts wrong sign (prediction opposes tb_label sign):
  PnL = -stop_ticks * $1.25 - RT_cost

If model predicts 0 (hold) or true label is 0 (expiry):
  PnL = 0 (no trade)
```

## Resource Budget

**Tier:** Standard

- Max GPU-hours: 0 (CPU only)
- Max wall-clock time: 2 hours
- Max training runs: 30 (5 folds × 2 horizons × CNN + 5 folds × XGBoost + 5 folds × GBT-only ablation + 5 folds × CNN-only ablation)
- Max seeds per configuration: 1 (seed=42; variance assessed across 5 temporal folds)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 87970
model_type: pytorch+xgboost
sequential_fits: 30
parallelizable: false
memory_gb: 4
gpu_type: none
estimated_wall_hours: 0.5
```

**Estimate breakdown**: CNN (~7.5k params) on ~74k rows trains in ~10s/fold on CPU. XGBoost (500 trees, 36 features, ~74k rows) trains in ~5s/fold. 30 total fits × ~10s average = ~5 minutes for training. Data loading + feature extraction + evaluation + I/O adds ~10 minutes. Total: ~15-20 minutes. Budget of 2 hours provides 6-8× headroom.

## Abort Criteria

- **CNN training diverges**: If any fold produces NaN loss within 5 epochs, abort. Check normalization pipeline.
- **CNN R² on train set < 0.01 after 50 epochs** (for h=5): The CNN cannot fit the training data. Architecture or data pipeline is broken. Do not proceed to Stage 2.
- **All 5 folds produce negative test R² at h=5**: The R3 signal does not reproduce in this pipeline. Investigate data differences before re-running.
- **XGBoost accuracy < 0.33 on all 5 folds**: The model learns nothing from the features. Check label encoding and feature assembly.
- **Wall-clock exceeds 90 minutes**: Something is pathologically slow. Abort and diagnose.
- **Time-based per-run abort**: If any single CNN training run exceeds 5 minutes (30× expected), kill and investigate. XGBoost: 2 minutes per fold (24× expected).

## Confounds to Watch For

1. **Feature name mismatch**: The 20 non-spatial features are specified by name from the synthesis/R4, but the actual CSV column names from the C++ exporter may differ (e.g., `weighted_imbalance` vs `book_weighted_imbalance`). The RUN agent must inspect actual column names and map accordingly. A column mapping error silently fills features with zeros after normalization, destroying signal.

2. **Book column ordering**: The (20,2) reshape assumes a specific column ordering in the CSV. If columns are interleaved differently (e.g., bid_price_0, bid_size_0, ask_price_0, ask_size_0 vs bid_price_0...bid_price_9, bid_size_0...bid_size_9), the CNN sees scrambled input. The RUN agent must verify ordering matches R3's convention: rows 0-9 = bids (deepest→best), rows 10-19 = asks (best→deepest), column 0 = price_offset, column 1 = size.

3. **Label distribution skew**: Triple barrier labels may be heavily imbalanced (e.g., >50% in class +1 due to the asymmetric 10:5 target:stop ratio). If so, XGBoost class weights or stratified sampling may be needed. The PnL model already accounts for the asymmetric payoff, but accuracy and F1 are sensitive to imbalance. Report the distribution before interpreting accuracy.

4. **h=5 target leakage through features**: `return_5` is both a non-spatial feature and the Stage 1 CNN target at h=5. The CNN sees only book snapshots (not return_5), so no leakage there. But XGBoost sees return_5 as one of 36 features while predicting tb_label, which is partially determined by the 5-bar forward return. If `return_5` leaks into tb_label prediction, XGBoost feature importance for return_5 will be artificially high. **Mitigation**: report return_5's importance rank. If it dominates (>20% gain share), re-run XGBoost excluding return_5 and compare.

5. **CNN architecture mismatch with R3**: R3 used ~12k params (Conv1d channels tuned to match Attention and MLP). This spec uses ~7.5k params (2→32→64). The lower param count may explain lower R² than R3. This is acceptable — the threshold (0.08 vs R3's 0.132) accounts for this. But if R² drops below 0.05, the architecture reduction is too aggressive and should be investigated.

6. **Regime concentration**: R3 showed fold 3 was an outlier (R²=0.049 vs mean 0.132). If this experiment's fold 3 is also an outlier, the signal may be concentrated in high-volatility regimes. Per-fold date ranges must be reported for interpretation.

7. **Expanding window non-stationarity**: Early folds have less training data (4 days) than later folds (16 days). If performance improves monotonically with fold number, it may reflect data quantity rather than signal quality. Report per-fold train set sizes alongside R².

## Deliverables

```
.kit/results/hybrid-model/
  fold_1/ through fold_5/
    cnn_h1_metrics.json       # {train_r2, test_r2, epochs_trained}
    cnn_h5_metrics.json       # {train_r2, test_r2, epochs_trained}
    xgb_metrics.json          # {accuracy, f1_macro, class_report}
    pnl_metrics.json          # {expectancy, pf, sharpe, trade_count} x 3 cost scenarios
    predictions.csv           # Test-set: bar_index, true_label, predicted_label, prob_neg, prob_zero, prob_pos
    cnn_encoder.pt            # Saved CNN weights (selected horizon)
    xgb_model.json            # Saved XGBoost model
  aggregate_metrics.json      # Pooled metrics across all folds + ablation deltas
  ablation_gbt_only.json      # GBT-only baseline: per-fold and aggregate metrics
  ablation_cnn_only.json      # CNN-only baseline: per-fold and aggregate metrics
  cost_sensitivity.json       # 3 cost scenarios x {expectancy, pf, trade_count}
  feature_importance.json     # XGBoost top-10 features by gain (per fold + aggregate)
  analysis.md                 # Human-readable summary with explicit SC pass/fail
```

### Required Outputs in analysis.md

1. Executive summary: does the hybrid model produce actionable signals?
2. CNN R² table: fold × horizon (h=1, h=5) — resolves R3 open h=1 question.
3. XGBoost accuracy and F1 table: fold × metric.
4. PnL table: fold × cost scenario → expectancy, profit factor.
5. Ablation comparison: hybrid vs GBT-only vs CNN-only (table with all metrics).
6. Label distribution: per-fold class counts for tb_label.
7. Feature importance: XGBoost top-10 features by gain.
8. Fold date ranges and train set sizes for regime interpretation.
9. Explicit pass/fail for each of SC-1 through SC-10.
