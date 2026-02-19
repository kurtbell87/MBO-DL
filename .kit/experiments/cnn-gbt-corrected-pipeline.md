# Experiment: CNN+GBT Hybrid with Corrected Normalization Pipeline

## Hypothesis

The CNN+GBT Hybrid architecture — R3-exact Conv1d spatial encoder on structured (20,2) book input (with TICK_SIZE normalization and per-day z-scoring) producing a 16-dim embedding, concatenated with 20 non-spatial hand-crafted features, fed to an XGBoost classifier on triple barrier labels — will achieve:

1. **CNN regression R² at h=5 >= 0.06** (mean across 5 expanding-window folds), reproducing the proper-validation R²≈0.084 discovered in 9D (±30% margin).
2. **XGBoost classification accuracy >= 0.38** on 3-class triple barrier labels (above 0.33 random baseline), with the hybrid feature set (CNN embedding + non-spatial features) outperforming GBT-only baseline.
3. **Aggregate per-trade expectancy >= $0.50** under the base transaction cost scenario ($3.74 round-trip).

These thresholds are calibrated to 9D's proper-validation R²=0.084 (not R3's leaked 0.132). The CNN signal is real but 36% weaker than originally believed.

## Background: Root Cause Resolution (9D)

Phase 9D (`r3-reproduction-pipeline-comparison`) definitively resolved the CNN R²=0.132→0.002 collapse:

- **R3's data is byte-identical to the C++ export** — there was never a "Python vs C++ pipeline" difference.
- **Root cause of 9B/9C failure**: (1) Missing TICK_SIZE division on prices, (2) Per-fold z-scoring instead of per-day z-scoring on sizes.
- **R3's R²=0.132 was inflated ~36%** by test-as-validation leakage (using test set for early stopping).
- **Proper-validation R²=0.084** — still 12× higher than R2's flattened MLP R²=0.007.

This experiment applies the corrected normalization to the full CNN+GBT hybrid pipeline.

## Independent Variables

1. **Model configuration** (3 levels):
   - **Hybrid** (primary): CNN 16-dim embedding + 20 non-spatial features → XGBoost classifier (36 input dims)
   - **GBT-only** (ablation baseline): All available hand-crafted features → XGBoost classifier (no CNN)
   - **CNN-only** (ablation baseline): CNN encoder → Linear(16, 3) classification head (no XGBoost, no non-spatial features)

2. **CNN regression horizon** (2 levels, Stage 1 only):
   - h=1 (1-bar, ~5s forward return)
   - h=5 (5-bar, ~25s forward return) — primary, matches R3/9D

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | `.kit/results/hybrid-model/time_5s.csv` (87,970 bars, 19 days) | Same C++ export used in 9B, 9C, 9D — byte-identical to R3's source |
| Bar type | time_5s | Locked by R1+R6 synthesis |
| Labels | Triple barrier (target=10, stop=5, vol_horizon=500) | Locked by R7 oracle expectancy ($4.00/trade, PF=3.30) |
| CV strategy | 5-fold expanding-window, split by day boundaries | Consistent with R2/R3/R4/9D |
| CNN architecture | **R3-exact**: Conv1d(2→59, k=3) → BN → ReLU → Conv1d(59→59, k=3) → BN → ReLU → AdaptiveAvgPool1d(1) → Linear(59→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1). **12,128 params.** | Confirmed working in 9D (R²=0.1317). 9B's smaller 2→32→64 (~7.5k params) was never tested with correct normalization |
| CNN optimizer | AdamW(lr=1e-3, weight_decay=1e-4), CosineAnnealingLR(T_max=50, eta_min=1e-5), batch_size=512 | R3-exact protocol, confirmed in 9D |
| Early stopping | **Patience=10 on validation loss. Validation = last 20% of TRAIN days (NOT test set).** | 9D showed R3 used test-as-validation, inflating R² by ~36%. This is the critical fix. |
| XGBoost hyperparameters | max_depth=6, lr=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0 | Reasonable defaults; no tuning |
| Random seed | 42 | Single seed; variance assessed across 5 temporal folds |
| Hardware | CPU only | CNN ~12k params; no GPU needed |

## CRITICAL: Normalization Protocol (Root Cause Fixes from 9D)

**These are the exact fixes that resolve the 0.132→0.002 gap. The RUN agent MUST implement these precisely.**

### Book Price Offsets (Channel 0 of CNN input)

```
raw_price_offset = bid_price_offset_N or ask_price_offset_N  (from CSV)
normalized = raw_price_offset / TICK_SIZE

where TICK_SIZE = 0.25  (MES tick size in index points)
```

**Result**: Integer-valued tick offsets from mid-price (e.g., -10, -5, -1, 0, 1, 5, 10).
**NO further normalization on channel 0.** Do NOT z-score price offsets.

### Book Sizes (Channel 1 of CNN input)

```
raw_size = bid_size_N or ask_size_N  (from CSV)
log_size = log1p(raw_size)
normalized = (log_size - day_mean) / day_std

where day_mean, day_std are computed PER TRADING DAY from ALL bars in that day
```

**Key**: Z-score is per-DAY, NOT per-fold. Each day's sizes are normalized using only that day's statistics. This is NOT a train/test leakage concern because day-level statistics are public knowledge (you know today's volume distribution during today's trading).

**NO per-fold normalization on channel 1.** Do NOT re-normalize sizes using fold-level train statistics.

### Non-Spatial Features (20 features for XGBoost)

```
For each feature:
  normalized = (feature - train_mean) / train_std

where train_mean, train_std are computed from the TRAINING FOLD only
```

**Standard per-fold normalization for non-spatial features.** NaN → 0.0 after normalization.

### Summary Table

| Input | Raw form | Normalization | Scope | Note |
|-------|----------|--------------|-------|------|
| Book prices (ch0) | price offset from mid | ÷ TICK_SIZE (0.25) | Global constant | Integer ticks. No z-score. |
| Book sizes (ch1) | raw lot size | log1p → per-day z-score | Per trading day | Day mean/std from ALL day's bars |
| Non-spatial features | various | per-fold z-score (train stats) | Per fold | Standard ML normalization |

## Metrics (ALL must be reported)

### Primary

1. **mean_cnn_r2_h5**: Mean out-of-sample R² of CNN regression at h=5 across 5 folds. Expected ≈0.084.
2. **aggregate_expectancy_base**: Per-trade expectancy (dollars) pooled across all test-set predictions under base costs ($3.74 RT).

### Secondary

| Metric | Description |
|--------|-------------|
| mean_cnn_r2_h1 | CNN R² at h=1 across 5 folds |
| per_fold_cnn_r2_h5 | Per-fold test R² at h=5 (compare with 9D: [0.163, 0.109, 0.049, 0.180, 0.159] leaked; expect ~36% lower) |
| per_fold_cnn_train_r2_h5 | Per-fold train R² (must be > 0.05 — 9B had 0.001, 9D had [0.157-0.196]) |
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
| CNN train R² at h=5 per fold | > 0.05 (9D range: 0.157–0.196) | Normalization is STILL wrong — the critical path fix failed |
| CNN test R² at h=5 for fold 5 | > 0 | If best-data fold is negative, pipeline is broken |
| Channel 0 sample values after normalization | Integer-like (±0.01 tolerance) | TICK_SIZE division not applied or wrong constant |
| Channel 1 sample values after normalization | Mean ≈ 0, std ≈ 1 per day | Per-day z-scoring not applied |
| XGBoost accuracy > 0.33 | Above random for 3-class | Model learns nothing |
| No NaN in CNN output | 0 NaN embeddings | Forward pass or normalization bug |
| Fold train/test day boundaries non-overlapping | No day in both | Temporal leakage |
| XGBoost accuracy ≤ 0.90 | Below implausible ceiling | Label leakage |
| CNN param count | 12,128 ± 5% | Architecture mismatch — not R3-exact |
| Validation set is from TRAIN days, not test days | last 20% of train days | Test-as-validation leakage (the R3 bug) |

## Baselines

### 1. 9D Proper-Validation CNN Result (primary comparison)
- **Source**: r3-reproduction-pipeline-comparison (9D)
- **Value**: R²=0.084 with proper validation (80/20 train/val from train days)
- **Note**: This is the ground truth. R3's R²=0.132 included ~36% leakage inflation.

### 2. R3 Original (leaked — for reference only)
- **Source**: R3 book-encoder-bias
- **Per-fold R²**: [0.163, 0.109, 0.049, 0.180, 0.159], mean=0.132
- **Note**: These used test-as-validation (early stopping on test set). Our proper-validation R² should be ~36% lower per fold.

### 3. Phase 9B (broken pipeline — negative reference)
- **Value**: CNN R²=-0.002, XGBoost acc=0.41, expectancy -$0.44/trade
- **Note**: Failed due to missing TICK_SIZE normalization + per-fold (not per-day) z-scoring.

### 4. Oracle Expectancy (ceiling, from R7)
- **Value**: Triple barrier: $4.00/trade, PF=3.30, WR=64.3%, Sharpe=0.362
- **Note**: Any model expectancy approaching oracle indicates leakage.

### 5. Random Baseline
- 3-class accuracy: 0.333, expectancy: negative

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: mean_cnn_r2_h5 >= 0.06 (proper-validation CNN signal reproduces within 30% of 9D's 0.084)
- [ ] **SC-2**: All fold train R² at h=5 > 0.05 (confirms normalization fix works — 9B had 0.001)
- [ ] **SC-3**: mean_xgb_accuracy >= 0.38 (above random by >= 5 percentage points)
- [ ] **SC-4**: aggregate_expectancy_base >= $0.50/trade (economically viable under base costs)
- [ ] **SC-5**: aggregate_profit_factor_base >= 1.5 (profitable with margin)
- [ ] **SC-6**: Hybrid outperforms GBT-only baseline on mean_xgb_accuracy OR aggregate_expectancy
- [ ] **SC-7**: Hybrid outperforms CNN-only baseline on mean_xgb_accuracy OR aggregate_expectancy
- [ ] **SC-8**: Cost sensitivity table produced for all 3 scenarios
- [ ] **SC-9**: No sanity check failures
- [ ] **SC-10**: Channel 0 values are integer-like after TICK_SIZE division (confirms the fix)

## Minimum Viable Experiment

Before running the full 5-fold CV pipeline:

1. **Data loading + normalization verification (MANDATORY)**:
   - Load `time_5s.csv`, verify shape (~87,970 rows).
   - Identify 40 book columns. Reshape to (N, 20, 2): price offsets and sizes.
   - Apply TICK_SIZE division on price offsets. **VERIFY**: print 10 sample values — must be integer-like (e.g., -10, -5, -1, 0, 1, 5, 10). If NOT integer-like, ABORT.
   - Apply per-day log1p + z-score on sizes. **VERIFY**: per-day mean ≈ 0, std ≈ 1. If mean deviates > 0.01 or std deviates > 0.1 from 1.0, ABORT.
   - Print label distribution (tb_label counts).

2. **Single-fold CNN overfit check (fold 5)**:
   - Train CNN (R3-exact, 12,128 params) on h=5, proper validation (last 20% of train days).
   - Print train R² and test R².
   - **Gate A**: train R² < 0.05 → normalization fix FAILED. The fix from 9D did not transfer. ABORT with diagnostics.
   - **Gate B**: test R² < 0.03 → signal much weaker than expected. Proceed cautiously.
   - **Gate C**: test R² >= 0.05 → normalization fix WORKS. Proceed to full pipeline.
   - Compare fold 5 test R² with 9D's fold 5 (0.161 leaked, expect ~0.10 proper).

3. **Single-fold XGBoost check**:
   - Using fold 5 CNN embeddings + 20 non-spatial features → XGBoost on tb_label.
   - Print accuracy (expect > 0.33). If < 0.33, feature pipeline is broken, ABORT.

If all MVE checks pass, proceed to full protocol.

## Full Protocol

### Step 0: Environment Setup
- Verify dependencies: torch, xgboost, pandas/polars, numpy, scikit-learn.
- Set seed=42 globally (torch, numpy, random).
- Log Python version, PyTorch version.

### Step 1: Data Loading and Normalization
- Load `.kit/results/hybrid-model/time_5s.csv`.
- Identify 19 unique days, sorted chronologically.
- Identify 40 book columns: reshape to (N, 20, 2) — column 0 = price_offset, column 1 = size.
  - Rows 0–9: bids (deepest → best bid at row 9)
  - Rows 10–19: asks (best ask at row 10 → deepest)
- **Apply TICK_SIZE normalization**: `book[:, :, 0] = book[:, :, 0] / 0.25`
- **Apply per-day size normalization**: For each day, `log_size = log1p(raw_size)`, then z-score using that day's mean and std.
- Verify normalization (print sample values, per-day statistics).
- Identify 20 non-spatial feature columns.
- Log dataset statistics: rows, days, label counts, feature NaN counts.

### Step 2: Define 5-Fold Expanding-Window Splits

| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | Days 1-4 | Days 5-7 |
| 2 | Days 1-7 | Days 8-10 |
| 3 | Days 1-10 | Days 11-13 |
| 4 | Days 1-13 | Days 14-16 |
| 5 | Days 1-16 | Days 17-19 |

### Step 3: Run MVE (fold 5 only)
Execute the three MVE checks above. Abort if Gate A triggers.

### Step 4: Full 5-Fold Pipeline
For each fold k in [1..5]:

**Stage 1a: CNN Training (h=1)**
- Book input: already normalized (TICK_SIZE on prices, per-day z-score on sizes).
- Train CNN encoder + Linear(16→1) head on fwd_return_1 (MSE loss).
- AdamW(lr=1e-3, weight_decay=1e-4), CosineAnnealingLR(T_max=50, eta_min=1e-5).
- Batch size: 512. Max epochs: 50.
- **Early stopping: patience=10 on VALIDATION loss. Validation = last 20% of TRAIN days.**
- Record train R² and test R².

**Stage 1b: CNN Training (h=5)**
- Same as 1a but target = fwd_return_5.
- Record train R² and test R².

**Stage 1c: Select Best Horizon CNN**
- Use h=5 CNN for embedding extraction (expected winner based on 9D).
- Report both h=1 and h=5 R² for comparison.

**Stage 2: Embedding Extraction**
- Freeze the h=5 CNN encoder. Remove the regression head.
- Extract 16-dim embeddings for all train and test bars.
- Sanity check: no NaN in embeddings.

**Stage 3: Feature Assembly**
- Normalize 20 non-spatial features using TRAIN-fold mean/std.
- NaN → 0.0 after normalization.
- Concatenate CNN 16-dim embedding + 20 non-spatial features → 36-dim input.

**Stage 4: XGBoost Classification**
- Train XGBoost (multi:softmax, num_class=3, eval_metric=mlogloss) on train: 36 features → tb_label.
- Predict on test set. Record predictions and probabilities.

**Stage 5: Evaluation**
- CNN R² (h=1, h=5) on test.
- XGBoost accuracy, F1_macro on test.
- PnL computation (see PnL model below).
- Save fold results.

### Step 5: Ablation — GBT-Only Baseline
For each fold k in [1..5]:
- Use ALL available non-book features from the CSV → XGBoost with same hyperparameters → tb_label.
- Compute same metrics (accuracy, F1, PnL under 3 cost scenarios).

### Step 6: Ablation — CNN-Only Baseline
For each fold k in [1..5]:
- CNN encoder → Linear(16, 3) classification head on tb_label (CrossEntropyLoss).
- Same CNN training protocol (AdamW, CosineAnnealingLR, 50 epochs, patience=10 on val loss).
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

**Note**: Map from actual CSV column names. If a specified feature is missing, substitute the closest available and document the mapping.

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
- Max wall-clock time: 120 minutes
- Max training runs: 30 (5 folds × 2 horizons × CNN + 5 folds × XGBoost + 5 folds × GBT-only + 5 folds × CNN-only)
- Max seeds per configuration: 1 (seed=42)

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

**Estimate**: CNN (~12k params) on ~74k rows trains in ~15s/fold. XGBoost (500 trees, 36 features, ~74k rows) in ~5s/fold. 30 total fits × ~10s = ~5 min training. Data loading + evaluation + I/O ≈ 15 min. Total ≈ 20-30 min. Budget provides 4-6× headroom.

## Abort Criteria

- **CNN train R² < 0.05 on fold 5**: The normalization fix from 9D did NOT work. This is the single most important gate. ABORT and report: "TICK_SIZE normalization and/or per-day z-scoring did not reproduce 9D's train R²≈0.17."
- **Channel 0 values not integer-like after TICK_SIZE division**: Division was not applied or used wrong constant. ABORT immediately.
- **NaN loss in any fold within 5 epochs**: Normalization or overflow bug. ABORT.
- **All 5 folds negative test R² at h=5**: Signal does not survive proper validation. Report this as a valid finding (R3's signal was entirely leakage-driven).
- **Wall-clock exceeds 90 minutes**: Abort remaining, report partial.
- **Per-run time**: CNN fit > 5 minutes or XGBoost > 2 minutes → investigate.

## Confounds to Watch For

1. **TICK_SIZE constant**: Must be 0.25 (MES tick size in index points). NOT 12.50 (dollar value per tick). NOT 1.0 (identity). A wrong constant produces non-integer channel 0 values — the sanity check catches this.

2. **Per-day vs per-fold z-scoring**: The critical normalization difference. 9B used per-fold. 9D proved per-day is correct. If the RUN agent defaults to per-fold normalization out of habit, the CNN will produce R²≈0 (the 9B failure mode). The sanity check (per-day mean≈0, std≈1) catches this.

3. **Validation split from train days, NOT test days**: R3 used test set for early stopping, inflating R² by ~36%. The validation set MUST be carved from TRAIN days only. Expected: R²≈0.084 (not 0.132). If R² exceeds 0.12, check for test-as-validation leakage.

4. **h=5 target leakage**: `return_5` is both a non-spatial feature and the CNN target at h=5. The CNN sees only book snapshots (no return_5), so no CNN leakage. But XGBoost sees return_5 while predicting tb_label, which depends partly on 5-bar return. Monitor return_5's importance rank. If > 20% gain share, re-run excluding it.

5. **Architecture must be R3-exact (12,128 params)**: 9B used 2→32→64 (~7.5k params) which was never tested with correct normalization. Use Conv1d(2→59→59) as in R3/9D. Param count sanity check enforces this.

6. **Book column ordering**: Reshape assumes specific CSV column ordering. Verify (N, 20, 2) layout matches R3 convention: rows 0-9 = bids (deepest→best), rows 10-19 = asks (best→deepest), col 0 = price_offset, col 1 = size.

7. **Label distribution skew**: Triple barrier with 10:5 target:stop may produce imbalanced classes. Report distribution. If >60% in one class, consider XGBoost class weights.

## Decision Rules

```
OUTCOME A — Full Success (EXPECTED):
  SC-1 PASS (R² >= 0.06) + SC-3 PASS (acc >= 0.38) + SC-4 PASS (expectancy >= $0.50)
  → CNN+GBT Hybrid is VIABLE with corrected normalization.
  → Proceed to multi-seed robustness study (5 seeds × 5 folds).
  → Production pipeline spec ready.

OUTCOME B — CNN Works, Hybrid Marginal:
  SC-1 PASS + SC-3 PASS + SC-4 FAIL (expectancy < $0.50)
  → CNN signal reproduces but doesn't convert to trading edge under current labeling.
  → Investigate: refine XGBoost features, try different label parameters, or accept GBT-only.

OUTCOME C — CNN Fails with Proper Validation:
  SC-1 FAIL (R² < 0.06) + SC-2 PASS (train R² > 0.05)
  → CNN overfits — signal was entirely leakage-driven in R3.
  → R² ≈ 0.084 was optimistic; true proper-validation R² may be < 0.06.
  → Pivot to GBT-only architecture.

OUTCOME D — Normalization Fix Failed:
  SC-2 FAIL (train R² < 0.05)
  → The TICK_SIZE / per-day z-score fix did NOT reproduce 9D's result.
  → Investigate: data loading, column mapping, or normalization implementation error.
  → This is NOT a valid experimental outcome — it's a pipeline bug.
```

## Deliverables

```
.kit/results/cnn-gbt-corrected-pipeline/
  normalization_verification.txt    # TICK_SIZE sample values, per-day stats
  fold_1/ through fold_5/
    cnn_h1_metrics.json             # {train_r2, test_r2, epochs_trained}
    cnn_h5_metrics.json             # {train_r2, test_r2, epochs_trained}
    xgb_metrics.json                # {accuracy, f1_macro, class_report}
    pnl_metrics.json                # {expectancy, pf, sharpe, trade_count} x 3 cost scenarios
    predictions.csv                 # Test-set: bar_index, true_label, predicted_label
  aggregate_metrics.json            # Pooled metrics across all folds + ablation deltas
  ablation_gbt_only.json            # GBT-only baseline: per-fold and aggregate metrics
  ablation_cnn_only.json            # CNN-only baseline: per-fold and aggregate metrics
  cost_sensitivity.json             # 3 cost scenarios x {expectancy, pf, trade_count}
  feature_importance.json           # XGBoost top-10 features by gain
  analysis.md                       # Human-readable summary with explicit SC pass/fail
```

### Required Outputs in analysis.md

1. Executive summary: does the hybrid model produce actionable signals with corrected normalization?
2. **Normalization verification section**: Confirm TICK_SIZE division produced integer ticks, per-day z-scoring produced mean≈0/std≈1.
3. CNN R² table: fold × horizon (h=1, h=5) — compare with 9D proper-validation baseline.
4. Comparison with 9B (broken) and 9D (reference): did the normalization fix close the gap?
5. XGBoost accuracy and F1 table: fold × metric.
6. PnL table: fold × cost scenario → expectancy, profit factor.
7. Ablation comparison: hybrid vs GBT-only vs CNN-only.
8. Label distribution: per-fold class counts.
9. Feature importance: XGBoost top-10 by gain (flag return_5 if dominant).
10. Fold date ranges and train set sizes.
11. Explicit pass/fail for SC-1 through SC-10.
12. Decision outcome (A/B/C/D) and recommended next action.

## Exit Criteria

- [ ] Data loaded and normalization verified (TICK_SIZE integer ticks + per-day z-score on sizes)
- [ ] MVE gate executed (fold 5 train R² > 0.05 confirms normalization fix works)
- [ ] Full 5-fold CNN training at h=1 and h=5 completed
- [ ] Per-fold CNN R² compared with 9D reference
- [ ] XGBoost hybrid trained on CNN embeddings + non-spatial features
- [ ] GBT-only ablation completed
- [ ] CNN-only ablation completed
- [ ] PnL computed under 3 cost scenarios
- [ ] Cost sensitivity table produced
- [ ] Feature importance reported (return_5 check)
- [ ] analysis.md written with all required sections
- [ ] All SC-1 through SC-10 evaluated explicitly
- [ ] Decision outcome stated (A/B/C/D)
