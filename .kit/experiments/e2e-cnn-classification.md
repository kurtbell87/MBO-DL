# Experiment: End-to-End CNN Classification on Full-Year MES Data

**Date:** 2026-02-21
**Priority:** P1 — highest-priority next experiment. Directly attacks the regression-to-classification bottleneck that killed 9E.
**Parent:** hybrid-model-corrected (9E, Outcome B), full-year-export (FYE)

---

## Hypothesis

A Conv1d CNN trained **end-to-end on 3-class triple barrier labels** (cross-entropy loss, no intermediate regression) on the full-year dataset (251 days, 1.16M bars) will achieve:

1. **Classification accuracy >= 0.42** (above 9E's 0.419 XGBoost accuracy, closing the 2pp gap toward 53.3% breakeven).
2. **Aggregate per-trade expectancy >= $0.00/trade** under base costs ($3.74 RT) — i.e., at least break even, demonstrating the end-to-end approach eliminates the regression-to-classification bottleneck.
3. **Profit factor >= 1.0** under base costs (at least break even).

These thresholds are deliberately conservative. 9E showed the gross edge is $3.37/trade — only $0.37 short of breakeven. The regression→frozen-embedding→XGBoost pipeline loses information at the handoff. End-to-end classification eliminates that handoff, and 13× more training data (1.16M vs 87K bars) should improve generalization. If end-to-end CNN cannot break even, the CNN spatial signal — while real (R²=0.089) — does not encode class boundaries that survive transaction costs.

**Key architectural change from 9E:** The CNN's loss function directly optimizes for the trading decision (which barrier was hit) rather than for return prediction (how far price moved). The 16-dim penultimate layer should learn class-discriminative spatial patterns, not return-variance features.

---

## Independent Variables

1. **Model configuration** (3 levels):
   - **E2E-CNN** (primary): Conv1d spatial encoder → Linear(16→3) classification head, trained end-to-end on tb_label with CrossEntropyLoss
   - **E2E-CNN + Features** (augmented): Conv1d encoder 16-dim embedding concatenated with 20 non-spatial features → Linear(36→3) classification head
   - **GBT-only** (baseline): 20 non-spatial features → XGBoost classifier on tb_label (reproduces 9E baseline on full-year data)

2. **Class weighting** (2 levels, E2E-CNN configs only):
   - **Uniform weights**: Standard CrossEntropyLoss
   - **Inverse-frequency weights**: Weight each class by N_total / (3 × N_class) — compensates for label distribution skew from asymmetric 10:5 target/stop ratio

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | `.kit/results/full-year-export/*.parquet` (251 days, 1,160,150 bars) | Full-year Parquet dataset (FYE confirmed) |
| Bar type | time_5s | Locked by R1/R6 |
| Labels | Triple barrier (target=10, stop=5, vol_horizon=500) | Locked by R7 ($4.00/trade oracle) |
| CNN architecture | Conv1d(2→59→59) + BN + ReLU ×2 → AdaptiveAvgPool1d(1) → Linear(59→16) + ReLU → Linear(16→3) | R3's exact encoder (12,128 params for encoder), classification head replaces regression head |
| CNN optimizer | AdamW(lr=1e-3, weight_decay=1e-4) | R3/9D/9E protocol |
| CNN LR schedule | CosineAnnealingLR(T_max=100, eta_min=1e-5) | T_max=100 (doubled from 50) — more data needs more epochs |
| CNN batch size | 1024 | Doubled from 512 — more data allows larger batches for stable gradients |
| CNN max epochs | 100 | Doubled — more data, more capacity |
| CNN early stopping | Patience=15 on **validation loss** (internal 80/20 train/val split, purged) | Proper validation per 9D protocol |
| CNN loss | CrossEntropyLoss on tb_label (3-class) | **Primary change from 9E** — end-to-end classification |
| XGBoost hyperparameters | max_depth=6, lr=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0, seed=42 | Same as 9E for comparability |
| Hardware | **GPU MANDATORY** — EC2 g5.xlarge (A10G, CUDA). Install `torch` with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu121`). Prior CPU runs took 180-320s/split; GPU should reduce to ~20-40s/split, fitting all 3 configs + 2 weight variants within budget. | GPU required for wall-clock feasibility |

### Normalization Protocol (verified in 9D/9E)

**1. Book price offsets (channel 0):**
- Raw Parquet values are in index points
- **DIVIDE by TICK_SIZE = 0.25** → tick offsets (range ±22.5, integer-quantized)
- Do NOT z-score. Raw tick integers are the correct representation.
- **Verification:** ≥99% of values are integer-valued after division (tolerance ±0.01).

**2. Book sizes (channel 1):**
- Apply `log1p()` to raw sizes
- **Z-score PER DAY** (compute mean/std of log1p(size) for each trading day independently)
- **Verification:** Each day has mean≈0 and std≈1.

**3. Non-spatial features (20 features, for E2E-CNN+Features and GBT-only):**
- Z-score using training-fold statistics only (no test leakage)
- NaN → 0.0 after normalization

---

## Validation Strategy: CPCV + Holdout

### Rationale

Prior experiments used 5-fold expanding-window CV on 19 days. This produced a single performance trajectory with high variance (fold 3 = -0.047, fold 5 = +0.135). With 251 days, we can use a statistically rigorous approach that generates a **distribution** of backtest results and quantifies overfitting probability.

The validation hierarchy:

```
|<------- 201 development days -------->|<-- 50-day holdout -->|
|  Day 1                       Day 201  | Day 202      Day 251 |
|                                       |  NEVER TOUCH         |
|  CPCV: N=10 groups, k=2              |  UNTIL FINAL EVAL    |
|  45 splits → 9 backtest paths         |  (one-shot)          |
```

### 50-Day Holdout (SACRED)

- **Days 202–251** (roughly mid-November through end of December 2022)
- **Never used** for training, validation, model selection, hyperparameter tuning, or any intermediate decision
- Evaluated **exactly once** after all CPCV-based development is complete
- The holdout falls in a specific regime (year-end consolidation, lower vol) — this is a feature, not a bug

### CPCV Configuration (Primary CV for Development)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N (groups) | 10 | ~20 days per group — meaningful test periods |
| k (test groups) | 2 | C(10,2) = 45 train/test splits, phi(10,2) = 9 backtest paths |
| Group assignment | Sequential by day (group 1 = days 1–20, group 2 = days 21–40, ..., group 10 = days 181–201) | Preserves temporal ordering within groups |
| Purge window | 500 bars | = max triple barrier label span (timeout). Removes training observations whose labels overlap the test fold boundary. |
| Embargo | 4,600 bars (~1 trading day) | Additional buffer after purge for serial correlation in order flow and book state |
| Sample weights | Uniqueness-weighted (per AFML Ch. 4) | Addresses label concurrency — overlapping triple barrier labels reduce per-observation information content |
| Fold boundaries | Day boundaries only | Never split mid-day. Each group is a contiguous block of complete trading days. |

**Per-split data loss from purging + embargo:**
- Each test-train boundary: purge 500 bars + embargo 4,600 bars ≈ 5,100 bars (~1.1 days)
- With 2 test groups per split, ~4 boundaries, ~20,400 bars excluded
- Out of ~925,000 training bars per split: ~2.2% excluded — negligible

### Internal Validation Split (Within Each CPCV Training Fold)

For early stopping / checkpoint selection within each of the 45 CPCV splits:
1. Take the training fold (8 groups = ~160 days)
2. Reserve the **last 20% of training days** (~32 days) as validation
3. **Apply purge (500 bars) and embargo (1 day) between internal train and val sets**
4. Early stopping monitors validation CrossEntropyLoss
5. The CPCV test fold is NEVER seen during training or checkpoint selection

### Walk-Forward Sanity Check (After CPCV Model Selection)

After the best model configuration is selected via CPCV, run a single expanding-window walk-forward pass as a reality check:

| Parameter | Value |
|-----------|-------|
| Training window | 120 days (expanding from 120) |
| Test window | 20 days |
| Purge gap | 500 bars between train end and test start |
| Embargo | 1 day after test end |
| Number of folds | 4 non-overlapping from 201 days |
| Purpose | Verify CPCV-selected model in temporally realistic deployment simulation |

---

## Metrics (ALL must be reported)

### Primary

1. **cpcv_mean_accuracy**: Mean classification accuracy across 9 CPCV backtest paths
2. **cpcv_mean_expectancy_base**: Mean per-trade expectancy ($) across 9 CPCV paths under base costs
3. **holdout_accuracy**: Accuracy on 50-day holdout (one-shot, reported last)
4. **holdout_expectancy_base**: Expectancy on holdout under base costs

### Secondary

| Metric | Description |
|--------|-------------|
| cpcv_path_accuracies | Per-path accuracy for all 9 backtest paths |
| cpcv_path_expectancies | Per-path expectancy for all 9 paths |
| cpcv_accuracy_std | Std of accuracy across 9 paths |
| cpcv_expectancy_std | Std of expectancy across 9 paths |
| cpcv_min_path_accuracy | Worst path — floor of performance |
| cpcv_pbo | Probability of Backtest Overfitting (fraction of paths where selected model underperforms median) |
| cpcv_deflated_sharpe | Deflated Sharpe Ratio accounting for trials and non-normality |
| per_regime_accuracy | Accuracy broken down by quarter (Q1-Q4 of development set) |
| per_regime_expectancy | Expectancy broken down by quarter |
| walkforward_accuracy | Walk-forward sanity check accuracy (4 folds) |
| walkforward_expectancy | Walk-forward expectancy |
| aggregate_profit_factor | Gross profit / gross loss pooled across all CPCV test predictions |
| aggregate_sharpe | Annualized Sharpe of daily PnL across CPCV test periods |
| class_weight_comparison | Uniform vs inverse-frequency accuracy and expectancy (E2E-CNN only) |
| ablation_delta_vs_gbt | E2E-CNN accuracy/expectancy minus GBT-only |
| ablation_delta_augmented | E2E-CNN+Features accuracy/expectancy minus E2E-CNN |
| cost_sensitivity_table | Expectancy and PF under optimistic ($2.49), base ($3.74), pessimistic ($6.25) |
| label_distribution | Class counts for tb_label across development and holdout sets |
| confusion_matrix | Per-class precision, recall, F1 (pooled across CPCV) |
| f1_macro | Macro-average F1 |
| trials_tested | Total number of model configurations evaluated (for DSR correction) |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| CNN param count | ~12,145 (encoder 12,128 + classification head 16×3+3=51) | Architecture mismatch |
| Channel 0 tick-quantized | >= 99% integer-valued after /0.25 | TICK_SIZE division not applied — **FATAL** |
| Channel 1 per-day z-scored | Per-day mean≈0, std≈1 | Normalization error |
| Train accuracy per fold > 0.40 | All CPCV splits | Model not learning — check loss function |
| Test accuracy > 0.33 on >= 80% of splits | At least 36/45 splits | Model worse than random — abort |
| No test-day data in validation set | All splits | Validation leakage — **FATAL** |
| Purge correctly applied | No training observation with label span overlapping test period | Purge failure — **FATAL** |
| No NaN in CNN outputs | 0 NaN | Forward pass bug |
| Holdout never seen during development | Holdout days excluded from all 45 CPCV splits and walk-forward | Holdout contamination — **FATAL** |
| PBO < 0.50 | More than half of paths beat median | Overfitting — report but do not abort |

---

## Baselines

### 1. 9E Hybrid Corrected (predecessor)
- **Source:** `.kit/results/hybrid-model-corrected/analysis.md`
- **CNN regression R²:** 0.089 (proper validation, 19 days)
- **XGBoost accuracy:** 0.419
- **Expectancy:** -$0.37/trade (base costs)
- **Pipeline:** Regression → frozen embedding → XGBoost
- **Note:** This experiment eliminates the regression-to-classification handoff

### 2. Oracle Expectancy (ceiling)
- **Source:** R7 oracle-expectancy
- **Value:** $4.00/trade, PF=3.30, WR=64.3%
- **Note:** Any model approaching oracle indicates leakage

### 3. Random Baseline
- 3-class accuracy: 0.333
- Expectancy: negative

### 4. 9E GBT-Only (hand-crafted features baseline)
- **Accuracy:** ~0.415 (GBT-nobook from 9E, 19 days)
- **Expectancy:** -$0.45/trade (base costs)

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: cpcv_mean_accuracy >= 0.42 (above 9E's XGBoost 0.419)
- [ ] **SC-2**: cpcv_mean_expectancy_base >= $0.00/trade (at least breakeven)
- [ ] **SC-3**: cpcv_pbo < 0.50 (more than half of backtest paths beat median — not overfit)
- [ ] **SC-4**: E2E-CNN outperforms GBT-only on accuracy OR expectancy (at least one)
- [ ] **SC-5**: holdout_accuracy >= 0.40 (above random by >= 7pp)
- [ ] **SC-6**: holdout_expectancy_base reported (any value — provides true OOS estimate)
- [ ] **SC-7**: No sanity check failures (including purge verification, holdout isolation, normalization checks)
- [ ] **SC-8**: Cost sensitivity table produced for all 3 cost scenarios × 3 model configs
- [ ] **SC-9**: Per-regime (quarterly) performance reported for both accuracy and expectancy
- [ ] **SC-10**: PBO and Deflated Sharpe Ratio computed and reported
- [ ] **SC-11**: Walk-forward sanity check completed (4 folds) and compared to CPCV results
- [ ] **SC-12**: Confusion matrix and per-class F1 reported (identifies which barrier class is hardest)

## Decision Rules

```
OUTCOME A — Breakeven or Better (SC-1 through SC-7 all PASS, expectancy >= $0):
  → End-to-end CNN classification eliminates the regression-to-classification bottleneck.
  → Next: (1) Multi-seed robustness (3 seeds × CPCV), (2) XGBoost hyperparameter tuning
    on CNN embeddings, (3) Ensemble E2E-CNN + GBT for potential further improvement.
  → If holdout expectancy is also positive: strong signal for live paper testing pipeline.

OUTCOME B — CNN Learns but Doesn't Break Even (accuracy > 0.40, expectancy < $0):
  → CNN learns class structure better than random but the edge is still consumed by costs.
  → Check: (1) Is class-weighted version better? (2) What's the breakeven cost?
    (3) Does E2E-CNN+Features outperform E2E-CNN?
  → If breakeven cost < $3.00: the edge exists but is not tradeable at retail.
  → Next: label design sensitivity (wider target, narrower stop to lower breakeven).

OUTCOME C — CNN Fails to Learn (accuracy <= 0.36 across CPCV):
  → The spatial book signal (R²=0.089 on returns) does not encode class-discriminative
    boundaries. The signal is real but too weak/noisy for 3-class classification.
  → This would mean the regression signal cannot be monetized through any classification
    approach at these barrier parameters.
  → Next: (1) Label design sensitivity as primary path, (2) 2-class formulation
    (directional only, merge tb_label=0 into abstain), (3) Regression-based sizing
    (continuous position sizing from return predictions instead of discrete classification).

OUTCOME D — GBT-Only Beats CNN (SC-4 FAIL):
  → Hand-crafted features are sufficient; CNN spatial encoding adds no value for
    classification even with end-to-end training.
  → Simplify to GBT-only pipeline. CNN spatial signal exists for regression but
    the information it captures is already in hand-crafted book features for
    classification purposes.
  → Next: XGBoost hyperparameter tuning on hand-crafted features (full year).

OUTCOME E — Overfit (PBO >= 0.50):
  → The selected model's performance is not robust across backtest paths.
  → Reduce model complexity: fewer conv filters, stronger regularization,
    larger embargo, or fewer features.
  → Do NOT trust holdout results if PBO >= 0.50.
```

---

## Minimum Viable Experiment

Before full CPCV (45 splits), validate the pipeline on a minimal subset:

**1. Data loading gate:**
- Load all 251 Parquet files via `polars.scan_parquet()`
- Verify shape ~(1,160,150, 149)
- Verify tb_label distribution, day count, column names
- **ABORT if shape or schema mismatch**

**2. Normalization verification (MANDATORY):**
- Apply TICK_SIZE division on channel 0. Verify ≥99% integer-valued.
- Apply log1p + per-day z-score on channel 1. Verify per-day mean≈0, std≈1.
- **ABORT if verification fails**

**3. Holdout isolation verification:**
- Confirm days 202–251 are excluded from all development splits
- Print holdout day range and bar count
- **ABORT if holdout days appear in any CPCV group**

**4. Single CPCV split — E2E-CNN classification (split 1 of 45):**
- Train CNN with CrossEntropyLoss on tb_label
- Apply purge (500 bars) and embargo (1 day)
- Print: train accuracy, val accuracy (early stopping), test accuracy
- **Gate A:** train accuracy < 0.35 → model not learning. Check loss function, class balance, normalization.
- **Gate B:** test accuracy < 0.30 → worse than random. Check for data pipeline bug.
- **Gate C:** test accuracy > 0.35 → pipeline working. Proceed.

**5. Single CPCV split — GBT-only:**
- Train XGBoost on same split. Print accuracy.
- **ABORT if accuracy < 0.33**

If all MVE gates pass, proceed to full protocol.

---

## Full Protocol

### Phase 0: Environment Setup
- Set global seed=42. Log library versions (torch, xgboost, polars, numpy, sklearn).
- Verify Parquet files readable and schema consistent.

### Phase 1: Data Loading and Preprocessing
1. Load all 251 Parquet files via `polars.scan_parquet(".kit/results/full-year-export/*.parquet").collect()`
2. Sort by day, then by timestamp within day
3. Identify 251 unique days, sorted chronologically
4. Separate development set (days 1–201) and holdout set (days 202–251)
5. Apply normalization protocol (TICK_SIZE, per-day z-score, feature z-score from dev-set stats)
6. Report: total bars, day count, label distribution (dev vs holdout), feature stats

### Phase 2: CPCV Group Assignment
1. Assign 201 development days to 10 sequential groups (~20 days each)
2. Print group assignments (day ranges per group)
3. For each of 45 splits (choose 2 of 10 groups as test):
   - Identify train groups (8 groups), test groups (2 groups)
   - Apply purge: remove training bars within 500 bars of any test-group boundary
   - Apply embargo: exclude additional 4,600 bars (~1 day) after each test-group boundary from training
   - Split training fold internally: last 20% of training days as validation (with purge + embargo)
   - Report per-split: train bars, val bars, test bars, purged bars, embargoed bars

### Phase 3: Run MVE
Execute MVE checks (Phase 0 items above). Abort if any gate fails.

### Phase 4: Full CPCV — E2E-CNN Classification
For each of 45 CPCV splits:
1. Set CNN seed = 42 + split_idx
2. Build CNN: Conv1d(2→59→59) + BN + ReLU ×2 → AdaptiveAvgPool1d(1) → Linear(59→16) + ReLU → Linear(16→3)
3. Train with CrossEntropyLoss (uniform weights first, then repeat with inverse-frequency weights)
4. AdamW(lr=1e-3, weight_decay=1e-4), CosineAnnealingLR(T_max=100, eta_min=1e-5)
5. Batch size 1024, max epochs 100, early stopping patience=15 on validation loss
6. Record: train accuracy, val accuracy, test accuracy, test predictions, epochs trained
7. Compute per-split PnL under 3 cost scenarios

### Phase 5: Full CPCV — E2E-CNN + Features
For each of 45 CPCV splits:
1. Same CNN encoder as Phase 4, but concatenate 16-dim penultimate output with 20 z-scored non-spatial features
2. Classification head: Linear(36→3) instead of Linear(16→3)
3. End-to-end training with CrossEntropyLoss (using best class weighting from Phase 4)
4. Same training protocol. Record same metrics.

### Phase 6: Full CPCV — GBT-Only Baseline
For each of 45 CPCV splits:
1. 20 non-spatial features → XGBoost (same hyperparameters as 9E)
2. Record accuracy, predictions, PnL

### Phase 7: CPCV Aggregation
1. For each of the 3 model configs:
   - Reconstruct 9 backtest paths from 45 splits (phi(10,2) = 9)
   - Compute per-path: accuracy, expectancy, profit factor, Sharpe
   - Compute across paths: mean, std, min, max
2. Compute PBO: fraction of paths where selected model underperforms the median of all configs tested
3. Compute Deflated Sharpe Ratio (correct for number of trials, skewness, kurtosis)
4. Select best model config based on mean CPCV expectancy

### Phase 8: Walk-Forward Sanity Check
1. Using the CPCV-selected best model config:
   - 4 expanding-window folds on 201 development days
   - Training window: starts at 120 days, expanding
   - Test window: 20 days, non-overlapping
   - Purge 500 bars + embargo 1 day between train and test
2. Report per-fold and mean accuracy/expectancy
3. Compare with CPCV results — large divergence suggests CPCV temporal mixing is biasing results

### Phase 9: Holdout Evaluation (ONE SHOT)
1. Train the selected model config on ALL 201 development days (with internal 80/20 val split for early stopping)
2. Evaluate on days 202–251 (50 holdout days)
3. Report: accuracy, expectancy, PF, Sharpe, confusion matrix, per-week performance
4. **This is the final, definitive result. No re-runs allowed.**

### Phase 10: Report
1. Write comprehensive analysis.md with all metrics, tables, and SC pass/fail
2. Per-regime breakdown (Q1-Q4 of dev set + holdout period characterization)
3. CPCV path distribution plots (describe in text if no plotting available)
4. Comparison table: E2E-CNN vs E2E-CNN+Features vs GBT-only vs 9E-Hybrid
5. PBO interpretation and DSR

---

## CNN Architecture (exact specification)

### E2E-CNN (primary)
```
Input: (B, 2, 20)                    # channels-first: (price_offset, log1p_size) × 20 levels
Conv1d(in=2, out=59, kernel_size=3, padding=1)  + BatchNorm1d(59) + ReLU
Conv1d(in=59, out=59, kernel_size=3, padding=1) + BatchNorm1d(59) + ReLU
AdaptiveAvgPool1d(1)                 # → (B, 59)
Linear(59, 16) + ReLU               # 16-dim spatial embedding
Linear(16, 3)                        # 3-class output (tb_label ∈ {-1, 0, +1})
```

**Parameter count:**
- Conv layers + BN: 11,151 (same as 9E)
- Linear(59→16): 960
- Linear(16→3): 51
- **Total: 12,162** (vs 9E's 12,128 — only the head differs: 51 vs 17)

### E2E-CNN + Features (augmented)
```
Same encoder as above through Linear(59→16) + ReLU → 16-dim embedding
Concatenate with 20 z-scored non-spatial features → 36-dim
Linear(36, 3)                        # 3-class output
```
**Additional params:** Linear(36→3) = 111 instead of Linear(16→3) = 51. Total: ~12,222.

### Non-Spatial Feature Set (20 dimensions)

Same as 9E — see 9E spec for full table. Key features: weighted_imbalance, spread, net_volume, volume_imbalance, trade_count, avg_trade_size, vwap_distance, return_1, return_5, return_20, volatility_20, volatility_50, high_low_range_50, close_position, cancel_add_ratio, message_rate, modify_fraction, time_sin, time_cos, minutes_since_open.

### Transaction Cost Scenarios

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

### PnL Model

```
tick_value = $1.25 per tick (MES)
target_ticks = 10 → win PnL = +$12.50 - RT_cost
stop_ticks = 5  → loss PnL = -$6.25 - RT_cost

Correct directional call (pred sign = label sign, both nonzero): +$12.50 - RT_cost
Wrong directional call (pred sign ≠ label sign, both nonzero):  -$6.25 - RT_cost
Predict 0 (hold): $0 (no trade)
True label = 0 but model predicted ±1: $0 (conservative simplification)
```

Report label=0 trade fraction. If >20% of directional predictions hit label=0, flag as unreliable.

---

## Resource Budget

**Tier:** Cloud GPU MANDATORY

### Compute Profile
```yaml
compute_type: gpu
estimated_rows: 1160150
model_type: pytorch+xgboost
sequential_fits: 135
parallelizable: true
memory_gb: 16
gpu_type: a10g
estimated_wall_hours: 1.5
```

### GPU Requirements — NON-NEGOTIABLE

**The prior CPU run (2026-02-22) took 180-320s/split and timed out before completing all configs.** GPU is mandatory for this experiment.

1. **Instance:** g5.xlarge (A10G GPU, 24GB VRAM, CUDA 12.x)
2. **PyTorch install:** `pip install torch --index-url https://download.pytorch.org/whl/cu121` — do NOT use default pip torch (installs CPU-only)
3. **Device:** All CNN training MUST use `device = torch.device('cuda')`. Verify `torch.cuda.is_available() == True` before proceeding. **ABORT if CUDA not available.**
4. **Expected speedup:** 5-10× per split (180-320s CPU → 20-40s GPU)

### Wall-Time Estimation (GPU)

| Component | Per-unit estimate | Count | Subtotal |
|-----------|------------------|-------|----------|
| Data loading (251 Parquet files) | 30s | 1 | 30s |
| CNN training per split (GPU) | 20–40s | 135 (45 splits × 3 configs incl. weight variants) | 45–90 min |
| XGBoost per split | 10s | 45 | 7.5 min |
| Walk-forward (4 folds × best config) | 40s | 4 | 2.5 min |
| Holdout evaluation | 30s | 1 | 30s |
| Aggregation + PBO/DSR computation | 5 min | 1 | 5 min |
| **Total estimated** | | | **~65–110 min** |

**Max budget:** 4 hours. With GPU, all 3 configs + 2 weight variants should complete well within budget.

### Other Limits
- Max wall-clock: 4 hours (abort)
- Max per-split CNN training (GPU): 60 seconds (investigate if exceeded — likely not using GPU)
- Max trials for DSR correction: 6 (3 configs × 2 class weightings)

---

## Abort Criteria

- **Normalization verification fails:** Channel 0 not integer-like after /0.25 → ABORT. Same error as 9B/9C.
- **Holdout contamination:** Any holdout day appears in development splits → ABORT.
- **Purge verification fails:** Any training observation's label span overlaps test period → ABORT.
- **MVE Gate A (train accuracy < 0.35):** Model not learning → ABORT. Check loss function.
- **CNN produces NaN loss:** Any split → ABORT.
- **All CPCV splits produce accuracy < 0.33:** Signal not learnable end-to-end → ABORT.
- **Wall-clock > 4 hours:** ABORT remaining, report completed work.
- **Per-split CNN training > 3 min:** Investigate. If unresolvable, switch to cloud GPU.

---

## Confounds to Watch For

1. **Label distribution skew.** The asymmetric 10:5 target/stop ratio may produce imbalanced classes. The class-weighted variant addresses this, but report the distribution prominently. If any class exceeds 50%, the uniform-weight model may degenerate to majority-class prediction.

2. **CPCV temporal mixing.** CPCV allows training on "future" data relative to some test groups. For non-stationary financial data, this is less realistic than walk-forward. The walk-forward sanity check (Phase 8) validates that CPCV isn't producing over-optimistic results. If walk-forward accuracy is substantially lower (>5pp), report prominently.

3. **Regime concentration in holdout.** Days 202–251 are Nov-Dec 2022 (year-end consolidation, lower vol). If the model learned primarily from high-vol periods (Mar-Sep), holdout performance may be systematically worse. Report per-regime metrics.

4. **CrossEntropyLoss vs regression MSE.** The CNN was previously trained on MSE for return prediction. Switching to CrossEntropyLoss changes the optimization landscape entirely. The 16-dim penultimate layer will learn different features. If the model fails to learn (Outcome C), it may need architecture changes (larger capacity, different activation, dropout) rather than indicating the signal is absent.

5. **More data ≠ better.** 251 days includes regime shifts that 19 days didn't. The CNN may struggle with non-stationarity across 2022. If CPCV path variance is very high (std > 0.05 on accuracy), the model is regime-sensitive.

6. **Label=0 simplification.** Same as 9E — PnL=0 for expired labels is conservative but may understate true performance if many label=0 bars resolve close to target. Report the fraction.

7. **Purge window size.** 500 bars assumes the max triple barrier timeout is 500 bars. If the actual timeout parameter in the Parquet data is different, the purge window is wrong. Verify from the data: check max(tb_bars_held) across the dataset.

8. **Feature leakage through return_5.** Same concern as 9E for the E2E-CNN+Features and GBT-only configs. `return_5` is backward-looking (legitimate) but if XGBoost assigns >20% gain share, investigate. The E2E-CNN config (book snapshots only) is immune to this.

---

## Deliverables

```
.kit/results/e2e-cnn-classification/
  cpcv/
    split_results.json             # Per-split: {train_acc, val_acc, test_acc, epochs, predictions} × 45
    path_results.json              # Per-path: {accuracy, expectancy, pf, sharpe} × 9 paths
    pbo.json                       # PBO value, DSR, trial count
    purge_audit.json               # Per-split: purged_bars, embargoed_bars, verification
  e2e_cnn/
    cpcv_aggregate.json            # Mean/std/min/max across 9 paths (uniform + weighted)
    confusion_matrix.json          # Pooled across CPCV test predictions
  e2e_cnn_features/
    cpcv_aggregate.json            # Same format
  gbt_only/
    cpcv_aggregate.json            # Same format
  walkforward/
    fold_results.json              # Per-fold: {accuracy, expectancy} × 4 folds
  holdout/
    results.json                   # One-shot: {accuracy, expectancy, pf, sharpe, confusion_matrix}
    per_week_performance.json      # Weekly breakdown within holdout
  regime_analysis.json             # Per-quarter metrics for all configs
  cost_sensitivity.json            # 3 scenarios × 3 configs × {expectancy, pf}
  comparison_vs_9e.json            # Direct comparison with 9E results
  analysis.md                      # Full writeup with all sections + SC pass/fail
```

### Required Outputs in analysis.md

1. Executive summary: does end-to-end CNN classification close the viability gap?
2. CPCV path distribution: per-path accuracy and expectancy for all 9 paths × 3 configs
3. PBO and DSR: is the selected model overfit?
4. Model comparison table: E2E-CNN vs E2E-CNN+Features vs GBT-only (accuracy, expectancy, PF)
5. Comparison with 9E: what changed from regression→XGBoost to end-to-end classification?
6. Walk-forward vs CPCV: do they agree? If not, which is more trustworthy?
7. Holdout results (reported last, clearly separated from development results)
8. Per-regime breakdown: Q1-Q4 accuracy and expectancy
9. Confusion matrix: which class is hardest? Does the model avoid trades (predict 0) or not?
10. Cost sensitivity table
11. Label distribution and label=0 simplification impact
12. Explicit pass/fail for each of SC-1 through SC-12

---

## Exit Criteria

- [ ] Data loaded and verified (251 days, ~1.16M bars, schema match)
- [ ] Normalization verified (TICK_SIZE, per-day z-score)
- [ ] Holdout isolation confirmed (days 202–251 excluded from all development)
- [ ] CPCV groups assigned (10 groups, ~20 days each)
- [ ] Purge and embargo applied and verified across all 45 splits
- [ ] MVE gates passed
- [ ] Full CPCV completed — E2E-CNN (45 splits × 2 weight configs)
- [ ] Full CPCV completed — E2E-CNN + Features (45 splits)
- [ ] Full CPCV completed — GBT-only (45 splits)
- [ ] 9 backtest paths reconstructed, PBO and DSR computed
- [ ] Walk-forward sanity check completed (4 folds)
- [ ] Holdout evaluation completed (one-shot)
- [ ] Per-regime breakdown reported
- [ ] Confusion matrix and per-class F1 reported
- [ ] Cost sensitivity table produced (3 scenarios × 3 configs)
- [ ] Comparison with 9E reported
- [ ] analysis.md written with all required sections
- [ ] All SC-1 through SC-12 evaluated explicitly
