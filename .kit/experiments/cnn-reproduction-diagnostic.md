# Experiment: CNN Reproduction Diagnostic

## Hypothesis

Phase B's CNN failure (mean OOS R² = −0.002 vs R3's +0.132) was caused by three protocol deviations: (1) z-scoring price offsets destroyed absolute tick magnitude information the Conv1d filters depend on, (2) Conv1d channel width 2→32→64 instead of R3's 2→32→32 reduced total parameters from ~12k to ~7.7k while misallocating capacity, and (3) a fixed learning rate prevented fine-tuning in later epochs. Correcting all three deviations will restore CNN mean out-of-sample R² on `fwd_return_5` to **≥ 0.10** across 5 expanding-window folds, reproducing ≥ 76% of R3's R² = 0.132.

Conditional on CNN reproduction (mean R² ≥ 0.10): a Hybrid pipeline (frozen 16-dim CNN embeddings + 20 non-spatial features → XGBoost classifier on triple barrier labels) will achieve **≥ $0.50/trade aggregate expectancy** under base transaction costs ($3.74 RT), demonstrating that R3's spatial signal converts to economically viable trading signals when the CNN is correctly implemented.

## Independent Variables

### Step 1: CNN Reproduction (single configuration)

No experimental manipulation — this is a faithful reproduction of R3's protocol on Phase 9A data. The single configuration is the R3-exact CNN pipeline. Comparison is against two historical references:
- R3's result (target): mean R² = 0.132
- Phase B's result (broken negative reference): mean R² = −0.002

### Step 2: Hybrid Integration (conditional on Step 1, 2 levels)

| Level | Description | Input dims |
|-------|-------------|------------|
| **Hybrid** | Frozen 16-dim CNN embeddings + 20 non-spatial features → XGBoost | 36 |
| **GBT-only** | All available non-book hand-crafted features → XGBoost (no CNN) | ~62 |

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data source | `.kit/results/hybrid-model/time_5s.csv` (87,970 bars, 19 days) | Same data as Phase B; eliminates data processing variance |
| Bar type | time_5s | Locked by R1 (subordination refuted), R6 synthesis |
| CV strategy | 5-fold expanding-window, day boundaries, no shuffling | Matches R3 exactly; no temporal leakage |
| Target (Step 1) | `fwd_return_5` (5-bar forward return) | Matches R3's primary metric |
| Target (Step 2) | `tb_label` (triple barrier: -1, 0, +1) | target=10, stop=5, vol_horizon=500 per R7 |
| Random seed | 42 (torch, numpy, random) | Reproducibility; single seed with 5 temporal folds |
| Hardware | CPU only | CNN ~12k params; XGBoost native CPU. No GPU needed |

### R3-Exact CNN Protocol (Step 1 — deviations MUST be logged)

| Parameter | R3 Value | Phase B Value (broken) |
|-----------|----------|------------------------|
| Architecture | Conv1d(2→32→32) + BN + ReLU × 2 → Pool → Linear(32→16→1) | Conv1d(2→32→64) → Linear(64→16→1) |
| Param count | ~12,128 | ~7,700 |
| Price normalization | Raw tick offsets (NO z-score) | Z-scored both channels (**FATAL**) |
| Size normalization | Z-score log1p(size) per fold (train stats only) | Z-score (correct) |
| Optimizer | AdamW | Adam (wrong — decoupled weight decay matters) |
| Learning rate | 1e-3 initial | 1e-3 fixed (wrong — no annealing) |
| LR schedule | CosineAnnealingLR(T_max=50, eta_min=1e-5) | None |
| Weight decay | 1e-4 | 1e-5 (wrong — 10× less regularization) |
| Batch size | 512 | 256 (wrong) |
| Early stop patience | 10, monitor val loss | 5 (wrong) |
| Loss function | MSE on fwd_return_5 | MSE (correct) |
| Validation split | Last 20% of train days | Last 20% (correct) |

**Full CNN architecture (must match exactly):**
```
Input: (B, 2, 20)              # channels-first: (price_offset, log1p_size) × 20 levels
Conv1d(in=2, out=32, kernel_size=3, padding=1) + BatchNorm1d(32) + ReLU
Conv1d(in=32, out=32, kernel_size=3, padding=1) + BatchNorm1d(32) + ReLU
AdaptiveAvgPool1d(1)            # → (B, 32)
Linear(32, 16) + ReLU
Linear(16, 1)                   # scalar return prediction
```

## Metrics (ALL must be reported)

### Primary

1. **mean_cnn_r2_h5**: Mean out-of-sample R² of CNN regression at h=5 across 5 folds. Directly tests whether R3's spatial signal reproduces when the three protocol deviations are fixed.
2. **aggregate_expectancy_base**: Per-trade expectancy ($) pooled across all test predictions under base costs ($3.74 RT). Tests whether the spatial signal converts to economic value. *Only evaluated if Step 1 passes (mean R² ≥ 0.10).*

### Secondary

| Metric | Description |
|--------|-------------|
| per_fold_cnn_r2_h5 | Per-fold test R² at h=5 (for direct comparison with R3's per-fold: [0.163, 0.109, 0.049, 0.180, 0.159]) |
| per_fold_cnn_train_r2_h5 | Per-fold train R² at h=5 (must be > 0.05 per fold — the Phase B smoking gun was train R² ≈ 0) |
| epochs_trained_per_fold | Epochs before early stopping triggered (R3 context: 50 max, patience 10) |
| mean_xgb_accuracy | XGBoost 3-class accuracy across 5 folds (Step 2 only) |
| mean_xgb_f1_macro | Macro F1 across 5 folds (Step 2 only) |
| aggregate_profit_factor | Gross profit / gross loss pooled under base cost (Step 2 only) |
| hybrid_vs_gbt_delta_accuracy | Hybrid accuracy − GBT-only accuracy |
| hybrid_vs_gbt_delta_expectancy | Hybrid expectancy − GBT-only expectancy |
| cost_sensitivity_table | Expectancy and PF under optimistic ($2.49), base ($3.74), pessimistic ($6.25) RT costs |
| xgb_top10_features | XGBoost feature importance (gain) top-10 — check for return_5 leakage |
| label_distribution | Per-fold class counts for tb_label {-1, 0, +1} |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| CNN param count | Within 5% of 12,128 | Architecture mismatch — spec deviation, results invalid |
| Channel 0 sample values | Integer-valued tick offsets (e.g., -10, -5, -1, 1, 5, 10) | Prices were z-scored — **FATAL**, same error as Phase B |
| Channel 1 sample values | Log-transformed sizes (e.g., 2.3, 3.1, 4.5) before z-scoring | Data format mismatch |
| LR at epochs 1 / 25 / 50 | Decays from ~1e-3 toward ~1e-5 | CosineAnnealingLR not applied |
| Train R² per fold at h=5 | > 0.05 | CNN cannot fit training data — pipeline still broken |
| No NaN in CNN outputs | 0 NaN embeddings | Normalization or forward pass bug |
| Fold day boundaries non-overlapping | No day in both train and test | Temporal leakage |
| XGBoost accuracy (Step 2) | > 0.33 and ≤ 0.90 | Below floor = learning nothing; above ceiling = leakage |

## Baselines

### 1. R3 CNN Result (reproduction target)
- **Source**: R3 book-encoder-bias experiment (`.kit/results/book-encoder-bias/analysis.md`)
- **Per-fold R²**: [0.163, 0.109, 0.049, 0.180, 0.159]
- **Mean**: 0.132 ± 0.048
- **Protocol**: Conv1d 2→32→32, ~12,128 params, raw tick offsets, CosineAnnealingLR, AdamW(lr=1e-3, wd=1e-4), batch=512, patience=10, MSE loss
- **Role**: Reproduction target. We require ≥ 0.10 (76% of 0.132), leaving margin for minor implementation differences (PyTorch version, data export pipeline).

### 2. Phase B CNN Result (broken pipeline — negative reference)
- **Source**: hybrid-model-training (`.kit/results/hybrid-model-training/analysis.md`)
- **Mean test R²**: −0.002 (train R²: 0.001 — smoking gun of total underfitting)
- **Protocol**: 3 deviations — z-scored prices, 2→32→64, no cosine LR, Adam, wd=1e-5, batch=256, patience=5
- **Role**: The broken state. Any result significantly above 0 confirms the fixes are working. The 0.132 → −0.002 gap is the target to close.

### 3. Phase B GBT-only (Step 2 comparison point)
- **Source**: hybrid-model-training
- **Value**: Accuracy 0.411, expectancy −$0.38/trade (base costs)
- **Role**: Hybrid must outperform this on ≥ 1 metric to justify CNN integration.

### 4. Random Baseline (Step 2 floor)
- 3-class accuracy: 0.333
- Expectancy: Negative (random direction selection loses to transaction costs)

## Success Criteria (immutable once RUN begins)

- [x] **SC-1**: mean_cnn_r2_h5 ≥ 0.10 — **FAIL**. Fold 5 test R²=0.0001 (MVE aborted full 5-fold). 1000× below threshold.
- [x] **SC-2**: No fold train R² < 0.05 — **FAIL**. Fold 5 train R²=0.002 (24× below). CNN cannot fit training data — same failure mode as Phase B despite all 3 deviations fixed.
- [ ] **SC-3**: aggregate_expectancy_base ≥ $0.50/trade — **NOT EVALUATED** (SC-1 failed, Step 2 not executed)
- [ ] **SC-4**: Hybrid outperforms GBT-only — **NOT EVALUATED** (SC-1 failed, Step 2 not executed)
- [x] **SC-5**: No sanity check failures — **FAIL**. 5/7 pass, 1 hard fail (train R²), 1 soft fail (channel 0 units).

## Minimum Viable Experiment

Before running the full 5-fold protocol, execute a single-fold gate check on **fold 5** (16 train days, 3 test days — maximum training data, best chance of reproducing R3):

**1. Data verification (MANDATORY before any training):**
- Load `time_5s.csv`, verify row count (~87,970) and 19 unique days.
- Identify 40 book columns, determine reshape order to (N, 2, 20).
- Print first 5 samples of channel 0 → must be integer-valued tick offsets.
- Print first 5 samples of channel 1 → must be log-transformed sizes.
- **ABORT if channel 0 values are near 0 with std ≈ 1 (z-scored).**

**2. Architecture verification:**
- Build CNN model, print parameter count → must be within 5% of 12,128.
- Print layer structure → must match R3 spec exactly.
- **ABORT if param count deviates > 10% from 12,128.**

**3. Single-fold training (fold 5):**
- Train CNN with R3-exact protocol (AdamW, CosineAnnealingLR, batch=512, etc.).
- Print LR at epoch 1, epoch 25, and final epoch → must decay from ~1e-3 toward ~1e-5.
- Print train R² and test R².
- **Gate**: train R² < 0.05 → pipeline is STILL broken. Print diagnostic details and **STOP**.
- **Gate**: train R² ≥ 0.05 AND test R² > 0 → **proceed** to full 5-fold protocol.

## Full Protocol

### Phase 0: Environment Setup
- Set seed=42 globally (torch, numpy, random).
- Log PyTorch version (for reproducibility documentation).
- Verify dependencies: torch, xgboost, pandas/polars, numpy, scikit-learn.

### Phase 1: Data Loading and Validation
- Load `.kit/results/hybrid-model/time_5s.csv`.
- Identify 19 unique days, sorted chronologically.
- Identify 40 book snapshot columns; determine reshape order to produce (N, 2, 20):
  - Channel 0 = price offsets (raw ticks from mid) — integers like -10, -5, -1, 1, 5, 10
  - Channel 1 = log1p(size) — positive reals like 2.3, 3.1, 4.5
  - **Print sample values and confirm** (mandatory sanity check).
- Identify 20 non-spatial feature columns (map from actual CSV column names to the feature list below).
- Log: row count, unique days, column names, NaN counts per column.

### Phase 2: Define Expanding-Window Splits

| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | Days 1–4 | Days 5–7 |
| 2 | Days 1–7 | Days 8–10 |
| 3 | Days 1–10 | Days 11–13 |
| 4 | Days 1–13 | Days 14–16 |
| 5 | Days 1–16 | Days 17–19 |

### Phase 3: Run MVE (fold 5 only)
Execute the Minimum Viable Experiment described above. Abort if any gate fails.

### Phase 4: Step 1 — Full 5-Fold CNN Training

For each fold k in [1..5]:
1. Split data by day boundaries per Phase 2.
2. **Channel 0 (price offsets): NO normalization. Raw tick integers.**
3. Channel 1 (sizes): z-score using train-fold statistics only (compute mean/std on train set, apply to both train and test).
4. Reserve last 20% of train days as validation set (for early stopping only — not used for final evaluation).
5. Train CNN with R3-exact protocol:
   - AdamW(lr=1e-3, weight_decay=1e-4)
   - CosineAnnealingLR(T_max=50, eta_min=1e-5)
   - Batch size: 512
   - Max epochs: 50, early stopping patience=10 on val loss
   - MSE loss on `fwd_return_5`
6. Save best checkpoint (lowest val loss).
7. Record: train R², test R², epochs trained, final LR, training loss curve.

### Phase 5: Step 1 Gate Evaluation

Compute mean test R² across 5 folds. Compare to R3 per-fold values.

| Outcome | Action |
|---------|--------|
| Mean R² ≥ 0.10 | **PASS** — Proceed to Step 2 |
| 0.05 ≤ Mean R² < 0.10 | **MARGINAL** — Proceed to Step 2 with caution. Flag gap vs R3. |
| Mean R² < 0.05 | **FAIL** — Do NOT proceed to Step 2. Report per-fold diagnostics and stop. |

### Phase 6: Step 2 — Hybrid Integration (conditional on Phase 5 PASS or MARGINAL)

For each fold k in [1..5]:

**6a. CNN Embedding Extraction**
1. Load best CNN checkpoint from Phase 4 (fold k).
2. **Freeze all CNN weights.** No gradient updates.
3. Remove the regression head (Linear(16→1)).
4. Forward pass all train and test bars through the frozen encoder.
5. Extract 16-dim output of Linear(32→16) + ReLU.
6. Verify: 0 NaN in embeddings.

**6b. Non-Spatial Feature Assembly**

Use these 20 non-spatial features (map from actual CSV column names):

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

Z-score per fold using train-fold statistics only. NaN → 0.0 after normalization.
Concatenate: 16-dim CNN embedding + 20 non-spatial features = 36-dim vector.

**6c. Hybrid XGBoost Training**
- Config: objective=multi:softmax, num_class=3, max_depth=6, learning_rate=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0, seed=42, eval_metric=mlogloss
- Train on 36-dim input → `tb_label`.
- Predict on test set. Record predictions and probabilities.

**6d. GBT-Only Ablation**
- Same XGBoost config, but input = all available non-book features from CSV.
- If Phase B used 62 features, use the same set for direct comparison.

**6e. PnL Computation**

For each fold, for each cost scenario (optimistic $2.49, base $3.74, pessimistic $6.25):
```
tick_value = $1.25 per tick (MES: $0.25 index pts × 5 multiplier)
target_ticks = 10 → win PnL = $12.50 - RT_cost
stop_ticks = 5  → loss PnL = -$6.25 - RT_cost

Correct directional call (prediction sign matches tb_label sign): PnL = +$12.50 - RT_cost
Wrong directional call (prediction sign opposes tb_label sign):  PnL = -$6.25 - RT_cost
Predict 0 or true label 0:                                       PnL = $0 (no trade)
```

### Phase 7: Aggregate Results and Write Analysis

- Pool test predictions across 5 folds.
- Compute all primary, secondary, and sanity check metrics.
- Produce R3 comparison table (per-fold, this experiment vs R3 side-by-side).
- Produce cost sensitivity table.
- Produce feature importance (top-10 by gain) — flag `return_5` if in top 3.
- Evaluate all success criteria (SC-1 through SC-5).
- Write analysis.md with all required sections (see below).

### Required Outputs in analysis.md

1. **Step 1 verdict**: PASS / MARGINAL / FAIL with mean R² and per-fold table.
2. **Normalization verification**: Printed sample values confirming channel 0 = raw ticks.
3. **Architecture verification**: Param count, layer structure confirmation.
4. **R3 comparison table**: This experiment vs R3, per-fold and mean ± std.
5. **Root cause confirmation**: Did fixing the 3 deviations (normalization, architecture, cosine LR) restore the signal?
6. **Step 2 verdict** (if applicable): Hybrid vs GBT-only comparison, expectancy, PF.
7. **Explicit pass/fail for each SC-1 through SC-5.**
8. **Deviations log**: Any deviation from R3 protocol, with justification (target: zero deviations).

### Non-Spatial Feature Column Name Mapping

The RUN agent must inspect actual CSV column names at data loading time and map to the 20 features above. If a specified feature name is missing, substitute the closest available feature and document the mapping in the deviations log. The Phase B analysis showed no individual feature was critical — the exact 20-feature set is secondary to the CNN integration.

### Transaction Cost Scenarios (Step 2)

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

## Resource Budget

**Tier:** Standard

- Max GPU-hours: 0 (CPU only)
- Max wall-clock time: 90 minutes
- Max training runs: 16 (Step 1: 1 MVE + 5 full folds = 6; Step 2: 5 hybrid XGBoost + 5 GBT-only = 10)
- Max seeds per configuration: 1 (seed=42; variance assessed across 5 temporal folds)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 87970
model_type: pytorch+xgboost
sequential_fits: 16
parallelizable: false
memory_gb: 4
gpu_type: none
estimated_wall_hours: 0.5
```

**Estimate breakdown**: CNN (~12k params) on ~70k rows trains in ~10-15s/fold on CPU. XGBoost (500 trees, 36 features, ~70k rows) trains in ~5s/fold. 16 total fits × ~10s average = ~160s training. Data loading, reshape verification, normalization, embedding extraction, evaluation, and I/O add ~15 minutes. Total: ~20-30 minutes. Budget of 90 minutes provides 3-4× headroom.

## Abort Criteria

- **NaN loss**: Any fold produces NaN loss within 5 epochs → normalization or overflow bug. ABORT.
- **MVE gate failure**: Fold 5 train R² < 0.05 after 50 epochs → pipeline is still broken despite fixes. Print diagnostic details (sample input values, param count, LR schedule, training loss curve) and **STOP**. Do not proceed to full 5-fold.
- **Step 1 failure**: Mean test R² < 0.05 across 5 folds → CNN reproduction failed. Do NOT proceed to Step 2. Report per-fold diagnostics.
- **NaN embeddings**: Any NaN in 16-dim CNN outputs during Step 2 → normalization or forward pass bug. ABORT Step 2.
- **Per-run time**: Any single CNN fit exceeds 3 minutes (12× expected) → investigate batch size or data loading bottleneck.
- **Wall-clock**: Total exceeds 90 minutes → pathological slowness. Abort and diagnose.

## Confounds to Watch For

1. **Data export pipeline difference**: R3 used Phase 4 Track B.1 export; this experiment uses Phase 9A C++ export (`time_5s.csv`). If the two export pipelines produce slightly different price offsets (different rounding, different mid-price calculation, different level ordering), the CNN sees a different input distribution than R3 trained on. The normalization verification step (printing sample values) partially mitigates this, but subtle numerical differences could reduce R² by 0.01-0.03. **If mean R² falls in the 0.08-0.10 marginal zone, this confound should be investigated as the primary alternative explanation.**

2. **Book column ordering in CSV**: The reshape from flat CSV columns to (N, 2, 20) tensor is the most fragile step. Columns could be ordered as interleaved (price, size) pairs or grouped (all prices then all sizes). If the RUN agent misidentifies the ordering, channel 0 contains a mix of prices and sizes, and the CNN sees structured noise identical to Phase B's failure mode. **This is the #1 implementation risk.** The mandatory verification step (print sample values, check integer tick offsets in channel 0) catches this.

3. **PyTorch version**: Different PyTorch versions have slightly different BatchNorm and AdamW implementations. Expected impact: ±0.01 R². Not a threat to the 0.10 threshold but could explain small per-fold deviations from R3. The RUN agent must log the PyTorch version.

4. **Label distribution skew (Step 2)**: Phase B showed fold 3 with 45.5% class 0 (vs ~33% balanced). XGBoost accuracy can be inflated by predicting the majority class. Mitigation: report distribution per fold; evaluate on expectancy (insensitive to class imbalance) as the primary economic metric.

5. **Feature leakage through return_5 (Step 2)**: `return_5` is both a non-spatial feature for XGBoost and partially determines `tb_label` (which is derived from forward price movement). If XGBoost exploits this correlation, accuracy is inflated. Phase B showed `return_5` was NOT in the top-10 by gain. Mitigation: report `return_5` importance rank; if it ranks top 3, flag and re-run XGBoost excluding it.

6. **Single seed**: With seed=42, results are deterministic but may represent a lucky or unlucky random initialization. The 5-fold temporal CV provides robustness across time periods but not across initializations. If results are near the 0.10 threshold (±0.02), a follow-up with 2-3 additional seeds would be warranted before making architecture decisions.

## Deliverables

```
.kit/results/cnn-reproduction-diagnostic/
  step1/
    mve_diagnostics.txt           # Fold 5 MVE: sample inputs, param count, LR schedule, train/test R²
    normalization_verification.txt # Sample values for channel 0 (raw ticks) and channel 1 (log1p sizes)
    fold_results.json             # Per-fold: {train_r2, test_r2, epochs_trained, final_lr}
    training_curves.csv           # epoch × fold → train_loss, val_loss, lr
    cnn_checkpoints/              # Best model per fold (for Step 2 embedding extraction)
  step2/                          # Only populated if Step 1 passes
    fold_results.json             # Per-fold: {accuracy, f1_macro, expectancy, pf} × 3 cost scenarios
    ablation_gbt_only.json        # GBT-only per-fold and aggregate
    predictions.csv               # Test-set: bar_index, true_label, predicted, probs
    feature_importance.json       # XGBoost top-10 by gain
    cost_sensitivity.json         # 3 cost scenarios × {expectancy, pf, trade_count}
  aggregate_metrics.json          # Pooled Step 1 + Step 2 metrics
  analysis.md                     # Human-readable with all required sections
```

## Deviations Log

Any deviation from R3's protocol MUST be logged here by the RUN agent:

| Parameter | R3 Value | Actual Value | Justification |
|-----------|----------|-------------|---------------|
| (none expected — target is zero deviations) | | | |

## Exit Criteria

- [x] Step 1 MVE check executed (fold 5 only) with diagnostics reported — train R²=0.002, test R²=0.0001. MVE gate FAIL (24× below threshold).
- [x] Input normalization verified (sample values printed, channel 0 = raw ticks confirmed) — channel 0 in index points (not z-scored), channel 1 z-scored log1p. Soft fail: units in index points not integer ticks, scale tested — no effect.
- [x] Architecture verified (param count logged, matches R3 within 5%) — 12,128 params, 0.0% deviation.
- [x] Full 5-fold Step 1 completed with per-fold R² reported — ABORTED by MVE gate (correct). 5 normalization variants tested diagnostically, all R² < 0.002.
- [x] Step 1 gate evaluated: **FAIL** (< 0.05). Fold 5 test R²=0.0001, 1000× below 0.10 threshold.
- [x] R3 comparison table (this experiment vs R3, per-fold) — Fold 5: R3=0.159, this=0.0001, delta=−0.159.
- [ ] ~~If Step 1 PASS/MARGINAL: Step 2 hybrid integration completed~~ — N/A (Step 1 FAIL)
- [ ] ~~If Step 1 PASS/MARGINAL: GBT-only ablation completed~~ — N/A (Step 1 FAIL)
- [ ] ~~If Step 1 PASS/MARGINAL: PnL and cost sensitivity computed~~ — N/A (Step 1 FAIL)
- [ ] ~~If Step 1 PASS/MARGINAL: Hybrid vs GBT-only delta reported~~ — N/A (Step 1 FAIL)
- [x] analysis.md written with all required sections — `.kit/results/cnn-reproduction-diagnostic/analysis.md`
- [x] Deviations log completed (target: empty) — 2 deviations documented: data source (C++ vs Python export), channel 0 units (index points vs ticks).
- [x] All SC-1 through SC-5 evaluated explicitly — SC-1 FAIL, SC-2 FAIL, SC-3 NOT EVAL, SC-4 NOT EVAL, SC-5 FAIL.
