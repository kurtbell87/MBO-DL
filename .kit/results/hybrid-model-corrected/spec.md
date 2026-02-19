# Experiment: CNN+GBT Hybrid Model (Corrected Pipeline)

## Hypothesis

The CNN+GBT Hybrid architecture — Conv1d spatial encoder on structured (20,2) book input with **corrected normalization** (TICK_SIZE division on prices, per-day z-scoring on sizes) and **proper validation** (80/20 train/val split for early stopping, never test data) — will achieve:

1. **CNN regression R² at h=5 >= 0.05** (mean across 5 expanding-window folds with proper validation). 9D's proper-validation run achieved R²=0.084; the 0.05 threshold accounts for seed variance and the known weak fold 3 (R²=-0.047 under proper validation).
2. **XGBoost classification accuracy >= 0.38** on 3-class triple barrier labels, with the hybrid feature set (CNN 16-dim embedding + 20 non-spatial features) outperforming both GBT-only (all non-book features, no CNN) and CNN-only (16-dim embedding, no hand-crafted features).
3. **Aggregate per-trade expectancy >= $0.50** under base costs ($3.74 RT), demonstrating the CNN spatial signal converts to economically viable trading signals when normalization is correct.

These thresholds account for: (a) R3's leaked R²=0.132 vs proper-validation R²=0.084 (36% deflation), (b) oracle ceiling of $4.00/trade (R7), and (c) the gap between regression R² and classification accuracy.

## Independent Variables

1. **Model configuration** (3 levels):
   - **Hybrid** (primary): CNN 16-dim embedding + 20 non-spatial features → XGBoost classifier (36 input dims)
   - **GBT-only** (ablation baseline): All available hand-crafted non-book features → XGBoost classifier (no CNN)
   - **CNN-only** (ablation baseline): CNN 16-dim embedding only → XGBoost classifier (16 input dims, no hand-crafted features)

2. **CNN regression horizon** (1 level):
   - h=5 (5-bar, ~25s forward return) — R3's primary horizon, confirmed by 9D reproduction

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | `.kit/results/hybrid-model/time_5s.csv` (87,970 bars, 19 days) | Same C++ export used by R3, confirmed byte-identical by 9D |
| Bar type | time_5s | Locked by R1 + R6 |
| Labels | Triple barrier (target=10, stop=5, vol_horizon=500) | Locked by R7 ($4.00/trade oracle) |
| CV strategy | 5-fold expanding-window, split by day boundaries, no shuffling | Consistent with R2/R3/R4 |
| CNN architecture | Conv1d(2→59→59) + BN + ReLU ×2 → AdaptiveAvgPool1d(1) → Linear(59→16) + ReLU → Linear(16→1). **12,128 params.** | R3's exact architecture, confirmed by 9C (0% deviation) and 9D (perfect reproduction) |
| CNN optimizer | AdamW(lr=1e-3, weight_decay=1e-4) | R3's exact optimizer |
| CNN LR schedule | CosineAnnealingLR(T_max=50, eta_min=1e-5) | R3's exact schedule |
| CNN batch size | 512 | R3's exact batch size |
| CNN early stopping | Patience=10 on **validation loss** (held-out 80/20 split from train days). **NEVER use test data for early stopping.** | Proper validation — R3 used test-as-val (leakage). This is the primary methodological fix. |
| CNN max epochs | 50 | R3's exact value |
| CNN loss | MSE on fwd_return_5 | R3's exact loss |
| CNN seed | seed = 42 + fold_idx (seeds 42, 43, 44, 45, 46 for folds 1-5) | **Matches 9D exactly.** Enables direct per-fold R² comparison as normalization verification. Any per-fold deviation > 0.01 from 9D signals a protocol error. |
| XGBoost hyperparameters | max_depth=6, lr=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0 | Same as Phase 9B for direct comparison |
| XGBoost seed | 42 | Fixed seed for XGBoost reproducibility |
| Hardware | CPU only | CNN ~12k params; XGBoost native CPU |

### Normalization Protocol (CRITICAL — root cause of 9B/9C failure)

These three normalization steps MUST be applied exactly as specified. 9D confirmed that R3's signal requires them:

**1. Book price offsets (channel 0):**
- Raw CSV values are in index points (range ±5.625)
- **DIVIDE by TICK_SIZE = 0.25** to convert to tick offsets (range ±22.5, integer-quantized)
- Do NOT z-score channel 0. Raw tick integers are the correct representation.
- **Verification:** After division, values should be integer-valued (tolerance ±0.01). Print sample values and report the fraction that are integer-valued.

**2. Book sizes (channel 1):**
- Raw CSV values are lot sizes (range 1–697)
- Apply `log1p()` to raw sizes
- **Z-score PER DAY** (compute mean and std of log1p(size) for each trading day independently, apply per-day stats)
- Do NOT z-score per fold or globally. Per-day granularity is required.
- **Verification:** After z-scoring, each day should have mean≈0 and std≈1. Print per-day stats.

**3. Non-spatial features (20 features for XGBoost):**
- Z-score per fold using train-fold statistics only (compute mean/std on train set, apply to both train and test).
- NaN → 0.0 after normalization.

### Validation Split Protocol (CRITICAL — prevents R3's leakage bug)

For each fold k:
1. Split training days into ~80% train / ~20% validation **by day boundaries**, using `n_val = max(1, round(n_train_days * 0.2))` validation days taken from the END of the train period.

   | Fold | Total train days | Actual train days | Val days (last N) | Test days |
   |------|-----------------|-------------------|-------------------|-----------|
   | 1 | 4 | 3 (days 1–3) | 1 (day 4) | Days 5–7 |
   | 2 | 7 | 6 (days 1–6) | 1 (day 7) | Days 8–10 |
   | 3 | 10 | 8 (days 1–8) | 2 (days 9–10) | Days 11–13 |
   | 4 | 13 | 10 (days 1–10) | 3 (days 11–13) | Days 14–16 |
   | 5 | 16 | 13 (days 1–13) | 3 (days 14–16) | Days 17–19 |

2. Early stopping monitors validation loss (NOT test loss).
3. Test data is NEVER seen during training or model selection.
4. **Verification:** Print the day indices for train/val/test splits for each fold. Confirm no test-day data in val set.

## Metrics (ALL must be reported)

### Primary

1. **mean_cnn_r2_h5**: Mean out-of-sample R² of CNN regression at h=5 across 5 folds (proper validation).
2. **aggregate_expectancy_base**: Per-trade expectancy ($) pooled across all test predictions under base costs ($3.74 RT).

### Secondary

| Metric | Description |
|--------|-------------|
| per_fold_cnn_r2_h5 | Per-fold test R² at h=5 (compare with 9D proper-validation: [0.134, 0.083, -0.047, 0.117, 0.135]) |
| per_fold_cnn_train_r2_h5 | Per-fold train R² (must be > 0.05 — 9D showed 0.157–0.196) |
| per_fold_delta_vs_9d | Per-fold |this_run - 9D_proper_val| — normalization verification signal |
| epochs_trained_per_fold | Epochs before early stopping |
| mean_xgb_accuracy | XGBoost 3-class accuracy across 5 folds (Hybrid config) |
| mean_xgb_f1_macro | Macro F1 across 5 folds |
| aggregate_profit_factor | Gross profit / gross loss pooled (base cost) |
| ablation_delta_accuracy_gbt | Hybrid accuracy - GBT-only accuracy |
| ablation_delta_accuracy_cnn | Hybrid accuracy - CNN-only accuracy |
| ablation_delta_expectancy_gbt | Hybrid expectancy - GBT-only expectancy |
| ablation_delta_expectancy_cnn | Hybrid expectancy - CNN-only expectancy |
| cost_sensitivity_table | Expectancy and PF under optimistic ($2.49), base ($3.74), pessimistic ($6.25) |
| xgb_top10_features | XGBoost feature importance (gain) top-10 |
| label_distribution | Per-fold class counts for tb_label {-1, 0, +1} |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| CNN param count | 12,128 ± 5% | Architecture mismatch |
| Channel 0 tick-quantized after TICK_SIZE division | >= 0.99 fraction integer-valued | TICK_SIZE division not applied — **FATAL** (same error as 9B/9C) |
| Channel 1 per-day z-scored | Each day mean≈0, std≈1 after log1p + z-score | Per-day normalization not applied |
| Train R² per fold > 0.05 | All folds > 0.05 | Pipeline still broken — ABORT |
| Validation split separate from test | No test-day data in validation set | Validation leakage — **FATAL** |
| No NaN in CNN outputs | 0 NaN | Normalization or forward pass bug |
| Fold boundaries non-overlapping | No day in both train and test | Temporal leakage |
| XGBoost accuracy > 0.33 and <= 0.90 | In range | Below = learning nothing; above = leakage |
| LR decays from ~1e-3 toward ~1e-5 | Cosine schedule active | CosineAnnealingLR not applied |
| Per-fold CNN R² delta vs 9D | |delta| < 0.02 for >= 4/5 folds | Normalization or protocol mismatch — investigate |

## Baselines

### 1. 9D R3 Reproduction with Proper Validation (primary reference)
- **Source:** `.kit/results/r3-reproduction-pipeline-comparison/analysis.md`
- **Per-fold R² (proper val):** [0.134, 0.083, -0.047, 0.117, 0.135]
- **Mean:** 0.084 ± 0.074
- **Seeds used:** seed = 42 + fold_idx (this experiment uses the same seeds)
- **Note:** This experiment should produce CNN R² values within ±0.02 per fold. Larger deviations indicate a normalization or protocol error. The proper-validation protocol (80/20 train/val split) is identical.

### 2. Phase 9B Original (broken pipeline reference)
- **Source:** `.kit/results/hybrid-model-training/analysis.md`
- **CNN R²:** -0.002 (train R²=0.001 — pipeline broken)
- **XGBoost accuracy:** 0.41
- **GBT-only expectancy:** -$0.38/trade
- **Note:** This experiment must beat all of these. The CNN normalization fix alone should produce R²≈0.084 (vs -0.002).

### 3. Oracle (ceiling)
- **Source:** R7 oracle-expectancy
- **Value:** $4.00/trade, PF=3.30, WR=64.3%

### 4. Random Baseline
- 3-class accuracy: 0.333, expectancy: negative

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: mean_cnn_r2_h5 >= 0.05 (with proper validation; 9D showed 0.084)
- [ ] **SC-2**: No fold train R² < 0.05 (9D showed min=0.157; ensures pipeline is working)
- [ ] **SC-3**: mean_xgb_accuracy >= 0.38 (above random by >= 5pp)
- [ ] **SC-4**: aggregate_expectancy_base >= $0.50/trade (economically viable)
- [ ] **SC-5**: aggregate_profit_factor_base >= 1.5 (profitable with margin)
- [ ] **SC-6**: Hybrid outperforms GBT-only on accuracy OR expectancy (at least one)
- [ ] **SC-7**: Hybrid outperforms CNN-only on accuracy OR expectancy (at least one)
- [ ] **SC-8**: Cost sensitivity table produced for all 3 scenarios
- [ ] **SC-9**: No sanity check failures (including TICK_SIZE verification, validation split verification, and per-fold CNN R² delta vs 9D)

## Decision Rules

```
OUTCOME A — Full Success (SC-1 through SC-9 all PASS):
  → CNN+GBT Hybrid is validated with corrected normalization.
  → Proceed to multi-seed robustness study (5 seeds × 5 folds) to confirm stability.
  → If multi-seed confirms, proceed to full-year training (312 days).
  → Architecture is locked: Conv1d(2→59→59) + XGBoost.

OUTCOME B — CNN Works, XGBoost Fails (SC-1+SC-2 PASS, SC-3 or SC-4 FAIL):
  → CNN spatial signal confirmed at R²≈0.084 with corrected normalization.
  → But CNN embeddings do not convert to economically viable classification.
  → Investigate: (1) XGBoost hyperparameter sensitivity, (2) label distribution
     effects, (3) embedding quality (linear probe on tb_label from 16-dim).
  → Do NOT abandon CNN. The spatial signal is real. The integration needs work.

OUTCOME C — CNN Fails (SC-1 or SC-2 FAIL):
  → Pipeline fix did NOT restore CNN signal despite corrected normalization.
  → This contradicts 9D (which showed proper-validation R²=0.084 with the
     same data, normalization, and seed strategy). Most likely explanation:
     implementation bug in the RUN agent's normalization code.
  → DO NOT abandon CNN path. Instead: diagnose by comparing this run's
     per-fold R² against 9D's exact values (the matching seed strategy
     enables fold-by-fold debugging). Any fold with |delta| > 0.02 has a
     normalization error.
  → If normalization is verified correct and R² still < 0.05: report as
     non-reproducible and escalate.

OUTCOME D — Hybrid Equals GBT-only (SC-1+SC-2 PASS, SC-6 FAIL):
  → CNN signal is real but adds nothing over hand-crafted features for
     classification on triple barrier labels.
  → Check CNN-only ablation: if CNN-only matches GBT-only, the spatial
     information the CNN captures is already present in hand-crafted book
     features (e.g., weighted_imbalance, spread). Simplify to GBT-only.
  → If CNN-only is worse than GBT-only but Hybrid = GBT-only, the CNN
     embedding is neither helpful nor harmful — XGBoost ignores it.
     Simplify to GBT-only.

OUTCOME E — Hybrid Equals CNN-only (SC-1+SC-2 PASS, SC-7 FAIL):
  → Hand-crafted features add nothing over CNN embedding alone.
  → Simplify to CNN-only pipeline (16-dim → XGBoost). Reduces
     feature engineering dependency.
```

## Minimum Viable Experiment

Before full 5-fold:

**1. Normalization verification (MANDATORY — prevents 9B/9C repeat):**
- Load `time_5s.csv`, identify 40 book columns.
- Apply TICK_SIZE division (/ 0.25) on channel 0. Print 5 sample values → must be integer-valued.
- Apply log1p + per-day z-score on channel 1. Print per-day mean/std → must be ≈0/≈1.
- **ABORT if channel 0 values are NOT integer-like after division.**

**2. Architecture verification:**
- Build CNN, print param count → must be 12,128 ± 5%.
- Print layer structure → must match R3 spec exactly.
- **ABORT if param count deviates > 10%.**

**3. Single-fold CNN training (fold 5 — maximum data, seed=46):**
- Split fold 5 train days (1–16) into 13 train + 3 val BY DAY.
- Train CNN with R3-exact protocol + proper validation.
- Print: train R², val R², test R², epochs trained.
- **Gate A:** train R² < 0.05 → normalization is still wrong. ABORT.
- **Gate B:** test R² < -0.10 → severe overfitting. Check validation protocol.
- **Gate C:** |test R² - 0.135| > 0.03 → significant deviation from 9D fold 5 (0.135). Investigate normalization before proceeding.
- **Gate D:** test R² > 0.05 → pipeline is working. Proceed.

**4. Single-fold XGBoost check (fold 5):**
- Extract 16-dim embeddings. Concatenate with 20 features. Train XGBoost on tb_label.
- Print accuracy → must be > 0.33.
- **ABORT if accuracy < 0.33.**

## Full Protocol

### Step 0: Environment Setup
- Set seed=42 globally for initial setup. Per-fold CNN seeds are 42+fold_idx.
- Log PyTorch version, XGBoost version, polars/pandas version.

### Step 1: Data Loading and Normalization
1. Load `.kit/results/hybrid-model/time_5s.csv`.
2. Identify 19 unique days, sorted chronologically.
3. Identify 40 book columns. Reshape to (N, 2, 20) channels-first.
4. **Apply TICK_SIZE normalization:** channel 0 = raw values / 0.25.
5. **Apply per-day z-scoring:** For each day, compute log1p(channel 1 raw), then z-score within that day (day mean, day std).
6. Construct `fwd_return_5` target from bar close prices.
7. Verify: print channel 0 samples (integer ticks), channel 1 per-day stats, bar count, day count.

### Step 2: Define 5-Fold Expanding-Window Splits

Print the exact day assignments below. Verify no test-day data in val set.

| Fold | Train Days | Val Days | Test Days | CNN Seed |
|------|-----------|----------|-----------|----------|
| 1 | Days 1–3 | Day 4 | Days 5–7 | 42 |
| 2 | Days 1–6 | Day 7 | Days 8–10 | 43 |
| 3 | Days 1–8 | Days 9–10 | Days 11–13 | 44 |
| 4 | Days 1–10 | Days 11–13 | Days 14–16 | 45 |
| 5 | Days 1–13 | Days 14–16 | Days 17–19 | 46 |

### Step 3: Run MVE
Execute MVE checks (Steps 1-4 above). Abort if any gate fails.

### Step 4: Full 5-Fold CNN Training

For each fold k in [1..5]:
1. Set CNN seed = 42 + fold_idx (matching 9D exactly).
2. Split by day boundaries per Step 2 table.
3. Channel 0: already tick-normalized from Step 1. No further normalization.
4. Channel 1: already per-day z-scored from Step 1. No further normalization.
5. Train CNN:
   - AdamW(lr=1e-3, weight_decay=1e-4)
   - CosineAnnealingLR(T_max=50, eta_min=1e-5)
   - Batch size: 512
   - Max epochs: 50, early stopping patience=10 on **validation loss**
   - MSE loss on `fwd_return_5`
6. Save best checkpoint (lowest val loss).
7. Record: train R², val R², test R², epochs trained.
8. **Verification:** Compare test R² with 9D reference values:

   | Fold | 9D Proper-Val R² | Acceptable range (±0.02) |
   |------|-----------------|-------------------------|
   | 1 | 0.134 | [0.114, 0.154] |
   | 2 | 0.083 | [0.063, 0.103] |
   | 3 | -0.047 | [-0.067, -0.027] |
   | 4 | 0.117 | [0.097, 0.137] |
   | 5 | 0.135 | [0.115, 0.155] |

   If > 1 fold falls outside the acceptable range, investigate normalization before proceeding to Step 5.

### Step 5: CNN Embedding Extraction

For each fold k:
1. Load best CNN checkpoint.
2. Freeze all weights. Remove regression head (Linear(16→1)).
3. Forward pass all train and test bars.
4. Extract 16-dim output of Linear(59→16) + ReLU.
5. Verify: 0 NaN in embeddings.

### Step 6: Hybrid XGBoost Classification

For each fold k:
1. Z-score 20 non-spatial features using train-fold stats. NaN → 0.0.
2. Concatenate: 16-dim CNN embedding + 20 non-spatial features = 36-dim.
3. Train XGBoost (multi:softmax, num_class=3, seed=42, same hyperparameters as Controls table) on train → tb_label.
4. Predict on test set. Record predictions and probabilities.

### Step 7: GBT-Only Ablation

For each fold k:
- Same XGBoost config (seed=42), but input = all available non-book features from CSV (no CNN embedding).
- Same metrics as Step 6.

### Step 8: CNN-Only Ablation

For each fold k:
- Same XGBoost config (seed=42), but input = 16-dim CNN embedding only (no hand-crafted features).
- Same metrics as Step 6.

### Step 9: PnL Computation

For each model config (Hybrid, GBT-only, CNN-only), for each fold, for each cost scenario:
```
tick_value = $1.25 per tick (MES: $0.25 × 5 multiplier)
target_ticks = 10 → win PnL = $12.50 - RT_cost
stop_ticks = 5  → loss PnL = -$6.25 - RT_cost

Correct directional call (pred sign = label sign, both nonzero): PnL = +$12.50 - RT_cost
Wrong directional call (pred sign != label sign, both nonzero):  PnL = -$6.25 - RT_cost
Predict 0 (hold):                                                PnL = $0 (no trade)
True label = 0 but model predicted ±1:                           PnL = $0 (simplified — treat expiry as flat; the true PnL is
                                                                  unknown without simulating the full path, but is bounded
                                                                  by [-$6.25, +$12.50] minus costs)
```

**Note on label=0 simplification:** When the true label is 0 (barrier expired without hitting target or stop) but the model predicted ±1, the real-world PnL depends on where price is at expiry — information not captured in the label. Setting PnL=0 for these cases is a conservative simplification. If label=0 trades are a substantial fraction (>20%) of all directional predictions, the aggregate expectancy estimate is unreliable. Report the fraction and flag if high.

### Step 10: Aggregate Results

- Pool test predictions across 5 folds for each model config.
- Compute all primary, secondary, and sanity check metrics.
- Produce: 9D comparison table, ablation comparison table, cost sensitivity table, feature importance top-10, label distribution.
- Evaluate all SC-1 through SC-9.
- Write analysis.md.

### CNN Architecture (exact specification)

```
Input: (B, 2, 20)                    # channels-first: (price_offset, log1p_size) × 20 levels
Conv1d(in=2, out=59, kernel_size=3, padding=1)  + BatchNorm1d(59) + ReLU
Conv1d(in=59, out=59, kernel_size=3, padding=1) + BatchNorm1d(59) + ReLU
AdaptiveAvgPool1d(1)                 # → (B, 59)
Linear(59, 16) + ReLU               # 16-dim embedding (extraction point for Step 5)
Linear(16, 1)                        # scalar return prediction (removed for embedding extraction)
```

**Parameter count breakdown:**
- Conv1d(2→59, k=3): 2×59×3 + 59 = 413
- BN(59): 59×2 = 118
- Conv1d(59→59, k=3): 59×59×3 + 59 = 10,502
- BN(59): 59×2 = 118
- Linear(59→16): 59×16 + 16 = 960
- Linear(16→1): 16×1 + 1 = 17
- **Total: 12,128**

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

Map from actual CSV column names at load time. Document any substitutions. Note: `return_5` is the *backward-looking* 5-bar return (a legitimate feature), not the forward return. The CNN target `fwd_return_5` is a separate column. No feature leaks the forward return.

### Transaction Cost Scenarios

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

## Resource Budget

**Tier:** Standard

- Max GPU-hours: 0 (CPU only)
- Max wall-clock time: 120 minutes
- Max training runs: 21 (1 MVE CNN + 5 folds CNN + 5 folds XGB hybrid + 5 folds XGB GBT-only + 5 folds XGB CNN-only)
- Max seeds: CNN seed=42+fold_idx (5 seeds); XGBoost seed=42 (1 seed)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 87970
model_type: pytorch+xgboost
sequential_fits: 21
parallelizable: false
memory_gb: 4
gpu_type: none
estimated_wall_hours: 0.5
```

**Estimate breakdown**: CNN (~12k params) on ~74k rows trains in ~10–15s/fold on CPU. XGBoost (500 trees, 36 features, ~74k rows) trains in ~5s/fold. 21 total fits × ~10s average = ~210s training. Data loading, reshape, normalization, embedding extraction, evaluation, and I/O add ~15 minutes. Total: ~20–30 minutes. Budget of 120 minutes provides 4–6× headroom.

### Wall-Time Estimation Guidance

| Component | Per-unit estimate | Count | Subtotal |
|-----------|------------------|-------|----------|
| Data loading + normalization | 2–5 min | 1 | 5 min |
| CNN training (12k params, ~74k rows) | 10–15s | 6 (1 MVE + 5 full) | 90s |
| CNN embedding extraction | 5s | 5 | 25s |
| XGBoost training (500 trees, 36 dim) | 5s | 15 (3 configs × 5 folds) | 75s |
| PnL computation + aggregation | 2 min | 1 | 2 min |
| File I/O + analysis | 5 min | 1 | 5 min |
| **Total estimated** | | | **~15–20 min** |

## Abort Criteria

- **TICK_SIZE verification fails:** Channel 0 not integer-like after /0.25 → ABORT immediately. Same error as 9B/9C.
- **Validation split uses test data:** Any test-day data in val set → ABORT. Same leak as R3.
- **CNN param count wrong:** > 10% deviation from 12,128 → ABORT.
- **MVE Gate A (train R² < 0.05):** Pipeline still broken despite normalization fix → ABORT. Investigate normalization.
- **MVE Gate C (|fold 5 test R² - 0.135| > 0.03):** Significant deviation from 9D → stop and verify normalization before full run.
- **NaN loss:** Any fold → ABORT.
- **All 5 folds negative test R²:** Signal absent → ABORT.
- **Wall-clock > 120 min:** ABORT remaining, report completed work.
- **Per-run time:** Any single CNN fit exceeds 5 minutes (20× expected) → investigate and abort if unresolvable.

## Confounds to Watch For

1. **Validation split reduces effective training data.** With proper 80/20 split, folds 1–2 have very few validation days (1 day each). Early stopping may be unreliable on small validation sets. Expected impact: higher variance on early folds, possibly lower R² than 9D's values for folds 1–2. This is inherent to the expanding-window design and not fixable without changing the CV strategy.

2. **Fold 3 regime weakness.** 9D showed fold 3 (Oct 2022) has R²=-0.047 under proper validation. Expect this fold to be weak. This is a market regime effect, not a pipeline bug. A negative R² on fold 3 alone does NOT indicate failure — evaluate SC-1 on the mean across all 5 folds.

3. **Label distribution skew.** Triple barrier labels may be imbalanced. The asymmetric target:stop ratio (10:5 ticks) biases toward +1 and -1 over 0. Report distribution per fold. If any class exceeds 50%, flag and report majority-class accuracy alongside model accuracy.

4. **Feature name mismatch.** CSV column names may differ from spec names. The RUN agent must map and document. A column mapping error silently fills features with zeros after normalization, destroying signal — the sanity check on XGBoost accuracy > 0.33 partially catches this.

5. **CNN embedding quality for classification.** The CNN is trained on regression (MSE on fwd_return_5) but used for classification (tb_label). The 16-dim embedding captures return prediction variance, not label boundaries. This is a known architectural compromise — the alternative (end-to-end CNN+XGBoost training) is not feasible with frozen embeddings. If SC-4/SC-5 fail while SC-1 passes, this compromise may be the limiting factor.

6. **return_5 feature importance.** `return_5` is the backward-looking 5-bar return and does NOT directly leak forward returns or tb_label. However, if XGBoost assigns > 20% gain share to `return_5`, it may indicate the feature proxies for recent momentum that partially predicts the next 5-bar direction. This is legitimate signal, not leakage, but report it prominently if it occurs.

7. **Label=0 PnL simplification.** The PnL model assigns $0 to trades where true_label=0 but model predicted ±1. In reality, the trader would be in a position that expires between stop and target — the PnL could be anywhere in [-$6.25, +$12.50] minus costs. If label=0 trades are a substantial fraction (>20%) of all directional predictions, the aggregate expectancy estimate is unreliable. Report the fraction and flag if high.

## Deliverables

```
.kit/results/hybrid-model-corrected/
  step1_cnn/
    fold_results.json              # Per-fold: {train_r2, val_r2, test_r2, epochs_trained, seed}
    normalization_verification.txt # Channel 0 samples, channel 1 per-day stats
    architecture_verification.txt  # Param count, layer structure
    r3_comparison_table.csv        # This run CNN R² vs 9D proper-validation R² per fold
  step2_hybrid/
    fold_results.json              # Per-fold: {accuracy, f1_macro, expectancy, pf} × 3 costs
    predictions.csv                # Test-set: bar_idx, true_label, pred_label, probs
    feature_importance.json        # XGBoost top-10 by gain
  ablation_gbt_only/
    fold_results.json              # GBT-only per-fold and aggregate
  ablation_cnn_only/
    fold_results.json              # CNN-only per-fold and aggregate
  cost_sensitivity.json            # 3 scenarios × 3 configs × {expectancy, pf, trade_count}
  label_distribution.json          # Per-fold class counts
  aggregate_metrics.json           # All pooled metrics
  analysis.md                      # Human-readable with all sections + SC pass/fail
```

### Required Outputs in analysis.md

1. Executive summary: does the hybrid model produce actionable signals?
2. CNN R² comparison table: this run vs 9D proper-validation, per fold and mean.
3. XGBoost accuracy and F1 table: fold × metric (all 3 configs).
4. PnL table: config × cost scenario → expectancy, profit factor.
5. Ablation comparison: Hybrid vs GBT-only vs CNN-only (table with all metrics).
6. Label distribution: per-fold class counts for tb_label.
7. Feature importance: XGBoost top-10 features by gain. Flag return_5 if top-3.
8. Fraction of label=0 trades in directional predictions (PnL simplification impact).
9. Explicit pass/fail for each of SC-1 through SC-9.

## Exit Criteria

- [ ] TICK_SIZE normalization verified (channel 0 = integer ticks after /0.25)
- [ ] Per-day z-scoring verified (channel 1 per-day mean≈0, std≈1)
- [ ] Validation split verified (no test data in val set, day boundaries printed)
- [ ] Architecture verified (12,128 params ± 5%)
- [ ] MVE gate passed (fold 5 train R² > 0.05, test R² within 0.03 of 9D)
- [ ] Full 5-fold CNN training completed with per-fold R² reported
- [ ] CNN R² comparison with 9D proper-validation reference (per-fold deltas)
- [ ] Hybrid XGBoost training completed (5 folds)
- [ ] GBT-only ablation completed (5 folds)
- [ ] CNN-only ablation completed (5 folds)
- [ ] PnL and cost sensitivity computed (3 scenarios × 3 configs)
- [ ] Feature importance top-10 reported (return_5 flagged if top-3)
- [ ] Label distribution per fold reported
- [ ] Label=0 simplification impact reported (fraction of affected trades)
- [ ] Hybrid vs GBT-only delta reported
- [ ] Hybrid vs CNN-only delta reported
- [ ] analysis.md written with all required sections
- [ ] All SC-1 through SC-9 evaluated explicitly
