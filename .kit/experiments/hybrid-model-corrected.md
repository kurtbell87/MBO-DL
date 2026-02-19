# Experiment: CNN+GBT Hybrid Model (Corrected Pipeline)

## Hypothesis

The CNN+GBT Hybrid architecture — Conv1d spatial encoder on structured (20,2) book input with **corrected normalization** (TICK_SIZE division on prices, per-day z-scoring on sizes) and **proper validation** (80/20 train/val split for early stopping) — will achieve:

1. **CNN regression R² at h=5 >= 0.05** (mean across 5 expanding-window folds with proper validation). 9D's proper-validation run achieved R²=0.084; the 0.05 threshold accounts for seed variance (this experiment uses seed=42 vs 9D's seed=42+fold_idx).
2. **XGBoost classification accuracy >= 0.38** on 3-class triple barrier labels, with the hybrid feature set (CNN 16-dim embedding + 20 non-spatial features) outperforming GBT-book (40 raw book columns + 20 non-spatial features) on accuracy or expectancy.
3. **Aggregate per-trade expectancy >= $0.50** under base costs ($3.74 RT), demonstrating the CNN spatial signal converts to economically viable trading signals.

These thresholds account for: (a) R3's leaked R²=0.132 vs proper-validation R²=0.084 (36% deflation), (b) oracle ceiling of $4.00/trade (R7), and (c) the gap between regression R² and classification accuracy.

## Independent Variables

1. **Model configuration** (3 levels):
   - **Hybrid** (primary): CNN 16-dim embedding + 20 non-spatial features -> XGBoost classifier (36 input dims)
   - **GBT-book** (ablation — CNN value test): 40 raw book columns + 20 non-spatial features -> XGBoost classifier (60 input dims). Tests whether CNN's learned 16-dim compression of the book adds value over XGBoost's native handling of raw book features.
   - **GBT-nobook** (ablation — book value test): 20 non-spatial features only -> XGBoost classifier (20 input dims). Tests baseline value of non-spatial features without any book information.

2. **CNN regression horizon** (1 level):
   - h=5 (5-bar, ~25s forward return) — R3's primary horizon, confirmed by 9D reproduction

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | `.kit/results/hybrid-model/time_5s.csv` (87,970 bars, 19 days) | Same C++ export used by R3, confirmed byte-identical by 9D |
| Bar type | time_5s | Locked by R1 + R6 |
| Labels | Triple barrier (target=10, stop=5, vol_horizon=500) | Locked by R7 ($4.00/trade oracle) |
| CV strategy | 5-fold expanding-window, split by day boundaries, no shuffling | Consistent with R2/R3/R4 |
| CNN architecture | Conv1d(2->59->59) + BN + ReLU x2 -> AdaptiveAvgPool1d(1) -> Linear(59->16) + ReLU -> Linear(16->1). **12,128 params.** | R3's exact architecture, confirmed by 9C (0% deviation) and 9D (perfect reproduction) |
| CNN optimizer | AdamW(lr=1e-3, weight_decay=1e-4) | R3's exact optimizer |
| CNN LR schedule | CosineAnnealingLR(T_max=50, eta_min=1e-5) | R3's exact schedule |
| CNN batch size | 512 | R3's exact batch size |
| CNN early stopping | Patience=10 on **validation loss** (held-out 80/20 split from train days). **NEVER use test data for early stopping.** | Proper validation — R3 used test-as-val (leakage). This is the primary methodological fix. |
| CNN max epochs | 50 | R3's exact value |
| CNN loss | MSE on fwd_return_5 | R3's exact loss |
| XGBoost hyperparameters | max_depth=6, lr=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0 | Same as Phase 9B for direct comparison |
| Seed | 42 (torch, numpy, random) | Fixed seed for reproducibility |
| Hardware | CPU only | CNN ~12k params; XGBoost native CPU |

### Normalization Protocol (CRITICAL — root cause of 9B/9C failure)

These normalization steps MUST be applied exactly as specified. 9D confirmed that R3's signal requires them:

**1. Book price offsets (channel 0) — for CNN input only:**
- Raw CSV values are in index points (range +/-5.625)
- **DIVIDE by TICK_SIZE = 0.25** to convert to tick offsets (range +/-22.5, integer-quantized)
- Do NOT z-score channel 0. Raw tick integers are the correct representation.
- **Verification:** After division, values should be integer-valued (tolerance +/-0.01). Print sample values.

**2. Book sizes (channel 1) — for CNN input only:**
- Raw CSV values are lot sizes (range 1-697)
- Apply `log1p()` to raw sizes
- **Z-score PER DAY** (compute mean and std of log1p(size) for each day independently, apply per-day stats)
- Do NOT z-score per fold or globally. Per-day granularity is required.
- **Verification:** After z-scoring, each day should have mean~=0 and std~=1. Print per-day stats.

**3. Non-spatial features (20 features — all XGBoost configs):**
- Z-score per fold using train-fold statistics only (compute mean/std on train set, apply to both train and test).
- NaN -> 0.0 after normalization.

**4. Book columns for GBT-book ablation:**
- Use raw CSV values (no TICK_SIZE division, no z-scoring). XGBoost is tree-based and scale-invariant — monotonic transforms do not change split points or ranking.

### Validation Split Protocol (CRITICAL — prevents R3's leakage bug)

For each fold k:
1. Split training days into train-sub and validation **by day boundaries**, with validation being the last ~20% of training days:

| Fold | Train-Sub Days | Val Days | Test Days | Train-Sub Bars (~) | Val Bars (~) |
|------|---------------|----------|-----------|-------------------|--------------|
| 1 | Days 1-3 | Day 4 | Days 5-7 | 13,890 | 4,630 |
| 2 | Days 1-5 | Days 6-7 | Days 8-10 | 23,150 | 9,260 |
| 3 | Days 1-8 | Days 9-10 | Days 11-13 | 37,040 | 9,260 |
| 4 | Days 1-10 | Days 11-13 | Days 14-16 | 46,300 | 13,890 |
| 5 | Days 1-13 | Days 14-16 | Days 17-19 | 60,190 | 13,890 |

2. Early stopping monitors validation loss (NOT test loss).
3. Test data is NEVER seen during training or model selection.
4. **Verification:** Print the day indices for train-sub/val/test splits for each fold.

## Metrics (ALL must be reported)

### Primary

1. **mean_cnn_r2_h5**: Mean out-of-sample R-squared of CNN regression at h=5 across 5 folds (proper validation).
2. **aggregate_expectancy_base**: Per-trade expectancy ($) pooled across all test predictions under base costs ($3.74 RT).

### Secondary

| Metric | Description |
|--------|-------------|
| per_fold_cnn_r2_h5 | Per-fold test R-squared at h=5 (compare with 9D proper-validation: [0.134, 0.083, -0.047, 0.117, 0.135]) |
| per_fold_cnn_train_r2_h5 | Per-fold train R-squared (must be > 0.05 — 9D showed 0.157-0.196) |
| epochs_trained_per_fold | Epochs before early stopping (9D showed 17-50 with test-as-val; proper val may differ) |
| mean_xgb_accuracy | XGBoost 3-class accuracy across 5 folds (Hybrid config) |
| mean_xgb_f1_macro | Macro F1 across 5 folds (Hybrid config) |
| aggregate_profit_factor | Gross profit / gross loss pooled (base cost) |
| ablation_delta_vs_gbt_book_accuracy | Hybrid accuracy - GBT-book accuracy |
| ablation_delta_vs_gbt_book_expectancy | Hybrid expectancy - GBT-book expectancy |
| ablation_delta_vs_gbt_nobook_accuracy | Hybrid accuracy - GBT-nobook accuracy |
| ablation_delta_vs_gbt_nobook_expectancy | Hybrid expectancy - GBT-nobook expectancy |
| gbt_book_accuracy | GBT-book 3-class accuracy across 5 folds |
| gbt_nobook_accuracy | GBT-nobook 3-class accuracy across 5 folds |
| cost_sensitivity_table | Expectancy and PF under optimistic ($2.49), base ($3.74), pessimistic ($6.25) for all 3 configs |
| xgb_top10_features | XGBoost feature importance (gain) top-10 for Hybrid config |
| label_distribution | Per-fold class counts for tb_label {-1, 0, +1} |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| CNN param count | 12,128 +/- 5% | Architecture mismatch |
| Channel 0 tick-quantized after TICK_SIZE division | >= 0.99 fraction integer-valued | TICK_SIZE division not applied — **FATAL** (same error as 9B/9C) |
| Channel 1 per-day z-scored | Each day mean~=0, std~=1 after log1p + z-score | Per-day normalization not applied |
| Train R-squared per fold > 0.05 | All folds > 0.05 | Pipeline still broken — ABORT |
| Validation split separate from test | No test-day data in validation set | Validation leakage — **FATAL** |
| No NaN in CNN outputs | 0 NaN | Normalization or forward pass bug |
| Fold boundaries non-overlapping | No day in both train and test | Temporal leakage |
| XGBoost accuracy > 0.33 and <= 0.90 | In range | Below = learning nothing; above = leakage |
| LR decays from ~1e-3 toward ~1e-5 | Cosine schedule | CosineAnnealingLR not applied |

## Baselines

### 1. 9D R3 Reproduction with Proper Validation (primary reference)
- **Source:** `.kit/results/r3-reproduction-pipeline-comparison/analysis.md`
- **Per-fold R-squared (proper val):** [0.134, 0.083, -0.047, 0.117, 0.135]
- **Mean:** 0.084 +/- 0.074
- **Note:** This is the corrected CNN R-squared without leakage. This experiment should approximately match these values for the CNN regression step, though small differences are expected due to different seed strategy (seed=42 vs 9D's seed=42+fold_idx).

### 2. Phase 9B Original (broken pipeline reference)
- **Source:** `.kit/results/hybrid-model-training/analysis.md`
- **CNN R-squared:** -0.002 (train R-squared=0.001 — pipeline broken)
- **XGBoost accuracy:** 0.41
- **GBT-only expectancy:** -$0.38/trade
- **Note:** This experiment must beat all of these. The CNN fix alone should produce R-squared~=0.084 (vs -0.002). XGBoost should benefit from the dramatically stronger CNN embeddings.

### 3. Oracle (ceiling)
- **Source:** R7 oracle-expectancy
- **Value:** $4.00/trade, PF=3.30, WR=64.3%

### 4. Random Baseline
- 3-class accuracy: 0.333, expectancy: negative

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: mean_cnn_r2_h5 >= 0.05 (with proper validation; 9D showed 0.084)
- [ ] **SC-2**: No fold train R-squared < 0.05 (9D showed min=0.157; ensures pipeline is working)
- [ ] **SC-3**: mean_xgb_accuracy >= 0.38 (above random by >= 5pp)
- [ ] **SC-4**: aggregate_expectancy_base >= $0.50/trade (economically viable)
- [ ] **SC-5**: aggregate_profit_factor_base >= 1.5 (profitable with margin)
- [ ] **SC-6**: Hybrid outperforms GBT-book on accuracy OR expectancy (CNN encoding adds value over raw book features in XGBoost — the meaningful architecture test)
- [ ] **SC-7**: Cost sensitivity table produced for all 3 scenarios x 3 configs
- [ ] **SC-8**: No sanity check failures (including TICK_SIZE verification and validation split verification)

## Decision Rules

```
OUTCOME A — Full Success (SC-1 through SC-8 all PASS):
  -> CNN+GBT Hybrid is validated with corrected normalization.
  -> CNN encoding beats raw book features in XGBoost (SC-6).
  -> Proceed to multi-seed robustness study (5 seeds x 5 folds) to confirm stability.
  -> If multi-seed confirms, proceed to full-year training (312 days).
  -> Architecture is locked: Conv1d(2->59->59) + XGBoost.

OUTCOME B — CNN Works, XGBoost Fails (SC-1+SC-2 PASS, SC-3 or SC-4 FAIL):
  -> CNN spatial signal confirmed at R-squared~=0.084 with corrected normalization.
  -> But CNN embeddings do not convert to economically viable classification.
  -> Investigate: (1) XGBoost hyperparameter sensitivity, (2) label distribution
     effects, (3) embedding quality (linear probe on tb_label from 16-dim).
  -> Do NOT abandon CNN. The spatial signal is real. The integration needs work.

OUTCOME C — CNN Fails (SC-1 or SC-2 FAIL):
  -> Pipeline fix did NOT restore CNN signal despite corrected normalization.
  -> This contradicts 9D (which showed proper-validation R-squared=0.084 with the
     same data and normalization). Most likely explanation: implementation bug in
     the RUN agent's normalization code (especially TICK_SIZE division or per-day
     z-score computation).
  -> DO NOT abandon CNN path. Instead: diagnose by comparing this run's
     normalization output against 9D's verified normalization output.
  -> If normalization is verified correct but R-squared still < 0.05: seed
     sensitivity is the remaining explanation. Run with seed=42+fold_idx
     (matching 9D exactly) as a follow-up.

OUTCOME D — CNN Works but Doesn't Beat Raw Book Features (SC-1+SC-2 PASS, SC-6 FAIL):
  -> CNN spatial signal is real but XGBoost handles raw book columns natively
     at least as well as CNN's 16-dim compression.
  -> GBT-book (60 features) is simpler and equally effective.
  -> DEPLOY GBT-book architecture. Drop CNN encoder — it adds complexity
     without marginal gain.
  -> Note: GBT-book may win partly due to having 60 features vs Hybrid's 36.
     If Hybrid accuracy is close (within 1pp), CNN compression is still
     valuable for dimensionality reduction. Only abandon CNN if GBT-book
     clearly dominates.
```

## Minimum Viable Experiment

Before full 5-fold:

**1. Normalization verification (MANDATORY — prevents 9B/9C repeat):**
- Load `time_5s.csv`, identify 40 book columns.
- Apply TICK_SIZE division (/ 0.25) on channel 0. Print 5 sample values -> must be integer-valued.
- Apply log1p + per-day z-score on channel 1. Print per-day mean/std -> must be ~=0/~=1.
- **ABORT if channel 0 values are NOT integer-like after division.**

**2. Architecture verification:**
- Build CNN, print param count -> must be 12,128 +/- 5%.
- Print layer structure -> must match R3 spec exactly (Conv1d 2->59->59, Linear 59->16->1).
- **ABORT if param count deviates > 10%.**

**3. Single-fold CNN training (fold 5 — maximum data):**
- Split fold 5 train days (1-16) into train-sub (days 1-13) + val (days 14-16).
- Train CNN with R3-exact protocol + proper validation.
- Print: train R-squared, val R-squared, test R-squared, epochs trained.
- **Gate A:** train R-squared < 0.05 -> normalization is still wrong. ABORT.
- **Gate B:** test R-squared < -0.10 -> severe overfitting. Check validation protocol.
- **Gate C:** test R-squared > 0.05 -> pipeline is working. Proceed.

**4. Single-fold XGBoost check (fold 5):**
- Extract 16-dim embeddings. Concatenate with 20 non-spatial features (z-scored). Train XGBoost on tb_label.
- Print accuracy -> must be > 0.33.
- **ABORT if accuracy < 0.33.**

## Full Protocol

### Step 0: Environment Setup
- Set seed=42 globally (torch, numpy, random).
- Log PyTorch version, XGBoost version, polars/pandas version.

### Step 1: Data Loading and Normalization
1. Load `.kit/results/hybrid-model/time_5s.csv`.
2. Identify 19 unique days, sorted chronologically.
3. Identify 40 book columns. Reshape to (N, 2, 20) channels-first for CNN input.
4. **Apply TICK_SIZE normalization on CNN book tensor:** channel 0 = raw values / 0.25.
5. **Apply per-day z-scoring on CNN book tensor:** For each day, compute log1p(channel 1 raw), then z-score within that day (day mean, day std).
6. Preserve raw book columns separately for GBT-book ablation (no normalization needed).
7. Construct `fwd_return_5` target.
8. Verify: print channel 0 samples (integer ticks), channel 1 per-day stats, bar count, day count.

### Step 2: Define 5-Fold Expanding-Window Splits

| Fold | Train-Sub Days | Val Days | Test Days | Note |
|------|---------------|----------|-----------|------|
| 1 | Days 1-3 | Day 4 | Days 5-7 | Smallest train set; highest variance expected |
| 2 | Days 1-5 | Days 6-7 | Days 8-10 | |
| 3 | Days 1-8 | Days 9-10 | Days 11-13 | Fold 3 test = Oct 2022; expect weakest R-squared |
| 4 | Days 1-10 | Days 11-13 | Days 14-16 | |
| 5 | Days 1-13 | Days 14-16 | Days 17-19 | Largest train set; most reliable estimate |

Print exact day assignments for each fold.

### Step 3: Run MVE
Execute MVE checks. Abort if any gate fails.

### Step 4: Full 5-Fold CNN Training

For each fold k in [1..5]:
1. Split by day boundaries per Step 2 table.
2. Channel 0: already tick-normalized from Step 1. No further normalization.
3. Channel 1: already per-day z-scored from Step 1. No further normalization.
4. Train CNN:
   - AdamW(lr=1e-3, weight_decay=1e-4)
   - CosineAnnealingLR(T_max=50, eta_min=1e-5)
   - Batch size: 512
   - Max epochs: 50, early stopping patience=10 on **validation loss**
   - MSE loss on `fwd_return_5`
5. Save best checkpoint (lowest val loss).
6. Record: train R-squared, val R-squared, test R-squared, epochs trained.

### Step 5: CNN Embedding Extraction

For each fold k:
1. Load best CNN checkpoint.
2. Freeze all weights. Remove regression head (Linear(16->1)).
3. Forward pass all train and test bars through the frozen encoder.
4. Extract 16-dim output of Linear(59->16) + ReLU.
5. Verify: 0 NaN in embeddings.

### Step 6: Hybrid XGBoost Classification

For each fold k:
1. Z-score 20 non-spatial features using train-fold stats.
2. Concatenate: 16-dim CNN embedding + 20 z-scored non-spatial features = 36-dim input.
3. Train XGBoost (multi:softmax, num_class=3, same hyperparameters as Controls table) on train -> tb_label.
4. Predict on test set. Record predictions and probabilities.

### Step 7a: GBT-Book Ablation (CNN value test)

For each fold k:
1. Z-score 20 non-spatial features using train-fold stats.
2. Concatenate: 40 raw book columns (from CSV, no normalization) + 20 z-scored non-spatial features = 60-dim input.
3. Same XGBoost config and hyperparameters as Step 6.
4. Predict on test set. Record predictions and probabilities.

### Step 7b: GBT-Nobook Ablation (book value test)

For each fold k:
1. Z-score 20 non-spatial features using train-fold stats.
2. Input = 20 z-scored non-spatial features only (no book columns, no CNN).
3. Same XGBoost config and hyperparameters as Step 6.
4. Predict on test set. Record predictions and probabilities.

### Step 8: PnL Computation

For each fold, each cost scenario (optimistic $2.49, base $3.74, pessimistic $6.25), each config (Hybrid, GBT-book, GBT-nobook):
```
tick_value = $1.25 per tick (MES: $0.25 x 5 multiplier)
target_ticks = 10 -> win PnL = $12.50 - RT_cost
stop_ticks = 5  -> loss PnL = -$6.25 - RT_cost

Correct directional call (pred sign = label sign): PnL = +$12.50 - RT_cost
Wrong directional call (pred sign != label sign):  PnL = -$6.25 - RT_cost
Predict 0 or true label 0:                         PnL = $0 (no trade)
```

### Step 9: Aggregate Results

- Pool test predictions across 5 folds for each config.
- Compute all primary, secondary, and sanity check metrics.
- Produce: R3/9D comparison table, cost sensitivity table (3 configs x 3 scenarios), feature importance top-10, label distribution.
- Compute ablation deltas: Hybrid vs GBT-book, Hybrid vs GBT-nobook.
- Evaluate all SC-1 through SC-8.
- State which Decision Rule outcome applies.
- Write analysis.md.

### CNN Architecture (exact specification)

```
Input: (B, 2, 20)                    # channels-first: (price_offset, log1p_size) x 20 levels
Conv1d(in=2, out=59, kernel_size=3, padding=1)  + BatchNorm1d(59) + ReLU
Conv1d(in=59, out=59, kernel_size=3, padding=1) + BatchNorm1d(59) + ReLU
AdaptiveAvgPool1d(1)                 # -> (B, 59)
Linear(59, 16) + ReLU               # 16-dim embedding (extraction point for Step 5)
Linear(16, 1)                        # scalar return prediction (removed for embedding extraction)
```

**Parameter count breakdown:**
- Conv1d(2->59, k=3): 2*59*3 + 59 = 413
- BN(59): 59*2 = 118
- Conv1d(59->59, k=3): 59*59*3 + 59 = 10,502
- BN(59): 59*2 = 118
- Linear(59->16): 59*16 + 16 = 960
- Linear(16->1): 16*1 + 1 = 17
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

Map from actual CSV column names at load time. Document any substitutions.

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
- Max training runs: 21 (1 MVE CNN + 5 folds CNN + 5 folds XGB Hybrid + 5 folds XGB GBT-book + 5 folds XGB GBT-nobook)
- Max seeds: 1 (seed=42)

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

**Estimate breakdown**: CNN (~12k params) on ~60k-74k rows trains in ~10-15s/fold on CPU. XGBoost (500 trees, 36-60 features, ~60k-74k rows) trains in ~5s/fold. 21 total fits x ~10s average = ~210s training. Data loading, reshape, normalization, embedding extraction, evaluation, and I/O add ~15 minutes. Total: ~20-30 minutes. Budget of 120 minutes provides 4-6x headroom.

## Abort Criteria

- **TICK_SIZE verification fails:** Channel 0 not integer-like after / 0.25 -> ABORT immediately. Same error as 9B/9C.
- **Validation split uses test data:** Any test-day data in val set -> ABORT. Same leak as R3.
- **CNN param count wrong:** > 10% deviation from 12,128 -> ABORT.
- **MVE Gate A (train R-squared < 0.05):** Pipeline still broken despite normalization fix -> ABORT. Investigate.
- **NaN loss:** Any fold -> ABORT.
- **All 5 folds negative test R-squared:** Signal absent -> ABORT.
- **Wall-clock > 120 min:** ABORT remaining, report completed work.
- **Per-run time:** Any single CNN fit exceeds 5 minutes (20x expected) -> investigate and abort if unresolvable.

## Confounds to Watch For

1. **Validation split reduces effective training data.** With proper 80/20 split, fold 1 has only 3 train-sub days (~13,890 bars) and 1 val day. Early stopping may be unreliable on such a small validation set. Expected impact: higher variance on early folds, possibly lower R-squared than 9D's proper-validation results (which used fold 5 = 16 train days). This is inherent to the expanding-window design and not fixable without changing the CV strategy.

2. **h=5 target leakage through features.** `return_5` is both a non-spatial feature and partially determines `tb_label`. If XGBoost exploits this, accuracy is inflated. Mitigation: report `return_5` importance rank; flag if top-3.

3. **Label distribution skew.** Triple barrier labels may be imbalanced. The asymmetric target:stop ratio (10:5 ticks) biases toward +1 and -1 over 0. Report distribution per fold. If any class exceeds 50%, flag and report majority-class accuracy alongside model accuracy.

4. **Feature name mismatch.** CSV column names may differ from spec names. The RUN agent must map and document. A column mapping error silently fills features with zeros after normalization, destroying signal — the sanity check on XGBoost accuracy > 0.33 partially catches this.

5. **Fold 3 regime weakness.** 9D showed fold 3 (Oct 2022) has R-squared=-0.047 under proper validation. Expect this fold to be weak. This is a market regime effect, not a pipeline bug. A negative R-squared on fold 3 alone does NOT indicate failure — evaluate SC-1 on the mean across all 5 folds.

6. **Seed strategy difference from 9D.** This experiment uses seed=42 for all folds; 9D used seed=42+fold_idx (matching R3). Per-fold CNN R-squared may differ from 9D's proper-validation values by +/-0.02 due to different weight initialization. The mean across 5 folds should be stable. If per-fold R-squared deviates dramatically (>0.05 per fold) from 9D, seed sensitivity is higher than expected and a multi-seed follow-up is warranted.

7. **CNN embedding quality for classification.** The CNN is trained on regression (MSE on fwd_return_5) but used for classification (tb_label). The 16-dim embedding captures return prediction variance, not label boundaries. The embedding may be suboptimal for the 3-class classification task. This is a known architectural compromise — the alternative (end-to-end CNN+XGBoost training) is not feasible with frozen embeddings.

8. **GBT-book feature dimensionality advantage.** GBT-book has 60 features vs Hybrid's 36. If GBT-book wins on accuracy, it could partly reflect having more features rather than proving raw features are better than CNN encoding. However, if Hybrid wins despite fewer features, it demonstrates that CNN compression genuinely adds value. Interpret SC-6 results in this context.

## Deliverables

```
.kit/results/hybrid-model-corrected/
  step1_cnn/
    fold_results.json            # Per-fold: {train_r2, val_r2, test_r2, epochs_trained}
    normalization_verification.txt  # Channel 0 samples, channel 1 per-day stats
    architecture_verification.txt   # Param count, layer structure
    r3_comparison_table.csv      # This run CNN R-squared vs 9D proper-validation R-squared
  step2_hybrid/
    fold_results.json            # Per-fold: {accuracy, f1_macro, expectancy, pf} x 3 costs
    predictions.csv              # Test-set: bar_idx, true_label, pred_label, probs
    feature_importance.json      # XGBoost top-10 by gain
  ablation_gbt_book/
    fold_results.json            # GBT-book (60-feature) per-fold and aggregate
  ablation_gbt_nobook/
    fold_results.json            # GBT-nobook (20-feature) per-fold and aggregate
  cost_sensitivity.json          # 3 configs x 3 scenarios x {expectancy, pf, trade_count}
  label_distribution.json        # Per-fold class counts
  aggregate_metrics.json         # All pooled metrics + ablation deltas
  analysis.md                    # Human-readable with all sections + SC pass/fail + decision rule outcome
```

## Exit Criteria

- [ ] TICK_SIZE normalization verified (channel 0 = integer ticks after / 0.25)
- [ ] Per-day z-scoring verified (channel 1 per-day mean~=0, std~=1)
- [ ] Validation split verified (no test data in val set, day boundaries printed)
- [ ] Architecture verified (12,128 params +/- 5%)
- [ ] MVE gate passed (fold 5 train R-squared > 0.05)
- [ ] Full 5-fold CNN training completed with per-fold R-squared reported
- [ ] CNN R-squared comparison with 9D proper-validation reference
- [ ] Hybrid XGBoost training completed (5 folds)
- [ ] GBT-book ablation completed (5 folds, 60 features)
- [ ] GBT-nobook ablation completed (5 folds, 20 features)
- [ ] PnL and cost sensitivity computed (3 configs x 3 scenarios)
- [ ] Feature importance top-10 reported (return_5 flagged if top-3)
- [ ] Label distribution per fold reported
- [ ] Hybrid vs GBT-book delta reported (SC-6 test)
- [ ] Hybrid vs GBT-nobook delta reported
- [ ] analysis.md written with all required sections
- [ ] All SC-1 through SC-8 evaluated explicitly
- [ ] Decision rule outcome stated explicitly
