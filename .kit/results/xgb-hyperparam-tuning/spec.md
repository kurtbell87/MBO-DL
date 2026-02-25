# Experiment: XGBoost Hyperparameter Tuning on Full-Year Data

**Date:** 2026-02-24
**Priority:** P1 — default params never optimized; GBT shows Q1-Q2 profitability with defaults
**Parent:** E2E CNN Classification (Outcome D — GBT-only baseline 0.449 accuracy, -$0.064 expectancy)

---

## Hypothesis

Systematic hyperparameter tuning of XGBoost on the full-year 1.16M-bar dataset will improve 3-class tb_label classification accuracy by at least 2.0 percentage points over the default-parameter baseline (0.449 → ≥0.469) and push CPCV mean per-trade expectancy from -$0.064 to at least breakeven ($0.00) under base costs ($3.74 RT).

**Rationale:** The default XGBoost hyperparameters (max_depth=6, lr=0.05, n_estimators=500) were inherited from 9B as "reasonable defaults" and never optimized. The GBT-only model is already marginally profitable in Q1 (+$0.003) and Q2 (+$0.029) with these defaults. Standard hyperparameter tuning on well-structured tabular data typically yields 1-5pp accuracy gains. A 2pp gain is a conservative, achievable target that would be economically meaningful. The specific default combination of high min_child_weight (10), moderate regularization (reg_alpha=0.1, reg_lambda=1.0), and fixed n_estimators=500 with lr=0.05 leaves substantial room for improvement — particularly via lower learning rate with more trees, and exploration of the depth-regularization tradeoff.

## Independent Variables

Seven XGBoost hyperparameters, searched via randomized sampling:

| Parameter | Default (from 9B/E2E) | Search Range | Scale |
|-----------|----------------------|--------------|-------|
| `max_depth` | 6 | {3, 4, 5, 6, 7, 8, 10} | discrete |
| `learning_rate` | 0.05 | [0.005, 0.3] | log-uniform |
| `min_child_weight` | 10 | {1, 3, 5, 10, 20, 50} | discrete |
| `subsample` | 0.8 | [0.5, 1.0] | uniform |
| `colsample_bytree` | 0.8 | [0.5, 1.0] | uniform |
| `reg_alpha` (L1) | 0.1 | [1e-3, 10.0] | log-uniform |
| `reg_lambda` (L2) | 1.0 | [0.1, 10.0] | log-uniform |

**Fixed (not searched):**
- `n_estimators` = 2000 (upper bound — actual count determined by early stopping)
- `early_stopping_rounds` = 50 (on validation mlogloss)
- `tree_method` = 'hist'
- `objective` = 'multi:softprob'
- `num_class` = 3
- `eval_metric` = 'mlogloss'
- `nthread` = -1 (use all available cores)
- `seed` = 42

**Search budget:** 48 random configs (Phase 1) + 15 fine configs (Phase 2) = 63 tuning evaluations + 1 default = 64 total.

**Why early stopping instead of searching n_estimators:** Early stopping with n_estimators=2000 is strictly more efficient than grid searching {200, 500, 1000}. Poor configs stop early (saving time), good configs use as many trees as needed. This eliminates one hyperparameter from the search and prevents both underfitting (too few trees) and overfitting (too many).

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Feature set | 20 non-spatial hand-crafted features (see list below) | Same as E2E CNN GBT-only baseline that produced 0.449 accuracy |
| Bar type | time_5s | Locked since R1/R6 |
| Label | tb_label (3-class: -1, 0, +1) | Triple barrier (target=10, stop=5, vol_horizon=500) |
| Data | Full-year Parquet (255 MB, 1,160,150 bars, 251 days) | `.kit/results/full-year-export/` |
| Dev set | Days 1–201 (201 days, ~925K bars) | Same as E2E CNN experiment |
| Holdout | Days 202–251 (50 days) | Sacred holdout, one-shot eval |
| Seed | 42 | Reproducibility |
| Feature normalization | Z-score per fold (train stats only). NaN → 0.0 after normalization | Same as E2E CNN baseline, no leakage |

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

**Note:** The RUN agent must verify these column names against the Parquet schema. If names differ, map to the closest match and document. The E2E CNN GBT-only baseline used these exact 20 features. Any mismatch invalidates baseline comparability.

## Metrics (ALL must be reported)

### Primary

1. **cpcv_mean_accuracy**: Mean 3-class accuracy across 45 CPCV splits (best tuned config)
2. **cpcv_mean_expectancy_base**: Mean per-trade expectancy ($) across 45 CPCV splits under base costs ($3.74 RT)

### Secondary

| Metric | Description |
|--------|-------------|
| search_best_cv_accuracy | Best config's mean accuracy from 5-fold search CV (Phase 1-2) |
| search_accuracy_landscape | All 64 configs' 5-fold CV accuracy (for landscape analysis) |
| cpcv_accuracy_std | Std of accuracy across 45 splits |
| cpcv_expectancy_std | Std of expectancy across 45 splits |
| per_class_recall | Recall for each class (-1, 0, +1), pooled across CPCV test predictions |
| profit_factor | Gross profit / gross loss pooled across CPCV test predictions |
| walkforward_accuracy | Walk-forward accuracy (3 folds, best config) |
| walkforward_expectancy | Walk-forward expectancy |
| per_quarter_expectancy | Per-quarter (Q1-Q4) expectancy in dev set (both configs) |
| feature_importance_top10 | Top 10 features by gain for best config vs default |
| best_n_estimators_mean | Mean actual number of trees used (from early stopping) across CPCV splits |
| cost_sensitivity | Expectancy and PF under optimistic ($2.49), base ($3.74), pessimistic ($6.25) RT costs |
| long_recall_vs_short | Long (+1) recall vs short (-1) recall — E2E CNN found 0.21 vs 0.45 asymmetry |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| Train accuracy > test accuracy | Yes | Underfitting or data leakage |
| No single feature > 50% gain share | Yes | Feature engineering issue — not genuine model improvement |
| Holdout accuracy within 5pp of CPCV mean | Yes | Overfitting to CV structure |
| Early stopping triggers (n_estimators < 2000) | At least 90% of fits | Early stopping broken or lr too low |
| Default config reproduces 0.449 ± 2pp on CPCV | Yes | Data or feature mismatch from E2E CNN experiment |

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| GBT default (CPCV accuracy) | E2E CNN experiment | 0.449 |
| GBT default (CPCV expectancy) | E2E CNN experiment | -$0.064/trade |
| GBT default (holdout accuracy) | E2E CNN experiment | 0.421 |
| GBT default (holdout expectancy) | E2E CNN experiment | -$0.204/trade |
| GBT default (Q1 expectancy) | E2E CNN experiment | +$0.003/trade |
| GBT default (Q2 expectancy) | E2E CNN experiment | +$0.029/trade |
| Random baseline | Theory | 0.333 accuracy, negative expectancy |
| Breakeven win rate | Cost analysis | ~53.3% at $3.74 RT (for directional trades) |

**Baseline reproduction is mandatory.** The default config is evaluated in Phase 1 (5-fold search CV) AND Phase 3 (full CPCV). Both must approximate the E2E CNN baseline (within 2pp accuracy). If they diverge, the data/feature pipeline differs and results are not comparable — ABORT.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: CPCV mean accuracy ≥ 0.469 (default 0.449 + 2.0pp)
- [ ] **SC-2**: CPCV mean per-trade expectancy ≥ $0.00 under base costs ($3.74 RT)
- [ ] **SC-3**: Holdout accuracy ≥ 0.441 (default 0.421 + 2.0pp)
- [ ] **SC-4**: At least 3 search configs outperform the default on 5-fold CV accuracy
- [ ] **SC-5**: Best config's CPCV accuracy std across 45 splits < 0.05 (stable)
- [ ] No regression on sanity checks

## Decision Rules

```
OUTCOME A — SC-1 AND SC-2 pass:
  → CONFIRMED. Tuning closes the breakeven gap. Deploy best config.
  → Next: label-design-sensitivity with tuned XGBoost params.
  → If holdout expectancy also positive: strong go signal for paper trading.

OUTCOME B — SC-1 passes but SC-2 fails:
  → PARTIAL. Accuracy improves but costs still dominate.
  → Record breakeven cost: at what RT cost does the tuned model break even?
  → Next: label-design-sensitivity (lower breakeven via wider target/narrower stop).

OUTCOME C — SC-4 passes but SC-1 fails:
  → MARGINAL. Some configs better, but < 2pp gain.
  → Conclusion: default params are near-optimal. Tuning is not the bottleneck.
  → Next: Combine modest tuning gain with label-design-sensitivity.

OUTCOME D — SC-4 fails (no config outperforms default):
  → REFUTED. Default params are at or near optimal for this feature set.
  → Next: label-design-sensitivity OR feature engineering. Tuning is not the path forward.
```

## Minimum Viable Experiment

1. **Data loading gate:** Load full-year Parquet. Assert ≥1,150,000 bars, ≥18 of 20 expected feature columns present, 3 tb_label classes present, ≥250 unique days. Print label distribution and day count. **ABORT if shape or schema mismatch.**

2. **Baseline reproduction:** Train XGBoost with exact default params (max_depth=6, lr=0.05, n_estimators=500, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0) on a single 80/20 time-series split of the dev set. Assert accuracy ≥ 0.40 (well above random). **ABORT if accuracy < 0.40.**

3. **Early stopping test:** Train one non-default config (max_depth=4, lr=0.01, n_estimators=2000, early_stopping_rounds=50) on same split with last 10% of training days as validation. Assert early stopping triggers before 2000 trees. Assert accuracy > 0.33. **ABORT if early stopping doesn't trigger or accuracy < 0.33.**

Pass all gates → proceed to full protocol.

## Full Protocol

### Search Strategy: Cheap CV for Search, CPCV for Final Evaluation

Using full 45-split CPCV for every hyperparameter config would require 64 × 45 = 2,880 XGBoost fits — far too expensive. Instead:

- **Phases 1-2 (search):** Use 5-fold blocked time-series CV with purge/embargo. ~5× cheaper per config than CPCV. Sufficient for relative ranking of configs.
- **Phase 3 (final evaluation):** Run full CPCV only for the best tuned config and the default. This produces statistically rigorous, directly comparable results.

### Phase 0: Data Loading and Preprocessing

1. Load full-year Parquet from `.kit/results/full-year-export/`.
   - If Parquet files are symlinks (S3 artifact store), run `orchestration-kit/tools/artifact-store hydrate` first.
2. Extract 20 non-spatial features and tb_label. Map column names to expected feature list. Document any name mismatches.
3. Separate dev set (days 1–201) and holdout (days 202–251). **Do NOT touch holdout until Phase 4.**
4. Report: total bars, day count, label distribution (dev vs holdout), feature NaN rates.

### Phase 1: Coarse Random Search (5-fold blocked time-series CV)

1. Define 5-fold blocked time-series CV on dev set (201 days):
   - Block 1: days 1–40, Block 2: days 41–80, Block 3: days 81–120, Block 4: days 121–160, Block 5: days 161–201
   - Each fold: test on one block, train on the remaining 4 blocks
   - Apply purge (500 bars) at each train/test boundary
   - Apply embargo (4,600 bars ≈ 1 day) after each test block boundary
   - Within each training fold, reserve the last 10% of training days as validation for XGBoost early stopping
2. Include the **default config** as config #0 (max_depth=6, lr=0.05, n_estimators=500, no early stopping — reproduces E2E CNN baseline exactly).
3. Generate 48 random hyperparameter configs from the search distributions.
4. Evaluate all 49 configs (48 random + 1 default) on 5-fold CV:
   - For each config: train XGBoost with n_estimators=2000 and early_stopping_rounds=50 on each fold
   - Default config: train with n_estimators=500, no early stopping (for exact baseline reproduction)
   - Record: per-fold accuracy, mean accuracy, std accuracy, actual n_estimators used per fold
5. Rank by mean 5-fold CV accuracy. **Verify default baseline reproduces ≈0.449 ± 2pp.** If not, ABORT.
6. Select top 5 configs for Phase 2.

### Phase 2: Fine Search (5-fold CV around top 5)

1. For each of the top 5 configs from Phase 1, generate 3 neighbors by perturbing one hyperparameter at a time (±20% for continuous params, ±1 grid step for discrete). Total: 15 fine-search configs.
2. Evaluate each on the same 5-fold CV with same purge/embargo protocol.
3. Rank all 64 configs (49 coarse + 15 fine) by mean 5-fold CV accuracy.
4. Select the single best config.

### Phase 3: CPCV Final Evaluation (best tuned vs default)

Run full CPCV (N=10, k=2, 45 splits) on the dev set for TWO configs:

1. **Best tuned config** (from Phase 2, with early stopping)
2. **Default config** (n_estimators=500, no early stopping — for direct comparison to E2E CNN baseline)

CPCV protocol (identical to E2E CNN experiment):
- 10 sequential groups of ~20 days each from 201 dev days
- 45 train/test splits (choose 2 of 10 groups as test)
- Purge: 500 bars at each train/test boundary
- Embargo: 4,600 bars (~1 day) after each test-group boundary
- Within each training fold: reserve last 20% of training days as validation for early stopping (tuned config only)
- Feature normalization: z-score using training fold statistics only

For each split, record:
- Accuracy, per-class predictions, confusion matrix
- PnL under 3 cost scenarios (optimistic $2.49, base $3.74, pessimistic $6.25)
- XGBoost feature importance (gain)
- Actual n_estimators used (tuned config only)

Aggregate across 45 splits:
- Mean and std of accuracy and expectancy
- Pooled per-class recall, confusion matrix, profit factor
- Per-quarter expectancy breakdown (assign each test-split prediction to its calendar quarter)

### Phase 4: Holdout Evaluation + Walk-Forward (one-shot)

**Holdout (SACRED — one shot only):**
1. Train best tuned config on ALL 201 dev days (internal 80/20 val split for early stopping)
2. Evaluate on days 202–251 (50 holdout days)
3. Record: accuracy, expectancy, per-quarter breakdown, confusion matrix
4. Train default config on ALL 201 dev days, evaluate on holdout (for comparison)

**Walk-forward sanity check (best config only):**
- 3-fold expanding window on dev set:
  - Fold 1: train days 1–100, test days 101–140 (40 days)
  - Fold 2: train days 1–140, test days 141–180 (40 days)
  - Fold 3: train days 1–180, test days 181–201 (21 days)
- Apply purge (500 bars) between train end and test start
- Record per-fold accuracy and expectancy
- Compare walk-forward accuracy to CPCV mean — divergence > 5pp indicates CPCV temporal mixing is biasing results

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

## Resource Budget

**Tier:** Standard

- Max wall-clock time: 2 hours
- Max training runs: ~420 (64 configs × 5-fold search + 2 configs × 45-split CPCV + 2 holdout + 3 walk-forward + MVE)
- Max seeds per configuration: 1 (seed=42; variance assessed across temporal folds)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 420
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 1.5
```

### Wall-Time Estimation (local Apple Silicon, ~12 cores)

| Phase | Fits | Per-fit estimate | Subtotal |
|-------|------|-----------------|----------|
| Data loading + preprocessing | — | 30s | 0.5 min |
| MVE (3 single-split fits) | 3 | 15s | 0.75 min |
| Phase 1: 49 configs × 5-fold CV | 245 | 12s | ~49 min |
| Phase 2: 15 configs × 5-fold CV | 75 | 12s | ~15 min |
| Phase 3: 2 configs × 45-split CPCV | 90 | 12s | ~18 min |
| Phase 4: 2 holdout + 3 walk-forward | 5 | 15s | ~1.25 min |
| Aggregation + reporting | — | — | ~5 min |
| **Total** | **~418** | | **~90 min** |

**Per-fit estimate breakdown:** ~720K training rows, 20 features, tree_method='hist'. XGBoost with nthread=-1 on Apple Silicon (8P + 4E cores). With early stopping, average ~300-500 trees per fit ≈ 10-15s. Conservative estimate: 12s/fit.

**Execution target: LOCAL (Apple Silicon).** Per CLAUDE.md compute policy: XGBoost / CPU-only / <1 GB data → run locally. No cloud instance needed. 255 MB Parquet fits comfortably in memory.

## Abort Criteria

- **Phase 1 default baseline diverges:** Default config reproduces < 0.43 accuracy on 5-fold CV (>2pp below expected ~0.449). ABORT — feature or data mismatch means results are not comparable to E2E CNN baseline.
- **All 49 Phase 1 configs have accuracy < 0.40:** ABORT. Something wrong with data/feature pipeline — baseline should be ~0.449.
- **NaN loss:** Any XGBoost training produces NaN — skip that config, log warning. If >10% of configs produce NaN, ABORT (data issue).
- **Per-fit time > 120 seconds:** Investigate. Likely early stopping not triggering or data loading issue. Force n_estimators=500 as fallback for remaining configs.
- **Wall-clock > 2.5 hours:** ABORT remaining phases, report completed work. Save all partial results.

## Confounds to Watch For

1. **Search CV vs CPCV disagreement.** The 5-fold blocked CV used for search (Phases 1-2) differs structurally from the 45-split CPCV used for final evaluation (Phase 3). If a config ranks #1 on 5-fold CV but drops below default on CPCV, the search CV and evaluation CV disagree. Report the rank-correlation between 5-fold CV accuracy and CPCV accuracy for the 2 evaluated configs (best + default). If they disagree, CPCV takes precedence.

2. **Feature importance shift.** If the tuned model relies on different features than the default, the improvement may be fragile. Report feature importance comparison: default top-10 vs tuned top-10. 9E found volatility_50 dominates (19.9 gain share). If the tuned model shifts dominance to a different feature, investigate whether the improvement generalizes.

3. **Class imbalance sensitivity.** The asymmetric 10:5 target/stop ratio produces imbalanced tb_label classes. Some hyperparameter configs may improve one class at the expense of others. Report per-class recall for default vs tuned. The E2E CNN experiment found Long (+1) recall of only 0.21 vs Short (-1) 0.45 — if tuned config improves long recall without sacrificing short, that's especially valuable.

4. **Early stopping vs fixed trees confound.** The tuned configs use early stopping (up to 2000 trees) while the default uses fixed n_estimators=500. A tuned config might outperform simply because it uses more trees, not because other hyperparameters are better. **Diagnostic:** Report the mean n_estimators from early stopping for the best config. If it's close to 500, other hyperparameters drove the gain. If it's >>500, the improvement may partly be from more trees. In either case, report the comparison honestly — the tuned model IS the model we'd deploy, so more trees is legitimate.

5. **Holdout regime.** Days 202-251 are Nov-Dec 2022 (year-end consolidation, lower vol). GBT is negative in Q3-Q4 with default params. Tuning may improve overall accuracy without improving the specific regime the holdout covers. Per-quarter breakdown is essential for interpretation — a result where CPCV improves but holdout doesn't is consistent with regime-dependent performance, not overfitting.

6. **return_5 feature dominance.** If return_5 gains >20% importance share in the tuned model, it may be partially leaking into tb_label prediction (return_5 is backward-looking but correlated with the label's forward return structure). This is the same confound flagged in 9B. If it occurs, report XGBoost performance with return_5 excluded as a diagnostic.

## Deliverables

```
.kit/results/xgb-hyperparam-tuning/
  metrics.json              # Best config params + all primary/secondary metrics
  analysis.md               # Full analysis with tables, comparisons, verdict
  coarse_search.csv         # All 49 Phase 1 configs + 5-fold CV results
  fine_search.csv           # All 15 Phase 2 configs + 5-fold CV results
  cpcv_results.json         # CPCV 45-split results for best tuned + default
  holdout_results.json      # Holdout evaluation (one-shot, both configs)
  walkforward_results.csv   # Walk-forward accuracy series (3 folds)
  feature_importance.json   # Top 10 features by gain: best tuned vs default
  cost_sensitivity.json     # 3 cost scenarios × 2 configs × {expectancy, PF}
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences: did tuning help? which outcome?)
2. Search landscape: distribution of 64 configs' 5-fold CV accuracy (is it flat or peaked?)
3. Best tuned hyperparameters vs default (side-by-side table)
4. CPCV comparison: best tuned vs default (accuracy, expectancy, PF, per-class recall)
5. Per-quarter expectancy breakdown (Q1-Q4, both configs)
6. Feature importance comparison (default vs tuned, top 10 by gain)
7. Walk-forward vs CPCV consistency check (divergence analysis)
8. Holdout evaluation (reported last, clearly separated from development results)
9. Cost sensitivity table (3 scenarios × 2 configs)
10. Explicit pass/fail for each SC-1 through SC-5

## Exit Criteria

- [ ] MVE gates passed (data loading, baseline reproduction, early stopping test)
- [ ] Coarse search complete (49 configs evaluated on 5-fold CV, including default)
- [ ] Fine search complete (15 configs evaluated on 5-fold CV)
- [ ] Best config identified and evaluated on full CPCV (45 splits)
- [ ] Default config evaluated on full CPCV (45 splits) for comparison
- [ ] Walk-forward validation complete (3 folds)
- [ ] Holdout evaluated (one-shot, best tuned + default)
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with all required sections and SC pass/fail
- [ ] Decision rule applied (Outcome A/B/C/D)
