# Experiment: XGBoost Hyperparameter Tuning on Full-Year Data

**Date:** 2026-02-24
**Priority:** P1 (HIGHEST) — default params never optimized; 2pp accuracy gain closes breakeven gap
**Parent:** E2E CNN Classification (Outcome D — GBT baseline established)

---

## Hypothesis

Systematic hyperparameter tuning of XGBoost on 1.16M full-year bars will improve 3-class tb_label classification accuracy by at least 2.0 percentage points over the default baseline (0.449), pushing per-trade expectancy to break-even or positive under base costs ($3.74 RT).

## Independent Variables

| Parameter | Default (9B) | Search Range | Scale |
|-----------|-------------|-------------|-------|
| `max_depth` | 6 | [3, 4, 6, 8] | linear |
| `learning_rate` | 0.05 | [0.01, 0.03, 0.05, 0.1] | log |
| `n_estimators` | 500 | [200, 500, 1000] | linear |
| `subsample` | 0.8 | [0.6, 0.8, 1.0] | linear |
| `colsample_bytree` | 0.8 | [0.6, 0.8, 1.0] | linear |
| `min_child_weight` | 10 | [1, 5, 10, 20] | log |
| `reg_alpha` | 0.1 | [0.0, 0.1, 1.0] | log |
| `reg_lambda` | 1.0 | [0.1, 1.0, 10.0] | log |

**Total configs:** Phase 1 (coarse): 48 configs (Latin hypercube or random sample from grid). Phase 2 (fine): 20 configs around Phase 1 best.

**Fixed:** `tree_method='hist'`, `objective='multi:softprob'`, `num_class=3`, `eval_metric='mlogloss'`, `seed=42`.

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Feature set | 16 hand-crafted GBT features from bar_feature_export | Same as E2E CNN baseline |
| Bar type | time_5s | Locked since R1/R6 |
| Label | tb_label (3-class: -1, 0, +1) | Same triple barrier (target=10, stop=5) |
| CV scheme | CPCV (N=10, k=2, 45 splits) | Same as E2E CNN experiment |
| Dev set | Days 1-201 (201 days) | Same split |
| Holdout | Days 202-251 (50 days) | Same split, one-shot eval |
| Seed | 42 | Reproducibility |
| Data | Full-year Parquet (255MB, 1.16M bars) | `.kit/results/full-year-export/` |

## Metrics (ALL must be reported)

### Primary
- **CPCV mean accuracy** (3-class) — direct test of hypothesis
- **CPCV mean per-trade expectancy** ($) under base costs ($3.74 RT)

### Secondary
- Per-class recall (long/flat/short)
- Profit factor (gross profit / gross loss)
- Walk-forward accuracy (201-day expanding window, 50-day test)
- Per-quarter expectancy breakdown (Q1-Q4)
- Feature importance ranking (top 10 by gain)

### Sanity Checks
- Training accuracy should be > test accuracy (no underfitting)
- Feature importance should not be dominated by a single feature (>50% gain)
- Holdout accuracy within 3pp of CPCV mean (no overfitting to CV splits)

| Check | Expected | Failure means |
|-------|----------|----------------|
| Train > test accuracy | Yes | Model underfitting or data leakage |
| No single feature >50% | Yes | Feature engineering issue |
| Holdout within 3pp of CPCV | Yes | Overfitting to CV structure |

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| GBT default (CPCV accuracy) | E2E CNN experiment | 0.449 |
| GBT default (CPCV expectancy) | E2E CNN experiment | -$0.064 |
| GBT default (holdout accuracy) | E2E CNN experiment | 0.421 |
| GBT default (holdout expectancy) | E2E CNN experiment | -$0.204 |
| GBT default (Q1 expectancy) | E2E CNN experiment | +$0.003 |
| GBT default (Q2 expectancy) | E2E CNN experiment | +$0.029 |
| Breakeven win rate | Cost analysis | 53.3% at $3.74 RT |

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: CPCV mean accuracy >= 0.469 (baseline 0.449 + 2.0pp)
- [ ] **SC-2**: CPCV mean per-trade expectancy >= $0.00 (break-even under base costs)
- [ ] **SC-3**: Holdout accuracy >= 0.441 (baseline 0.421 + 2.0pp)
- [ ] **SC-4**: At least 3 hyperparameter configs outperform default on CPCV accuracy
- [ ] **SC-5**: Best config's CPCV accuracy std across 45 splits < 0.05 (stable)
- [ ] No regression on sanity checks

## Decision Rules

```
OUTCOME A — SC-1 AND SC-2 pass:
  -> CONFIRMED. Tuning closes gap. Deploy best config.
  -> Next: label-design-sensitivity (can it widen the margin further?)

OUTCOME B — SC-1 passes but SC-2 fails:
  -> PARTIAL. Accuracy improves but costs still dominate.
  -> Next: label-design-sensitivity (lower breakeven via label geometry)

OUTCOME C — SC-4 passes but SC-1 fails:
  -> MARGINAL. Some configs better, but <2pp gain.
  -> Next: Combine best tuning with label-design-sensitivity

OUTCOME D — SC-4 fails (no config outperforms default):
  -> REFUTED. Default params are near-optimal for this feature set.
  -> Next: label-design-sensitivity OR feature engineering
```

## Minimum Viable Experiment

1. **Data loading gate:** Load full-year Parquet. Assert 1,160,150 bars, 16 GBT features, 3 tb_label classes.
2. **Baseline reproduction:** Train XGBoost with exact default params. Assert CPCV mean accuracy within 1pp of 0.449.
3. **Single-config test:** Train one non-default config (max_depth=4, lr=0.03). Assert CPCV completes without error.
4. Pass all gates -> proceed to full protocol.

## Full Protocol

### Phase 1: Coarse Search (48 configs)

1. Load full-year Parquet data from `.kit/results/full-year-export/`.
   - If Parquet files are symlinks (S3 artifact store), run `orchestration-kit/tools/artifact-store hydrate` first.
2. Extract 16 GBT features and tb_label from Parquet columns.
3. Split: days 1-201 (dev), days 202-251 (holdout). DO NOT touch holdout until Phase 3.
4. Generate 48 hyperparameter configs via Latin hypercube sampling from the grid.
5. For each config:
   a. Run CPCV (N=10, k=2, 45 splits) on dev set.
   b. Record: mean accuracy, std accuracy, mean expectancy, per-class recall, profit factor.
6. Rank configs by CPCV mean accuracy. Select top 5 for Phase 2.

### Phase 2: Fine Search (20 configs around top 5)

1. For each of top 5 configs from Phase 1, generate 4 neighbors (perturb each param by +/- one grid step).
2. Run CPCV for each of 20 fine-search configs.
3. Select overall best config by CPCV mean accuracy.

### Phase 3: Holdout Evaluation (one-shot)

1. Train best config on full dev set (201 days).
2. Evaluate on holdout (50 days).
3. Record: accuracy, expectancy, per-quarter breakdown, feature importance.
4. Compare to baseline holdout (0.421 accuracy, -$0.204 expectancy).

### Phase 4: Walk-Forward Validation

1. Expanding window: train on days 1-N, test on days N+1 to N+50.
2. Start with N=100, step by 25 days.
3. Record accuracy at each step. Compute mean walk-forward accuracy.

## Resource Budget

**Tier:** Standard (2-3 hours)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 68
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 2.5
```

**Parallelism:** Each hyperparameter config is independent. Use `joblib` or `multiprocessing` to parallelize across Apple Silicon cores (8-10 cores). Estimated: ~3 min per CPCV run (45 splits x ~4s each), 48 configs = ~144 min serial, ~25 min with 6-way parallelism.

## Abort Criteria

- XGBoost training produces NaN loss for any config: skip that config, log warning.
- Phase 1 completes and ALL 48 configs have accuracy < 0.440 (worse than default minus 1pp): ABORT, declare REFUTED.
- Wall-clock exceeds 4 hours: save partial results, evaluate what's available.

## Confounds to Watch For

1. **Overfitting to CPCV structure:** Holdout evaluation (Phase 3) catches this. If holdout drops >5pp below CPCV, the CV is overfit.
2. **Feature importance shift:** If tuned model relies on different features than default, the improvement may be fragile. Report feature importance comparison.
3. **Class imbalance sensitivity:** Some hyperparams may improve one class at the expense of others. Report per-class recall for all configs.

## Deliverables

```
.kit/results/xgb-hyperparam-tuning/
  metrics.json           # Best config params + all primary/secondary metrics
  analysis.md            # Full analysis with tables, comparisons, verdict
  coarse_search.csv      # All 48 coarse configs + CPCV results
  fine_search.csv        # All 20 fine configs + CPCV results
  holdout_results.json   # Holdout evaluation (one-shot)
  walkforward_results.csv # Walk-forward accuracy series
  feature_importance.json # Top 10 features for best config
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. Coarse search results table (sorted by CPCV accuracy)
3. Fine search results table
4. Best config vs default comparison (all metrics)
5. Holdout evaluation
6. Per-quarter expectancy breakdown
7. Feature importance comparison (default vs tuned)
8. Walk-forward accuracy plot data
9. Explicit pass/fail for each SC-1 through SC-5

## Exit Criteria

- [ ] MVE gates passed (data loading, baseline reproduction, single-config test)
- [ ] Coarse search complete (48 configs evaluated)
- [ ] Fine search complete (20 configs evaluated)
- [ ] Best config identified and evaluated on holdout
- [ ] Walk-forward validation complete
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with verdict and SC pass/fail
- [ ] Decision rule applied (Outcome A/B/C/D)
