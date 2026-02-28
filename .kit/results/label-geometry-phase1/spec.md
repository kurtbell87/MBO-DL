# Experiment: Label Geometry Phase 1 — Model Training at Breakeven-Favorable Ratios

**Date:** 2026-02-26
**Priority:** P0 — highest-priority experiment from synthesis-v2 (55-60% prior)
**Parent:** Label Design Sensitivity (Phase 0 oracle sweep, PR #29), Synthesis-v2 (GO verdict, PR #30)
**Depends on:**
1. Phase 0 oracle sweep (123 geometries) — DONE (`.kit/results/label-design-sensitivity/oracle/`)
2. bar_feature_export --target/--stop CLI flags — DONE (PR #28)
3. Bidirectional TB labels (152-col schema) — DONE (PR #26, #27)
4. Full-year Parquet re-export — DONE (312/312, S3)
5. XGBoost tuned params — DONE (XGB tuning experiment)

**All prerequisites DONE. Experiment is FULLY UNBLOCKED.**

---

## Context

Synthesis-v2 identified this experiment as the single highest-priority next step (55-60% prior of positive expectancy). The key insight: **the project was optimizing the wrong variable** — breakeven win rate (not oracle ceiling) is the binding constraint. At 15:3 geometry, breakeven WR drops from 53.3% to 33.3%, providing 12pp of theoretical margin below the model's ~45% accuracy. This has never been tested with actual model training.

Phase 0 oracle sweep is complete (123 geometries mapped). Phase 1 training was prevented by a miscalibrated abort criterion ($5.00 oracle net exp) — this experiment skips that gate entirely and proceeds directly to model training.

**Why this is a new spec (not updating label-design-sensitivity.md):**
- The existing spec is committed with PR #29, has logged results (metrics.json, analysis.md, 123 oracle files), and a RESEARCH_LOG entry. Modifying it post-hoc breaks audit trail.
- The existing run_experiment.py has Phase 0 oracle gate hard-coded (lines 1048-1062) that aborts when `oracle_viable_count == 0`. This is structural, not a patch.
- This experiment has different geometry selection criteria (breakeven WR diversity), different success criteria, and no Phase 0.

---

## Hypothesis

XGBoost trained at high-ratio geometries (15:3, 19:7, 20:3) achieves CPCV directional accuracy > breakeven WR + 2pp at at least one geometry, producing positive per-trade expectancy after base costs ($3.74 RT).

**Mechanism:** At high target:stop ratios, the breakeven WR drops dramatically (33.3% at 15:3 vs 53.3% at 10:5). If the model's directional signal (~45% accuracy) persists — even partially — under the new label distribution, the favorable payoff structure converts that signal into positive expectancy.

**Key uncertainty:** Accuracy on new labels is NOT guaranteed to remain at ~45%. The label distribution shifts substantially (more holds at wider targets), changing the classification problem. This experiment measures that empirically.

---

## Independent Variables

### 4 Geometries (selected by breakeven WR diversity, NOT oracle net exp)

| Geometry | Target:Stop | Ratio | BEV WR | Model @ 45% Acc (theoretical) | Oracle from Phase 0 |
|----------|------------|-------|--------|-------------------------------|---------------------|
| 10:5 (control) | 10 ticks : 5 ticks | 2:1 | 53.3% | -$1.56/trade | oracle_10_5.json |
| 15:3 | 15 ticks : 3 ticks | 5:1 | 33.3% | +$2.63/trade | oracle_15_3.json |
| 19:7 | 19 ticks : 7 ticks | 2.71:1 | 38.4% | +$2.14/trade | oracle_19_7.json |
| 20:3 | 20 ticks : 3 ticks | 6.67:1 | 29.6% | +$5.45/trade | oracle_20_3.json |

**Selection rationale:** Diverse breakeven WR coverage (29.6%–53.3%) spanning the model's accuracy range. 10:5 is the calibration control. 15:3 and 20:3 test narrow-stop high-ratio. 19:7 tests wider-stop moderate-ratio. Together they span the most informative region of geometry space.

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Model | XGBoost with tuned params (LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014) | Best available from XGB tuning experiment |
| Feature set | 20 non-spatial features (same as tuning experiment) | Isolate label design effect |
| Bar type | time_5s | Locked since R1/R6 |
| Label type | Bidirectional triple barrier (compute_bidirectional_tb_label) | Corrected labels; NOT long-perspective-only |
| n_estimators | 2000 (upper bound, early stopping determines actual) | Same as tuned XGB protocol |
| Seed | 42 | Reproducibility |
| Data source | Raw .dbn.zst for C++ label/feature computation; C++-produced Parquet for Python model training | C++ is canonical source (CLAUDE.md rule #8) |

---

## Baselines

| Baseline | Source | Value | Notes |
|----------|--------|-------|-------|
| 10:5 CPCV accuracy (tuned, long-perspective) | xgb-tuning | 0.4504 | Long-perspective labels — NOT directly comparable to bidirectional |
| 10:5 CPCV expectancy (tuned, long-perspective) | xgb-tuning | -$0.001/trade | Long-perspective labels |
| 10:5 WF expectancy (tuned, long-perspective) | xgb-tuning | -$0.140/trade | Deployment-realistic estimate |
| 10:5 oracle net exp (bidirectional, 19d) | label-sensitivity | $2.747/trade | From Phase 0 oracle sweep |
| 10:5 oracle WR (bidirectional, 19d) | label-sensitivity | 64.29% | Reproduces R7 exactly |
| 15:3 oracle net exp (bidirectional, 19d) | label-sensitivity | $1.744/trade | Lower oracle $/trade but more favorable BEV WR |
| 19:7 oracle net exp (bidirectional, 19d) | label-sensitivity | $3.878/trade | Best oracle margin (+11.71pp) |
| 20:3 oracle net exp (bidirectional, 19d) | label-sensitivity | ~$1.920/trade | Most extreme ratio (6.67:1) |

**The 10:5 control trained IN THIS EXPERIMENT (with bidirectional labels) is the proper baseline — not historical long-perspective numbers.** Prior baselines are reference points only. Any cross-experiment comparison is confounded by the label correction.

---

## CV Protocol

### Primary: CPCV
- N=10, k=2, 45 splits
- Dev days 1-201, holdout days 202-251
- Purge: 500 bars at each train/test boundary
- Embargo: 4,600 bars (~1 day) after each test-group boundary
- Early stopping: 50 rounds, last 20% of training days as validation for mlogloss
- Feature normalization: z-score using training fold stats only; NaN -> 0.0 after normalization

### Secondary: Walk-Forward
- 3 expanding-window folds (expanding training, fixed-size test):
  - Fold 1: Train on days 1-100, test on days 101-150 (50 days)
  - Fold 2: Train on days 1-150, test on days 151-201 (51 days)
  - Fold 3: Train on days 1-201, test on days 202-251 (50 days, = holdout)
- Early stopping: same protocol as CPCV (last 20% of training days as val)
- Must report geometry-specific PnL (tick values change per geometry!)
- Walk-forward is the primary metric for deployment decisions (XGB tuning showed CPCV -$0.001 vs WF -$0.140 divergence)

---

## Metrics (ALL must be reported)

### Primary

1. **cpcv_accuracy_per_geometry**: CPCV mean accuracy for each of the 4 geometries
2. **cpcv_expectancy_per_geometry**: CPCV mean per-trade expectancy ($) under base costs for each geometry
3. **walkforward_expectancy_per_geometry**: Walk-forward mean per-trade expectancy ($) for each geometry

### Secondary

| Metric | Description |
|--------|-------------|
| breakeven_margin_per_geometry | (model accuracy - breakeven WR) for each geometry |
| class_distribution_per_geometry | -1/0/+1 fractions for each of the 4 geometries |
| per_class_recall_per_geometry | Long/flat/short recall for each geometry |
| per_direction_oracle | Long and short oracle WR/expectancy for 4 geometries |
| both_triggered_rate | Fraction of bars where both long and short races trigger |
| time_of_day_breakdown | Oracle/model metrics for Bands A (09:30-10:00), B (10:00-15:00), C (15:00-15:30) |
| profit_factor_per_geometry | PF for each geometry |
| feature_importance_shift | Top 10 features by gain: each geometry vs baseline (10:5) |
| cost_sensitivity | Expectancy under optimistic ($2.49), base ($3.74), pessimistic ($6.25) RT costs |
| long_recall_vs_short | Long (+1) vs short (-1) recall asymmetry per geometry |
| holdout_expectancy | Holdout expectancy for best geometry + baseline |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: Baseline (10:5) CPCV accuracy > 0.40 | Yes | Pipeline is broken |
| SC-S2: No single feature > 60% gain share | Yes | Degenerate model |
| SC-S3: Re-export produces 152 columns | Yes | Export pipeline broken |
| SC-S4: Label distribution check (no >95% single class) | Yes | Degenerate labels at that geometry |

---

## Minimum Viable Experiment

Before the full protocol, validate the tool chain end-to-end:

1. **Export CLI gate.** Run `bar_feature_export --bar-type time --bar-param 5 --target 15 --stop 3 --output /tmp/test_export.parquet` on 1 .dbn.zst file. Assert:
   - Parquet output has 152 columns (bidirectional schema)
   - tb_label column exists with values in {-1, 0, +1}
   - Row count in [3000, 6000]

2. **Training pipeline gate.** Load the 1-day test Parquet. Extract 20 features + tb_label. Train XGBoost with tuned params on an 80/20 split. Assert:
   - No NaN in features or predictions
   - Accuracy > 0.33 (above random)

Pass both gates -> proceed to full protocol.

---

## Full Protocol (NO Phase 0 Oracle Sweep — Already Done)

### Step 1: Re-Export at 4 Geometries

Re-export 251 RTH days at all 4 geometries via `bar_feature_export --target T --stop S`:
- 4 geometries x 251 days = 1,004 export runs
- Output: `.kit/results/label-geometry-phase1/geom_T_S/*.parquet`
- Parallelize at 8+ workers (~4 min estimated; prior full-year-export did 11-way in 77s)
- Each file ~1MB (zstd-compressed Parquet), total ~1 GB across all 4 geometries

### Step 2: Train XGBoost CPCV

For each of 4 geometries:
- Load all 251-day Parquet for that geometry
- CPCV (N=10, k=2, 45 splits) with tuned XGB params
- 4 x 45 = 180 fits (~36 min estimated)

### Step 3: Walk-Forward Evaluation

For each of 4 geometries:
- 3 expanding-window folds
- 4 x 3 = 12 fits (~2 min estimated)
- **Critical:** Geometry-specific PnL computation (target/stop tick values differ!)

### Step 4: Holdout

Best geometry by CPCV expectancy + baseline (10:5):
- Train on full dev set (201 days), internal 80/20 val split
- Evaluate on holdout (days 202-251, 50 days)
- 2 fits

### Step 5: Per-Direction Analysis

From bidirectional label columns in re-exported Parquet:
- Long oracle: WR and expectancy for bars where tb_long_triggered=1
- Short oracle: WR and expectancy for bars where tb_short_triggered=1
- Both-triggered rate and correlation with time-of-day

### Step 6: Time-of-Day Bands

- Band A: First 30 min RTH (09:30-10:00 ET) — opening range
- Band B: Mid-session (10:00-15:00 ET) — steady state
- Band C: Last 30 min (15:00-15:30 ET) — close

### Step 7: Cost Sensitivity

3 scenarios applied to all 4 geometries:

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

### PnL Model (geometry-dependent)

```
tick_value = $1.25 per tick (MES)

For geometry (target=T, stop=S):
  Correct directional call (pred sign = label sign, both nonzero):
    PnL = +(T x $1.25) - RT_cost
  Wrong directional call (pred sign != label sign, both nonzero):
    PnL = -(S x $1.25) - RT_cost
  Predict 0 (hold): $0 (no trade)
  True label=0, model predicted +/-1: $0 (conservative simplification)
```

Report label=0 trade fraction per geometry. If >25% of directional predictions hit label=0, flag as unreliable.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Re-export completed for all 4 geometries (4 x 251 = 1,004 files)
- [ ] **SC-2**: CPCV + walk-forward evaluation completed for all 4 geometries
- [ ] **SC-3**: At least one geometry CPCV accuracy > BEV WR + 2pp
- [ ] **SC-4**: At least one geometry CPCV expectancy > $0.00 (base costs)
- [ ] **SC-5**: Best geometry holdout expectancy > -$0.10
- [ ] **SC-6**: Walk-forward expectancy reported for all 4 geometries
- [ ] **SC-7**: Per-direction + time-of-day analysis reported

---

## Abort Criteria

- **MVE gate failure:** STOP. Diagnose before proceeding.
- **Baseline (10:5) CPCV accuracy < 0.40 on bidirectional labels:** STOP. Pipeline broken.
- **Degenerate labels:** Any geometry produces >95% one class -> skip that geometry, log warning.
- **Per-fit time > 60s:** Investigate. Expected ~12s.
- **NaN loss:** Any XGB fit produces NaN -> skip, log warning. If >10% NaN, ABORT.
- **Wall-clock > 4 hours:** Abort remaining, report partial results. (3.75x estimated 64 min per research-engineering-practices.md guidance.)
- **NO oracle net exp gate.** Phase 0 is already done. We proceed directly to training.

---

## Decision Rules

```
OUTCOME A — SC-3 AND SC-4 pass:
  -> CONFIRMED. A viable geometry exists.
  -> Record: best geometry, accuracy margin over breakeven, expectancy, per-class recall.
  -> Next: Regime-conditional experiment + multi-year data validation.

OUTCOME B — SC-3/SC-4 fail, accuracy stable across geometries:
  -> PARTIAL. Payoff structure insufficient at current accuracy.
  -> The model has a fixed directional signal that doesn't change with geometry.
  -> Next: 2-class formulation (short/no-short), class-weighted loss, or feature engineering.

OUTCOME C — Accuracy drops >10pp at all non-baseline geometries:
  -> REFUTED. Label distribution change breaks the model.
  -> The wider-target classification problem is fundamentally harder.
  -> Next: Per-direction asymmetric strategy (different geometry for longs vs shorts).
```

---

## Resource Budget

**Tier:** Standard

- Max wall-clock time: 4 hours
- Max training runs: ~194 (180 CPCV + 12 walk-forward + 2 holdout)
- Max export runs: 1,004 (4 geometries x 251 days)
- Max seeds: 1 (seed=42)
- COMPUTE_TARGET: local
- GPU hours: 0

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 194
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 1.5
```

### Wall-Time Estimation

| Phase | Work | Per-unit | Units | Subtotal |
|-------|------|----------|-------|----------|
| MVE (export + train gates) | 2 gates | — | 2 | ~2 min |
| Re-export: 4 geometries x 251 days | 1,004 C++ runs | ~2s | 1,004 | ~4 min (parallel at 8+ workers) |
| CPCV training: 4 x 45 splits | 180 XGB fits | ~12s | 180 | ~36 min |
| Walk-forward: 4 x 3 folds | 12 XGB fits | ~12s | 12 | ~2 min |
| Holdout: 2 configs | 2 XGB fits | ~15s | 2 | ~1 min |
| Per-direction + time-of-day analysis | 4 geometries | ~1 min | 4 | ~5 min |
| Cost sensitivity + reporting | — | — | — | ~10 min |
| **Total** | | | | **~60 min** |

---

## Key References

- **Phase 0 oracle data:** `.kit/results/label-design-sensitivity/oracle/` (123 JSON files)
- **Prior run_experiment.py:** `.kit/results/label-design-sensitivity/run_experiment.py` (RUN sub-agent should adapt this — remove Phase 0, hard-code 4 geometries, add walk-forward with geometry-specific PnL)
- **Walk-forward pattern:** `.kit/results/xgb-hyperparam-tuning/run_experiment.py` (walk-forward implementation)
- **Synthesis-v2 analysis:** `.kit/results/synthesis-v2/analysis.md` (GO verdict, geometry selection rationale)
- **Tuned XGB params:** LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014

---

## Confounds to Watch For

1. **Accuracy does NOT transfer across geometries.** Each geometry is a different classification problem. Compare accuracy to per-geometry breakeven WR, not a fixed number.
2. **Narrow stops (2-3 ticks) get swept by noise.** 2-3 ticks = $0.50-$0.75, within MES spread. Expect high trade counts but potentially degraded signal quality.
3. **Hold-majority predictions.** Model may shift to predicting mostly holds at wider targets (class imbalance). Monitor per-class recall for degenerate predictions.
4. **Walk-forward divergence.** CPCV may again show near-breakeven while WF shows negative (as in XGB tuning: -$0.001 vs -$0.140). Both must be reported; WF is primary for deployment.
5. **Time-of-day effects may dominate.** If opening-range is 3x better, the optimal strategy may be "trade only the open" regardless of geometry.
6. **PnL model simplification for label=0 predictions.** The PnL model assigns $0 when the model predicts directional (+/-1) but the true label is 0 (hold). In reality, entering a position that doesn't reach target or stop within the bar window means an uncertain exit P&L. This simplification is consistent with all prior experiments (enabling comparison) and is conservative in the sense that it doesn't charge costs, but it may understate losses if the model frequently predicts directional on hold-labeled bars. The `label=0 trade fraction` metric monitors this.

---

## Deliverables

```
.kit/results/label-geometry-phase1/
  metrics.json                    # All SC statuses + per-geometry metrics
  analysis.md                     # Comparative analysis, verdict, SC pass/fail
  run_experiment.py               # Experiment script (created by RUN phase)
  spec.md                         # Local copy of spec
  geom_10_5/                      # Re-exported Parquet (251 files)
  geom_15_3/                      # Re-exported Parquet (251 files)
  geom_19_7/                      # Re-exported Parquet (251 files)
  geom_20_3/                      # Re-exported Parquet (251 files)
  cpcv_results.csv                # CPCV metrics per geometry
  walkforward_results.csv         # Walk-forward metrics per geometry
  holdout_results.json            # Best geometry + baseline holdout
  per_direction_oracle.csv        # Long/short oracle metrics
  time_of_day.csv                 # Band A/B/C metrics
  cost_sensitivity.csv            # 3 cost scenarios x 4 geometries
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **CPCV results table** (4 geometries x accuracy, expectancy, breakeven margin)
3. **Walk-forward results table** (4 geometries x expectancy) — CRITICAL comparison to CPCV
4. **Breakeven margin analysis** (accuracy - breakeven_WR per geometry)
5. **Class distribution** (-1/0/+1 fractions per geometry)
6. **Per-class recall** — did long/short asymmetry persist?
7. **Per-direction oracle analysis** — optimal long vs short geometry
8. **Time-of-day breakdown** — Bands A, B, C
9. **Cost sensitivity table** (3 scenarios x 4 geometries)
10. **Key diagnostic:** Does accuracy track geometry, or is it geometry-invariant?
11. **Holdout evaluation** (reported last, clearly separated)
12. **Explicit SC-1 through SC-7 pass/fail**
13. **Outcome verdict** (A/B/C per decision rules)

---

## Exit Criteria

- [ ] MVE gates passed (export CLI, training pipeline)
- [ ] Re-export completed for all 4 geometries (1,004 files)
- [ ] CPCV completed for all 4 geometries (180 fits)
- [ ] Walk-forward completed for all 4 geometries (12 fits)
- [ ] Holdout evaluated for best geometry + baseline
- [ ] Per-direction oracle metrics reported for all 4 geometries
- [ ] Time-of-day analysis computed for all 4 geometries
- [ ] Cost sensitivity (3 scenarios) reported for all 4 geometries
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with comparison tables and verdict
- [ ] Decision rule applied (Outcome A/B/C)
- [ ] SC-1 through SC-7 explicitly evaluated
