# Experiment: Stage 1 Threshold Sweep — Optimizing Hold-Bar Exposure

**Date:** 2026-02-26
**Priority:** P0 — highest priority from PnL realized return analysis
**Parent:** PnL Realized Return (REFUTED SC-2, $0.90/trade, PR #35)
**Depends on:**
1. PnL realized return results and run_experiment.py — DONE (PR #35)
2. 2-class directional results — DONE (PR #34)
3. Label geometry 1h Parquet data (3600s horizon, 152-col) — DONE (PR #33)

**All prerequisites DONE. Same data, same models — only the Stage 1 decision threshold changes.**

---

## Context

The PnL realized return experiment (PR #35) revealed a clear economic structure:
- **Directional bars contribute +$2.10/trade** ($3.77 x 55.6% fraction) — real, stable across folds
- **Hold bars drag -$1.19/trade** (-$2.68 x 44.4% fraction) — destroys 57% of the edge
- **Net realized expectancy: $0.90/trade** — positive but fragile (Fold 2 outlier, CV=129%)

The PnL decomposition implies a direct lever: **reducing hold-bar fraction recovers edge**. Each 10pp reduction in hold-bar fraction recovers ~$0.27/trade. The Stage 1 model already outputs P(directional) as a continuous probability — raising the threshold from 0.5 trades fewer bars but concentrates on higher-confidence directional predictions.

**No re-training needed.** Stage 1 and Stage 2 models are already trained in each walk-forward fold. This experiment just re-scores existing predictions at different thresholds.

---

## Hypothesis

There exists a Stage 1 probability threshold T* > 0.5 such that the two-stage pipeline at 19:7 achieves walk-forward realized expectancy > $1.50/trade with trade rate > 15%, by reducing hold-bar fraction below 25%.

**Direction:** Higher threshold -> higher expectancy (up to a point where trade rate becomes too low for reliable metrics).
**Magnitude:** At threshold ~0.70 with estimated 15% hold fraction, expectancy ~$2.81/trade (from PnL decomposition extrapolation: $0.90 + (44.4% - 15%) x $0.27/pp = $0.90 + $7.93 ... that's the linear extrapolation; reality will deviate — see Confound #4).

**Mechanism:** Higher threshold = stricter Stage 1 filter = fewer bars predicted "directional" = lower hold-bar fraction among traded bars. Since directional-bar PnL ($3.77) is 6.4x the magnitude of hold-bar drag (-$2.68 net), reducing hold fraction mechanically improves expectancy — unless the threshold also degrades directional-bar quality (selecting harder-to-predict directional bars).

---

## Independent Variables

### Stage 1 threshold (primary IV — 9 levels)

| Threshold | Expected Trade Rate | Expected Hold Fraction | Expected Exp |
|-----------|-------------------|----------------------|-------------|
| 0.50 (baseline) | 85.2% | 44.4% | $0.90 |
| 0.55 | ~75% | ~38% | ~$1.25 |
| 0.60 | ~65% | ~32% | ~$1.60 |
| 0.65 | ~55% | ~26% | ~$1.95 |
| 0.70 | ~45% | ~20% | ~$2.30 |
| 0.75 | ~35% | ~15% | ~$2.65 |
| 0.80 | ~25% | ~10% | ~$3.00 |
| 0.85 | ~15% | ~6% | ~$3.30 |
| 0.90 | ~8% | ~3% | ~$3.50 |

Estimates are linear extrapolations from the PnL decomposition. Reality will deviate due to: (a) non-linear relationship between threshold and hold fraction, (b) directional-bar quality may degrade at high thresholds, (c) per-fold variance.

### Geometry (2 levels)

| Geometry | Role |
|----------|------|
| **19:7** | Primary — hold bars are the dominant problem |
| **10:5** | Control — lower hold fraction at baseline (32.6%), less room for improvement |

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Pipeline | Two-stage (Stage 1: reachability, Stage 2: direction) | Identical to pnl-realized-return |
| XGB params | Tuned params (LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014) | No re-tuning |
| Feature set | 20 non-spatial features | Identical |
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` and `geom_10_5/` | Same Parquet as pnl-realized-return |
| Walk-forward folds | 3 expanding-window (same splits as pnl-realized-return) | Direct comparison to baseline |
| Stage 2 threshold | 0.5 (always) | Only Stage 1 threshold varies |
| Seed | 42 | Reproducibility |
| PnL model | Realized return (from PR #35) | Same corrected PnL: directional bars get barrier payoffs, hold bars get forward-return PnL |

---

## Baselines

| Baseline | Source | Value | Notes |
|----------|--------|-------|-------|
| Realized WF expectancy at T=0.50 (19:7) | pnl-realized-return | $0.90/trade | Must reproduce exactly (SC-S1) |
| Realized WF expectancy at T=0.50 (10:5) | pnl-realized-return | -$1.65/trade | Control geometry |
| Trade rate at T=0.50 (19:7) | pnl-realized-return | 85.18% | Reference for monotonicity |
| Hold fraction at T=0.50 (19:7) | pnl-realized-return | 44.4% | Starting point; goal is <25% |
| Dir-bar PnL at T=0.50 (19:7) | pnl-realized-return | $3.77/trade | Check for selection-bias degradation |
| Per-fold CV at T=0.50 (19:7) | pnl-realized-return | 129% | Goal: reduce via threshold optimization |
| Break-even RT at T=0.50 (19:7) | pnl-realized-return | $4.64 | Floor — should improve at T* |
| Hold-bar mean net PnL at T=0.50 (19:7) | pnl-realized-return | -$2.68/trade | Hold-bar cost per trade |

**The baseline is T=0.50 at 19:7 from the pnl-realized-return experiment (PR #35). This experiment must reproduce the baseline exactly before evaluating higher thresholds.**

---

## Metrics (ALL must be reported)

### Primary

1. **optimal_threshold_19_7**: The threshold T* that maximizes realized expectancy while maintaining trade rate > 15%. THE metric.
2. **optimal_realized_expectancy_19_7**: Realized WF expectancy at T* under base costs ($3.74 RT).
3. **optimal_trade_rate_19_7**: Trade rate at T*.

### Secondary

| Metric | Description |
|--------|-------------|
| threshold_curve_19_7 | Full curve: threshold -> (trade rate, hold fraction, realized exp, dir-bar exp, hold-bar exp) for all 9 thresholds |
| threshold_curve_10_5 | Same curve for 10:5 control |
| pareto_frontier | Thresholds on the Pareto frontier of expectancy x trade_rate |
| per_fold_at_optimal | Per-fold breakdown at T* (3 folds) — check if fold instability is reduced |
| hold_fraction_at_optimal | Hold-bar fraction at T* (target: < 25%) |
| cost_sensitivity_at_optimal | 3 cost scenarios at T* (optimistic $2.49, base $3.74, pessimistic $6.25) |
| dir_bar_quality_vs_threshold | Does directional-bar PnL ($3.77) degrade at higher thresholds? (selection bias check) |
| daily_pnl_at_optimal | Mean daily PnL at T* |
| break_even_rt_at_optimal | Break-even RT cost at T* |
| p_directional_distribution | Histogram/percentiles of Stage 1 P(directional) — determines whether thresholds differentiate meaningfully |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: Threshold 0.50 reproduces baseline | Realized exp $0.90 +/- $0.01, trade rate 85.18% | Code bug |
| SC-S2: Trade rate monotonically decreases with threshold | Yes | Threshold logic is inverted or broken |
| SC-S3: Hold fraction monotonically decreases with threshold | Yes | Stage 1 calibration issue |
| SC-S4: At least 1000 trades per fold at optimal threshold | Yes | Threshold too aggressive |

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: There exists a threshold where realized WF expectancy > $1.50/trade at 19:7 (base costs) with trade rate > 15%
- [ ] **SC-2**: Per-fold CV of realized expectancy at optimal threshold < 80% (improved stability vs baseline 129%)
- [ ] **SC-3**: Hold fraction at optimal threshold < 25% (vs baseline 44.4%)
- [ ] **SC-4**: Directional-bar PnL at optimal threshold > $3.00/trade (no severe degradation from $3.77 baseline)

---

## Minimum Viable Experiment

Run 3 thresholds (0.50, 0.70, 0.90) on Fold 1 only at 19:7. Verify:
1. Threshold 0.50 reproduces baseline (sanity)
2. Trade rate decreases monotonically
3. Hold fraction decreases monotonically
4. Expectancy increases from 0.50 to 0.70

**MVE pass:** All 4 checks pass. **ABORT if** threshold 0.50 doesn't reproduce baseline or monotonicity fails.

---

## Full Protocol

### Step 0: Adapt the PnL Realized Return Script

Start from `.kit/results/pnl-realized-return/run_experiment.py`. Key adaptation:

1. After training Stage 1 and Stage 2 models per fold (IDENTICAL to existing code), extract the raw P(directional) predictions from Stage 1.
2. For each of the 9 thresholds (0.50, 0.55, 0.60, ..., 0.90):
   a. Apply threshold to Stage 1 P(directional) — bars with P > threshold are "directional"
   b. For predicted-directional bars: use Stage 2 direction prediction
   c. For predicted-hold bars: predict 0 (no trade)
   d. Compute realized-return PnL (same model as PR #35)
   e. Record: trade rate, hold fraction, realized expectancy, dir-bar exp, hold-bar exp, per-fold details

3. Models are trained ONCE per fold. Only the thresholding changes. This is purely a post-hoc re-scoring exercise.

### Step 1: Walk-Forward at 19:7 (Primary)

3 expanding-window folds x 9 thresholds = 27 evaluation points. Per fold:
- Train Stage 1 + Stage 2 (once, at seed 42)
- Extract raw P(directional) for all test bars
- Evaluate at each of 9 thresholds
- Compute realized-return PnL at each threshold

### Step 2: Walk-Forward at 10:5 (Control)

Same protocol. 3 folds x 9 thresholds = 27 evaluation points.

### Step 3: Analysis

1. **Threshold curve plot data:** For each threshold, report mean across 3 folds of: trade rate, hold fraction, realized expectancy, per-fold values.
2. **Identify T*:** Threshold that maximizes realized expectancy subject to trade rate > 15%.
3. **Pareto frontier:** Thresholds where no other threshold has both higher expectancy AND higher trade rate.
4. **Selection bias check:** Does directional-bar PnL degrade at higher thresholds? If the Stage 1 model's high-confidence predictions correspond to "easy" directional bars (ones that would have been correctly predicted anyway), the marginal directional bars dropped at higher thresholds are the "hard" ones — and directional-bar PnL should be stable or slightly improve.
5. **Fold stability check:** Does per-fold CV decrease at higher thresholds? (Expected: yes, because removing hold-bar noise reduces variance.)
6. **P(directional) distribution:** Report histogram and percentiles (p10, p25, p50, p75, p90) of raw Stage 1 probabilities. If >80% of probabilities are within [0.45, 0.55], thresholds above 0.55 will have minimal effect (poorly calibrated model).

### Step 4: Cost Sensitivity at Optimal Threshold

3 scenarios x 2 geometries at T*.

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

### Step 5: Comparison to Baseline

| Metric | Baseline (T=0.50) | Optimal (T=T*) | Delta |
|--------|-------------------|----------------|-------|
| Realized exp | $0.90 | ? | ? |
| Trade rate | 85.2% | ? | ? |
| Hold fraction | 44.4% | ? | ? |
| Per-fold CV | 129% | ? | ? |
| Break-even RT | $4.64 | ? | ? |
| Daily PnL | ? | ? | ? |
| Dir-bar PnL | $3.77 | ? | ? |

---

## Resource Budget

**Tier:** Quick

- Max wall-clock time: 15 min
- Max training runs: 14 (models trained once per fold; thresholds are post-hoc)
- Max seeds: 1 (seed=42)
- COMPUTE_TARGET: local
- GPU hours: 0

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 14
parallelizable: false
memory_gb: 2
gpu_type: none
estimated_wall_hours: 0.05
```

Training is identical to pnl-realized-return (~32s for 14 XGB fits). Threshold sweep is pure numpy — adds <5s. Data loading adds ~30-60s. Total: ~2 min.

---

## Abort Criteria

- **Baseline reproduction fails:** STOP. Threshold 0.50 must reproduce $0.90 +/- $0.01 and 85.18% trade rate.
- **Trade rate at 0.90 is still > 50%:** STOP. Stage 1 model is poorly calibrated — probabilities don't spread.
- **Monotonicity violation:** STOP. If trade rate or hold fraction increases at higher thresholds, threshold logic is wrong.
- **Wall-clock > 15 min:** ABORT. (5x expected ~3 min.)

---

## Decision Rules

```
OUTCOME A — SC-1 AND SC-2 pass (exp > $1.50 at >15% trade rate, CV < 80%):
  -> CONFIRMED. Threshold optimization produces a robust, economically viable strategy.
  -> Record optimal threshold, expectancy, trade rate, daily PnL.
  -> Next: CPCV validation at optimal threshold (45 splits for proper CI and PBO).

OUTCOME B — SC-1 passes but SC-2 fails (exp > $1.50 but CV > 80%):
  -> PARTIAL. Better economics but still fold-unstable.
  -> Diagnose: Is fold instability driven by hold bars or directional bars?
  -> Next: CPCV at optimal threshold to determine if fold instability is sampling noise.

OUTCOME C — SC-1 fails (no threshold achieves exp > $1.50 at >15% trade rate):
  -> REFUTED. Threshold optimization alone is insufficient.
  -> Diagnose: Does directional-bar PnL degrade at high thresholds? (selection bias)
  -> If dir-bar PnL stable but expectancy still < $1.50: hold-bar drag is structurally
     unresolvable at 19:7 without fixing the volume horizon (hold returns unbounded +-63 ticks).
  -> Next: Volume horizon fix (re-export with unlimited volume horizon to bound hold-bar returns),
           or intermediate geometry (14:6) where direction signal exists.

OUTCOME D — Monotonicity fails or baseline doesn't reproduce:
  -> INVALID. Code bug. Fix and retry.
```

---

## Confounds to Watch For

1. **Stage 1 probability calibration.** XGBoost `binary:logistic` probabilities may not be well-calibrated. If all P(directional) are clustered near 0.5 (e.g., IQR within [0.45, 0.55]), higher thresholds will abruptly cut off trade volume rather than gradually filtering. The P(directional) distribution diagnostic (Step 3.6) is specifically designed to detect this. If >80% of probabilities are within 0.05 of 0.5, thresholds above 0.55 will annihilate trade rate and the linear extrapolation in the IV table will be wildly wrong.

2. **Directional-bar selection bias.** At higher thresholds, the model selects "high-confidence directional" bars. If these are disproportionately volatile bars where direction is random (high reachability but no directional signal), directional-bar PnL could degrade. The dir-bar PnL vs threshold curve will detect this. If dir-bar PnL drops below $3.00 at any threshold with >15% trade rate, SC-4 fails and the selection bias confound is active.

3. **Reduced sample size at high thresholds.** At threshold 0.90, ~8% trade rate -> ~37K trades at 19:7 across 3 folds. Still ample for point estimates but per-fold counts may drop to ~10K, reducing per-fold reliability. SC-S4 (>1000 trades per fold at T*) guards against this.

4. **Non-linear hold fraction curve.** The linear extrapolation in the IV table assumes uniform P(directional) distribution. The actual curve may be concave (diminishing returns at high thresholds: all the "easy" holds removed early) or convex (accelerating improvement: high-probability predictions are disproportionately directional). Report the actual curve shape and compare to the linear prediction.

5. **Volume horizon confound persists.** All PnL computations inherit the unbounded hold-bar return issue from PR #35 (p10/p90 = -63/+63 ticks due to 50K volume horizon truncating barrier race). This confound is *reduced* (not eliminated) at higher thresholds because hold fraction is lower — fewer trades carry unbounded risk. But the remaining hold trades still have fat-tailed returns. If T* reduces hold fraction to <10%, the volume horizon confound becomes negligible (<10% of trades affected). If T* is at 30%+ hold, the confound remains material.

---

## Deliverables

```
.kit/results/threshold-sweep/
  metrics.json                     # All SC statuses + optimal threshold metrics
  analysis.md                      # Threshold curve, optimal selection, verdict
  run_experiment.py                # Adapted from pnl-realized-return with threshold sweep
  spec.md                          # Local copy of spec
  threshold_curve.csv              # 9 thresholds x 2 geometries: all metrics
  per_fold_at_optimal.csv          # Per-fold details at T*
  cost_sensitivity.csv             # 3 cost scenarios at T*
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **P(directional) distribution** — histogram/percentiles of Stage 1 probabilities (report FIRST — determines whether the experiment is informative at all)
3. **Threshold curve table** — 9 thresholds x (trade rate, hold fraction, realized exp, dir-bar exp, per-fold CV) at 19:7
4. **10:5 control curve** — validates pipeline
5. **Optimal threshold selection** — T*, rationale, comparison to baseline
6. **Per-fold consistency at T*** — reduced fold instability?
7. **Directional-bar quality check** — does dir-bar PnL degrade at higher thresholds?
8. **Pareto frontier** — expectancy x trade_rate
9. **Cost sensitivity at T***
10. **Comparison table: baseline (T=0.50) vs optimal (T=T*)**
11. **Explicit SC-1 through SC-4 pass/fail**
12. **Outcome verdict (A/B/C/D)**

---

## Exit Criteria

- [ ] P(directional) distribution analyzed (histogram, percentiles)
- [ ] Threshold curve computed for all 9 thresholds x 2 geometries x 3 folds
- [ ] Baseline (T=0.50) reproduces PR #35 results (SC-S1)
- [ ] Monotonicity checks passed (SC-S2, SC-S3)
- [ ] Optimal threshold T* identified with rationale
- [ ] Per-fold details at T* reported
- [ ] Directional-bar PnL vs threshold curve reported (selection bias check)
- [ ] Cost sensitivity at T* computed (3 scenarios x 2 geometries)
- [ ] Comparison table: baseline (T=0.50) vs optimal (T=T*)
- [ ] All metrics in metrics.json
- [ ] analysis.md with threshold curve, P(directional) distribution, and verdict
- [ ] SC-1 through SC-4 explicitly evaluated

---

## Key References

- **PnL realized return script:** `.kit/results/pnl-realized-return/run_experiment.py` — starting point, add threshold sweep loop
- **PnL realized return results:** `.kit/results/pnl-realized-return/metrics.json` — baseline metrics at T=0.50
- **PnL realized return analysis:** `.kit/results/pnl-realized-return/analysis.md` — PnL decomposition, fold details
- **2-class results:** `.kit/results/2class-directional/metrics.json` — original 2-class metrics
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/` and `geom_10_5/`
- **PnL constants:** tick_value=$1.25, tick_size=0.25. RT costs: optimistic $2.49, base $3.74, pessimistic $6.25
