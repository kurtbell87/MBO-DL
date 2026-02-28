# Experiment: Class-Weighted Stage 1 — Spreading the Probability Surface

**Date:** 2026-02-27
**Priority:** P0 — direct fix for threshold sweep's root cause (compressed probability distribution)
**Parent:** Threshold Sweep (REFUTED SC-1, PR #36) + PnL Realized Return (PR #35)
**Depends on:**
1. PnL realized return results and pipeline — DONE (PR #35)
2. 2-class directional results — DONE (PR #34)
3. Threshold sweep results — DONE (PR #36)
4. Label geometry 1h Parquet data (3600s horizon, 152-col) — DONE (PR #33)

**All prerequisites DONE. Same data, new `scale_pos_weight` parameter for Stage 1 XGBoost training.**

---

## Context

The threshold sweep (PR #36) revealed that XGBoost `binary:logistic` produces a near-degenerate probability distribution — 80.6% of P(directional) predictions cluster in [0.50, 0.60]. This makes post-hoc threshold optimization impossible (cliff from 64.5% to 4.5% trade rate between T=0.55 and T=0.60).

However, the directional-bar signal IS real: dir-bar PnL improves at moderate thresholds ($3.77 -> $6.07), confirming the model's confidence is informative. The problem is the compressed probability surface, not the absence of signal.

**Class-weighted training addresses this at the source.** By penalizing false positives (predicting directional when truly hold) more heavily via `scale_pos_weight`, we reshape the model's decision boundary. This should:
1. Spread the probability distribution (wider IQR, more usable tail)
2. Increase Stage 1 precision at the cost of recall
3. Reduce hold-bar fraction among traded bars without needing post-hoc thresholds
4. Potentially enable the threshold sweep that was impossible with the default model

The realized expectancy at T=0.50 is $0.90/trade. Hold bars drag -$1.19/trade (57% of the directional-bar edge). If class weighting shifts the model to higher precision (fewer false positives = fewer hold-bar trades), the economics should improve.

---

## Hypothesis

There exists a (`scale_pos_weight`, threshold) pair such that the two-stage pipeline at 19:7 achieves walk-forward realized expectancy > $1.50/trade with trade rate > 15%, by spreading the Stage 1 probability distribution and enabling fine-grained threshold control.

**Direction:** Lower `scale_pos_weight` (< 1.0) increases precision, decreases recall, spreads probability distribution.
**Mechanism:** In XGBoost `binary:logistic` with label `is_directional`, positive class = directional. `scale_pos_weight < 1` makes the model MORE conservative about predicting directional — higher precision, lower recall — fewer trades, fewer hold-bar trades. Critically, the decision boundary shift should spread the probability distribution away from the [0.50, 0.60] compression observed at `scale_pos_weight=1.0`.

---

## Independent Variables

### Stage 1 `scale_pos_weight` (primary IV — 5 levels)

| Weight | Interpretation | Expected Effect |
|--------|---------------|-----------------|
| 1.0 (baseline) | Equal penalty | Reproduces PR #35/#36 results: 80.6% in [0.50, 0.60] |
| 0.5 | FP costs 2x FN | Moderate precision boost, some probability spread |
| 0.33 | FP costs 3x FN | Strong precision boost, significant probability spread |
| 0.25 | FP costs 4x FN | Aggressive precision boost, wide probability spread |
| 0.20 | FP costs 5x FN | Most aggressive — may over-suppress trading |

**Note:** `scale_pos_weight < 1.0` penalizes false positives (predicting directional when truly hold) more than false negatives (predicting hold when truly directional). This shifts the model toward conservative directional predictions — exactly what we want to reduce hold-bar exposure.

### Threshold sweep at each weight (9 levels)

For each weight, sweep T = 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90. This tests whether class weighting spreads the probability distribution enough to unlock threshold control (which was impossible at weight=1.0).

### Geometry

| Geometry | Role |
|----------|------|
| **19:7** | Primary — the only viable geometry |

**10:5 is excluded.** 10:5 is definitively non-viable (-$1.65/trade at all thresholds, consistent across 3 folds with std $0.064). No further validation needed.

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Pipeline | Two-stage (Stage 1: reachability, Stage 2: direction) | Identical to PR #34/#35/#36 |
| XGB params (both stages) | LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014 | Same tuned params; `scale_pos_weight` is the ONLY change to Stage 1 |
| Stage 2 XGB | Unchanged (weight=1.0, same params) | Only Stage 1 weighting changes |
| Feature set | 20 non-spatial features | Identical to all prior experiments |
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` (existing 3600s Parquet) | Same data as PR #34/#35/#36 |
| Walk-forward folds | 3 expanding-window (days 1-100/101-150, 1-150/151-201, 1-201/202-251) | Same splits for direct comparison |
| Early stopping | 50 rounds, logloss, val = last 20% of training days | Standard protocol (both stages) |
| n_estimators | 2000 (upper bound) | Same as prior |
| Seed | 42 | Reproducibility |
| PnL model | Realized return (from PR #35) | Same corrected hold-bar treatment |
| Stage 2 threshold | 0.5 (always) | Only Stage 1 weight + threshold vary |

---

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| Realized WF expectancy at weight=1.0, T=0.50 (19:7) | PR #35 | $0.90/trade |
| Trade rate at weight=1.0, T=0.50 (19:7) | PR #35 | 85.18% |
| Hold fraction at weight=1.0, T=0.50 (19:7) | PR #35 | 44.4% |
| P(directional) IQR at weight=1.0 | PR #36 | 4.0pp [0.536, 0.576] |
| P(directional) in [0.50, 0.60] at weight=1.0 | PR #36 | 80.6% |
| Dir-bar PnL at weight=1.0, T=0.50 (19:7) | PR #35 | $3.77/trade |
| Per-fold CV at weight=1.0, T=0.50 (19:7) | PR #35 | 129% |
| Break-even RT at weight=1.0, T=0.50 (19:7) | PR #35 | $4.64 |

**The weight=1.0 baseline must reproduce PR #35/#36 results exactly before evaluating other weights.**

---

## Metrics (ALL must be reported)

### Primary

1. **best_weight_threshold_pair**: The (weight, threshold) pair that maximizes realized expectancy with trade rate > 15%.
2. **best_realized_expectancy**: Realized WF expectancy at the optimal (weight, threshold) pair.
3. **best_trade_rate**: Trade rate at the optimal pair.

### Secondary

| Metric | Description |
|--------|-------------|
| p_directional_distribution_per_weight | P(directional) distribution at each weight: mean, std, IQR, p10, p25, p50, p75, p90, fraction in [0.50, 0.60] |
| threshold_curve_per_weight | For each weight: threshold -> (trade rate, hold fraction, realized exp, dir-bar exp, hold-bar exp, per-fold CV) for all 9 thresholds |
| stage1_accuracy_per_weight | Binary accuracy (directional vs hold) at each weight |
| stage1_precision_recall_per_weight | Precision and recall for directional class at each weight |
| hold_fraction_at_best | Hold fraction at optimal (weight, threshold) |
| cost_sensitivity_at_best | 3 cost scenarios at optimal pair |
| dir_bar_pnl_stability | Dir-bar PnL at each (weight, threshold) — check for degradation |
| per_fold_at_best | Per-fold breakdown at optimal pair (3 folds) |
| daily_pnl_at_best | Mean daily PnL at optimal pair |
| break_even_rt_at_best | Break-even RT cost at optimal pair |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: Weight 1.0 reproduces baseline | Realized exp $0.90 +/- $0.01, trade rate 85.18% | Code bug |
| SC-S2: Precision increases as weight decreases | Monotonic (with possible plateau) | Weight logic is inverted |
| SC-S3: Recall decreases as weight decreases | Monotonic (with possible plateau) | Weight logic is inverted |
| SC-S4: IQR monotonically increases as weight decreases | Yes (or at least wider than baseline 4pp) | Class weighting doesn't affect probability distribution |

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: There exists a (weight, threshold) pair with realized WF expectancy > $1.50/trade and trade rate > 15%
- [ ] **SC-2**: P(directional) IQR at best weight > 10pp (vs baseline 4pp) — probability distribution actually spread
- [ ] **SC-3**: Per-fold CV < 80% at optimal (weight, threshold) — improved stability vs baseline 129%
- [ ] **SC-4**: Dir-bar PnL > $3.00/trade at optimal (weight, threshold) — no severe degradation

---

## Minimum Viable Experiment

Run 3 weights (1.0, 0.5, 0.25) on Fold 1 only at 19:7, threshold 0.50. Verify:
1. Weight 1.0 reproduces baseline (sanity)
2. Stage 1 precision increases as weight decreases (1.0 -> 0.5 -> 0.25)
3. P(directional) IQR increases as weight decreases
4. Trade rate decreases as weight decreases (fewer directional predictions)

**MVE pass:** All 4 checks pass. **ABORT if** weight 1.0 doesn't reproduce baseline or precision doesn't increase.

---

## Full Protocol

### Step 0: Adapt the PnL Realized Return Script

Start from `.kit/results/pnl-realized-return/run_experiment.py` (or `.kit/results/threshold-sweep/run_experiment.py`). Key adaptations:

1. Add outer loop over `scale_pos_weight` values: [1.0, 0.5, 0.33, 0.25, 0.20]
2. For each weight: train Stage 1 with `scale_pos_weight=<weight>` in XGBoost params. Stage 2 is unchanged (weight=1.0).
3. After training at each weight: extract raw P(directional) predictions from Stage 1
4. For each of 9 thresholds (0.50 through 0.90): compute realized PnL (same as PR #36)
5. Record P(directional) distribution diagnostics per weight (IQR, histogram bins, percentiles)

**Critical implementation detail:** `scale_pos_weight` is passed directly to `xgb.XGBClassifier(scale_pos_weight=<weight>, ...)` for Stage 1 only. Stage 2 training is completely independent.

### Step 1: Walk-Forward at Each Weight (19:7 only)

For each of the 3 folds:
- Train Stage 2 ONCE with default params (Stage 2 trains on true-label directional bars — independent of Stage 1 weight)
- For each of the 5 weights:
  - Train Stage 1 with `scale_pos_weight=<weight>` (re-training required — this is NOT post-hoc)
  - Extract P(directional) distribution from Stage 1
  - Sweep 9 thresholds (post-hoc on trained model)
  - Apply Stage 2 direction predictions (from the single Stage 2 model) to bars that pass threshold

Stage 1: 5 weights × 3 folds = 15 XGB fits (retrained per weight).
Stage 2: 3 folds × 1 = 3 XGB fits (trained once per fold — Stage 2 trains on true-label directional bars, independent of Stage 1 weight).
Total: 18 XGB fits. Threshold sweep is post-hoc numpy on Stage 1 probabilities.

### Step 2: P(directional) Distribution Analysis (THE key diagnostic)

For each weight, report:
- N (total test bars across 3 folds)
- Mean, std, median, IQR [p25, p75]
- p10, p90
- Fraction in [0.50, 0.60] (baseline: 80.6%)
- Fraction in [0.60, 1.00] (baseline: 4.5%)
- Fraction in [0.70, 1.00] (baseline: 0.13%)

**This is the primary diagnostic.** If the IQR doesn't expand beyond 10pp at any weight, class weighting doesn't help and the probability surface is structurally compressed. If it expands, threshold control becomes possible.

### Step 3: Threshold Curve at Each Weight

For each (weight, threshold) pair, report:
- Trade rate, hold fraction
- Realized expectancy (base costs $2.49 RT)
- Dir-bar PnL, hold-bar PnL
- Per-fold values, per-fold CV

Identify the optimal (weight, threshold) pair:
- Maximize realized expectancy subject to trade rate > 15%
- Report Pareto frontier across all (weight, threshold) pairs

### Step 4: Cost Sensitivity at Optimal

3 cost scenarios at the optimal (weight, threshold) pair:

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic (limits, RTH) | $0.62/side | — | — | $1.24 |
| Base (markets, RTH) | $0.62/side | 1 tick | 0 | $2.49 |
| Pessimistic (markets, fast) | $0.62/side | 1 tick | 1 tick/side | $4.99 |

### Step 5: Comparison to Baseline

| Metric | Baseline (w=1.0, T=0.50) | Optimal (w=w*, T=T*) | Delta |
|--------|--------------------------|---------------------|-------|
| Realized exp | $0.90 | ? | ? |
| Trade rate | 85.2% | ? | ? |
| Hold fraction | 44.4% | ? | ? |
| Per-fold CV | 129% | ? | ? |
| Break-even RT | $4.64 | ? | ? |
| Dir-bar PnL | $3.77 | ? | ? |
| P(dir) IQR | 4.0pp | ? | ? |
| P(dir) in [0.50, 0.60] | 80.6% | ? | ? |

---

## Resource Budget

**Tier:** Quick

- Max wall-clock time: 15 min
- Max training runs: 18 (5 weights x 3 folds Stage 1 + 3 folds Stage 2; threshold sweep is post-hoc)
- Max seeds: 1 (seed=42)
- COMPUTE_TARGET: local
- GPU hours: 0

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 18
parallelizable: false
memory_gb: 2
gpu_type: none
estimated_wall_hours: 0.08
```

18 XGB fits (15 Stage 1 + 3 Stage 2) at ~10-15s each ≈ 3-4 min. Threshold sweep is post-hoc numpy (<5s). Data loading ~30-60s. Total: ~5 min.

---

## Abort Criteria

- **Baseline reproduction fails:** STOP. Weight 1.0 must reproduce $0.90 +/- $0.01 and 85.18% trade rate.
- **Precision doesn't increase at lower weights:** STOP. If precision is constant or decreasing as weight decreases, the `scale_pos_weight` parameter is being applied incorrectly.
- **Wall-clock > 20 min:** ABORT. (4x expected ~5 min.)

---

## Decision Rules

```
OUTCOME A — SC-1 pass (exp > $1.50 at >15% trade rate at some (weight, threshold)):
  -> CONFIRMED. Class weighting + threshold produces viable strategy.
  -> Record optimal (weight, threshold), expectancy, trade rate.
  -> Next: CPCV validation at optimal (weight, threshold) for proper CI and PBO.

OUTCOME B — SC-2 pass (IQR > 10pp) but SC-1 fail (no viable (weight, threshold)):
  -> PARTIAL. Probability spread works but the underlying signal is insufficient.
  -> The distribution IS controllable, but the edge doesn't survive threshold filtering.
  -> Next: Try intermediate geometry (14:6) where direction signal may be stronger,
     or long-perspective labels with higher oracle edge.

OUTCOME C — SC-2 fail (IQR still < 10pp at all weights):
  -> REFUTED. Class weighting doesn't spread the probability surface.
  -> The compression is structural to XGBoost binary:logistic on this problem.
  -> Next: Try calibrated probabilities (Platt scaling, isotonic regression),
     or feature engineering to increase Stage 1 discriminability,
     or long-perspective labels for different label structure.
```

---

## Confounds to Watch For

1. **Precision-recall tradeoff overshoot.** At aggressive weights (0.20, 0.25), the model may become so conservative that it barely predicts any bar as directional even at T=0.50. If trade rate at T=0.50 drops below 15% for weight <= 0.25, the weight is too aggressive. Report trade rate at T=0.50 for each weight as the first-order diagnostic.

2. **Stage 2 inference distribution shift.** Stage 2 trains on true-label directional bars (independent of Stage 1 weight), but at inference time it scores bars that Stage 1 predicts as directional. At lower weights, the set of bars reaching Stage 2 at inference shifts — it's smaller and potentially more homogeneous. Stage 2's accuracy may differ on this filtered population vs its training distribution. Report Stage 2 directional accuracy per weight at T=0.50.

3. **Volume horizon confound persists.** All PnL computations inherit the unbounded hold-bar return issue from PR #35 (p10/p90 = -63/+63 ticks). Reduced hold fraction at lower weights partially mitigates this. If optimal (weight, threshold) has hold fraction < 10%, the confound is negligible.

4. **Directional-bar quality vs weight.** Lower weights produce fewer directional predictions. The retained directional bars may be "easier" (more confident predictions) or "harder" (more extreme probabilities that pass threshold but on noisier bars). Dir-bar PnL per weight is the diagnostic.

5. **IQR increase without tail improvement.** The IQR could increase by spreading probabilities from [0.50, 0.60] to [0.40, 0.70] — wider but still without a usable high-confidence tail (>0.80). Report the full distribution, not just IQR. The fraction in [0.70, 1.00] is the critical tail metric.

---

## Deliverables

```
.kit/results/class-weighted-stage1/
  metrics.json                     # All SC statuses + optimal (weight, threshold) metrics
  analysis.md                      # P(dir) distributions, threshold curves per weight, verdict
  run_experiment.py                # Adapted from threshold-sweep with weight loop
  spec.md                          # Local copy of spec
  threshold_curve.csv              # 5 weights x 9 thresholds: all metrics
  p_directional_distribution.csv   # Distribution stats per weight
  per_fold_at_optimal.csv          # Per-fold details at optimal (weight, threshold)
  cost_sensitivity.csv             # 3 cost scenarios at optimal
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **P(directional) distribution per weight** — the primary diagnostic. Table showing IQR, histogram concentration, tail fractions at each weight. Report FIRST.
3. **Threshold curve per weight** — trade rate, hold fraction, realized exp at each (weight x threshold) point
4. **Best (weight, threshold) identification** — selection rationale, comparison to baseline
5. **Per-fold consistency at optimal** — reduced fold instability?
6. **Dir-bar quality check** — does dir-bar PnL degrade at lower weights?
7. **Stage 1 precision/recall per weight** — confirm precision increases
8. **Cost sensitivity at optimal**
9. **Comparison table: baseline (w=1.0, T=0.50) vs optimal (w=w*, T=T*)**
10. **Explicit SC-1 through SC-4 pass/fail**
11. **Outcome verdict (A/B/C)**

---

## Exit Criteria

- [ ] Baseline (weight=1.0, T=0.50) reproduces PR #35 results ($0.90 +/- $0.01, 85.18%)
- [ ] P(directional) distribution analyzed for all 5 weights (IQR, histogram, tail fractions)
- [ ] Threshold curves computed for all 5 weights x 9 thresholds x 3 folds
- [ ] Stage 1 precision/recall reported per weight
- [ ] Optimal (weight, threshold) pair identified with rationale
- [ ] Per-fold details at optimal pair reported
- [ ] Dir-bar PnL per weight reported (quality check)
- [ ] Cost sensitivity at optimal computed (3 scenarios)
- [ ] Comparison table: baseline vs optimal
- [ ] All metrics in metrics.json
- [ ] analysis.md with P(directional) distributions, threshold curves, and verdict
- [ ] SC-1 through SC-4 explicitly evaluated

---

## Key References

- **Threshold sweep script:** `.kit/results/threshold-sweep/run_experiment.py` — starting point, add weight loop
- **Threshold sweep results:** `.kit/results/threshold-sweep/metrics.json` — P(directional) distribution at weight=1.0
- **PnL realized return script:** `.kit/results/pnl-realized-return/run_experiment.py` — PnL model reference
- **PnL realized return results:** `.kit/results/pnl-realized-return/metrics.json` — baseline metrics
- **2-class results:** `.kit/results/2class-directional/metrics.json` — pipeline structure reference
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/`
- **PnL constants:** tick_value=$1.25, tick_size=0.25. RT costs: optimistic $1.24, base $2.49, pessimistic $4.99
