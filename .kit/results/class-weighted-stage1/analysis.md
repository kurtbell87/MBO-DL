# Analysis: Class-Weighted Stage 1 — Spreading the Probability Surface

## Verdict: REFUTED (Outcome C)

Class weighting does not spread the Stage 1 probability distribution. Instead of widening the P(directional) surface to enable threshold control, `scale_pos_weight < 1` shifts the entire distribution downward and compresses it further. At weight=0.20, the distribution collapses to a 0.79pp IQR around P=0.224 — 5x MORE degenerate than the baseline's already-compressed 4.00pp IQR. The probability compression is structural to XGBoost `binary:logistic` on this reachability task and cannot be addressed by loss reweighting. The baseline (weight=1.0, T=0.50, $0.90/trade) remains the ceiling for this pipeline + geometry combination.

---

## Results vs. Success Criteria

- [ ] **SC-1: FAIL** — Best (weight, threshold) pair is (1.0, 0.50) with realized WF exp = $0.90/trade at 85.2% trade rate. No pair achieves exp > $1.50 at >15% trade rate. At lower weights, trade rate collapses to <0.3% or zero.
- [ ] **SC-2: FAIL** — Maximum P(directional) IQR is 4.00pp at weight=1.0 (the baseline). IQR *decreases* at every lower weight: 3.83pp (0.50) -> 3.25pp (0.33) -> 2.02pp (0.25) -> 0.79pp (0.20). The distribution shrank, not spread.
- [ ] **SC-3: FAIL** — Per-fold CV = 128.6% at optimal (weight=1.0, T=0.50). Identical to baseline — no improvement possible since optimal IS baseline.
- [x] **SC-4: PASS** — Dir-bar PnL = $3.77/trade at optimal (weight=1.0, T=0.50) > $3.00. But this is the unchanged baseline, not a class-weighted improvement.
- [x] Sanity checks: **SC-S1/S1b PASS**, **SC-S2 FAIL**, **SC-S3 PASS**, **SC-S4 FAIL** (details below)
- [ ] Reproducibility: N/A — single seed (42), 3 folds. Baseline reproduces exactly (diff = 0.0).

**Summary: 1/4 SC pass. SC-4 passes trivially (baseline unchanged). SC-1, SC-2, SC-3 all FAIL.**

---

## Metric-by-Metric Breakdown

### 1. P(directional) Distribution per Weight (THE Primary Diagnostic)

| Weight | N | Mean | Median | IQR (pp) | p10 | p25 | p75 | p90 | Max | Frac [0.50,0.60] | Frac [0.60,1.0] | Frac [0.70,1.0] |
|--------|------|------|--------|----------|-----|-----|-----|-----|-----|------------------|-----------------|-----------------|
| **1.00** | 697,150 | 0.518 | 0.560 | **4.00** | 0.378 | 0.536 | 0.576 | 0.588 | 0.907 | **80.6%** | 4.5% | 0.13% |
| **0.50** | 697,150 | 0.359 | 0.390 | **3.83** | 0.238 | 0.367 | 0.405 | 0.416 | 0.693 | 0.29% | 0.02% | 0.0% |
| **0.33** | 697,150 | 0.275 | 0.297 | **3.25** | 0.179 | 0.278 | 0.311 | 0.320 | 0.583 | 0.01% | 0.0% | 0.0% |
| **0.25** | 697,150 | 0.233 | 0.247 | **2.02** | 0.178 | 0.234 | 0.254 | 0.259 | 0.393 | 0.0% | 0.0% | 0.0% |
| **0.20** | 697,150 | 0.224 | 0.226 | **0.79** | 0.219 | 0.219 | 0.227 | 0.228 | 0.229 | 0.0% | 0.0% | 0.0% |

**The hypothesis predicted IQR would INCREASE. It DECREASED monotonically.**

The mechanism is clear: `scale_pos_weight < 1` increases the cost of false positives relative to false negatives for the directional class. XGBoost responds not by becoming more discriminating (spreading probabilities) but by uniformly suppressing all directional predictions. The entire distribution shifts below 0.5 and compresses toward a single point.

At weight=0.20: ALL 697,150 predictions fall in a single histogram bin [0.2, 0.3], with a total range of 0.217-0.229 (1.2pp). The model has become a constant predictor — the most degenerate possible outcome.

At weight=0.50: The distribution shifts to center around 0.39, with 86.5% of predictions in [0.3, 0.45]. The "spreading" that was supposed to happen simply became "shifting below 0.5."

**Critical insight:** XGBoost `binary:logistic` uses the sigmoid function on raw logits. `scale_pos_weight` modifies the gradient computation, effectively shifting the logit threshold. This shifts the sigmoid curve's operating point rather than stretching it. The probability surface is structurally compressed because the model lacks the capacity to distinguish between "confidently directional" and "marginally directional" — it treats most bars as roughly equally (un)directional.

### 2. Stage 1 Accuracy, Precision, and Recall per Weight

| Weight | Accuracy | Precision | Recall |
|--------|----------|-----------|--------|
| **1.00** | 0.586 | 0.556 | **0.930** |
| **0.50** | 0.491 | 0.426 | **0.002** |
| **0.33** | 0.491 | 0.281 | **0.0002** |
| **0.25** | 0.491 | 0.000 | **0.000** |
| **0.20** | 0.491 | 0.000 | **0.000** |

**Recall collapses catastrophically.** From 93% to 0.2% in a single step (weight 1.0 -> 0.50). This is a 465x reduction. The model transitions from "predict almost everything as directional" to "predict almost nothing as directional."

**Precision DECREASES, not increases** (SC-S2 FAIL). At weight=1.0: 55.6%. At weight=0.50: 42.6%. This is the opposite of the hypothesis. The few positive predictions at weight=0.50 are *worse quality* than the many predictions at weight=1.0. The model doesn't select "the best" directional bars when forced to be conservative — it selects essentially random bars from a tiny tail of the distribution.

At weights 0.25 and 0.20, precision = 0 and recall = 0 because the model predicts ZERO bars as directional. Accuracy stabilizes at ~0.491 — the hold class prevalence — because predicting "hold" for everything achieves this level.

**SC-S2 (precision increases): FAIL** — precision decreases monotonically from 0.556 to 0.0.
**SC-S3 (recall decreases): PASS** — recall does decrease monotonically: 0.930 -> 0.002 -> 0.0002 -> 0.0 -> 0.0.

### 3. Threshold Curves per Weight

#### Weight 1.00 (Baseline)

| Threshold | Trade Rate | Hold Frac | Realized Exp | Dir-bar Exp | Per-fold CV | N trades |
|-----------|-----------|-----------|-------------|-------------|-------------|----------|
| 0.50 | **85.2%** | 44.4% | **$0.90** | $3.77 | 128.6% | 197,841 |
| 0.55 | 64.5% | 43.8% | $0.83 | $3.87 | 133.5% | 149,902 |
| 0.60 | **4.5%** | 42.0% | -$4.29 | $4.49 | 101.9% | 10,543 |
| 0.65 | 0.43% | 53.8% | -$9.11 | $6.07 | 96.7% | 994 |
| 0.70 | 0.13% | 47.1% | $1.30 | $5.08 | 1043.5% | 312 |
| 0.75 | 0.07% | 41.0% | -$0.29 | $2.32 | 927.5% | 160 |
| 0.80 | 0.006% | 1.7% | $0.20 | $0.68 | 141.4% | 13 |
| 0.85 | 0.001% | 0.0% | -$3.08 | -$3.08 | 141.4% | 3 |
| 0.90 | 0.0004% | 0.0% | -$4.16 | -$4.16 | 141.4% | 1 |

The cliff from 64.5% (T=0.55) to 4.5% (T=0.60) persists unchanged — this is the same as the threshold-sweep experiment. Reproduces exactly.

#### Weight 0.50

| Threshold | Trade Rate | Hold Frac | Realized Exp | Dir-bar Exp | N trades |
|-----------|-----------|-----------|-------------|-------------|----------|
| 0.50 | **0.30%** | 57.4% | -$2.51 | $5.77 | 705 |
| 0.55 | 0.09% | 48.1% | $6.13 | $4.88 | 221 |
| 0.60 | 0.02% | 40.3% | -$9.10 | $2.25 | 42 |
| 0.65 | 0.007% | 1.9% | $2.81 | $2.55 | 18 |
| 0.70-0.90 | **0.0%** | — | — | — | **0** |

At weight=0.50, T=0.50 generates only 705 trades (0.3% trade rate). T=0.55 shows $6.13/trade — but this is on 221 trades, overwhelmingly from Fold 2 only (290 of 370 non-zero fold trades), with enormous CV (165%). T>=0.70 has zero trades.

#### Weights 0.33, 0.25, 0.20

| Weight | T=0.50 trades | T=0.50 trade rate | T=0.55+ trades |
|--------|--------------|-------------------|----------------|
| 0.33 | 34 (Fold 2 only) | 0.01% | 9 (Fold 2, T=0.55) then 0 |
| 0.25 | **0** | **0.0%** | **0** at all thresholds |
| 0.20 | **0** | **0.0%** | **0** at all thresholds |

Weights 0.25 and 0.20 are completely dead — zero trades at any threshold. The model's max P(dir) is 0.393 and 0.229 respectively, both far below T=0.50.

### 4. Best (Weight, Threshold) Pair

**Best: (1.0, 0.50)** — the unchanged baseline. $0.90/trade at 85.2% trade rate.

No other (weight, threshold) pair achieves exp > $1.50 at >15% trade rate. The only pair with higher per-trade expectancy that has >0 trades is (0.33, T=0.55) at $4.63/trade — but on 9 trades total across 3 folds (all from Fold 2). This is noise on noise.

Selection rationale: The baseline IS the optimal. Class weighting uniformly degrades all outcomes.

### 5. Per-Fold Consistency at Optimal (Weight=1.0, T=0.50)

| Fold | N test | N trades | Trade Rate | Hold Frac | Realized Exp | Dir-bar Exp | Hold-bar Exp | Dir Acc |
|------|--------|----------|-----------|-----------|-------------|-------------|-------------|---------|
| Fold 1 | 231,500 | 196,699 | 84.97% | 44.49% | **$0.01** | $4.01 | -$4.98 | 50.77% |
| Fold 2 | 236,130 | 189,962 | 80.45% | 43.82% | **$2.54** | $3.81 | $0.91 | 50.16% |
| Fold 3 (holdout) | 229,520 | 206,862 | 90.13% | 44.85% | **$0.16** | $3.50 | -$3.96 | 49.21% |

Per-fold CV = 128.6%. Fold 2 drives the mean ($2.54 vs $0.01/$0.16). This is unchanged from PR #35 — the instability is intrinsic to the signal, not addressable by class weighting.

Fold 2's anomalous hold-bar PnL (+$0.91 vs -$4.98/-$3.96) is the dominant factor. Dir-bar PnL is more stable ($3.50-$4.01 range). Directional accuracy is ~50% in all folds (50.77%, 50.16%, 49.21%) — essentially coin-flip.

### 6. Dir-bar PnL Stability

| Weight | T=0.50 Dir-bar Exp | T=0.55 Dir-bar Exp | Assessment |
|--------|-------------------|-------------------|------------|
| 1.00 | $3.77 | $3.87 | Healthy — improving with threshold |
| 0.50 | $5.77 | $4.88 | Higher but on 705/221 trades — small sample noise |
| 0.33 | $2.46 | $4.59 | 34/9 trades — meaningless |
| 0.25 | — | — | Zero trades |
| 0.20 | — | — | Zero trades |

At weight=1.0, dir-bar PnL is stable and healthy ($3.77-$6.07 from T=0.50 to T=0.65). This confirms the directional signal exists in the bars the model identifies. The problem is sample size collapse at lower weights, not signal degradation.

### 7. Cost Sensitivity at Optimal (Weight=1.0, T=0.50)

| Scenario | RT Cost | Mean Exp | Fold 1 | Fold 2 | Fold 3 |
|----------|---------|----------|--------|--------|--------|
| Optimistic | $2.49 | **$2.15** | $1.26 | $3.79 | $1.41 |
| Base | $3.74 | **$0.90** | $0.01 | $2.54 | $0.16 |
| Pessimistic | $6.25 | **-$1.61** | -$2.50 | $0.03 | -$2.35 |

Break-even RT = $4.64 (unchanged from PR #35). Strategy is viable under base costs (all folds positive) but fragile — Fold 1 is $0.01 (essentially zero). Under pessimistic costs, only Fold 2 survives.

### 8. Sanity Checks

| Check | Expected | Result | Status |
|-------|----------|--------|--------|
| SC-S1: Weight 1.0 reproduces baseline exp | $0.90 +/- $0.01 | $0.9015 (diff = 0.0) | **PASS** |
| SC-S1b: Weight 1.0 reproduces baseline trade rate | 85.18% | 85.18% (diff = 0.0) | **PASS** |
| SC-S2: Precision increases as weight decreases | Monotonic increase | **0.556 -> 0.426 -> 0.281 -> 0.0 -> 0.0** | **FAIL** |
| SC-S3: Recall decreases as weight decreases | Monotonic decrease | 0.930 -> 0.002 -> 0.0002 -> 0.0 -> 0.0 | **PASS** |
| SC-S4: IQR increases as weight decreases | Monotonic increase | **4.00 -> 3.83 -> 3.25 -> 2.02 -> 0.79** | **FAIL** |

SC-S2 failure is diagnostic: precision decreasing means the model's few remaining positive predictions at lower weights are LESS accurate, not more. The model doesn't gain discrimination — it loses it.

SC-S4 failure is the core finding: IQR contracts instead of expanding. The probability surface cannot be spread by loss reweighting.

### 9. Comparison Table: Baseline vs Optimal

| Metric | Baseline (w=1.0, T=0.50) | Optimal (w=1.0, T=0.50) | Delta |
|--------|--------------------------|------------------------|-------|
| Realized exp | $0.90 | $0.90 | $0.00 |
| Trade rate | 85.2% | 85.2% | 0.0pp |
| Hold fraction | 44.4% | 44.4% | 0.0pp |
| Per-fold CV | 128.6% | 128.6% | 0.0pp |
| Break-even RT | $4.64 | $4.64 | $0.00 |
| Dir-bar PnL | $3.77 | $3.77 | $0.00 |
| P(dir) IQR | 4.0pp | 4.0pp | 0.0pp |
| P(dir) in [0.50,0.60] | 80.6% | 80.6% | 0.0pp |

**Optimal = baseline in all dimensions.** Class weighting provides zero improvement.

---

## Resource Usage

| Resource | Budgeted | Actual | Assessment |
|----------|----------|--------|------------|
| Wall-clock | 15 min | **0.63 min (38s)** | 24x under budget |
| Training runs | 18 | 18 | Exact match |
| Stage 1 fits | 15 | 15 | Exact match |
| Stage 2 fits | 3 | 3 | Exact match |
| Threshold evaluations | 135 | 135 | 5 weights x 9 thresholds x 3 folds |
| GPU hours | 0 | 0 | CPU-only on Apple Silicon |
| Seeds | 1 | 1 | Single seed (42) |

Budget was appropriate. Experiment completed in 38 seconds — well within the "Quick" tier.

---

## Confounds and Alternative Explanations

1. **The hypothesis mechanism was wrong, not the execution.** `scale_pos_weight < 1` in XGBoost `binary:logistic` modifies the gradient by multiplying positive-class gradients by the weight. This shifts the effective decision boundary but does NOT change the model's capacity to distinguish probability levels. The sigmoid transformation on logits is monotonic — shifting logits down shifts ALL probabilities down uniformly. There is no mechanism for "spreading."

2. **Could the probabilities be miscalibrated rather than truly compressed?** Even if the raw probabilities are shifted, if the RANK ORDER of predictions is preserved, post-hoc calibration (isotonic regression, Platt scaling) could theoretically remap them to a wider range. However, the IQR contraction suggests the rank-order resolution is also degrading — at weight=0.20, there is only 0.79pp of spread to work with. Calibration cannot create information that isn't there.

3. **Is 3 folds sufficient?** The result is so clear (IQR decreasing monotonically across all 5 weights) that additional folds would not change the conclusion. The effect is deterministic, not stochastic.

4. **Fold 2 dominance persists.** All non-baseline positive results come from Fold 2. This isn't a confound specific to class weighting — it's the same Fold 2 outlier seen in PR #35. The train period (days 1-150) and test period (days 151-201) of Fold 2 appear to have a favorable regime alignment not present in other folds.

5. **Could `scale_pos_weight > 1` work instead?** The spec tested only weight < 1 (to increase precision). Weight > 1 would increase recall (predict MORE bars as directional). This would likely further compress the distribution around high values near 1.0. However, it was not tested and cannot be formally ruled out from this data alone. The structural limitation of sigmoid compression almost certainly applies in both directions.

6. **Could the weight values be too aggressive?** The jump from 1.0 to 0.5 already kills recall (93% -> 0.2%). A gentler sweep (e.g., 0.9, 0.8, 0.7) might show a more gradual transition. But the IQR ALREADY decreases from 4.00 to 3.83 at weight=0.50 — the direction is wrong from the first step. A gentler sweep would only confirm the same monotonic decrease with finer resolution.

---

## What This Changes About Our Understanding

1. **XGBoost `binary:logistic` probability compression is structural, not tunable.** Three independent experiments have now failed to produce a usable probability gradient on this task: (a) baseline produces 80.6% in [0.50, 0.60] (threshold-sweep), (b) class weighting further compresses rather than spreads (this experiment), (c) threshold optimization fails due to the cliff (threshold-sweep). The model fundamentally cannot distinguish confidence levels for the reachability task.

2. **The `scale_pos_weight` mechanism is a logit-shift, not a spread.** This is a well-understood property of XGBoost's gradient computation, but the spec's hypothesis treated it as a calibration tool. For future reference: in XGBoost, `scale_pos_weight` modifies the gradient as `grad = sigmoid(pred) - weight * label` for positive samples. This shifts the zero-gradient point, causing the model to learn a different intercept, not a different variance of the output distribution.

3. **The baseline (weight=1.0, T=0.50, $0.90/trade) is the absolute ceiling for this pipeline architecture.** Class weighting, threshold optimization, and geometry variation (10:5 vs 19:7) have all been exhausted on the two-stage pipeline. No parameter-level intervention can improve beyond $0.90/trade without changing the model formulation, feature set, or labeling scheme.

4. **The two-stage pipeline's value is the 2.71:1 payoff ratio, not prediction skill.** With directional accuracy at ~50% across all configurations, the pipeline is economically viable only because the 19:7 geometry provides favorable risk/reward. The signal is in the geometry, not the model.

5. **Precision-recall tradeoff is degenerate on this task.** The normal expectation — lower recall, higher precision — does not hold. This suggests the model's internal representation lacks a meaningful "confidence axis" for reachability. All bars are treated as roughly equally (un)likely to be directional.

---

## Proposed Next Experiments

1. **Probability recalibration (Platt scaling / isotonic regression) on Stage 1 output** — This addresses the compressed probability surface directly. If the rank-order of Stage 1 predictions contains useful information (even if the raw probabilities are compressed), calibration could unlock it. Quick to test: apply isotonic regression to the weight=1.0 predictions, then re-sweep thresholds. Expected wall-clock: ~2 minutes. Low confidence this helps given the IQR contraction evidence, but it's the cheapest test remaining.

2. **Long-perspective labels at 19:7 geometry** — Changes the labeling scheme rather than the model. Long-perspective labels produce balanced classes by construction, avoiding the ~50% hold-class dominance of bidirectional labels. If the model achieves >38.4% directional accuracy (the 19:7 breakeven WR) on long-perspective labels, the favorable payoff ratio converts to positive expectancy. This is the P0 open question in QUESTIONS.md and the highest-priority path forward.

3. **Regime-conditional trading (Q1-Q2 only)** — GBT is marginally profitable in Q1 (+$0.003) and Q2 (+$0.029) under base costs. A simple H1-only strategy avoids Q3-Q4 losses. Limited by single year of data, but can be validated on the existing 3-fold walk-forward results.

---

## SC-1 through SC-4 Summary

| Criterion | Threshold | Observed | **PASS/FAIL** |
|-----------|-----------|----------|--------------|
| SC-1: Exp > $1.50 at >15% trade rate | $1.50 / 15% | $0.90 / 85.2% (baseline only) | **FAIL** |
| SC-2: IQR > 10pp at best weight | 10pp | 4.0pp (max, at weight=1.0) | **FAIL** |
| SC-3: Per-fold CV < 80% | 80% | 128.6% | **FAIL** |
| SC-4: Dir-bar PnL > $3.00 | $3.00 | $3.77 | **PASS** (baseline only) |

**Outcome: C — REFUTED.** SC-2 fails definitively. Class weighting does not spread the probability surface.

---

## Program Status

- Questions answered this cycle: 0 (no open question directly maps; the P2 "class-weighted XGBoost at 19:7" question is about forcing directional predictions at maintained accuracy, which this experiment partially addresses but from the probability-spreading angle)
- New questions added this cycle: 0
- Questions remaining (open, not blocked): 4
- Handoff required: NO
