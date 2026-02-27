# Analysis: Stage 1 Threshold Sweep — Optimizing Hold-Bar Exposure

## Verdict: REFUTED (Outcome C)

No threshold achieves realized WF expectancy > $1.50/trade at >15% trade rate. The optimal threshold is the baseline (T=0.50). SC-1, SC-2, and SC-3 all FAIL. The root cause is **catastrophic probability calibration**: 80.6% of Stage 1 P(directional) predictions cluster in [0.50, 0.60], creating a cliff where trade rate collapses from 64.5% to 4.5% between T=0.55 and T=0.60. Gradual threshold filtering is mechanically impossible with this model. The hypothesis (reduce hold-bar exposure via threshold) is sound in principle — dir-bar PnL improves at moderate thresholds — but the XGBoost probability surface does not provide a usable control lever.

---

## Results vs. Success Criteria

- [ ] **SC-1: FAIL** — Realized WF exp > $1.50 at 19:7 (base) with trade rate > 15%. Observed: $0.90 at 85.2% (T=0.50). Next best with >15% trade rate: $0.83 at 64.5% (T=0.55). No threshold above 0.55 has >15% trade rate.
- [ ] **SC-2: FAIL** — Per-fold CV < 80% at optimal threshold. Observed: 128.6% (identical to baseline — optimal IS the baseline).
- [ ] **SC-3: FAIL** — Hold fraction at optimal < 25%. Observed: 44.4% (identical to baseline).
- [x] **SC-4: PASS** — Dir-bar PnL at optimal > $3.00/trade. Observed: $3.77.
- [x] **SC-S1: PASS** — T=0.50 reproduces baseline. Exp $0.9015 (diff: $0.00), trade rate 85.18% (diff: 0.00%).
- [x] **SC-S2: PASS** — Trade rate monotonically decreases with threshold (85.2% → 64.5% → 4.5% → ... → 0.0%).
- [ ] **SC-S3: FAIL** — Hold fraction does NOT monotonically decrease. Observed: 44.4% → 43.8% → 42.0% → **53.8%** → 47.1% → 41.0% → 1.7% → 0% → 0%. Non-monotonicity at T=0.65 (see Confounds).
- [x] **SC-S4: PASS** — At least 1,000 trades per fold at optimal (T=0.50). Min fold: 189,962.

**Score: 1/4 primary pass, 3/4 sanity pass.**

---

## Metric-by-Metric Breakdown

### P(directional) Distribution — THE Critical Diagnostic

This distribution determines whether the experiment can possibly succeed. It cannot.

| Statistic | Value |
|-----------|-------|
| N (total test bars across 3 folds) | 697,150 |
| Mean | 0.5178 |
| Std | 0.1304 |
| IQR [p25, p75] | [0.536, 0.576] — **4.0 percentage points wide** |
| p10 | 0.378 |
| p90 | 0.588 |

**Histogram concentration:**
- [0.50, 0.60): **80.6%** of all predictions (561,895 bars)
- [0.55, 0.60): **60.0%** alone (418,079 bars)
- [0.60, 1.00): **4.5%** total (31,628 bars)
- [0.70, 1.00): **0.13%** total (935 bars)

The spec warned about this scenario in Confound #1: "If >80% of probabilities are within 0.05 of 0.5, thresholds above 0.55 will have minimal effect." The reality is even worse: 80.6% are within [0.50, 0.60], and the spike at [0.55, 0.60) is so extreme that T=0.60 annihilates trade rate from 64.5% to 4.5% — a 14:1 collapse in a single 5-point step.

**Implication:** XGBoost `binary:logistic` on this Stage 1 problem produces a near-binary output compressed into a 10-point probability band. The model "knows" which bars are directional with ~58.6% accuracy (from 2class-directional), but expresses this knowledge through a narrow probability range — there is no high-confidence tail to exploit. The linear extrapolation in the spec's IV table (estimated trade rates of 75%, 65%, 55%... at T=0.55, 0.60, 0.65) was off by an order of magnitude for T≥0.60.

### Primary Metrics

**1. optimal_threshold_19_7 = 0.50 (the baseline)**

The optimization found no improvement over baseline. The only thresholds with trade rate > 15% are T=0.50 (85.2%) and T=0.55 (64.5%). T=0.55 has LOWER expectancy ($0.83 vs $0.90). This is because T=0.55 cuts 20 percentage points of trade volume (dropping from 85% to 64.5%) but barely reduces hold fraction (44.4% → 43.8% — only 0.6pp). The trades removed at T=0.55 were actually net-positive contributors.

**2. optimal_realized_expectancy_19_7 = $0.90/trade**

Unchanged from baseline. The experiment produced zero improvement.

**3. optimal_trade_rate_19_7 = 85.2%**

Unchanged from baseline.

### Threshold Curve (19:7) — Full Detail

| Threshold | Trade Rate | Hold Frac | Realized Exp | Dir-Bar Exp | Hold-Bar Exp | CV% | Break-even RT | Daily PnL | Mean Trades |
|-----------|-----------|-----------|-------------|-------------|-------------|-----|---------------|-----------|-------------|
| 0.50 | 85.2% | 44.4% | $0.90 | $3.77 | -$2.68 | 129% | $4.64 | $3,380 | 197,841 |
| 0.55 | 64.5% | 43.8% | $0.83 | $3.87 | -$3.06 | 134% | $4.57 | $2,291 | 149,902 |
| 0.60 | **4.5%** | 42.0% | -$4.29 | $4.49 | -$16.89 | 102% | -$0.55 | -$519 | 10,543 |
| 0.65 | 0.43% | 53.8% | -$9.11 | $6.07 | -$25.89 | 97% | -$5.37 | -$286 | 994 |
| 0.70 | 0.13% | 47.1% | $1.30 | $5.08 | -$14.35 | 1044% | $5.04 | -$52 | 312 |
| 0.75 | 0.07% | 41.0% | -$0.29 | $2.32 | -$4.84 | 927% | $2.20 | -$6 | 160 |
| 0.80 | 0.006% | 1.7% | $0.20 | $0.68 | -$8.96 | 141% | $1.45 | $0.16 | 13 |
| 0.85 | 0.001% | 0.0% | -$3.08 | -$3.08 | $0.00 | 141% | -$1.83 | -$0.60 | 3 |
| 0.90 | 0.0004% | 0.0% | -$4.16 | -$4.16 | $0.00 | 141% | -$2.92 | -$0.24 | 1 |

**Key pattern:** The curve has two regimes:
1. **T=0.50–0.55:** High trade rate, positive expectancy, stable. Hold fraction barely changes (44.4%→43.8%).
2. **T≥0.60:** Trade rate collapses (<5%), sample sizes become statistically meaningless, expectancy becomes wildly noisy. The model becomes dominated by a handful of trades with unbounded hold-bar returns.

There is no intermediate regime. The cliff between T=0.55 and T=0.60 is absolute.

### Threshold Curve (10:5 Control)

| Threshold | Trade Rate | Hold Frac | Realized Exp | Dir-Bar Exp |
|-----------|-----------|-----------|-------------|-------------|
| 0.50 | 93.7% | 31.3% | -$1.65 | -$0.51 |
| 0.55 | 92.7% | 31.1% | -$1.66 | -$0.52 |
| 0.60 | 91.7% | 30.9% | -$1.68 | -$0.54 |
| 0.65 | 89.7% | 30.8% | -$1.73 | -$0.55 |
| 0.70 | 43.1% | 30.2% | -$1.98 | -$0.53 |
| 0.75 | 1.6% | 33.4% | -$6.65 | -$0.76 |
| 0.80 | 0.3% | 38.0% | -$9.62 | -$1.09 |
| 0.85 | 0.02% | 46.4% | -$21.21 | $0.24 |
| 0.90 | 0.0% | 0.0% | $0.00 | $0.00 |

10:5 behaves as expected: uniformly negative at all thresholds. The cliff is shifted to T=0.70 (93.7%→43.1%→1.6%) — the Stage 1 model is slightly better calibrated at the 10:5 geometry (wider usable range 0.50–0.65 before collapse), but this doesn't matter because the economics are negative throughout. 10:5 directional-bar PnL is -$0.51 (negative!) vs 19:7's +$3.77 — the 10:5 payoff ratio is simply unfavorable. The control validates the pipeline: it confirms threshold sweep mechanics work correctly on a geometry where we know the answer is negative.

### Per-Fold Details at T* (T=0.50)

| Fold | Realized Exp | Trades |
|------|-------------|--------|
| Fold 1 | $0.01 | 196,699 |
| Fold 2 | $2.54 | 189,962 |
| Fold 3 | $0.16 | 206,862 |

**CV = 128.6%.** The Fold 2 outlier ($2.54 vs $0.01 and $0.16) dominates the mean. This is identical to the pnl-realized-return baseline — threshold optimization provided zero improvement in fold stability. The 3-fold structure with expanding windows is too coarse to distinguish signal from noise at this effect size.

### Directional-Bar Quality vs. Threshold (Selection Bias Check)

| Threshold | Dir-Bar Exp | N Dir Bars (est.) | Interpretation |
|-----------|-------------|-------------------|----------------|
| 0.50 | $3.77 | ~110K | Baseline |
| 0.55 | $3.87 | ~84K | Slight improvement (+2.4%) |
| 0.60 | $4.49 | ~6.1K | Meaningful improvement (+19%) |
| 0.65 | $6.07 | ~460 | Large improvement (+61%) — but tiny sample |
| 0.70 | $5.08 | ~165 | Degradation begins — sample noise |
| 0.75 | $2.32 | ~95 | Below $3.00 — SC-4 would fail here |
| 0.80 | $0.68 | ~13 | Statistically meaningless |
| 0.85 | -$3.08 | ~3 | Negative — pure noise |
| 0.90 | -$4.16 | ~1 | Degenerate |

**Key finding:** Dir-bar PnL DOES improve at moderate thresholds (T=0.55–0.65). The Stage 1 model's high-confidence predictions correspond to genuinely better directional trades — the selection mechanism works in principle. At T=0.60, dir-bar PnL rises 19% to $4.49/trade. At T=0.65, it reaches $6.07/trade.

**But:** This improvement is swamped by hold-bar drag escalation. Hold-bar exp goes from -$2.68 (T=0.50) to -$16.89 (T=0.60) to -$25.89 (T=0.65). The hold bars that survive the threshold are the WORST ones — they are high-probability-directional bars that happen to hit the hold barrier, meaning the barrier race was extremely close and the hold outcome was maximally adverse. This is selection bias operating in reverse on hold bars.

**Confound assessment:** The dir-bar quality improvement confirms the hypothesis mechanism is sound. The problem is not that high-threshold directional bars are worse (they're better). The problem is that high-threshold hold bars are catastrophically worse, and the threshold doesn't remove them — it concentrates them.

### Hold Fraction Non-Monotonicity (SC-S3 Failure)

Hold fraction at T≥0.65 is dominated by sampling noise. At T=0.65, only 994 trades remain (mean across folds), with per-fold counts of [154, 1884, 943]. Fold 1 has 154 trades — at this sample size, hold fraction is ±7pp by chance alone (binomial CI). The non-monotonicity (53.8% at T=0.65 vs 42.0% at T=0.60) does not indicate broken threshold logic; it indicates that small-sample noise overwhelms the statistic.

This SC-S3 failure is informational, not diagnostic. The threshold logic is correct (SC-S2 passes — trade rate is strictly monotonic). The hold fraction failure reflects sample size degradation, which is itself the core problem.

### Cost Sensitivity at T* (T=0.50)

| Scenario | RT Cost | Mean Exp | Fold 1 | Fold 2 | Fold 3 |
|----------|---------|----------|--------|--------|--------|
| Optimistic | $2.49 | $2.15 | $1.26 | $3.79 | $1.41 |
| Base | $3.74 | $0.90 | $0.01 | $2.54 | $0.16 |
| Pessimistic | $6.25 | -$1.61 | -$2.50 | $0.03 | -$2.35 |

Under optimistic costs ($2.49 RT), all 3 folds are positive — this is the most encouraging finding in the data. Under pessimistic costs, only Fold 2 survives (barely). The strategy lives or dies on a $1.25/RT cost difference ($2.49 vs $3.74), which highlights how thin the edge is.

### Pareto Frontier

19:7 Pareto-optimal thresholds: [0.50, 0.70]. T=0.70 has higher expectancy ($1.30 vs $0.90) but 0.13% trade rate (655x less volume) — this is a degenerate Pareto point driven by 312 trades with CV=1044%. The practical Pareto frontier is {T=0.50} — a single point.

### Daily PnL at T*

$3,380/day (mean across folds). But Fold 1 contributes near-zero ($0.01/trade × ~787/day = ~$8/day) while Fold 2 contributes ~$12,000/day. This is not a stable daily PnL estimate.

### Break-Even RT at T*

$4.64/trade. Identical to baseline. This means the strategy survives if all-in execution costs stay below $4.64 RT — achievable with $0.62/side commission, 1-tick spread, and <1.5 ticks slippage. Not unreasonable but not comfortable.

---

## Resource Usage

| Resource | Budget | Actual |
|----------|--------|--------|
| Wall-clock | 15 min | **0.6 min** (36.4 seconds) |
| Training runs | 14 | 12 |
| GPU hours | 0 | 0 |
| Seeds | 1 | 1 |
| Total evaluation points | 54 (9 thresholds x 2 geometries x 3 folds) | 54 |

Execution was 25x under budget — appropriate for a post-hoc re-scoring experiment.

---

## Confounds and Alternative Explanations

### 1. Probability Calibration (CONFIRMED — dominant confound)

The spec identified this as Confound #1 and it is the primary cause of the REFUTED verdict. XGBoost `binary:logistic` with these features and data produces a near-degenerate probability distribution. 80.6% of predictions in [0.50, 0.60] means the model has functionally two states: "slightly favors directional" (p~0.56) and "slightly favors hold" (p~0.45). There is no gradual confidence gradient to exploit.

**Is this fixable?** Potentially. Platt scaling or isotonic regression could stretch the probability distribution. But calibration post-hoc cannot create discriminative power that isn't there — it can only redistribute it. If the model's 58.6% accuracy on the binary reachability task is expressed through a narrow probability band, recalibrating could create a wider band, but the underlying signal-to-noise ratio is fixed. The more likely interpretation is that 58.6% accuracy is achieved through weak, diffuse signals across many features (consistent with volatility_50 dominating at 19.9 gain share), not through confident high/low predictions on individual bars.

### 2. Hold-Bar Adverse Selection (CONFIRMED — secondary confound)

At high thresholds, the surviving hold bars have increasingly adverse returns (-$2.68 → -$16.89 → -$25.89). This is not a bug — it's mechanically correct. A bar with P(directional)=0.62 that hits the hold barrier represents a case where the model was moderately confident a barrier would be breached, but it wasn't. These are the bars where the barrier race was closest to being won — and the hold outcome is maximally painful (the price moved enough to suggest directionality but didn't quite reach the barrier). This is a structural feature of the threshold mechanism, not a resolvable confound.

### 3. Three-Fold Variance

With only 3 expanding-window folds, the Fold 2 outlier ($2.54 vs $0.01/$0.16) makes the mean unreliable. A CPCV design with 10+ groups would give much better fold stability estimates. However, this confound does not change the verdict — even Fold 2's best-case $2.54/trade at T=0.50 would still show the same probability calibration problem at higher thresholds.

### 4. Volume Horizon Confound (PERSISTS but reduced in relevance)

The spec noted that hold-bar returns are unbounded (±63 ticks at p10/p90 due to volume horizon truncation). This confound persists but is less relevant because the experiment failed on a more fundamental issue (probability calibration) before the volume horizon confound could materially affect the verdict. If the threshold could have reduced hold fraction to <10%, the volume horizon confound would have become negligible. It didn't.

### 5. Could T=0.55 be strictly better than T=0.50?

T=0.55 gives $0.83 vs $0.90 at T=0.50 — a $0.07 decrease. But T=0.55 has lower daily PnL ($2,291 vs $3,380) because it trades 24% fewer bars while barely improving per-trade economics. The daily PnL metric is arguably more decision-relevant than per-trade expectancy, and it confirms T=0.50 dominates.

---

## What This Changes About Our Understanding

### 1. Threshold optimization is not a viable path for the two-stage pipeline.

The pnl-realized-return analysis predicted that threshold optimization could reach $3.45/trade at 5% hold fraction. This was based on a linear extrapolation that assumed smooth, gradual control over hold-bar exposure. The actual probability distribution makes this impossible — there is no intermediate regime between "take 85% of bars" and "take 4.5% of bars."

### 2. XGBoost probability outputs are functionally binary on this problem.

The Stage 1 model achieves 58.6% reachability accuracy, but expresses this through a narrow [0.50, 0.60] probability band. This is consistent with the model learning a weak, diffuse signal across many features (volatility_50 dominating) rather than identifying distinct subpopulations of "clearly directional" vs "clearly hold" bars. The implication is that there is no high-confidence directional subpopulation to isolate — the ~58.6% accuracy is uniformly distributed across the prediction space.

### 3. The directional-bar signal is real and threshold-responsive.

Dir-bar PnL at T=0.55–0.65 ($3.87–$6.07) demonstrates that the model's confidence IS informative — higher-confidence directional predictions yield better payoffs. The signal exists. The problem is that the model can't generate enough high-confidence predictions to build a viable trading strategy. This is a precision problem (narrow probability band), not a signal problem (the signal is there).

### 4. The $0.90/trade edge at T=0.50 is the ceiling for this pipeline + geometry.

Without changing the model (calibration, features, or architecture), the Stage 1 threshold knob provides no improvement. The two-stage pipeline at 19:7 geometry produces $0.90/trade under base costs — take it or leave it.

### 5. The hold-bar problem is structural, not filterable.

Hold bars are not "bad predictions that should have been filtered." They are bars where the barrier race was genuinely uncertain — the model correctly identified them as marginal. Filtering them requires discriminative power that the model doesn't have. The alternative is to change the game: modify what "hold" means (volume horizon fix), change the label geometry, or eliminate hold bars from the loss function entirely.

---

## Proposed Next Experiments

### 1. Probability Calibration Study (if pursuing threshold path)

Apply Platt scaling and isotonic regression to Stage 1 probabilities. Measure whether the recalibrated distribution has a meaningful tail (>5% of bars above p=0.70). If yes, re-run threshold sweep on calibrated probabilities. If no (most likely), the threshold path is permanently closed.

**Priority: LOW.** The underlying signal structure makes this unlikely to succeed.

### 2. Volume Horizon Fix + Re-export (if pursuing 19:7)

Re-export with `--max-time-horizon 3600 --volume-horizon 50000` is already done (label-geometry-1h data). But the volume horizon causes hold-bar returns to be unbounded (±63 ticks). A tighter volume horizon (or infinite, letting the time horizon alone bound returns) would cap hold-bar drag. This is a data re-export, not a model change.

**Priority: MEDIUM.** This could reduce the -$2.68 hold-bar drag but doesn't address the calibration problem.

### 3. Long-Perspective Labels at Varied Geometries (P0 open question)

The P0 question "Does XGBoost at high-ratio geometries achieve positive expectancy using long-perspective labels?" remains unanswered. Long-perspective labels avoid the bidirectional hold-dominance problem. The label-geometry-1h experiment used bidirectional labels where the model refused to trade at high ratios. Long-perspective labels produce balanced classes (the original 45% accuracy was on these labels).

**Priority: HIGH.** This is the most promising direction — changes the labeling scheme, not the model or threshold.

### 4. Regime-Conditional Trading (P3 question)

GBT is marginally profitable in Q1 (+$0.003) and Q2 (+$0.029) under base costs. A regime filter (VIX, realized vol, calendar) could restrict trading to H1-like periods. Limited by single year of data.

**Priority: MEDIUM.** Quick to test but scientifically weak with N=1 year.

### 5. Class-Weighted Loss or Cost-Sensitive Learning

Instead of post-hoc threshold optimization, train the Stage 1 model with asymmetric costs: penalize false positives (predicting directional when hold) more than false negatives. This could shift the probability distribution to have a more useful shape. `scale_pos_weight` in XGBoost is the simplest lever.

**Priority: MEDIUM.** Addresses the calibration problem at the source rather than post-hoc.

---

## Program Status

- Questions answered this cycle: 1 (Stage 1 threshold optimization — REFUTED)
- New questions added this cycle: 0 (probability calibration study is a potential sub-question but not worth formalizing — the threshold path appears closed)
- Questions remaining (open, not blocked): 5 (long-perspective labels at geometries P0, class-weighted XGBoost P2, regime-stratified message P2, cost sensitivity P2, regime-conditional P3)
- Handoff required: NO (probability calibration is a research/model issue, not infrastructure)
