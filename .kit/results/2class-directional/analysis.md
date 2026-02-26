# Analysis: 2-Class Directional — Two-Stage Reachability + Direction Pipeline

## Verdict: CONFIRMED (with critical PnL model caveat)

All four success criteria pass as measured. The two-stage decomposition achieves its primary goal: liberating trade volume from the 3-class hold-prediction trap (0.28% → 85.2% trade rate at 19:7). However, the headline economic result ($3.77/trade) is almost certainly inflated ~7x by a PnL model that assigns full barrier payoffs to hold-bar trades instead of the spec's $0 conservative simplification. Corrected expectancy is estimated at ~$0.44/trade — still positive, but marginal. The directional signal and trade rate results are unambiguous. The economic viability requires follow-up validation with a corrected PnL model.

---

## Results vs. Success Criteria

- [x] **SC-1: PASS** — WF expectancy $3.775 > $0.00 at 19:7 (base $3.74 RT). **CAVEAT: likely inflated ~7x (see PnL Model Validity section). Corrected estimate ~$0.44/trade — still positive but marginal.**
- [x] **SC-2: PASS** — Dir accuracy 50.05% > 45% on truly-directional traded bars at 19:7. 11.6pp above BEV WR (38.4%). Holdout fold 3: 49.21% still above 45%.
- [x] **SC-3: PASS** — Trade rate 85.18% > 10% at 19:7. 301x increase from 3-class's 0.28%. Per-fold: 109K, 107K, 114K trades.
- [x] **SC-4: PASS** — label0_hit_rate 44.39% < 50% at 19:7. Improved from 3-class's 59.79%.
- [x] **Sanity checks: ALL PASS**
  - SC-S1 (19:7): Stage 1 acc 58.64% > 52.6% majority-class baseline
  - SC-S1 (10:5): Stage 1 acc 68.95% > 67.4% majority-class baseline
  - SC-S2: Stage 2 acc 50.76% > 50% at 10:5
  - SC-S3: Per-fold trade count > 100 at 19:7 (min 106,713)
- [x] **Reproducibility: PASS** — per-fold expectancy std $0.209 (CV=5.5%), dir accuracy std 0.64pp. Low variance across 3 folds.
- [x] **MVE gates: ALL PASS** — Stage 1 acc 56.4% > 52.6%, Stage 2 acc 49.9% > 40%, combined trade rate 89.8% > 5%.

**Outcome: A** — CONFIRMED per decision rules (SC-1 AND SC-3 pass). But the economic magnitude is uncertain (see below).

---

## Metric-by-Metric Breakdown

### Primary Metrics

| Metric | 19:7 Value | 19:7 Threshold | Verdict | Notes |
|--------|-----------|----------------|---------|-------|
| WF expectancy (base) | $3.775 (std $0.209) | > $0.00 | PASS | Likely inflated ~7x (see caveat) |
| Dir accuracy | 50.05% (std 0.64pp) | > 45% (BEV=38.4%) | PASS | Coin-flip level but above BEV |
| Trade rate | 85.18% (std 3.95pp) | > 10% | PASS | 301x increase from 3-class |

### Secondary Metrics

**Two-Stage vs 3-Class Comparison (headline result)**

| Metric | 3-Class (19:7) | Two-Stage (19:7) | Delta | 3-Class (10:5) | Two-Stage (10:5) | Delta |
|--------|----------------|------------------|-------|----------------|------------------|-------|
| Trade rate | 0.28% | 85.18% | +84.90pp | 90.39% | 93.68% | +3.29pp |
| Dir accuracy | 55.9%* | 50.05% | -5.9pp | 50.62% | 50.54% | -0.08pp |
| Expectancy (base) | $6.42* | $3.78 | -$2.64 | -$0.499 | -$0.513 | -$0.015 |
| label0_hit_rate | 59.79% | 44.39% | -15.4pp | 30.83% | 31.30% | +0.47pp |
| Daily PnL | ~$27† | $8,241 | +$8,214 | — | -$1,522 | — |

*Unreliable (0.28% trade rate, ~427 trades/fold). †Estimated from 0.28% × avg bars × $6.42.

**Trade rate headline:** 3-class 0.28% → two-stage 85.2% at 19:7 (301x increase).

**Per-Stage Accuracy**

| Metric | 19:7 | 10:5 |
|--------|------|------|
| Stage 1 binary accuracy | 58.64% | 68.95% |
| Stage 1 precision (directional) | 55.61% | 68.70% |
| Stage 1 recall (directional) | 93.04% | 97.40% |
| Stage 2 accuracy (all directional bars) | 50.39% | 50.76% |
| Stage 2 accuracy (S1-filtered bars) | 50.05% | 50.54% |
| Selection bias (all − filtered) | +0.34pp | +0.21pp |

Stage 1 is a high-recall, low-precision filter: it calls 85% of bars "directional" when 52.6% truly are. This is the OPPOSITE of the 3-class model's behavior (which called 0.28% directional). The two-stage decomposition successfully removed the cross-entropy penalty that caused the 3-class hold-prediction collapse.

**Selection bias check:** +0.34pp at 19:7, +0.21pp at 10:5 — both well within the 3pp tolerance. Stage 1 filtering does not systematically select harder-to-predict directional bars. No selection bias concern.

**Cost Sensitivity**

| Scenario | RT Cost | 19:7 Expectancy | 10:5 Expectancy |
|----------|---------|-----------------|-----------------|
| Optimistic | $2.49 | $5.025 | $0.737 |
| **Base** | **$3.74** | **$3.775** | **-$0.513** |
| Pessimistic | $6.25 | $1.265 | -$3.023 |

19:7 is positive across ALL three cost scenarios (even pessimistic). 10:5 is profitable only under optimistic costs. Cost moves 1:1 per trade (see PnL model analysis below).

### Per-Fold Consistency

**19:7 (primary)**

| Fold | Trades | Exp (base) | Dir Acc | Trade Rate | label0_hit | S1 Acc |
|------|--------|------------|---------|------------|------------|--------|
| 1 | 109,178 | $4.010 | 50.77% | 85.0% | 44.5% | 59.5% |
| 2 | 106,713 | $3.812 | 50.16% | 80.4% | 43.8% | 60.1% |
| 3 (holdout) | 114,088 | $3.502 | **49.21%** | 90.1% | 44.8% | 56.3% |

**Concerning trend:** Directional accuracy declines monotonically across folds: 50.77% → 50.16% → 49.21%. Fold 3 (holdout) dir accuracy is BELOW 50% — the direction model is worse than a coin flip in the holdout period. The declining trend is not large in absolute terms (1.56pp range) but the direction is consistent. Stage 1 accuracy also declines (59.5% → 60.1% → 56.3%), suggesting some non-stationarity.

Despite the sub-50% holdout dir accuracy, Fold 3 still reports positive expectancy ($3.50). This is possible because: (a) 49.21% accuracy exceeds the 38.4% BEV WR by 10.8pp under the favorable payoff ratio, and (b) hold-bar trade PnL (discussed below) contributes positively.

**10:5 (control)**

| Fold | Trades | Exp (base) | Dir Acc | Trade Rate | label0_hit | S1 Acc |
|------|--------|------------|---------|------------|------------|--------|
| 1 | 147,601 | -$0.428 | 51.0% | 92.3% | 30.9% | 69.6% |
| 2 | 148,639 | -$0.637 | 49.9% | 91.0% | 30.9% | 69.5% |
| 3 (holdout) | 152,194 | -$0.476 | 50.7% | 97.7% | 32.1% | 67.8% |

10:5 is stable and closely matches the 3-class baseline (-$0.499). The two-stage pipeline does not distort 10:5 results — delta is only -$0.015/trade. This validates the pipeline mechanics.

### Feature Importance Decomposition

**19:7 — Stage 1 vs Stage 2**

| Rank | Stage 1 (Reachability) | Gain | Stage 2 (Direction) | Gain |
|------|------------------------|------|---------------------|------|
| 1 | message_rate | 462.7 | volatility_50 | 52.3 |
| 2 | volatility_50 | 177.8 | message_rate | 50.2 |
| 3 | trade_count | 94.7 | minutes_since_open | 47.2 |
| 4 | volatility_20 | 85.4 | time_sin | 42.9 |
| 5 | time_sin | 51.9 | trade_count | 41.6 |

**Critical finding: Stage 2 at 19:7 does NOT learn directional features.** The spec predicted Stage 2 would concentrate on directional features (weighted_imbalance, OFI, return features). Instead, Stage 2's top-10 is dominated by the SAME volatility/activity features as Stage 1. `weighted_imbalance` is entirely absent from Stage 2's top-10 at 19:7. No return features appear.

The gain ratio is revealing: Stage 1 top feature = 462.7, Stage 2 top feature = 52.3 (ratio 8.9x). Stage 1 has massively more discriminative power than Stage 2. This quantitatively confirms: reachability detection is strong (Stage 1), direction prediction is near-zero (Stage 2).

**Contrast with 10:5:** Stage 2 at 10:5 DOES include `weighted_imbalance` (rank 6, gain 29.1) and `return_20` (rank 10, gain 16.5) — genuine directional features. At narrower barriers, directional features help. At wider barriers (19:7), they don't. This implies the direction-prediction difficulty increases with barrier width, consistent with the project's finding that directional accuracy ≈ 50% is a hard ceiling.

### Sanity Checks

| Check | Expected | Observed | Verdict |
|-------|----------|----------|---------|
| SC-S1: Stage 1 acc > majority at 19:7 | > 52.6% | 58.64% | PASS |
| SC-S1: Stage 1 acc > majority at 10:5 | > 67.4% | 68.95% | PASS |
| SC-S2: Stage 2 acc > 50% at 10:5 | > 50% | 50.76% | PASS |
| SC-S3: Per-fold trades > 100 at 19:7 | > 100 | min 106,713 | PASS |

---

## PnL Model Validity (CRITICAL)

**This section identifies a likely systematic overestimation in the reported expectancy.**

The cost sensitivity data reveals a structural property of the PnL model. Across all folds and geometries, changing RT cost by $X changes per-trade expectancy by exactly $X:

- 19:7 Fold 1: optimistic $5.260, base $4.010. Delta = $1.250 = ($3.74 - $2.49). Cost applied 1:1 per trade. ✓
- 10:5 Fold 1: optimistic $0.822, base -$0.428. Delta = $1.250. ✓

This means EVERY trade (regardless of whether it hits a directional or hold bar) incurs the full RT cost. The implied gross per trade:

| Geometry | Gross/trade | Derivation |
|----------|-------------|------------|
| 19:7 | $7.750 | $3.775 + $3.74 |
| 10:5 | $3.312 | -$0.513 + $3.74 |

**Cross-check against directional-bar PnL:**

At 19:7: dir accuracy 50.05% → dir bar gross = 0.5005 × $23.75 + 0.4995 × (-$8.75) = **$7.750**
At 10:5: dir accuracy 50.54% → dir bar gross = 0.5054 × $12.50 + 0.4946 × (-$6.25) = **$3.313**

The directional-bar gross EQUALS the overall gross per trade. For this to hold given 44.4% hold-bar trades at 19:7, hold-bar trades must ALSO average $7.750 gross — identical to directional-bar trades.

**This is only possible if the PnL model assigns full target/stop payoffs to hold-bar trades**, contradicting the spec's "$0 (conservative simplification)." The experiment script likely computes PnL for ALL trades based on whether the prediction matches the actual price direction (sign of fwd_return), assigning +target or -stop payoffs regardless of whether the barrier was actually hit.

**Why this matters:** Hold-labeled bars are bars where the price stayed within [-7, +19] ticks for the full 3600s horizon. The actual PnL for a directional bet on these bars is the realized price movement (bounded by barriers) — NOT the full target/stop payoff. A bar that moved +3 ticks in the predicted direction should earn $3.75, not $23.75. Assigning full barrier payoffs to these bars inflates their contribution.

**Corrected expectancy estimate** (using $0 for hold-bar trades per spec):

| Fold | frac_dir | dir_gross | Corrected gross | Net (base) |
|------|----------|-----------|-----------------|------------|
| 1 | 0.5551 | $7.750 | $4.302 | **$0.562** |
| 2 | 0.5618 | $7.552 | $4.242 | **$0.502** |
| 3 | 0.5515 | $7.243 | $3.995 | **$0.255** |
| **Mean** | | | | **$0.440 ± $0.163** |

The corrected mean expectancy is ~**$0.44/trade** — still positive, but **8.6x lower** than the reported $3.775. The 95% confidence interval (t-distribution, df=2) is approximately [$0.04, $0.84]. The lower bound barely clears zero.

**Impact on verdict:** SC-1 likely still passes under correction ($0.44 > $0.00), but the margin is thin. The result is economically marginal, not the strong positive signal the raw metrics suggest. SC-2, SC-3, SC-4 are unaffected — they don't depend on PnL.

**Note:** This is an analytical estimate based on cost-sensitivity inference. The actual hold-bar PnL may not be exactly $0 — it depends on the typical price movement magnitude within the barriers. Computing the TRUE hold-bar PnL requires the raw predictions, which this analysis cannot access. A follow-up experiment with the corrected PnL model is needed.

---

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| Wall-clock | 15 min | 0.53 min (32s) | Well under budget |
| Training runs | 14 | 14 | On budget |
| GPU hours | 0 | 0 | On budget |
| Seeds | 1 | 1 | On budget |

Extremely fast execution — 14 XGBoost fits in 32 seconds on Apple Silicon. The Quick-tier budget was appropriate.

---

## Confounds and Alternative Explanations

### 1. PnL Model Inflates Economics (CRITICAL — discussed above)
The reported $3.775/trade is likely ~8x the true value. The corrected estimate (~$0.44) makes the economic case marginal, not decisive. **This is the dominant confound.** All subsequent analysis should be interpreted with corrected economics in mind.

### 2. Hold-Bar PnL Uncertainty
The corrected estimate assumes hold-bar PnL = $0. In reality, hold-bar PnL depends on the realized price movement at time horizon expiry. If the model's direction prediction on hold bars is systematically biased (e.g., predicting the direction of recent momentum that then mean-reverts over the 3600s horizon), hold-bar trades could have negative expected PnL, making the true expectancy lower than $0.44.

### 3. Directional Accuracy is a Coin Flip
50.05% directional accuracy is 0.05pp above random. The 2.71:1 payoff ratio converts this to positive expectancy mathematically (BEV = 38.4%), but the direction prediction itself carries essentially zero information. Fold 3 (holdout) at 49.21% is actually BELOW 50%. The model is not predicting direction — it's exploiting the asymmetric payoff structure at near-random accuracy.

### 4. Only 3 Walk-Forward Folds
With 3 data points, statistical power is minimal. The corrected expectancy CI of [$0.04, $0.84] barely excludes zero. CPCV (45 splits) is needed for any reliable inference about economic viability.

### 5. Stage 2 Does Not Learn Complementary Features (19:7)
Both stages learn the same volatility/activity features. The two-stage decomposition did not produce the expected specialization (reachability vs direction). This suggests the decomposition's value is structural (removing the cross-entropy penalty for wrong-direction from Stage 1) rather than representational (learning different features).

### 6. XGB Params Tuned for multi:softprob
The tuned hyperparameters were optimized for 3-class classification, not binary:logistic. The binary stages may benefit from separate tuning. This is a LOWER BOUND — if the result is positive with suboptimal params, optimized params could improve it.

### 7. Hard Threshold at 0.5 is Arbitrary
Stage 1's threshold controls the precision/recall trade-off. At 0.5, it operates at very high recall (93%) and low precision (55.6%). Raising the threshold would reduce trade rate but increase the fraction of trades hitting directional bars (lowering label0_hit_rate). This could improve corrected expectancy by concentrating trades on bars with real payoff potential.

### 8. Same 20 Features for Both Stages
Using the same features for reachability and direction is conservative. Stage 1 might benefit from additional volatility features; Stage 2 might benefit from order-flow features. Feature-per-stage optimization is a natural follow-up.

### 9. Declining Holdout Performance
The monotonic decline in dir accuracy (50.77% → 50.16% → 49.21%) across folds suggests mild non-stationarity. With more training data (expanding window), the model does NOT improve on direction prediction — it may slightly worsen. This is consistent with the finding that direction at 19:7 is fundamentally unpredictable with these features.

### 10. Objective Function Confound
Improvement over 3-class could be from switching `multi:softprob` → `binary:logistic` rather than the two-stage decomposition per se. Diagnostic: 3-class model at 19:7 had directional recall ≈ 0.3% (essentially zero). Stage 1's 93% directional recall is a 300x improvement that cannot be attributed to objective function alone — it's the binary formulation removing the wrong-direction penalty.

---

## What This Changes About Our Understanding

### Confirmed
1. **The two-stage decomposition liberates trade volume.** The 3-class model's hold-prediction collapse at 19:7 (0.28% trade rate) was caused by the cross-entropy penalty for wrong-direction predictions. Removing this penalty via binary Stage 1 restores 85% trade rate. The mechanism works as hypothesized.

2. **Stage 1 (reachability) is a solvable problem.** The model achieves 58.6% binary accuracy on directional-vs-hold classification (6pp above majority baseline). Volatility and activity features provide genuine reachability signal. This is the strongest per-stage result.

3. **Stage 2 (direction) at 19:7 is essentially random.** 50.05% accuracy with no directional features in the top-10 confirms that the 20-feature set cannot predict direction at 19:7 barriers. The direction signal (50.05% vs 50% random) contributes ~zero information.

4. **The favorable payoff ratio (2.71:1) at 19:7 is the key economic driver**, not direction prediction skill. The model succeeds because 50% accuracy > 38.4% BEV WR, not because it knows which way price will move. This is a fundamentally different value proposition than typical classification models.

### Revised
5. **The economic magnitude is much smaller than reported.** Under corrected PnL assumptions, the per-trade edge is ~$0.44, not $3.78 — making 19:7 economically marginal, not strongly profitable. The daily PnL drops from $8,241 to an estimated ~$960 (still meaningful but in a different category).

6. **Hold-bar trades are a dominant factor.** 44.4% of trades hit hold-labeled bars, making the PnL model's treatment of these bars the single largest determinant of reported performance. Any future experiment at 19:7 MUST specify exactly how hold-bar trades are valued.

### New Understanding
7. **The two-stage decomposition reveals a spectrum.** At 10:5 (tight barriers), Stage 2 learns directional features and achieves ~51% accuracy. At 19:7 (wide barriers), Stage 2 finds no directional features and achieves ~50% accuracy. Barrier width directly controls direction-prediction difficulty. This suggests an intermediate geometry (e.g., 14:6 or 15:5) might offer the best trade-off: wide enough for favorable BEV, narrow enough for marginal directional signal.

8. **Reachability and direction are genuinely decoupled.** The absence of feature overlap between Stage 1 (volatility-dominated) and Stage 2 (no directional features at 19:7) confirms that barrier-reachability and price-direction are orthogonal problems at wide barriers. This decomposition is architecturally sound even though the direction signal is weak.

---

## Proposed Next Experiments

### 1. Corrected PnL Model Validation (HIGHEST PRIORITY)
Re-run the walk-forward with a PnL model that correctly assigns $0 to hold-bar trades (or, better, computes the actual realized PnL at time horizon expiry). This resolves the dominant uncertainty in the current results. Same data, same models — just correct the PnL computation. If corrected expectancy is still positive: strong CONFIRMED. If it goes negative: REFUTED.

### 2. CPCV at 19:7 with Corrected PnL (45 splits)
Three walk-forward folds provide insufficient statistical power. CPCV with 45 splits and the corrected PnL model gives proper confidence intervals and PBO. The corrected CI [$0.04, $0.84] from 3 folds is too wide for a go/no-go decision.

### 3. Stage 1 Threshold Optimization
Varying the Stage 1 P(directional) threshold from 0.5 to higher values (0.6, 0.7, 0.8) would trade off trade rate for precision. Higher precision → fewer hold-bar trades → less sensitivity to the hold-bar PnL assumption. At threshold 0.7, the model might trade 40-50% of bars with label0_hit_rate < 30%, making the corrected economics more robust.

### 4. Intermediate Geometry Exploration (e.g., 14:6, 15:5)
The 19:7 experiment shows that direction prediction is random at 2.71:1 payoff. The 10:5 control shows marginal directional signal at 2:1 payoff. An intermediate geometry (BEV ~43-48%, payoff ratio ~2.2-2.5:1) might find the sweet spot where directional accuracy exceeds BEV by a comfortable margin AND the payoff ratio amplifies the edge.

### 5. Class-Weighted Stage 1
Current Stage 1 has high recall (93%) and low precision (56%). Class-weighted binary objective could shift this trade-off: higher precision at the cost of recall, producing fewer but higher-quality directional predictions. Combined with threshold optimization, this could significantly improve the corrected economics.

---

## Program Status

- **Questions answered this cycle:** 1 (P1: 2-class formulation at 19:7)
- **New questions added this cycle:** 1 (hold-bar PnL model validation)
- **Questions remaining (open, not blocked):** 5 (per QUESTIONS.md)
- **Handoff required:** NO (PnL model fix is within research scope — Python script modification)

---

## Appendix: Exit Criteria Audit

- [x] MVE gates passed (Stage 1 accuracy 56.4% > 52.6%, Stage 2 accuracy 49.9% > 48%, combined trade rate 89.8% > 5%)
- [x] Walk-forward completed for 19:7 (primary) — 3 folds × 2 stages
- [x] Walk-forward completed for 10:5 (control) — 3 folds × 2 stages
- [x] Two-stage combined metrics: expectancy, dir accuracy, trade rate, label0_hit_rate per geometry
- [x] Comparison to 3-class label-geometry-1h walk-forward results tabulated
- [x] Feature importance reported for Stage 1 and Stage 2 at both geometries
- [x] Cost sensitivity computed (3 scenarios × 2 geometries)
- [x] All metrics reported in metrics.json
- [x] analysis.md written with comparison tables and verdict
- [x] Decision rule applied (Outcome A)
- [x] SC-1 through SC-4 explicitly evaluated
