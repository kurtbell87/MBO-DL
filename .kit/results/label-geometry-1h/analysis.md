# Analysis: Label Geometry 1h — Model Training with Corrected Time Horizon

## Verdict: INCONCLUSIVE

Primary success criteria SC-1 through SC-8 technically pass. However, SC-3, SC-4, and SC-5 pass only at geometry 19:7 where the model makes directional predictions on **0.28% of bars** (~3,280 directional pairs out of 1.16M), producing unreliable metrics with CPCV directional accuracy std=0.208 (37% relative). The holdout collapses to **50.0% directional accuracy on 52 trades** with 96.8% label0_hit_rate — indistinguishable from a coin flip. Sanity check SC-S1 failed (baseline accuracy 0.384 < 0.40 abort threshold), which was specified as an abort criterion in the spec. The only geometry with meaningful trade volume (10:5) shows directional accuracy 50.67% — **2.6pp below breakeven** (53.28%), with consistent -$0.49/trade expectancy across CPCV and walk-forward. The time horizon fix is validated (SC-8 PASS: hold rate dropped 90.7%->32.6% at 10:5), but the model's response to high-ratio geometries was to become a near-total hold predictor, making the favorable payoff structure unexploitable.

---

## Executive Summary

The 1-hour time horizon fix succeeded in producing meaningful label distributions — hold rates dropped 29-58pp across all geometries vs Phase 1's 91-99%. At 10:5 (control), labels are near-perfectly balanced (33.2%/32.6%/34.2%). The oracle is profitable at all geometries at 3600s ($3.22-$9.44/trade). **However, the model cannot exploit the favorable payoff structure**: at the only geometry with real trade volume (10:5), directional accuracy is ~50.7% (below breakeven); at high-ratio geometries (15:3, 19:7, 20:3), the model refuses to trade (0.003-0.28% directional prediction rate). Reported positive expectancy at those geometries is a small-sample artifact on 0-100 trades per fold with Infinity profit factors. The practical reality is Outcome B: the model's directional signal is ~50-51% regardless of geometry, and it cannot distinguish directional from hold bars at wider barriers.

---

## Results vs. Success Criteria

- [x] **SC-1: PASS** — Re-export completed: 1,004 files (4 x 251), 0 failures
- [x] **SC-2: PASS** — CPCV (180 fits) + WF (12 fits) completed for all 4 geometries
- [~] **SC-3: TECHNICALLY PASS, UNRELIABLE** — 19:7 CPCV directional accuracy 55.9% > BEV WR 38.4% + 2pp = 40.4% (margin +17.5pp). **BUT:** computed on 0.28% of bars (~3,280 directional pairs), std=0.208 (37% relative). Holdout drops to 50.0% on 52 trades. Not trustworthy evidence of an edge.
- [~] **SC-4: TECHNICALLY PASS, UNRELIABLE** — 15:3 ($4.87), 19:7 ($5.68), 20:3 ($1.72) CPCV expectancy > $0. **BUT:** all Infinity profit factors (zero losses in some splits). Based on 0.003-0.28% directional prediction rates. Meaningless as economic evidence.
- [~] **SC-5: TECHNICALLY PASS, UNRELIABLE** — 19:7 holdout expectancy $3.76 > -$0.10. **BUT:** 52 trades total (0.023% of holdout bars), 96.8% label0_hit_rate, 50.0% directional accuracy = coin flip.
- [x] **SC-6: PASS** — Walk-forward reported for all 4 geometries
- [x] **SC-7: PASS** — Per-direction + time-of-day analysis reported
- [x] **SC-8: PASS** — Hold rate at 10:5 = 32.6% < 80% (vs Phase 1's 90.7%)

### Sanity Checks

- [ ] **SC-S1: FAIL** — Baseline (10:5) CPCV overall accuracy = **0.384 < 0.40**. Spec says "STOP. Pipeline broken." Abort was not enforced. The failure reflects that balanced bidirectional labels are genuinely harder to classify than expected (prior 0.449 was on long-perspective labels with different target semantics), not a pipeline bug. But the abort criterion was pre-committed and should have been respected.
- [x] **SC-S2: PASS** — No single feature > 60% gain share (max: f15 at 43.5% of top-10 at 19:7)
- [x] **SC-S3: PASS** — 152 columns confirmed
- [x] **SC-S4: PASS** — No >95% single class (max: 70.1% hold at 20:3)
- [x] **SC-S5: PASS** — Hold rate 10:5 = 32.6% < 80%
- [x] **SC-S6: PASS** — Oracle net exp > $0 at all 4 geometries ($3.22-$9.44)

---

## Metric-by-Metric Breakdown

### Primary Metrics

#### 1. Directional Accuracy per Geometry (CPCV)

| Geometry | Dir Acc | Std | BEV WR | Margin | Dir Pred Rate | N Dir Pairs (est) | Verdict |
|----------|---------|-----|--------|--------|---------------|-------------------|---------|
| 10:5 (control) | 0.5067 | 0.0039 | 0.5328 | **-2.6pp** | 90.4% | ~1,049K | **FAIL** — below breakeven, reliable |
| 15:3 | 0.3424 | **0.4665** | 0.3329 | +0.9pp | **0.006%** | ~75 | **FAIL** — margin < 2pp, N meaningless |
| 19:7 | 0.5590 | **0.2081** | 0.3843 | +17.5pp | **0.28%** | ~3,280 | **TECHNICALLY PASS** — unreliable |
| 20:3 | 0.0889 | **0.2846** | 0.2605 | **-17.2pp** | **0.003%** | ~30 | **FAIL** — below breakeven |

**Critical finding:** At 10:5, where the model actually trades (90.4% directional prediction rate, ~1.05M directional pairs), directional accuracy is 50.67% — 2.6pp below the 53.28% breakeven. The low std (0.0039) and high N confirm this is a precise, reliable estimate of the model's directional skill on balanced bidirectional labels.

At all other geometries, the model essentially stops trading (<0.3% prediction rate), making reported metrics statistically meaningless. The 15:3 directional accuracy std of 0.4665 (136% relative!) reflects that most CPCV splits have 0 or 1 directional prediction pairs — the metric swings between 0% and 100% by chance.

The 19:7 "pass" is driven by ~3,280 directional predictions across all 45 CPCV splits. While a 95% CI on binary accuracy with N=3,280 at 55.9% spans [54.2%, 57.6%], this CI does not capture the severe **selection bias**: the model selected 0.28% of bars as "trade-worthy." If forced to trade all bars (as at 10:5), directional accuracy at 19:7 would likely revert to ~50% — as confirmed by the holdout (50.0% on 52 trades, 96.8% label0_hit_rate).

#### 2. CPCV Expectancy per Geometry (Base Costs)

| Geometry | CPCV Exp ($) | Profit Factor | N Trades (est) | Interpretation |
|----------|-------------|---------------|----------------|----------------|
| 10:5 | **-$0.490** | 0.901 | ~1,049K | Reliable — consistent loss |
| 15:3 | +$4.873 | **Infinity** | ~75 | Artifact — zero losses from near-zero trades |
| 19:7 | +$5.678 | **Infinity** | ~3,280 | Artifact — extreme selection bias |
| 20:3 | +$1.723 | **Infinity** | ~30 | Artifact — near-zero trades |

**Infinity profit factor** means no losing trades were recorded in at least some splits. This is not because the model is perfect — it's because 0.003-0.28% directional prediction rates produce per-split sample sizes of 0-100 trades, where having zero losses is common by chance.

**10:5 is the only reliable economic number:** -$0.490/trade. This is 7x worse than the xgb-tuning experiment's long-perspective label result (-$0.064), confirming that balanced bidirectional labels are a fundamentally harder classification target.

#### 3. Walk-Forward Expectancy per Geometry

| Geometry | WF Mean Exp ($) | Fold 1 (N trades) | Fold 2 (N trades) | Fold 3/Holdout (N trades) |
|----------|----------------|-------------------|-------------------|---------------------------|
| 10:5 | **-$0.499** | -$0.514 (142,482) | -$0.473 (142,922) | -$0.510 (148,678) |
| 15:3 | +$0.257 | $0.00 (**0**) | +$8.26 (**100**) | -$7.49 (**2**) |
| 19:7 | +$6.416 | +$7.49 (**218**) | +$7.35 (**1,014**) | +$4.41 (**50**) |
| 20:3 | +$7.087 | $0.00 (**0**) | +$21.26 (**18**) | $0.00 (**0**) |

**10:5 walk-forward** is consistent and reliable: -$0.499/trade across 3 folds with ~143-149K trades each. CPCV/WF alignment is excellent (-$0.490 vs -$0.499) — no divergence.

**All other walk-forward results are unreliable.** 15:3 has 0, 100, and 2 trades across folds. 20:3 has 0, 18, and 0 trades — **18 total trades, all in one fold.** The reported 20:3 WF mean expectancy of $7.09 comes from exactly 18 trades in a single test window. 19:7 has 218, 1,014, and 50 trades — the best of the high-ratio geometries but still very thin, and the WF Fold 3 (holdout period) drops to only 50 trades with 52% directional accuracy.

### Secondary Metrics

#### Hold Rate Comparison — KEY DIAGNOSTIC

| Geometry | Phase 1 (300s) | This Exp (3600s) | Delta | SC-S4 |
|----------|---------------|------------------|-------|-------|
| 10:5 | 90.7% | **32.6%** | **-58.1pp** | PASS |
| 15:3 | 97.1% | **62.1%** | **-35.0pp** | PASS |
| 19:7 | 98.6% | **47.4%** | **-51.2pp** | PASS |
| 20:3 | 98.9% | **70.1%** | **-28.8pp** | PASS |

**The time horizon fix is unambiguously successful.** All 4 geometries show massive hold rate reductions. At 10:5, labels are near-perfectly balanced (33.2%/32.6%/34.2%). No geometry exceeds 95% hold — the SC-S4 degenerate label failure from Phase 1 is fully resolved.

The hold rate ordering reveals a structural pattern: wider targets and narrower stops produce higher hold rates even at 3600s. At 20:3 (stop=3 ticks, $0.75), 70% of bars don't reach either barrier within 1 hour. The 3-tick stop is narrower than typical bid-ask bounce at 5-second resolution — most bars don't traverse even $0.75 in either direction within the forward window.

#### Class Distribution per Geometry

| Geometry | Short (-1) | Hold (0) | Long (+1) | Long:Short Ratio |
|----------|------------|----------|-----------|------------------|
| 10:5 | 33.2% | 32.6% | 34.2% | 1.03:1 (balanced) |
| 15:3 | 18.7% | 62.1% | 19.2% | 1.03:1 |
| 19:7 | 26.1% | 47.4% | 26.5% | 1.02:1 |
| 20:3 | 14.8% | 70.1% | 15.1% | 1.02:1 |

Near-perfect long:short symmetry (1.02-1.03:1) at all geometries confirms the bidirectional label computation is unbiased.

#### Per-Class Recall per Geometry

| Geometry | Short Recall | Hold Recall | Long Recall | Dir Pred Rate |
|----------|-------------|-------------|-------------|---------------|
| 10:5 | 0.390 | 0.182 | 0.568 | 90.4% |
| 15:3 | 0.00002 | 0.9999 | 0.0001 | 0.006% |
| 19:7 | 0.0016 | 0.9971 | 0.0017 | 0.28% |
| 20:3 | 0.00001 | 0.9999 | 0.00001 | 0.003% |

**At 10:5:** The model actively trades — 90.4% directional prediction rate. Long-biased: long recall 0.568 vs short recall 0.390 (1.46x). Hold recall only 0.182 — the model correctly identifies only 18% of actual hold bars (inverse of Phase 1's 99.86% hold recall).

**At 15:3, 19:7, 20:3:** Near-perfect hold predictors. Short/long recall <0.2% at all three. The model has learned that the optimal cross-entropy-minimizing strategy at these geometries is to predict hold for everything.

**The 19:7 puzzle:** Hold is only 47.4% — predicting hold for everything yields only 47.4% accuracy, while predicting directional for everything yields 52.6%. The model would achieve HIGHER accuracy by predicting directional. Yet it predicts hold 99.7% of the time. This is not majority-class bias — it's genuine inability to learn the directional boundary at 19-tick barriers. The minimum-loss strategy for a model that cannot distinguish directions is to predict hold (avoiding the larger penalty of wrong-direction predictions), even when hold is the minority class.

#### Directional Prediction Rate — The Core Failure Mode

| Geometry | Dir Pred Rate | Interpretation |
|----------|---------------|----------------|
| 10:5 | **90.4%** | Active trading — model engages with balanced classes |
| 15:3 | 0.006% | Degenerate hold predictor |
| 19:7 | 0.28% | Degenerate hold predictor |
| 20:3 | 0.003% | Degenerate hold predictor |

There is a cliff between 10:5 (90.4%) and all other geometries (<0.3%). The model engages with the classification problem only at 10:5, where the balanced classes (32.6% hold) make hold-prediction a poor strategy. At wider barriers, even when hold is not the majority class (47.4% at 19:7), the model defaults to hold because the feature-to-wider-barrier correlation is too weak to exploit.

#### Feature Importance Shift

Top 3 features by gain at each geometry (feature names from RUN phase mapping):

| Geometry | #1 (gain) | #2 (gain) | #3 (gain) | Top-feature % of top-10 |
|----------|-----------|-----------|-----------|-------------------------|
| 10:5 | volatility_50 (75.7) | high_low_range_50 (45.1) | volatility_20 (28.6) | 29% |
| 15:3 | volatility_50 (68.6) | high_low_range_50 (50.0) | volatility_20 (29.6) | 28% |
| 19:7 | **high_low_range_50 (226.8)** | volatility_50 (71.2) | **depth_imbalance_1 (59.5)** | **43.5%** |
| 20:3 | high_low_range_50 (90.9) | volatility_50 (53.6) | volatility_20 (30.7) | 32% |

**Key shift at 19:7:** high_low_range_50 gain explodes to 226.8 (3-5x higher than other geometries), and depth_imbalance_1 spikes to 59.5 (4-5x higher). The model at 19:7 concentrates on features that predict barrier reachability (volatility, range) rather than direction. The Phase 1 pattern (76% volatility dominance) persists in spirit: the model's primary learning at wider barriers is "will any barrier trigger?" not "which direction?"

At 10:5, importance is more distributed — the model has enough signal to spread weight across volatility, momentum, and microstructure features. Volatility share of top 10: 57.1% (10:5), 60.1% (15:3), 64.7% (19:7), 61.3% (20:3) — increasing concentration at wider barriers.

#### Per-Direction Oracle (3600s)

| Geometry | Long Triggered | Long WR | Short Triggered | Short WR | Both Rate |
|----------|---------------|---------|-----------------|----------|-----------|
| 10:5 | 398,320 | 99.56% | 387,338 | 99.55% | 0.15% |
| 15:3 | 224,028 | 99.48% | 217,586 | 99.47% | 0.10% |
| 19:7 | 309,783 | 99.22% | 304,845 | 99.21% | 0.21% |
| 20:3 | 176,296 | 99.37% | 172,999 | 99.36% | 0.10% |

Per-direction oracle WR (among bars where the respective barrier IS triggered) is >99% by construction. Both-triggered rate is <0.25% — negligible label ambiguity from simultaneous barrier hits. Long and short are symmetric across all geometries.

#### Oracle Comparison: 300s vs 3600s

| Geometry | Oracle WR (300s) | Oracle WR (3600s) | Oracle Exp (300s) | Oracle Exp (3600s) |
|----------|-----------------|-------------------|-------------------|-------------------|
| 10:5 | 64.29% | **75.19%** | $2.75 | **$6.47** |
| 15:3 | — | **42.82%** | $1.74 | **$3.29** |
| 19:7 | — | **61.81%** | $3.88 | **$9.44** |
| 20:3 | — | **34.57%** | ~$1.92 | **$3.22** |

All geometries improved substantially with 3600s. Oracle net expectancy at all 4 is positive (SC-S6 PASS). The 19:7 geometry has the highest oracle expectancy ($9.44), consistent with its moderate hold rate and favorable 2.71:1 ratio.

#### Time-of-Day Breakdown

| Band | Period | 10:5 Dir Rate | 15:3 Dir Rate | 19:7 Dir Rate | 20:3 Dir Rate |
|------|--------|---------------|---------------|---------------|---------------|
| A | 09:30-10:00 | 71.2% | 41.5% | 57.8% | 33.4% |
| B | 10:00-15:00 | 70.1% | 39.7% | 56.3% | 31.8% |
| C | 15:00-15:30 | 61.5% | 32.2% | 38.2% | 22.8% |

Closing band (C) consistently has the lowest directional rate — less time remaining means fewer barriers reached. Band A vs B difference is modest (1-2pp), unlike Phase 1's 2.8x opening-range effect (which was a hold-rate artifact at 300s).

#### label0_hit_rate (Directional Predictions Hitting Hold Bars)

| Geometry | CPCV label0_hit | Holdout label0_hit | Flag (>25%) |
|----------|----------------|-------------------|-------------|
| 10:5 | 29.4% | 32.0% | **FLAG** |
| 15:3 | 31.6% | — | **FLAG** |
| 19:7 | 47.9% | **96.8%** | **SEVERE** |
| 20:3 | 28.9% | — | **FLAG** |

All geometries exceed the 25% flag threshold. At 10:5, 29.4% of directional predictions hit hold-labeled bars — the PnL model assigns $0 to these (conservative simplification per spec).

**19:7 holdout is catastrophic:** 96.8% of the model's 52 directional predictions hit hold-labeled bars. Only ~2 of 52 predictions coincided with actual directional bars. The model at 19:7 cannot distinguish which bars WILL be directional — its rare directional predictions are essentially random bar selections.

#### Cost Sensitivity

| Geometry | Optimistic ($2.49) | Base ($3.74) | Pessimistic ($6.25) | N Trades |
|----------|-------------------|-------------|---------------------|----------|
| 10:5 | **+$0.760** | **-$0.490** | -$3.000 | ~1.05M |
| 15:3 | +$5.346 | +$4.873 | +$3.925 | ~75 |
| 19:7 | +$6.928 | +$5.678 | +$3.168 | ~3,280 |
| 20:3 | +$1.862 | +$1.723 | +$1.444 | ~30 |

**10:5 is the only reliable cost sensitivity:** positive under optimistic costs (+$0.76), negative under base and pessimistic. Breakeven RT cost ~$3.74, matching the xgb-tuning finding. If execution costs can be reduced to $2.49 RT (zero slippage), 10:5 becomes marginally viable — but this requires perfect execution.

High-ratio geometry cost sensitivity numbers are artifacts of near-zero trade counts.

#### Holdout Evaluation

**19:7 (best by CPCV expectancy):**
- Accuracy: 0.464
- Directional accuracy: **0.500** (coin flip) on **52 trades**
- Directional prediction rate: 0.71%
- Per-class recall: short=0.0001, hold=0.985, long=0.0003
- Expectancy: +$3.76 (base) — but only 52 trades out of 229,520 bars (0.023%)
- label0_hit_rate: **96.8%** — nearly ALL directional predictions hit hold bars
- Confusion matrix: model predicted 401 shorts, 167,893 holds, 61,226 longs out of 229,520 bars. Of ~1,627 directional predictions, only 52 coincided with actual directional bars.

**10:5 (control):**
- Accuracy: 0.351
- Directional accuracy: **0.506** on **148,642 trades** (reliable)
- Directional prediction rate: 95.2%
- Per-class recall: short=0.337, hold=0.071, long=0.636
- Expectancy: **-$0.509** (base)
- label0_hit_rate: 32.0%
- Consistent with CPCV and walk-forward

The 10:5 holdout confirms: ~50.6% directional accuracy, -$0.51 expectancy, strong long bias. The 19:7 holdout confirms: the model is a hold predictor with no meaningful directional signal at wider barriers.

---

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| Wall clock | 4 hours (240 min) | **29.6 min** | Well within budget |
| Export runs | 1,004 | 1,004 (0 failures) | Complete |
| Oracle runs | 80 (4x20) | 4 (aggregated runs) | Complete |
| CPCV fits | 180 | 180 | Complete |
| WF fits | 12 | 12 | Complete |
| Holdout fits | 2 | 2 | Complete |
| GPU hours | 0 | 0 | CPU-only as specified |

Export: 67-69s per geometry (251 files each). Mean fit time: 5.97-7.42s (well under 120s abort). Total 29.6 min — 2.5x under estimate, 8x under budget. Budget was appropriate.

---

## Confounds and Alternative Explanations

### 1. The model genuinely cannot classify at wider barriers (most likely)
XGBoost's 20 features predict 5-bar (~25 second) return direction at ~50.7% accuracy. This signal does not improve (and the model refuses to engage) when the target changes from "move 10 ticks" to "move 15-20 ticks within 1 hour." The features capture microstructure state on a ~5-second timescale, poorly correlated with 19-tick moves requiring minutes to hours.

### 2. XGBoost loss minimization favors hold prediction, but this does not explain 19:7
At 15:3 (62% hold) and 20:3 (70% hold), hold is the majority class — predicting hold achieves 62-70% accuracy. **But at 19:7, hold is only 47.4%** (directional = 52.6%). The model would achieve higher accuracy by predicting directional more often, yet it predicts hold 99.7% of the time. This is NOT simple majority-class bias. The model's cross-entropy loss function penalizes wrong-direction predictions more than wrong-hold predictions (wrong direction loses the full target payout; wrong hold loses nothing). When the model cannot reliably distinguish directions, predicting hold is the minimum-loss strategy even when hold is the minority class.

### 3. Narrow-stop geometries are noise-swept
At 15:3 and 20:3, the 3-tick stop ($0.75) is within typical bid-ask bounce. Many stop-hit labels may be triggered by transient spread widening rather than genuine adverse moves, making the classification target noisier. Evidence: 15:3 and 20:3 have the highest hold rates (62%, 70%) despite the narrow stop — paradoxically, the narrow stop should trigger MORE often, but the combined barrier geometry produces high hold rates because the target is so wide that most bars don't reach it.

### 4. SC-S1 failure indicates a fundamentally different classification problem
Baseline 10:5 CPCV accuracy of 0.384 vs prior 0.449 (long-perspective labels) is a **6.5pp decline**. Both label types produce ~33% balanced classes, so the difference is in target semantics: bidirectional labels define direction by which barrier (long or short) triggers first within a time horizon, while long-perspective labels use return sign. The bidirectional target is harder because it couples direction with barrier reachability — the model must predict both "will the barrier be reached?" and "which direction?" jointly.

### 5. Selection bias inflates directional accuracy at high-ratio geometries
At 19:7, the model selects 3,280 of 1.16M bars (0.28%) for directional prediction. These are bars where the model is most confident. Directional accuracy on this highly selected subset (55.9%) is NOT representative of what would happen with broader trading. The holdout confirms this: when forced to evaluate the model's full prediction set, 96.8% of directional predictions hit hold bars and directional accuracy drops to exactly 50%.

### 6. Could 10:5 improve with class weighting or threshold optimization?
The 2.6pp gap to breakeven at 10:5 (50.7% vs 53.3%) is small enough that class-weighted training or prediction threshold tuning might bridge it. However, the xgb-tuning experiment showed the accuracy landscape is a 0.33pp plateau across 64 hyperparameter configs — fundamental accuracy gains are unlikely from XGBoost alone. The binding constraint is the feature-label correlation ceiling, not the model.

---

## What This Changes About Our Understanding

### Confirmed
1. **The time horizon fix works.** Hold rates dropped 29-58pp across all geometries. The 5-minute cap was the root cause of Phase 1's degenerate distributions. SC-8 PASS.
2. **Bidirectional labels at 10:5 with 3600s produce near-perfect class balance** (33.2%/32.6%/34.2%). This is a testable, non-degenerate classification problem.
3. **The oracle is profitable at all geometries at 3600s** ($3.22-$9.44/trade). The label design is sound — perfect information generates positive expectancy.
4. **Long:short symmetry holds** across all geometries (1.02-1.03:1). Bidirectional label computation is unbiased.

### New findings
5. **XGBoost directional accuracy on balanced bidirectional labels is ~50.7% at 10:5** — 2.6pp below the 53.3% breakeven. On long-perspective labels, the model achieved ~45% 3-class accuracy, which corresponds to ~50% directional accuracy (excluding hold predictions). The directional accuracy is essentially ~50-51% regardless of label type — the model is at the ceiling of what 20 microstructure features can predict about near-term direction.
6. **The model refuses to trade at high-ratio geometries** even when labels are near-balanced (47% hold at 19:7). This is not majority-class bias — it is genuine inability to learn the directional boundary at wider barriers, causing the model to adopt a hold-prediction strategy to minimize cross-entropy loss.
7. **Feature importance concentrates on volatility at wider barriers.** At 19:7, high_low_range_50 accounts for 43.5% of top-10 gain (vs 17% at 10:5). The model at wider barriers learns barrier reachability (volatility detection), not direction.
8. **Balanced bidirectional labels are harder than long-perspective labels.** 3-class accuracy drops 6.5pp (0.449→0.384) despite both having ~33% class balance. The bidirectional target couples direction with barrier reachability, making it a harder joint prediction problem.

### Revised understanding
- The model's directional accuracy is ~50-51% regardless of label type, geometry, or hyperparameters. This appears to be a hard ceiling of the 20-feature set on MES time_5s bars.
- The geometry hypothesis (lower breakeven via high ratio → convert marginal signal to profit) fails because the model cannot be induced to trade at high-ratio geometries. The favorable payoff structure is unexploitable with XGBoost on these features.
- The binding constraint is NOT label design, NOT hyperparameters (0.33pp plateau from xgb-tuning), NOT geometry — it is the feature-label correlation ceiling. The next step must either change the feature set, change the model's willingness to predict directional, or change the label formulation.

---

## Proposed Next Experiments

### 1. 2-Class Formulation: Directional vs Hold (highest priority)
Train XGBoost on a binary "will this bar be directional?" target at 19:7 labels (47.4% directional). If the model can predict which bars will be directional with >60% accuracy, a second-stage direction model (only on predicted-directional bars) could exploit the 19:7 payoff. This decouples barrier-reachability prediction from direction prediction — addressing the model's demonstrated strength (volatility/barrier detection) and weakness (direction).

### 2. Class-Weighted XGBoost at 19:7
Force the model to make directional predictions by up-weighting directional classes (e.g., 3:1 weight for +/-1 vs 0). Accept lower overall accuracy in exchange for higher directional prediction rate. The question: would forced directional predictions maintain >38.4% directional accuracy (19:7 breakeven)? Given 50.7% accuracy at 10:5 with active trading, this is plausible if the directional signal is geometry-invariant.

### 3. Long-Perspective Labels at Varied Geometries
The original P0 question remains open. Long-perspective labels (--legacy-labels) produce balanced distributions by design and showed ~45% 3-class / ~50% directional accuracy. Test at high-ratio geometries to determine if directional accuracy transfers when labels avoid the hold-prediction trap.

### 4. Feature Engineering for Wider Barriers
Current 20 features capture ~5s microstructure. Wider barriers (19-20 ticks) require multi-minute to multi-hour prediction. Candidates: rolling VWAP slope, cumulative order flow over longer windows (50-500 bars), intraday trend indicators, volatility regime markers. This is the highest-effort path but addresses the root cause (feature-label correlation ceiling).

---

## Program Status

- Questions answered this cycle: 1 (bidirectional labels + 1h horizon at high-ratio geometries — INCONCLUSIVE)
- New questions added this cycle: 1 (Does 2-class directional-vs-hold formulation enable high-ratio geometry exploitation?)
- Questions remaining (open, not blocked): 5
- Handoff required: NO

---

## Outcome Assessment

The metrics.json reports **Outcome A (CONFIRMED)**. After critical analysis:

**Arguments for Outcome A (as reported):**
- SC-3 and SC-4 numerically pass at 19:7
- SC-5 holdout passes

**Arguments for Outcome B (the correct practical interpretation):**
- The passing metrics are based on 0.28% directional prediction rate (unreliable)
- Holdout directional accuracy = 50.0% on 52 trades (coin flip)
- 96.8% of holdout directional predictions hit hold bars
- SC-S1 abort criterion was triggered but not enforced
- The model has a fixed directional signal (~50%) that doesn't change with geometry
- At the only geometry with real trade volume (10:5), expectancy is -$0.49/trade
- Infinity profit factors at 3 of 4 geometries are small-sample artifacts

**Overall verdict: INCONCLUSIVE, practically Outcome B.**

The time horizon fix succeeded (SC-8), the pipeline works (SC-1/2/6/7), but the economic hypothesis is neither cleanly confirmed (unreliable evidence at 19:7) nor cleanly refuted (SC-3/4/5 technically pass). The practical reality is Outcome B: "The model has a fixed directional signal that doesn't change with geometry. Next: 2-class formulation, class-weighted loss, or feature engineering." The favorable payoff structure at high-ratio geometries is real (oracle confirms it) but unexploitable with the current model and feature set.
