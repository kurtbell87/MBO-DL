# Full Research Synthesis v2 — Strategic Pivot Assessment

**Date:** 2026-02-26
**Spec:** `.kit/experiments/synthesis.md`
**Experiments analyzed:** 23 (17 research + 6 infrastructure/engineering)
**Prior synthesis:** R6 (2026-02-17, CONDITIONAL GO) + SYNTHESIS.md (2026-02-24)

---

## 1. Executive Summary

**Verdict: GO** — with label geometry training as the primary justification.

The MBO-DL research program has completed 23 phases spanning 10 engineering and 17 research experiments. The central finding of this synthesis is a **strategic pivot**: the project was optimizing the wrong variable. Oracle net expectancy ($/trade) was used as the viability gate, but **breakeven win rate vs. model accuracy** is the actual binding constraint. At the current 10:5 geometry, breakeven WR is 53.3% — above the model's ~45% accuracy, guaranteeing negative expectancy. At 15:3 geometry, breakeven WR drops to 33.3% — 12pp below model accuracy, with positive PnL projections down to 35% directional accuracy. This lever has never been tested with actual model training.

| Key Number | Value | Source |
|-----------|-------|--------|
| Best model accuracy (CPCV) | 0.4504 | xgb-tuning |
| Best model expectancy (CPCV) | -$0.001/trade | xgb-tuning |
| Best model expectancy (WF) | -$0.140/trade | xgb-tuning |
| Breakeven WR at 10:5 | 53.3% | label-sensitivity |
| Breakeven WR at 15:3 | 33.3% | label-sensitivity |
| Breakeven WR at 20:3 | 29.6% | label-sensitivity |
| Oracle edge | $4.00/trade, WR 64.3% | R7 |
| Accuracy gap (10:5) | 45.0% - 53.3% = **-8.3pp** | xgb-tuning + label-sensitivity |
| Accuracy gap (15:3) | 45.0% - 33.3% = **+11.7pp** | xgb-tuning + label-sensitivity |
| Architecture | GBT-only on 20 features | e2e-cnn, xgb-tuning |
| Highest-priority next | Phase 1 label geometry training | This synthesis |
| Prior probability of success | 55-60% | Calibrated estimate (see T3) |

**Single highest-priority next experiment:** Phase 1 label geometry training — XGBoost at 4 geometries (10:5 control, 15:3, 19:7, 20:3) selected by breakeven WR diversity. Estimated compute: local CPU, 2-4 hours. This resolves the central unresolved question: does model accuracy transfer to high-ratio geometries?

---

## 2. Complete Evidence Matrix (23 Experiments x 5 Questions)

### Legend
- **++** Strong positive evidence for GO / narrows options / high-confidence conclusion
- **+** Moderate positive evidence
- **0** Neutral / infrastructure / no direct bearing
- **-** Negative evidence (for GO) / limitation identified
- **--** Strong negative evidence

| # | Experiment | Q1: Go/No-Go | Q2: Next Experiment | Q3: Architecture | Q4: Eliminated | Q5: Limitations |
|---|-----------|-------------|-------------------|-----------------|---------------|----------------|
| 1 | R1 subordination-test | 0 (time bars adequate) | Closes event-bar-first | time_5s canonical | Event bars for subordination | In-sample AR R² inflated |
| 2 | R2 info-decomposition | 0 (features sufficient) | Skip encoders | Book snapshot = sufficient statistic | Message encoder, temporal lookback | 500-event truncation |
| 3 | R3 book-encoder-bias | + (CNN R²=0.084 proper) | Led to hybrid chain | CNN best regression; non-viable classification | N/A (opened investigation) | Leakage: 0.132→0.084 (36%) |
| 4 | R4 temporal-predictability | 0 (closes temporal) | Redirect resources | Drop SSM/temporal | All temporal (time_5s) | 36 configs, thorough |
| 5 | R4b temporal-event-bars | 0 | Gap closure | Dollar_25k 140ms (non-actionable) | Temporal for vol/dollar bars | Dollar sub-second = HFT |
| 6 | R4c temporal-completion | 0 | Gap closure | No extended-horizon signal | Tick_50/100/250 temporal; h=200-1000 | Super-linear R² degradation |
| 7 | R4d temporal-actionable | 0 | Closes R4 permanently | Calibration table (10 thresholds) | Actionable-timescale temporal (7s-300s) | SC6 expected failure |
| 8 | R6 synthesis-v1 | + (CONDITIONAL GO) | Led to 9B hybrid | CNN+GBT hybrid (since revised) | Initial closed lines | Pre-tuning, pre-label-design |
| 9 | R7 oracle-expectancy | ++ (oracle $4.00/trade) | Confirms edge exists | TB > first-to-hit | N/A | 19-day subsample |
| 10 | 9B hybrid-model-training | - (CNN R²=-0.002) | Led to CNN debug chain | Pipeline broken | N/A (failure, not architecture) | Missing TICK_SIZE |
| 11 | 9C cnn-reproduction | 0 (diagnostic) | Narrowed to normalization | Protocol deviations NOT cause | "3 deviations" hypothesis | MVE gate saved compute |
| 12 | 9D pipeline-comparison | 0 (diagnostic) | Identified root cause | Data byte-identical | "Python vs C++ pipeline" hypothesis | Leakage quantified |
| 13 | 9E hybrid-corrected | - (exp -$0.37/trade) | Closes hybrid path | CNN denoiser, not viable | CNN+GBT hybrid for trading | 19-day subsample |
| 14 | R3b-original event-bar-cnn | 0 (voided) | Led to TB-Fix | Results void | Pre-fix tick bar results | Bar construction defect |
| 15 | R3b-genuine tick-bars | + weak (R²=0.124, p=0.21) | Not actionable alone | tick_100 inverted-U | N/A | Single-fold driven |
| 16 | Full-year-export | + (infrastructure) | Unblocks full-year experiments | 251d, 1.16M bars, 149 cols | N/A | 10/10 SC pass |
| 17 | E2E CNN classification | -- for CNN, + for GBT | Closes CNN; focus GBT | GBT 0.449, CNN 0.390 | CNN for classification | PBO=0.222; WF exp worse |
| 18 | XGB hyperparameter tuning | mixed (plateau + near-breakeven) | Tuning exhausted; geometry next | 0.33pp range; feature-binding | Hyperparameter tuning as lever | WF vs CPCV: -$0.140 vs -$0.001 |
| 19 | Label design sensitivity | ++ (BEV WR 33.3% at 15:3) | Phase 1 training = #1 priority | 123 geometries; BEV WR is correct metric | Oracle net exp as viability metric | Abort miscalibrated; accuracy transfer untested |
| 20 | Bidir reexport | 0 (infrastructure) | Unblocks bidir experiments | 152 columns | N/A | 312/312 files |
| 21 | TB-Fix | 0 (infrastructure) | Unblocks genuine tick bars | bar_feature_export counts trades | Pre-fix tick bar results | N/A |
| 22 | Oracle params CLI | 0 (infrastructure) | Unblocks geometry sweep | --target/--stop/--take-profit | N/A | N/A |
| 23 | Geometry CLI | 0 (infrastructure) | Unblocks per-geometry export | --target/--stop on bar_feature_export | N/A | N/A |

### Column Totals

| Question | Strong Positive | Moderate Positive | Neutral | Moderate Negative | Strong Negative |
|----------|----------------|-------------------|---------|-------------------|-----------------|
| Q1: Go/No-Go | 2 (R7, label-sensitivity) | 4 (R3, R6, R3b-g, FYE) | 13 | 2 (9B, 9E) | 1 (E2E CNN for CNN) |
| Q4: Eliminated | — | — | 8 (infrastructure/NA) | — | — |

The evidence matrix shows 6 positive-or-above vs. 3 negative-or-below on Q1. The positives are structurally stronger: R7 confirms exploitable edge exists, label-sensitivity reveals the correct optimization variable. The negatives (9B pipeline failure, 9E hybrid non-viable, CNN classification) all point to the same conclusion: CNN is not the path, GBT is. This is consistent with GO — the negatives narrow the approach, they don't close the project.

---

## 3. Strategic Pivot: Breakeven WR vs Oracle Ceiling

### Why the Project Was Optimizing the Wrong Variable

The label-design-sensitivity experiment mapped 123 valid triple barrier geometries across a (target=[8-20], stop=[2-12]) grid. The pre-committed abort criterion was "oracle net expectancy > $5.00/trade." Zero geometries passed at base costs ($3.74 RT). Peak: $4.126 at (16:10).

The abort criterion was **fundamentally miscalibrated** because it confused two distinct quantities:
1. **Oracle per-trade profit** (how much a perfect trader earns) — depends on geometry, favors moderate ratios (1.6:1 to 2.7:1)
2. **Model viability** (can an imperfect model profit?) — depends on breakeven WR vs. model accuracy, favors HIGH ratios (5:1+)

### Inverse Correlation Between Oracle Net Exp and Model Viability

| Geometry | Ratio | Oracle Net Exp | Rank (of 123) | BEV WR | Model at 45% |
|----------|-------|---------------|----------------|--------|--------------|
| 16:10 | 1.6:1 | $4.126 | 1st | 49.97% | **-$1.56/trade** |
| 18:8 | 2.25:1 | $3.962 | 2nd | 42.28% | **+$0.56/trade** |
| 19:7 | 2.71:1 | $3.878 | 3rd | 38.43% | **+$2.14/trade** |
| 15:3 | 5:1 | $1.744 | ~80th | 33.29% | **+$2.63/trade** |
| 20:3 | 6.67:1 | $1.920 | ~60th | 29.60% | **+$5.45/trade** |
| 10:5 (ctrl) | 2:1 | $2.747 | ~40th | 53.28% | **-$1.56/trade** |

The top-10 by oracle net exp (ratios 1.6:1-2.7:1) produce NEGATIVE model PnL because their breakeven WRs (42-50%) are at or above the model's ~45% accuracy. The geometries that produce POSITIVE model PnL (15:3, 20:3) are ranked 60th-80th by oracle net exp. The ranking variables are inversely correlated in the high-ratio region.

### Oracle Margin Is Geometry-Invariant

Oracle margin (WR - breakeven WR) across all top-10 geometries: 9.55pp to 11.71pp (range 2.16pp). The oracle captures a similar relative edge at ALL geometries. The model's challenge is not the oracle's edge — it's the absolute breakeven threshold.

### PnL Projections at Candidate Geometries

Using: Win = +(T × $1.25) - $3.74; Lose = -(S × $1.25) - $3.74.

| Geometry | BEV WR | @ 45% acc | @ 40% acc | @ 35% acc | @ 30% acc |
|----------|--------|-----------|-----------|-----------|-----------|
| 10:5 (ctrl) | 53.3% | -$1.56 | -$2.81 | -$4.06 | -$5.31 |
| 15:3 | 33.3% | **+$2.63** | +$1.38 | **+$0.13** | -$1.12 |
| 19:7 | 38.4% | **+$2.14** | +$0.73 | -$0.68 | -$2.09 |
| 20:3 | 29.6% | **+$5.45** | +$3.73 | +$2.01 | **+$0.29** |

At 15:3, the model remains profitable down to ~35% directional accuracy (10pp tolerance). At 20:3, profitable down to ~30% (15pp tolerance). The current 10:5 geometry requires >53% — explaining ALL prior negative expectancy results.

**Critical caveat:** These projections assume the model's ~45% 3-class accuracy transfers intact. This is the central unresolved hypothesis. See Tension T3.

---

## 4. Tension Resolutions (T1-T4)

### T1: What Is the Go/No-Go Status After All Evidence?

**Verdict: GO**
**Confidence: Moderate (55-60%)**

Decision rule applied:
> GO: At least one unexplored high-leverage intervention exists with >50% prior probability of closing the viability gap.

**Evidence for GO:**
- Oracle edge: $4.00/trade (R7), stable across all 4 quarters — exploitable edge exists
- At 15:3, model profitable down to 35% accuracy — 10pp tolerance below current 45%
- At 20:3, model profitable down to 30% — 15pp tolerance
- XGB accuracy plateau (0.33pp across 64 configs) suggests features determine accuracy, not labels — supports accuracy transfer
- CPCV expectancy -$0.001/trade at 10:5 (xgb-tuning) — knife-edge of breakeven before geometry change
- Q1-Q2 positive expectancy (+$0.137, +$0.058) — seasonal edge exists
- All infrastructure prerequisites complete — Phase 1 costs 2-4 hours
- Multiple independent levers (geometry, regime, cost reduction, class-weighting) are additive

**Evidence against GO:**
- Walk-forward expectancy -$0.140/trade — deployment estimate is 14x worse than CPCV
- Accuracy transfer is UNTESTED — 45% may not hold at 15:3 or 20:3
- Label distribution shifts at wider targets (more holds, fewer directional)
- Long recall 0.149 at 10:5 — may worsen at wider targets
- volatility_50 feature monopoly (49.7% gain share) — fragile model
- Single year of data (2022) — regime effects may not generalize

**Resolution:** The label geometry lever is the unexplored high-leverage intervention. The structural tolerance (10-15pp accuracy margin at 15:3/20:3) is large relative to the expected accuracy degradation. The XGB plateau evidence suggests features are the binding constraint on accuracy, not labels, supporting transfer. The intervention cost is low (2-4 hours compute). Prior probability of Phase 1 yielding positive expectancy at at least one geometry: **55-60%**, which satisfies the >50% GO criterion.

The walk-forward divergence (-$0.140 vs -$0.001) is the strongest counterargument. It means the CPCV near-breakeven result may be optimistic. However, the geometry lever changes the breakeven threshold by 20pp (53.3% → 33.3%) — an effect size that dwarfs the CPCV-WF divergence.

### T2: Was Optimizing Oracle Net Exp the Wrong Variable?

**Verdict: CONFIRMED**
**Confidence: High**

The label-design-sensitivity experiment provides direct evidence. The $5.00 oracle net exp abort criterion:
- Selected for geometries where the oracle makes the most absolute dollars (moderate ratios 1.6:1-2.7:1)
- These are exactly the geometries where the model LOSES money (BEV WR 42-50%, above model accuracy)
- The correct viability metric (breakeven WR) selects fundamentally different geometries (5:1+ ratios)

The oracle net exp ranking and model viability ranking are **inversely correlated** in the high-ratio region. This is structural: higher T:S ratios reduce per-trade dollar profit (wider targets hit less often) but reduce breakeven WR faster than they reduce oracle WR.

Oracle margin (WR - BEV WR) is remarkably stable at 10-12pp across ALL geometries. The oracle captures a similar fractional edge everywhere. Only the absolute breakeven threshold changes — and that determines model viability.

**Implication:** The prior synthesis (R6, CONDITIONAL GO) recommended CNN+GBT Hybrid based partly on oracle ceiling analysis. This was the wrong frame. GBT-only with geometry optimization is the correct path.

### T3: Can the Model's ~45% Accuracy Transfer to High-Ratio Geometries?

**Verdict: UNRESOLVED — estimated 55-60% probability of transfer**
**Confidence: Low (this is the central uncertainty)**

Evidence supporting transfer:
1. **XGB accuracy plateau** (xgb-tuning): 0.33pp range across 64 configs, std=0.0006. The feature set determines accuracy, not the learning algorithm or hyperparameters. If features are the binding constraint, changing labels should not dramatically alter accuracy.
2. **Oracle margin is geometry-invariant** (label-sensitivity): 10-12pp across all geometries. The signal-to-noise ratio for the oracle is similar across geometries.
3. **Trade counts remain adequate** (label-sensitivity): 15:3 has ~210 trades/day (3,990/19d), 20:3 has ~160/day. Sufficient for XGBoost learning.
4. **Wide tolerance**: At 15:3, profitable down to 35% (10pp drop tolerated). At 20:3, profitable down to 30% (15pp drop tolerated).

Evidence against transfer:
1. **Label distribution shift**: At wider targets, more trades exit via take-profit (20 ticks), session end, or expiry rather than target/stop. This produces more hold labels, fewer directional labels, and more imbalanced classes.
2. **Long recall already 0.149** (xgb-tuning): The model barely identifies longs at 10:5. Wider targets require larger bullish moves → long prediction may become even harder.
3. **Model may predict mostly holds**: At high-ratio geometries, hold becomes the majority class. The model may achieve reasonable accuracy by predicting holds — but take few directional trades, making the payoff structure irrelevant.
4. **Zero empirical data**: Accuracy transfer has never been measured. All projections are theoretical.

**Probability calibration:** The structural tolerance (10-15pp) is large. Even significant accuracy degradation (10pp) leaves the model profitable at 15:3. The accuracy plateau argument is the strongest positive signal — if features determine accuracy and features are geometry-invariant (they are: book snapshot features don't change with label construction), then accuracy should be approximately stable. The main risk is behavioral: the model may shift to hold-majority predictions, reducing trade count rather than accuracy. Assigning 55-60% probability.

### T4: Is the CNN Line Truly Closed?

**Verdict: CLOSED for classification. Regression signal acknowledged but non-actionable.**
**Confidence: High**

Evidence chain (5 independent experiments):
1. **E2E CNN classification** (e2e-cnn): Accuracy 0.390 vs GBT 0.449 — CNN is 5.9pp WORSE. Expectancy -$0.146 vs -$0.064. Three mechanisms identified: spatial signal encodes variance not boundaries, long recall 0.21, hold prediction dominance.
2. **9E hybrid-corrected**: Corrected CNN R²=0.089 (proper validation). End-to-end expectancy -$0.37/trade. Hybrid beats GBT-nobook (+$0.075) and GBT-book (+$0.013), but delta too small to flip sign.
3. **9B hybrid-training**: CNN R²=-0.002 (broken pipeline). Root cause: TICK_SIZE normalization, not architecture. But even after normalization fix (9E), expectancy remains negative.
4. **9D pipeline-comparison**: Proper-validation R²=0.084 (3rd independent reproduction). Confirms regression signal is real but ~36% smaller than initially reported (0.132 included leakage).
5. **R3b-genuine**: tick_100 CNN R²=0.124, but p=0.21, single-fold driven (fold 5 anomaly). Not actionable.

The CNN captures genuine spatial structure (R²=0.084, p<0.05) but this structure encodes return variance, not class boundaries. The 16-dim embedding learns a denoiser, not a classifier. No path exists to convert this regression signal into profitable trading decisions.

---

## 5. Architecture Status (Updated)

### CNN Line: CLOSED

| Evidence | Result | Experiment |
|----------|--------|------------|
| CNN vs GBT classification accuracy | CNN 5.9pp WORSE | e2e-cnn |
| CNN regression R² (proper) | 0.084 | 9D, 9E, R3b-genuine |
| CNN+GBT hybrid expectancy | -$0.37/trade | 9E |
| CNN end-to-end expectancy | -$0.146/trade | e2e-cnn |
| CNN penultimate layer function | Variance encoder, not class discriminator | e2e-cnn |

### XGB Hyperparameter Tuning: EXHAUSTED

| Evidence | Result | Experiment |
|----------|--------|------------|
| Accuracy range across 64 configs | 0.33pp (std=0.0006) | xgb-tuning |
| Best accuracy delta from default | +0.15pp | xgb-tuning |
| Expectancy delta from default | +$0.065/trade | xgb-tuning |
| Feature set binding? | Yes — same features, same ordering | xgb-tuning |

### GBT-Only on 20 Features: CANONICAL

| Component | Specification | Evidence |
|-----------|--------------|---------|
| Model | XGBoost, tuned params (LR=0.013, L2=6.6, depth=6) | xgb-tuning |
| Features | 20 hand-crafted (volatility_50 dominates at 49.7%) | xgb-tuning |
| Bar type | time_5s | R1, R4 chain |
| Input | Book snapshot at bar close | R2 |
| Temporal features | None (0/168+ passes) | R4 chain |
| Message features | None (0/40 passes) | R2 |
| Evaluation | CPCV (45 splits, 10-group, k=2) + walk-forward | e2e-cnn, xgb-tuning |
| Primary metric | Walk-forward expectancy (deployment-realistic) | xgb-tuning |

### Feature Set: BINDING CONSTRAINT

The 20-feature set has extracted approximately all information XGBoost can use. Evidence: hyperparameter surface is a plateau (0.33pp range). Improving the model requires either (a) new features, (b) different label construction (geometry), or (c) different evaluation framework (regime-conditional, class-weighted).

---

## 6. Closed Lines of Investigation (8 Lines with Evidence Chains)

### Line 1: Temporal Encoder
**Status:** PERMANENTLY CLOSED (highest confidence)
**Evidence:** R4, R4b, R4c, R4d — 0/168+ dual threshold passes across 7 bar types, timescales 0.14s-300s
**Detail:** MES 5s returns are a martingale difference sequence. All AR configs produce negative OOS R². Dollar_25k has sub-second (~140ms) positive AR R² — non-actionable HFT microstructure.

### Line 2: Message Encoder
**Status:** PERMANENTLY CLOSED (high confidence)
**Evidence:** R2 — 0/40 dual threshold passes. LSTM and Transformer both worse than plain book MLP.
**Detail:** Book snapshot at bar close is a sufficient statistic. Intra-bar message sequences add zero marginal information.

### Line 3: Event-Driven Bars Over Time Bars (Subordination Theory)
**Status:** PERMANENTLY CLOSED (high confidence)
**Evidence:** R1 (0/3 pairwise tests), R4b (volume_100 ≈ time_5s on all metrics)
**Detail:** Clark/Ane-Geman subordination theory is refuted for MES. Dollar bars have catastrophically non-Gaussian returns (JB=109M). time_5s is simplest and performs equivalently.

### Line 4: CNN for Classification
**Status:** PERMANENTLY CLOSED (high confidence)
**Evidence:** e2e-cnn-classification (5.9pp worse), 9E (expectancy -$0.37), 9B (pipeline failure), R3b-genuine (p=0.21)
**Detail:** CNN spatial signal (R²=0.084) encodes return variance, not class boundaries. No viable path from regression to classification.

### Line 5: CNN+GBT Hybrid for Trading
**Status:** PERMANENTLY CLOSED (high confidence)
**Evidence:** 9E (expectancy -$0.37/trade base), e2e-cnn (GBT-only superior by 5.9pp accuracy)
**Detail:** CNN 16-dim embedding acts as denoiser. Delta over GBT-only too small to flip sign under any cost assumption.

### Line 6: XGBoost Hyperparameter Tuning as Accuracy Lever
**Status:** PERMANENTLY CLOSED (high confidence)
**Evidence:** xgb-tuning — 0.33pp accuracy range across 64 configs, std=0.0006
**Detail:** The hyperparameter surface is a plateau. Default params near-optimal. Feature set is the binding constraint.

### Line 7: Oracle Net Expectancy as Viability Metric
**Status:** PERMANENTLY CLOSED (high confidence)
**Evidence:** label-design-sensitivity — inverse correlation between oracle net exp ranking and model viability in high-ratio region
**Detail:** Oracle net exp selects for moderate-ratio geometries (1.6-2.7:1) where the model LOSES money. Breakeven WR is the correct viability metric. The $5.00 abort criterion was fundamentally miscalibrated.

### Line 8: "Python vs C++ Pipeline" Hypothesis for CNN Failure
**Status:** PERMANENTLY CLOSED (very high confidence)
**Evidence:** 9D (byte-identical data, identity rate=1.0), 9C (protocol deviations not root cause)
**Detail:** R3 and 9B/9C loaded from the same C++ export. Root cause: missing TICK_SIZE division + test-as-validation leakage. Three independent reproductions confirm proper R²=0.084.

---

## 7. Open Questions (Ranked by Priority)

### Priority 1: Label Geometry Training (Phase 1)
**Question:** Does model accuracy transfer to high-ratio geometries, producing positive expectancy?
**Hypothesis:** XGBoost at 15:3 geometry achieves directional accuracy > breakeven WR (33.3%) + 2pp = 35.3%, and CPCV expectancy > $0.00 at base costs.
**Success criterion:** Positive CPCV expectancy at at least one of the 4 candidate geometries (10:5, 15:3, 19:7, 20:3).
**Compute:** Local CPU, 2-4 hours (4 geometries × 45 CPCV splits × ~15s/split).
**Prior probability of success:** 55-60%
**Justification:** 10-15pp accuracy tolerance at high-ratio geometries. Feature-binding argument (XGB plateau) supports accuracy stability. Low cost, maximum information value.

### Priority 2: Regime-Conditional Trading
**Question:** Does trading only in favorable regimes (Q1-Q2 or volatility-gated) produce positive expectancy?
**Hypothesis:** Regime-filtered model produces walk-forward expectancy > $0.00 on favorable-regime subsets.
**Success criterion:** Positive WF expectancy on regime-selected days, with regime identified ex-ante (not post-hoc).
**Compute:** Local CPU, 1-2 hours.
**Prior probability of success:** 40%
**Justification:** GBT Q1 (+$0.137) and Q2 (+$0.058) are positive. But: single year of data, risk of overfitting regime definition to 2022.

### Priority 3: 2-Class Short/Not-Short Reformulation
**Question:** Does binary classification focusing on the model's strong short-prediction capability produce positive expectancy?
**Hypothesis:** Short-only strategy with model short recall 0.634 achieves positive expectancy net of costs.
**Success criterion:** CPCV short-trade expectancy > $0.00.
**Compute:** Local CPU, 1-2 hours.
**Prior probability of success:** 45%
**Justification:** Model has strong short recall (0.634 vs long 0.149). Binary formulation simplifies the problem. Risk: short-only may not have enough trades or may inherit Q3-Q4 losses.

### Priority 4: Cost Reduction Investigation
**Question:** Can execution costs be reduced by >$0.63/RT through limit orders or maker rebates?
**Hypothesis:** Limit order execution at MES reduces effective RT cost to <$3.11.
**Success criterion:** Demonstrated or analytically justified cost reduction.
**Compute:** Minimal (analysis of existing MBO data).
**Prior probability of success:** 60%
**Justification:** Limit orders typically improve execution quality. But: fill rate uncertainty, adverse selection risk.

### Priority 5: Class-Weighted XGBoost with PnL-Aligned Loss
**Question:** Does asymmetric cost weighting improve expectancy beyond $0.10/trade?
**Hypothesis:** PnL-aligned sample weights (wrong long costs more than wrong short under 10:5) shift prediction distribution favorably.
**Success criterion:** CPCV expectancy improvement > $0.10/trade over tuned baseline.
**Compute:** Local CPU, 1-2 hours.
**Prior probability of success:** 35%
**Justification:** Expectancy is more sensitive to class distribution than accuracy (xgb-tuning finding). But: XGB plateau suggests limited room. The tuned model already implicitly suppresses longs via regularization.

---

## 8. Statistical Limitations (Updated)

### 8.1 Abort Criterion Miscalibration
**Source:** label-design-sensitivity
**Issue:** The $5.00 oracle net exp abort criterion was fundamentally miscalibrated. It measured the wrong variable (oracle ceiling instead of breakeven WR). This prevented Phase 1 training from running, which was the only way to answer the central viability question.
**Impact:** High — the experiment was designed to test geometry's effect on viability but was aborted before the test could run.
**Mitigation:** Phase 1 should not use oracle-based abort criteria. New abort: "Baseline (10:5) CPCV accuracy < 0.40 on bidirectional labels."

### 8.2 Walk-Forward vs. CPCV Expectancy Divergence
**Source:** xgb-tuning
**Issue:** CPCV expectancy -$0.001/trade vs. walk-forward -$0.140/trade — a $0.139 gap. CPCV's combinatorial splits allow training on future regime patterns. Walk-forward is more deployment-realistic.
**Impact:** Critical — the near-breakeven CPCV result is an optimistic bound. The true deployment expectancy likely falls between -$0.14 and -$0.001.
**Mitigation:** Report both CPCV and WF expectancy. Use WF as primary for deployment decisions. Use CPCV for model selection.

### 8.3 Long/Short Recall Asymmetry
**Source:** xgb-tuning, e2e-cnn
**Issue:** Long recall 0.149, short recall 0.634 (tuned XGB). The model is overwhelmingly short-biased. Under tuning, this asymmetry worsened (long recall dropped from 0.201 to 0.149).
**Impact:** Moderate — the model's "near-breakeven" CPCV expectancy is achieved partly by avoiding long trades (which lose more under 10:5 asymmetry). At different geometries, this pattern may shift.
**Mitigation:** Track per-class recall at each geometry in Phase 1. Consider 2-class reformulation if long recall collapses at high-ratio geometries.

### 8.4 Single-Year Regime Dependence
**Source:** e2e-cnn, xgb-tuning, R7
**Issue:** All experiments use 2022 MES data only. GBT's Q1-Q2 profitability and Q3-Q4 losses may be year-specific. The oracle's quarter stability ($3.16-$5.39) may not generalize.
**Impact:** High — the single largest threat to external validity. No amount of model improvement on 2022 data validates deployment in 2023+.
**Mitigation:** Multi-year data required before any deployment decision. Phase 1 geometry training is valid within 2022; deployment requires out-of-sample validation.

### 8.5 volatility_50 Feature Monopoly
**Source:** xgb-tuning
**Issue:** volatility_50 accounts for 49.7% of XGBoost gain share — just under the 50% sanity threshold. The model's performance is almost entirely determined by one feature's relationship to the label.
**Impact:** Moderate-high — if the volatility-label relationship is regime-dependent (which Q3-Q4 losses suggest), the model is fragile to volatility regime shifts.
**Mitigation:** Monitor volatility_50 gain share across geometries in Phase 1. If it increases above 60%, flag as structural risk.

### 8.6 Accuracy-Transfer Uncertainty
**Source:** This synthesis (T3)
**Issue:** The model's 45% accuracy at 10:5 has never been measured at high-ratio geometries (15:3, 20:3). Label distribution shifts (more holds) and the model's existing long-side weakness create genuine uncertainty about whether the payoff-structure improvement will translate to profitable trading.
**Impact:** Critical — this is the central unresolved question that determines GO vs NO-GO.
**Mitigation:** Phase 1 label geometry training. This is the single highest-priority experiment.

---

## 9. Recommendation

### Immediate Next Action: Phase 1 Label Geometry Training

**Experiment:** Train XGBoost at 4 triple barrier geometries (10:5, 15:3, 19:7, 20:3) using CPCV evaluation on full-year data (251 days, 1.16M bars).

**Rationale:** This is the highest information-value experiment in the program:
- Resolves the central unresolved question (accuracy transfer across geometries)
- Has the largest structural lever (20pp breakeven WR reduction)
- Costs 2-4 hours of local CPU time
- All infrastructure prerequisites are complete (bidirectional labels, geometry CLI, full-year data)
- 55-60% prior probability of success

**Protocol:**
1. Re-export data at 4 geometries using bar_feature_export --target T --stop S
2. Train tuned XGBoost (LR=0.013, L2=6.6, depth=6) on each geometry
3. Evaluate via CPCV (45 splits, 10-group, k=2) + walk-forward
4. Report: accuracy, expectancy, per-class recall, feature importance, per-quarter stability
5. Abort: Baseline (10:5) CPCV accuracy < 0.40

**Decision tree after Phase 1:**
- If positive expectancy at any geometry → Proceed to multi-year validation
- If accuracy drops >10pp at all geometries → Evaluate 2-class reformulation (Priority 3)
- If accuracy holds but expectancy still negative → Evaluate regime-conditional (Priority 2)
- If all paths exhausted → Close project (Priority 6)

---

## Appendix: Changes Since Prior Synthesis (2026-02-24)

| Item | Prior (2026-02-24) | Updated (2026-02-26) |
|------|-------------------|---------------------|
| Experiments analyzed | 17 | 23 |
| Verdict | GO (implicit, CONDITIONAL GO from R6) | **GO** (explicit, 55-60% prior) |
| Architecture | CNN+GBT Hybrid (R6) → GBT-only (2024-02-24) | **GBT-only confirmed** |
| Highest priority | XGB tuning (#1), label design (#2) | **Label geometry Phase 1 (#1)** — tuning is done |
| XGB tuning status | Not started | **DONE** — Outcome C (MARGINAL), plateau confirmed |
| Label design status | Not started | **Phase 0 done** — 123 geometries mapped, breakeven WR is key |
| Key insight | GBT $0.06/trade from breakeven | **Breakeven WR, not oracle ceiling, is the correct metric** |
| Accuracy gap | 45% - 53.3% = -8.3pp (at 10:5) | +11.7pp (at 15:3), +15.4pp (at 20:3) |
| Closed lines | 5 | **8** |
| Open questions | 4 | **5** (ranked with priors) |
| Walk-forward divergence | Not measured | **-$0.139 gap** (CPCV -$0.001 vs WF -$0.140) |
