# Experiment: 2-Class Directional — Two-Stage Reachability + Direction Pipeline

**Date:** 2026-02-26
**Priority:** P0 — highest priority from label-geometry-1h analysis (NEXT_STEPS §1)
**Parent:** Label Geometry 1h (INCONCLUSIVE — model refuses to trade at high-ratio geometries, PR #33)
**Depends on:**
1. Label geometry 1h re-exports at 19:7 and 10:5 (3600s horizon, 152-col schema) — DONE (PR #33)
2. XGBoost tuned params — DONE (XGB tuning experiment)
3. Bidirectional TB labels — DONE (PR #26, #27)

**All prerequisites DONE. No re-export needed — uses existing Parquet from label-geometry-1h.**

---

## Context

Label Geometry 1h (PR #33) revealed a fundamental failure mode: at 19:7 geometry, the 3-class XGBoost model becomes a near-total hold predictor (99.7% hold predictions) despite hold being the minority class (47.4%). The root cause: 3-class cross-entropy penalizes wrong-direction predictions more than wrong-hold predictions. When the model cannot distinguish direction at wider barriers, it minimizes loss by predicting hold — even when hold is the minority class.

**Evidence that the 2-class decomposition should work:**
- Feature importance at 19:7 concentrates on volatility: `high_low_range_50` gain 226.8 (3-5x other geometries). The model IS detecting barrier-reachability — it just can't use this for directional trading under 3-class loss.
- At 10:5 (where the model actively trades), directional accuracy is 50.67%. This exceeds 19:7's 38.4% breakeven WR by 12.3pp. If the direction signal transfers, the favorable 2.71:1 payoff converts it to positive expectancy.
- Oracle at 19:7/3600s: WR=61.81%, exp=$9.44/trade, ~614K directional bars. The edge exists.
- 19:7 class distribution is near-balanced for binary: 52.6% directional / 47.4% hold.

**Key economic insight:** Under the PnL model (hold-bar trades = $0), two-stage profitability depends ONLY on directional accuracy among truly-directional traded bars exceeding 38.4% (19:7 BEV WR). Stage 1 precision affects profit magnitude, not sign. This is testable.

---

## Hypothesis

A two-stage XGBoost pipeline (Stage 1: binary "directional vs hold" filter, Stage 2: binary "long vs short" on predicted-directional bars) at 19:7 geometry achieves walk-forward per-trade expectancy > $0.00 after base costs ($3.74 RT) with >10% trade rate. The 3-class model produces 0.28% trade rate at 19:7; the two-stage model will produce meaningful trade volume by removing the cross-entropy penalty for wrong-direction predictions from the reachability decision.

**Direction:** Positive per-trade expectancy at 19:7.
**Magnitude:** > $0.00 per trade (any positive edge given the 2.71:1 payoff is a meaningful result).

**Mechanism:** Stage 1 uses `binary:logistic` (no direction penalty) to classify bars as directional vs hold — directly targeting the volatility/reachability signal. Stage 2 uses `binary:logistic` on directional-only training bars to predict direction, where long:short is perfectly balanced (1.02:1 at 19:7). Combined: Stage 1 filters → Stage 2 assigns direction → PnL evaluated at 19:7 payoff.

---

## Independent Variables

### Pipeline architecture (primary IV)

| Pipeline | Stage 1 | Stage 2 | Combination Rule |
|----------|---------|---------|-----------------|
| **Two-stage** (this experiment) | binary:logistic on ALL bars (directional vs hold) | binary:logistic on directional-only train bars (long vs short) | Stage 1 predicts directional → Stage 2 assigns direction; Stage 1 predicts hold → predict 0 |
| **3-class baseline** (reference) | N/A — single multi:softprob | N/A | label-geometry-1h walk-forward results |

### Geometry (2 levels)

| Geometry | Hold Rate | Dir Rate | BEV WR | Payoff Ratio | Binary Stage 1 Balance | Role |
|----------|-----------|----------|--------|--------------|----------------------|------|
| **19:7** | 47.4% | 52.6% | 38.4% | 2.71:1 | Near-balanced | Primary |
| **10:5** | 32.6% | 67.4% | 53.3% | 2:1 | Directional-majority | Control — 3-class already trades here |

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| XGB params (both stages) | LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014 | Same tuned params; binary objective replaces multi:softprob |
| Feature set | 20 non-spatial features | Identical to all prior experiments |
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` and `geom_10_5/` (existing 3600s Parquet) | No re-export — relabel existing data |
| Walk-forward folds | 3 expanding-window (days 1-100/101-150, 1-150/151-201, 1-201/202-251) | Same as label-geometry-1h for direct comparison |
| Early stopping | 50 rounds, logloss, val = last 20% of training days | Standard protocol (applies to both stages) |
| n_estimators | 2000 (upper bound) | Same as prior |
| Seed | 42 | Reproducibility |
| Stage 1 threshold | 0.5 (hard classification) | Baseline — threshold tuning is a Standard-tier followup |

---

## Baselines

| Baseline | Source | 19:7 | 10:5 |
|----------|--------|------|------|
| 3-class WF expectancy | label-geometry-1h | +$6.42 (unreliable — 1,282 trades across 3 folds) | -$0.499 (reliable — ~434K trades) |
| 3-class WF dir pred rate | label-geometry-1h | 0.28% (218+1014+50 trades per fold) | 90.4% |
| 3-class WF dir accuracy | label-geometry-1h | 55.9% (unreliable — high variance) | 50.67% (reliable) |
| 3-class label0_hit_rate (holdout) | label-geometry-1h | 96.8% (52 of 1,627 dir preds hit dir bars) | 32.0% |
| 3-class CPCV expectancy | label-geometry-1h | +$5.68 (Inf PF — artifact on 0.28% trade rate) | -$0.490 |
| Majority-class binary accuracy | Class prior | 52.6% (predict all "directional") | 67.4% |

**The 3-class WF results at 19:7 are unreliable** due to 0.28% trade rate. The two-stage model's primary advantage is producing enough trades for reliable measurement.

---

## Metrics (ALL must be reported)

### Primary

1. **two_stage_wf_expectancy_19_7**: Walk-forward mean per-trade expectancy ($) at 19:7 under base costs ($3.74 RT). THE metric that tests the hypothesis.
2. **two_stage_dir_accuracy_19_7**: Directional accuracy among truly-directional traded bars (both pred ≠ 0 AND true label ≠ 0) at 19:7. Must exceed 38.4% BEV WR.
3. **two_stage_trade_rate_19_7**: Fraction of test bars where the two-stage model produces a directional prediction (pred ≠ 0). Must be >10% for reliable metrics.

### Secondary

| Metric | Description |
|--------|-------------|
| stage1_binary_accuracy | Binary accuracy (directional vs hold) per geometry per fold |
| stage1_precision_directional | P(truly directional \| predicted directional) per geometry |
| stage1_recall_directional | P(predicted directional \| truly directional) per geometry |
| stage2_binary_accuracy | Binary accuracy (long vs short) on truly-directional test bars per geometry |
| stage2_dir_acc_on_filtered | Stage 2 accuracy ONLY on Stage 1-filtered directional test bars (selection bias check) |
| stage1_feature_importance | Top 10 features by gain — expect volatility/range dominance |
| stage2_feature_importance | Top 10 features by gain — expect directional features (OFI, imbalance, spread) |
| two_stage_wf_expectancy_10_5 | Walk-forward expectancy at 10:5 (control) |
| two_stage_trade_rate_10_5 | Trade rate at 10:5 |
| two_stage_label0_hit_rate | Fraction of directional predictions hitting hold-labeled bars, per geometry |
| two_stage_daily_pnl | Mean daily PnL (total PnL / trading days) per geometry |
| per_fold_details | Per-fold trade count, expectancy, dir accuracy — check consistency across folds |
| comparison_3class_delta | Two-stage minus 3-class metrics (expectancy, trade rate, label0_hit_rate) per geometry |
| cost_sensitivity | Expectancy under optimistic ($2.49), base ($3.74), pessimistic ($6.25) RT costs |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: Stage 1 accuracy > majority-class baseline | 19:7: >52.6%, 10:5: >67.4% | Stage 1 adds no value over always-predict-directional |
| SC-S2: Stage 2 accuracy > 50% at 10:5 on directional bars | >50% on balanced long:short data | Direction model has no signal — pipeline broken |
| SC-S3: Per-fold trade count > 100 at 19:7 | Yes (10% of ~230K test bars = ~23K) | Too few trades — model still defaulting to hold |

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Two-stage WF mean per-trade expectancy > $0.00 at 19:7 (base costs $3.74 RT)
- [ ] **SC-2**: Two-stage directional accuracy > 45% on truly-directional traded bars at 19:7 (BEV WR is 38.4%; 45% gives 6.6pp margin for per-fold variance)
- [ ] **SC-3**: Two-stage trade rate > 10% at 19:7 (vs 3-class's 0.28% — confirms the decoupling produces real trade volume)
- [ ] **SC-4**: Two-stage label0_hit_rate < 50% at 19:7 (vs 3-class's 96.8% holdout — Stage 1 identifies truly-directional bars, not random bars)
- [ ] No regression on sanity checks

---

## Minimum Viable Experiment

Before the full protocol, validate on a single temporal split at 19:7:

1. **Data loading gate.** Load `.kit/results/label-geometry-1h/geom_19_7/` Parquet files. Assert: files exist, 152 columns, tb_label in {-1, 0, +1}, total rows ~1.16M. **ABORT if data not found** — run `orchestration-kit/tools/artifact-store hydrate` and retry.

2. **Stage 1 gate.** Split 80/20 by days (first ~201 days train, last ~50 days test). Create binary label: `is_directional = (tb_label != 0)`. Train XGBoost (`binary:logistic`, tuned params, early stopping 50 rounds). Assert:
   - Accuracy > 52.6% (majority-class baseline at 19:7)
   - Both classes predicted
   - **ABORT if accuracy ≤ 52.6%.** Stage 1 adds no value — 2-class decomposition not viable.

3. **Stage 2 gate.** Filter training data to directional bars only (tb_label ≠ 0, ~52.6% of bars). Create binary label: `is_long = (tb_label == 1)`. Train XGBoost (`binary:logistic`, same params). Assert:
   - Accuracy > 48% on directional test bars
   - **ABORT if accuracy ≤ 40%.** Direction model is broken.

4. **Combined gate.** Combine: Stage 1 predicts directional → use Stage 2 direction, else hold. Assert:
   - Trade rate > 5%
   - **ABORT if trade rate ≤ 5%.** Model still collapses to hold predictor even with binary decomposition.

Pass all 4 gates → proceed to full protocol.

---

## Full Protocol

### Step 1: Walk-Forward at 19:7 (Primary)

For each of 3 expanding-window folds:

| Fold | Train Days | Test Days | Approx Test Bars |
|------|-----------|-----------|-----------------|
| 1 | 1-100 | 101-150 | ~230K |
| 2 | 1-150 | 151-201 | ~235K |
| 3 | 1-201 | 202-251 | ~230K (holdout period) |

Per fold:
1. Load all Parquet for 19:7 geometry. Split into train/test by day index.
2. **Stage 1:** Create binary label `is_directional = (tb_label != 0)` on ALL bars. Train XGBoost (`binary:logistic`, tuned params, n_estimators=2000, early stopping 50 rounds on last 20% of training days by logloss). Predict P(directional) on all test bars. Hard threshold at 0.5.
3. **Stage 2:** Filter training data to directional bars only (tb_label ≠ 0). Create binary label `is_long = (tb_label == 1)`. Train XGBoost (`binary:logistic`, same params). Predict P(long) on all test bars (regardless of Stage 1 — compute predictions universally, filter at combination step). Hard threshold at 0.5 → map to {-1: short, +1: long}.
4. **Combine:** For each test bar: if Stage 1 predicts directional (P > 0.5) → use Stage 2 direction. If Stage 1 predicts hold → predict 0.
5. **PnL:** Evaluate combined prediction against original 3-class `tb_label` using the geometry-specific PnL model (below).
6. **Record:** Stage 1 accuracy/precision/recall for directional class, Stage 2 accuracy on ALL directional test bars AND on Stage 1-filtered directional test bars (selection bias check), combined expectancy, trade rate, directional accuracy, label0_hit_rate, feature importance (both stages), n_estimators used.

### Step 2: Walk-Forward at 10:5 (Control)

Same protocol as Step 1 using `.kit/results/label-geometry-1h/geom_10_5/` data. The 3-class model already trades 90.4% of bars at 10:5 — expect minimal improvement from two-stage. This validates the pipeline and checks for bugs.

### Step 3: Comparison Analysis

Construct comparison table (two-stage vs 3-class from label-geometry-1h walk-forward):

| Metric | 3-Class (19:7) | Two-Stage (19:7) | Delta | 3-Class (10:5) | Two-Stage (10:5) | Delta |
|--------|----------------|------------------|-------|----------------|------------------|-------|
| Trade rate | 0.28% | ? | ? | 90.4% | ? | ? |
| Dir accuracy | 55.9%* | ? | ? | 50.67% | ? | ? |
| Expectancy | +$6.42* | ? | ? | -$0.499 | ? | ? |
| label0_hit_rate | 47.9% | ? | ? | 29.4% | ? | ? |
| Daily PnL | — | ? | ? | — | ? | ? |

*Unreliable (0.28% trade rate)

Key diagnostics:
- Did trade rate increase by >30x at 19:7? (0.28% → >10%)
- Did directional accuracy hold above 38.4% BEV WR at 19:7?
- Does Stage 1 feature importance concentrate on volatility and Stage 2 on directional features?
- Is 10:5 unchanged? (validates no pipeline artifacts)

### Step 4: Cost Sensitivity

3 scenarios × 2 geometries:

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

### PnL Model (geometry-dependent — same as label-geometry-1h)

```
tick_value = $1.25 per tick (MES)

For geometry (target=T, stop=S):
  Correct directional call (pred sign = label sign, both nonzero):
    PnL = +(T × $1.25) - RT_cost
  Wrong directional call (pred sign ≠ label sign, both nonzero):
    PnL = -(S × $1.25) - RT_cost
  Predict 0 (hold): $0 (no trade)
  True label=0, model predicted ±1: $0 (conservative simplification)

Per-trade expectancy = total PnL / number of trades (bars where pred ≠ 0)
Daily PnL = total PnL / number of test trading days
```

Report label0_trade_fraction per geometry. Flag if >25% of directional predictions hit hold-labeled bars.

---

## Resource Budget

**Tier:** Quick

- Max wall-clock time: 15 min
- Max training runs: 14 (3 folds × 2 stages × 2 geometries + 2 MVE)
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
estimated_wall_hours: 0.2
```

### Wall-Time Estimation

| Phase | Work | Per-unit | Units | Subtotal |
|-------|------|----------|-------|----------|
| Data loading (Parquet → memory) | 2 geometries × 251 files | — | 502 | ~2 min |
| MVE gates (4 checks) | 2 XGB fits + evaluation | ~12s | 2 | ~1 min |
| Step 1: WF 19:7 | 3 folds × 2 stages | ~10s | 6 | ~1 min |
| Step 2: WF 10:5 | 3 folds × 2 stages | ~10s | 6 | ~1 min |
| Steps 3-4: Analysis + cost sensitivity + reporting | — | — | — | ~4 min |
| **Total** | | | | **~9 min** |

Stage 2 fits are faster (~6-8s) because training on ~52-67% of rows (directional-only).

---

## Abort Criteria

- **MVE Stage 1 failure (accuracy ≤ 52.6% at 19:7):** STOP. Binary model cannot beat majority-class prediction. 2-class decomposition is not viable for this feature set.
- **MVE Stage 2 failure (accuracy ≤ 40%):** STOP. Direction model fundamentally broken.
- **MVE combined trade rate ≤ 5%:** STOP. Two-stage model still collapses to hold predictor — no improvement over 3-class.
- **Data loading failure:** STOP. Run `orchestration-kit/tools/artifact-store hydrate` and retry.
- **NaN in predictions:** Skip fold, log warning. If >1 fold affected, ABORT.
- **Wall-clock > 30 min:** ABORT (3× Quick-tier budget). Report partial results.

---

## Decision Rules

```
OUTCOME A — SC-1 AND SC-3 pass (positive expectancy + meaningful trade volume):
  -> CONFIRMED. Two-stage pipeline is viable at 19:7.
  -> Record: per-trade expectancy, daily PnL, trade rate, directional accuracy.
  -> Next: Standard-tier CPCV evaluation (45 splits for confidence intervals),
           Stage 1 threshold tuning (vary P(directional) cutoff for precision/recall trade-off),
           class-weighted Stage 1 (boost directional recall).

OUTCOME B — SC-2 passes but SC-1 fails (adequate dir accuracy, negative economics):
  -> PARTIAL. Direction signal exists but two-stage filtering doesn't improve economics.
  -> Diagnose: Is label0_hit_rate still high? Is precision too low (too many hold-bar trades)?
  -> Next: Stage 1 threshold optimization (raise threshold → fewer but higher-precision trades).
           Alternatively: class-weighted 3-class at 19:7.

OUTCOME C — SC-2 fails (directional accuracy < 45% at 19:7):
  -> REFUTED. Direction prediction at 19:7 barriers is fundamentally harder than at 10:5.
  -> The wider barrier creates a different, harder classification problem.
  -> Next: Class-weighted 3-class XGBoost (force directional predictions without decomposition).
           Or: feature engineering for wider barriers (rolling VWAP, cumulative order flow).

OUTCOME D — SC-3 fails (trade rate < 10% despite two-stage):
  -> REFUTED. Even binary Stage 1 collapses to hold prediction at 19:7.
  -> The features genuinely cannot distinguish directional from hold bars at 19:7.
  -> Next: Feature engineering (longer-horizon features addressing root cause: feature-label
           correlation ceiling at wider barriers).
```

---

## Confounds to Watch For

1. **Stage 2 selection bias.** Stage 2 trains on ALL truly-directional bars but at test time predicts only on Stage 1-filtered bars. If Stage 1 systematically selects bars where direction is harder to predict (e.g., high-volatility bars with random direction), Stage 2 accuracy on filtered bars may be lower than on all directional bars. **Mitigation:** Report Stage 2 accuracy BOTH on all directional test bars AND on Stage 1-filtered directional test bars. Flag if they differ by >3pp.

2. **Hard threshold at 0.5 is arbitrary.** Stage 1's P(directional) threshold determines the trade-rate / precision trade-off. A lower threshold → more trades, lower precision; higher → fewer trades, higher precision. This experiment uses 0.5 as the baseline. If SC-1 fails narrowly, threshold tuning (Outcome B path) may recover it.

3. **Walk-forward has only 3 folds — limited statistical power.** Each metric is based on 3 data points. Appropriate for a Quick-tier gate check; if promising, CPCV (45 splits) provides proper confidence intervals.

4. **Same 20 features for both stages.** Stage 1 (reachability) and Stage 2 (direction) likely benefit from different feature subsets. Using the same 20 features is a conservative lower bound. If Stage 1 and Stage 2 feature importance diverge substantially (volatility vs directional features), this validates the decomposition AND motivates per-stage feature selection in a followup.

5. **XGB hyperparameters tuned for multi:softprob, not binary:logistic.** The tuned params may be suboptimal for binary classification. This is a lower bound — if the gate passes with suboptimal params, optimized params will be better.

6. **PnL model assigns $0 to hold-bar trades.** Entering a position on a hold-labeled bar has uncertain exit PnL — the PnL model's conservative simplification understates both risk and reward on these bars. `label0_hit_rate` monitors severity. If >50%, the PnL estimate is unreliable.

7. **Objective function confound.** Any improvement over 3-class could be from switching `multi:softprob` → `binary:logistic` (removing the multi-class penalty) rather than the two-stage decomposition. **Diagnostic:** Compare Stage 1 hold-detection accuracy (1 - hold recall) to the 3-class model's. At 19:7, 3-class hold recall = 0.997 → hold-detection = 0.003. If Stage 1 substantially improves this (even just to 20-30% directional recall), the decomposition has value beyond the objective change.

8. **10:5 control should be stable.** At 10:5, the 3-class model trades 90.4% of bars with 50.67% dir accuracy. Two-stage should produce similar results. If 10:5 metrics change dramatically (>5pp accuracy, >$0.20 expectancy), suspect a pipeline bug.

---

## Deliverables

```
.kit/results/2class-directional/
  metrics.json                     # All SC statuses + per-stage/per-geometry metrics
  analysis.md                      # Two-stage vs 3-class comparison, verdict, SC pass/fail
  run_experiment.py                # Experiment script (created by RUN phase)
  spec.md                          # Local copy of spec
  walkforward_results.csv          # Per-fold × per-geometry: both stages + combined metrics
  stage1_feature_importance.csv    # Stage 1 feature importance per geometry
  stage2_feature_importance.csv    # Stage 2 feature importance per geometry
  cost_sensitivity.csv             # 3 cost scenarios × 2 geometries
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **Two-stage vs 3-class comparison table** (19:7 and 10:5 × trade rate, dir accuracy, expectancy, label0_hit_rate) — the headline result
3. **Per-stage accuracy breakdown** (Stage 1: binary accuracy + precision/recall for directional class; Stage 2: direction accuracy)
4. **Selection bias check:** Stage 2 accuracy on ALL directional test bars vs Stage 1-filtered directional test bars — flag if >3pp difference
5. **Feature importance decomposition** — Stage 1 top-10 vs Stage 2 top-10 (key diagnostic: do stages learn complementary signals?)
6. **Per-fold consistency** (3 folds × 2 geometries — are results stable or driven by one fold?)
7. **Cost sensitivity table** (3 scenarios × 2 geometries)
8. **Explicit SC-1 through SC-4 pass/fail**
9. **Outcome verdict (A/B/C/D)**
10. **Trade rate comparison headline:** 3-class 0.28% → two-stage ?% at 19:7

---

## Exit Criteria

- [ ] MVE gates passed (Stage 1 accuracy > 52.6%, Stage 2 accuracy > 48%, combined trade rate > 5%)
- [ ] Walk-forward completed for 19:7 (primary) — 3 folds × 2 stages
- [ ] Walk-forward completed for 10:5 (control) — 3 folds × 2 stages
- [ ] Two-stage combined metrics: expectancy, dir accuracy, trade rate, label0_hit_rate per geometry
- [ ] Comparison to 3-class label-geometry-1h walk-forward results tabulated
- [ ] Feature importance reported for Stage 1 and Stage 2 at both geometries
- [ ] Cost sensitivity computed (3 scenarios × 2 geometries)
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with comparison tables and verdict
- [ ] Decision rule applied (Outcome A/B/C/D)
- [ ] SC-1 through SC-4 explicitly evaluated

---

## Key References

- **Label geometry 1h analysis:** `.kit/results/label-geometry-1h/analysis.md` — 3-class WF results, feature importance, hold-prediction root cause diagnosis
- **Label geometry 1h Parquet:** `.kit/results/label-geometry-1h/geom_19_7/` and `geom_10_5/` (251 files each, 152-col schema, 3600s horizon)
- **Label geometry 1h run_experiment.py:** `.kit/results/label-geometry-1h/run_experiment.py` — reference for data loading, feature extraction, walk-forward splits, PnL model
- **Tuned XGB params:** LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014
- **Class distributions (3600s):** 19:7 = 26.1%/47.4%/26.5% (short/hold/long), 10:5 = 33.2%/32.6%/34.2%
- **Breakeven WR formula:** `(stop × $1.25 + RT_cost) / ((target + stop) × $1.25)`. 19:7 = 38.4%, 10:5 = 53.3%
