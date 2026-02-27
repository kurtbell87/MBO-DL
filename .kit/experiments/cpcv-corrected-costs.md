# Experiment: CPCV Validation at Corrected Costs

**Date:** 2026-02-27
**Priority:** P0 — statistical validation of the only positive-expectancy pipeline configuration
**Parent:** Class-Weighted Stage 1 (REFUTED, PR #37), Threshold Sweep (REFUTED, PR #36), PnL Realized Return (PR #35)
**Depends on:**
1. PnL realized return pipeline and results — DONE (PR #35)
2. Threshold sweep results (probability distribution analysis) — DONE (PR #36)
3. Class-weighted stage 1 (exhausted parameter-level interventions) — DONE (PR #37)
4. Label geometry 1h Parquet data (3600s horizon, 152-col) — DONE (PR #33)

**All prerequisites DONE. Same pipeline (w=1.0, T=0.50, 19:7), same data — validated under corrected costs with 45-split CPCV.**

---

## Context

Three experiments (threshold sweep, class weighting, hyperparameter tuning) have exhausted all parameter-level interventions on the two-stage pipeline. The baseline configuration (weight=1.0, T=0.50) at 19:7 geometry produces $0.90/trade on 3 walk-forward folds — but with CV=129% (Fold 2 outlier: $2.54 vs $0.01/$0.16). Three folds provide insufficient statistical power (t-stat ~1.35, p~0.31).

**The cost model was corrected (2026-02-27):** Prior experiments used inflated costs ($3.74 base). Corrected AMP volume-tiered costs are:

| Scenario | Old RT | Corrected RT | Description |
|----------|--------|-------------|-------------|
| Optimistic | $2.49 | **$1.24** | Limit orders, RTH — commission only ($0.62/side) |
| Base | $3.74 | **$2.49** | Market orders, RTH — commission + 1 tick spread crossing |
| Pessimistic | $6.25 | **$4.99** | Market orders, fast market — commission + spread + 1 tick slippage/side |

Under corrected-base costs ($2.49), the pipeline gains $1.25/trade in margin. The 3-fold WF expectancy shifts from $0.90 to ~$2.15/trade. **But 3 folds cannot validate this.** CPCV with 45 splits provides:
- Proper confidence interval (not just a point estimate from 3 data points)
- Probability of Backtest Overfitting (PBO) — critical for deployment decision
- Per-split stability analysis (is the edge real or driven by a single temporal regime?)
- t-test for mean > $0 with 44 degrees of freedom (vs 2 df from WF)

---

## Hypothesis

The two-stage XGBoost pipeline at 19:7 (w=1.0, T=0.50) achieves CPCV mean realized expectancy > $0.00 under corrected-base costs ($2.49 RT) with PBO < 0.50 across 45 CPCV splits.

**Direction:** Positive mean expectancy.
**Magnitude:** Expected ~$2.15/trade based on 3-fold WF result + $1.25 cost reduction. CPCV typically gives a more conservative estimate than WF due to averaging across more temporal configurations, but can also be optimistic due to bidirectional training. Prior: 50-65% probability of SC-1 passing.

---

## Independent Variables

### Cost model (primary IV — 3 levels, evaluated on identical predictions)

| Scenario | Commission/side | Spread | Slippage | Total RT |
|----------|----------------|--------|----------|----------|
| **Optimistic** (limits, RTH) | $0.62 | — | — | **$1.24** |
| **Base** (markets, RTH) | $0.62 | 1 tick | 0 | **$2.49** |
| **Pessimistic** (markets, fast) | $0.62 | 1 tick | 1 tick/side | **$4.99** |

### Pipeline configuration (FIXED — not swept)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Stage 1 weight | 1.0 | Optimal from class-weighted experiment (PR #37) |
| Stage 1 threshold | 0.50 | Optimal from threshold sweep (PR #36) |
| Geometry | 19:7 | Only viable geometry (10:5 definitively -$1.65) |
| Pipeline | Two-stage (reachability + direction) | Established architecture |

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| XGB params (both stages) | LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014 | Tuned params from XGB tuning experiment |
| Feature set | 20 non-spatial features | Identical to all prior experiments |
| Data source | `.kit/results/label-geometry-1h/geom_19_7/` (3600s Parquet, 152-col) | Same data as PR #34/#35/#36/#37 |
| PnL model | Realized return (from PR #35) | Corrected hold-bar treatment: forward-return PnL at horizon |
| Early stopping | 50 rounds, logloss, val = last 20% of training days | Standard protocol |
| n_estimators | 2000 (upper bound) | Same as prior |
| Seed | 42 (per-split: 42 + split_idx) | Reproducibility with per-split variation |

---

## CPCV Protocol

**N=10 groups, k=2 test groups per split, C(10,2) = 45 splits.**

### Development / Holdout Split

```
|<------- ~201 development days -------->|<-- ~50-day holdout -->|
|  Day 1                       Day ~201  | Day ~202      Day 251 |
|                                        |  NEVER TOUCH          |
|  CPCV: N=10 groups, k=2               |  UNTIL FINAL EVAL     |
|  45 splits                             |  (one-shot)           |
```

Holdout days are excluded from ALL CPCV groups. Exact day boundaries determined from the Parquet data.

### Group Assignment

Divide the ~201 development days into 10 sequential groups of ~20 days each:
- Group 1: days 1-20, Group 2: days 21-40, ..., Group 10: days 181-201
- Exact boundaries depend on actual day count in Parquet; groups should be as equal-sized as possible
- Group assignment is by COMPLETE DAYS — never split mid-day

### Purge and Embargo

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Purge window | 500 bars | Max triple barrier label span (timeout) |
| Embargo | 4,600 bars (~1 trading day) | Serial correlation in order flow and book state |

- Apply purge at EVERY train/test boundary (remove training bars within 500 bars of a test-group boundary)
- Apply embargo after each test-group boundary (exclude additional 4,600 bars from training)
- With 2 non-contiguous test groups per split, up to 4 boundaries → ~20,400 bars excluded (~2.8% of training set — negligible)

### Per-Split Protocol

For each of 45 splits:
1. Identify train groups (8 groups, ~160 days) and test groups (2 groups, ~40 days)
2. Apply purge and embargo between train and test
3. Within training fold: reserve last 20% of training days as internal validation (with purge + embargo between internal train/val)
4. Feature normalization: z-score using training fold statistics only (no leakage to test)
5. **Stage 1:** Binary `is_directional = (tb_label != 0)` on ALL training bars. XGBoost `binary:logistic`, tuned params, early stopping on internal val. Predict P(directional) on test bars.
6. **Stage 2:** Binary `is_long = (tb_label == 1)` on directional-only training bars (tb_label != 0). XGBoost `binary:logistic`, tuned params, early stopping on directional val bars. Predict P(long) on test bars.
7. **Combine:** If Stage 1 P(directional) > 0.50 → use Stage 2 direction (map P(long) > 0.50 → +1, else → -1). Else → predict 0 (no trade).
8. **PnL (realized-return model):**
   - Pred=0: $0 (no trade)
   - Pred != 0, true_label != 0, correct (pred sign = label sign): +19 × $1.25 - RT_cost
   - Pred != 0, true_label != 0, wrong (pred sign != label sign): -7 × $1.25 - RT_cost
   - Pred != 0, true_label = 0 (hold bar): fwd_return_ticks × $1.25 × sign(pred) - RT_cost
9. Compute PnL at ALL 3 cost levels ($1.24, $2.49, $4.99 RT)
10. Record: per-split accuracy, trade rate, hold fraction, realized expectancy (3 costs), dir-bar PnL, hold-bar PnL, hold-bar directional accuracy, predictions

---

## Baselines

| Baseline | Source | Value (old costs $3.74) | Value (corrected costs $2.49) |
|----------|--------|------------------------|-------------------------------|
| 3-fold WF mean expectancy | PR #35 | $0.90 | ~$2.15 |
| 3-fold WF per-fold | PR #35 | $0.01, $2.54, $0.16 | ~$1.26, $3.79, $1.41 |
| 3-fold WF trade rate | PR #35 | 85.18% | 85.18% (unchanged) |
| 3-fold WF hold fraction | PR #35 | 44.4% | 44.4% (unchanged) |
| 3-fold WF dir-bar PnL (gross) | PR #35 | $3.77 | $3.77 (unchanged) |
| 3-fold WF hold-bar PnL (net) | PR #35 | -$2.68 | -$1.43 (at corrected costs) |
| 3-fold WF per-fold CV | PR #35 | 129% | ~TBD |
| Break-even RT cost | PR #35 | $4.64 | $4.64 (unchanged — property of gross edge) |
| 3-class GBT CPCV (10:5) | e2e-cnn-classification | -$0.064 (base $3.74) | — (different geometry, different pipeline) |

---

## Metrics (ALL must be reported)

### Primary

1. **cpcv_mean_expectancy_base**: Mean per-trade expectancy across 45 CPCV splits at corrected-base costs ($2.49 RT). THE metric.
2. **cpcv_pbo**: Probability of Backtest Overfitting — fraction of 45 splits where the strategy produces negative expectancy at corrected-base costs. PBO < 0.50 means more profitable splits than not.
3. **cpcv_ci_95_lower**: Lower bound of 95% CI on mean expectancy (t-distribution, 44 df).
4. **holdout_expectancy_base**: Realized expectancy on ~50-day holdout at corrected-base costs (one-shot, reported last).

### Secondary

| Metric | Description |
|--------|-------------|
| cpcv_std_expectancy_base | Standard deviation of per-split expectancy at corrected-base |
| cpcv_fraction_positive_base | Fraction of 45 splits with positive expectancy at corrected-base |
| cpcv_mean_expectancy_optimistic | Mean expectancy at optimistic costs ($1.24 RT) |
| cpcv_mean_expectancy_pessimistic | Mean expectancy at pessimistic costs ($4.99 RT) |
| cpcv_t_stat | t-statistic for mean expectancy > $0 (one-sided, 44 df) |
| cpcv_p_value | p-value for mean expectancy > $0 |
| cpcv_mean_trade_rate | Mean trade rate across 45 splits |
| cpcv_mean_hold_fraction | Mean hold fraction across 45 splits |
| cpcv_mean_dir_bar_pnl | Mean directional-bar PnL (gross) across 45 splits |
| cpcv_mean_hold_bar_pnl | Mean hold-bar PnL (net at corrected-base) across 45 splits |
| per_group_expectancy | Per-group (1-10) mean expectancy when group is in test — temporal regime analysis |
| per_quarter_expectancy | Per-quarter (Q1-Q4) expectancy — map each test bar to its calendar quarter |
| worst_5_splits | 5 worst-performing splits: which group combinations, expectancy |
| best_5_splits | 5 best-performing splits: which group combinations, expectancy |
| wf_reproduction | 3-fold WF expectancy at both old-base ($3.74) and corrected-base ($2.49) |
| feature_importance_pooled | Top 10 features by gain pooled across 45 splits (Stage 1 and Stage 2 separately) |
| dir_accuracy_pooled | Directional accuracy pooled across all 45 test sets |
| profit_factor | Gross profit / gross loss pooled across all 45 splits |
| deflated_sharpe | Deflated Sharpe Ratio (DSR) corrected for trials |
| holdout_trade_rate | Trade rate on holdout |
| holdout_dir_accuracy | Directional accuracy on holdout |
| holdout_cost_sensitivity | Holdout expectancy at all 3 cost levels |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: 3-fold WF reproduces PR #35 | Exp $0.90 +/- $0.01 at old-base ($3.74) | Code bug — pipeline produces different predictions |
| SC-S2: Trade rate consistent across splits | Mean ~85% +/- 10pp across 45 splits | Data splitting or feature normalization issue |
| SC-S3: CPCV mean expectancy within 2x of WF mean | At corrected-base: within [$0, $4.30] | Structural divergence between CPCV and WF estimates |
| SC-S4: No single group dominates | No group appears in >80% of the top-10 splits by expectancy | Temporal regime overfitting — edge concentrated in one period |
| SC-S5: Holdout trade rate within 10pp of CPCV mean | Yes | Pipeline behavior changes on OOS data |

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: CPCV mean realized expectancy > $0.00 at corrected-base costs ($2.49 RT)
- [ ] **SC-2**: CPCV PBO < 0.50 (more than half of 45 splits are profitable at corrected-base)
- [ ] **SC-3**: 95% CI lower bound on mean expectancy > -$0.50 at corrected-base (not confidently negative)
- [ ] **SC-4**: CPCV fraction of positive splits > 0.60 at corrected-base (robust majority)
- [ ] **SC-5**: cpcv_mean_expectancy_pessimistic > -$1.00 (not catastrophic under worst-case costs)
- [ ] **SC-6**: Holdout realized expectancy > -$1.00 at corrected-base (not catastrophically negative on OOS)

---

## Minimum Viable Experiment

Run 3-fold walk-forward (identical to PR #35) at corrected costs. Verify:

1. Expectancy at old-base ($3.74) reproduces $0.90 +/- $0.01 (SC-S1)
2. Expectancy at corrected-base ($2.49) is ~$2.15 (= $0.90 + $1.25 cost reduction, applied per traded bar)
3. Trade rate matches 85.18% +/- 0.1pp
4. Hold fraction matches 44.4% +/- 0.1pp

**MVE pass:** Checks 1 and 3 pass. **ABORT if** baseline reproduction fails (exp off by >$0.05 or trade rate by >1pp). This catches data loading, feature normalization, or PnL computation bugs before scaling to 45 splits.

**Expected MVE wall-clock:** <60 seconds (6 XGB fits + PnL computation).

---

## Full Protocol

### Step 0: Adapt the Pipeline Script

Start from `.kit/results/pnl-realized-return/run_experiment.py`. Key adaptations:

1. **Add CPCV splitting logic:** Generate all C(10,2) = 45 split configurations from 10 sequential groups of development days
2. **Add purge/embargo:** 500-bar purge at train/test boundaries, 4,600-bar embargo after test boundaries
3. **Add internal validation split:** Last 20% of training days within each split (with purge/embargo to internal val)
4. **Run 3-fold WF first (MVE gate)** — reproduce PR #35, then proceed to CPCV
5. **Compute PnL at all 3 corrected cost levels** per split (post-hoc on identical predictions)
6. **Compute PBO, CI, t-test** from per-split expectancy distribution
7. **Pre-compute forward returns:** Compute the 720-bar forward return column ONCE for all hold bars (reused across splits, avoids redundant lookahead computation)
8. **Incremental result saving:** Write per-split results to disk as each split completes (no batch write at end)

### Step 1: 3-Fold Walk-Forward (MVE Gate)

Reproduce PR #35 at both old and corrected costs:

| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | 1-100 | 101-150 |
| 2 | 1-150 | 151-201 |
| 3 | 1-201 | 202-251 |

Verify reproduction at old-base ($3.74) before proceeding. Report corrected-base ($2.49) expectancy per fold.

### Step 2: 45-Split CPCV (Primary Analysis)

For each of 45 splits:
1. Apply purge and embargo
2. Split training fold into internal train (80%) and validation (20%)
3. Train Stage 1 + Stage 2 with early stopping on internal validation
4. Predict on test set (2 groups)
5. Combine predictions at T=0.50
6. Compute realized-return PnL at $1.24, $2.49, $4.99 RT costs
7. Record all per-split metrics
8. Save split result to disk immediately

### Step 3: Aggregate and Analyze

1. **Mean and std of expectancy** across 45 splits at each cost level
2. **95% CI** on mean expectancy (t-distribution with 44 df)
3. **t-test** — one-sided test for mean expectancy > $0
4. **PBO** — fraction of splits with negative expectancy at corrected-base
5. **Per-group analysis** — which temporal groups appear in the best/worst splits?
6. **Per-quarter mapping** — assign each test bar to its calendar quarter (based on trading date), report per-quarter expectancy
7. **Temporal stability** — is the edge concentrated in specific groups/quarters?
8. **Profit factor** — pooled across all test predictions
9. **DSR** — Deflated Sharpe corrected for number of trials (1 strategy, but 45 evaluations)
10. **Feature importance** — top 10 by gain, pooled across 45 splits (Stage 1 and Stage 2 separately)

### Step 4: Cost Sensitivity

Report mean expectancy at all 3 cost levels:

| Scenario | RT Cost | Mean Exp | Std | 95% CI | Frac Positive | PBO |
|----------|---------|----------|-----|--------|---------------|-----|
| Optimistic | $1.24 | ? | ? | ? | ? | ? |
| Base | $2.49 | ? | ? | ? | ? | ? |
| Pessimistic | $4.99 | ? | ? | ? | ? | ? |

Also report the **break-even RT cost** from the CPCV data: the cost at which mean CPCV expectancy = $0. Compare to the WF break-even ($4.64).

### Step 5: Holdout Evaluation (ONE SHOT)

1. Train Stage 1 + Stage 2 on ALL ~201 development days (with internal 80/20 val split for early stopping)
2. Evaluate on holdout days (~202-251)
3. Compute realized-return PnL at all 3 cost levels
4. Report: expectancy, trade rate, hold fraction, dir accuracy, dir-bar PnL, hold-bar PnL
5. **This is the final, definitive OOS result. No re-runs.**

Note: WF Fold 3 already tests on the holdout period (train days 1-201, test days 202-251). The holdout result should closely match WF Fold 3's expectancy ($0.16 at old-base, ~$1.41 at corrected-base). Any discrepancy indicates a data splitting or feature normalization difference.

### Step 6: Comparison — CPCV vs WF

| Metric | 3-Fold WF (corrected-base) | CPCV (corrected-base) | Delta |
|--------|---------------------------|----------------------|-------|
| Mean exp | ~$2.15 | ? | ? |
| Std exp | ~$1.16 (est.) | ? | ? |
| Frac positive | 3/3 = 100% | ? | ? |
| Trade rate | 85.2% | ? | ? |
| Hold fraction | 44.4% | ? | ? |
| Dir-bar PnL (gross) | $3.77 | ? | ? |
| Hold-bar PnL (net) | -$1.43 | ? | ? |
| Break-even RT | $4.64 | ? | ? |

---

## Resource Budget

**Tier:** Standard

- Max wall-clock time: 30 min
- Max training runs: 96 (45 CPCV splits × 2 stages + 3 WF folds × 2 stages = 96)
- Max seeds: 1 (seed=42, per-split: 42 + split_idx)
- COMPUTE_TARGET: local
- GPU hours: 0

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 96
parallelizable: true
memory_gb: 2
gpu_type: none
estimated_wall_hours: 0.35
```

### Wall-Time Estimation

| Phase | Work | Per-unit | Units | Subtotal |
|-------|------|----------|-------|----------|
| Data loading (Parquet → memory) | Read 19:7 Parquet, compute fwd returns | — | 1 | ~60s |
| Step 1: MVE (3-fold WF) | 6 XGB fits + PnL | ~2.5s | 6 | ~15s |
| Step 2: CPCV splits | 90 XGB fits (purge/embargo + train + predict + PnL) | ~5s | 90 | ~8 min |
| Step 3: Aggregation + PBO/CI | Numpy computation | — | 1 | ~30s |
| Step 4: Cost sensitivity | Re-evaluate at 2 additional cost levels | — | 1 | ~15s |
| Step 5: Holdout | 2 XGB fits + PnL | — | 2 | ~15s |
| Step 6: Analysis + reporting | Comparison tables, feature importance | — | 1 | ~2 min |
| **Total** | | | | **~13 min** |

Prior experiments: pnl-realized-return ran 14 XGB fits in 32s (~2.3s/fit). CPCV training sets are ~740K rows (vs WF's variable 460K-925K), so ~3-5s/fit is conservative. Stage 2 fits on ~390K directional-only rows: ~2-3s each.

---

## Abort Criteria

- **MVE fails:** STOP. 3-fold WF must reproduce $0.90 +/- $0.05 at old-base ($3.74). Wider tolerance than $0.01 to account for any minor numerical differences in the adapted script.
- **CPCV wall-clock > 30 min:** ABORT. Something wrong with split logic or data loading.
- **All 45 splits have identical expectancy (std < $0.01):** STOP. Splitting or purge/embargo logic is broken — all splits are seeing the same data.
- **Trade rate < 50% on >5 splits:** WARNING. Investigate — may indicate data gaps in some group combinations.
- **NaN in expectancy:** STOP. PnL computation bug.
- **Purge audit: any training bar overlaps test period label span:** STOP. Purge implementation is incorrect — results would be invalid.

---

## Decision Rules

```
OUTCOME A — SC-1 AND SC-2 AND SC-4 pass (mean exp > $0, PBO < 0.50, >60% positive):
  -> CONFIRMED. The two-stage pipeline at 19:7 is statistically validated as
     positive-expectancy under corrected costs.
  -> Record: mean exp, CI, PBO, fraction positive, per-quarter stability.
  -> If per-quarter analysis shows Q1-Q2 positive and Q3-Q4 negative:
     regime-conditional deployment is the next step.
  -> Next: Paper trading infrastructure (Rithmic R|API+ integration).
     Begin with 1 /MES contract, Q1/Q2 regime if quarterly analysis supports.

OUTCOME B — SC-1 passes but SC-2 or SC-4 fails (mean > $0 but <60% positive or PBO > 0.50):
  -> PARTIAL. Mean is positive but driven by a minority of high-return splits.
  -> The edge is concentrated in specific temporal regimes.
  -> Diagnose: per-group analysis to identify which regimes drive profitability.
  -> Next: Regime-conditional trading — trade only during identified profitable
     regimes. Requires out-of-sample regime identification.

OUTCOME C — SC-1 fails (mean exp ≤ $0 at corrected-base):
  -> REFUTED. The pipeline is not profitable even with corrected costs.
  -> The 3-fold WF $0.90/trade was driven by fold instability (Fold 2 outlier).
  -> Check: Does pessimistic-cost CPCV expectancy suggest the gross edge exists
     but costs consume it? If mean exp at optimistic ($1.24) is positive, the
     model has edge at limit-order execution costs.
  -> Next: Long-perspective labels (P0 open question — changes labeling scheme,
     not model parameters). Feature engineering for wider barriers. Or close the
     two-stage line entirely.
```

---

## Confounds to Watch For

1. **CPCV temporal mixing.** Unlike walk-forward, CPCV train sets include groups from both before AND after the test period. The model trains on "future" data relative to some test observations. This is standard for CPCV (it measures strategy robustness across temporal configurations, not strict forward predictability), but may produce optimistic estimates vs pure walk-forward. The 3-fold WF reproduction (Step 1) and holdout evaluation (Step 5) provide temporal-purity baselines for comparison. If CPCV mean is significantly higher than WF mean, temporal mixing is inflating the estimate.

2. **Group boundary effects.** With 500-bar purge and 4,600-bar embargo at each boundary, the effective training set is ~2.8% smaller than nominal. More importantly, bars at group boundaries may have systematically different characteristics (e.g., day transitions between groups). Verify that per-split accuracy and trade rate are stable across splits with different group boundaries.

3. **Volume horizon confound persists.** Hold-bar returns remain unbounded (-63 to +63 ticks, p10/p90 from PR #35) due to the 50K volume horizon truncating the barrier race. This affects all splits equally, so it doesn't bias the CPCV comparison vs WF. But the absolute expectancy of hold-bar trades may be distorted — the realized-return model treats hold bars as "hold to horizon" while a real trader would use stop-loss exits.

4. **Fold 2 correspondence.** The 3-fold WF Fold 2 (days 151-201) drives the positive WF mean ($2.54 of the $0.90 average). In CPCV, groups 8-10 (roughly days 141-201) correspond to the Fold 2 test period. If groups 8-10 appear disproportionately in the top CPCV splits, the same temporal regime is driving both results. This is NOT overfitting — it's regime dependence. But it means the strategy may only work in that specific market regime. The per-group analysis (Step 3.5) directly tests this.

5. **Simplified PBO.** True PBO (Bailey et al. 2016) compares multiple strategy candidates using a combinatorial search to measure the probability that the best-performing strategy is overfit. We have a single strategy, so PBO simplifies to the fraction of splits where the strategy underperforms $0 (no trading). This is an approximate lower bound on the full PBO. A more conservative interpretation: PBO < 0.50 means the strategy is profitable in the majority of temporal configurations.

6. **Cost correction uncertainty.** The corrected costs assume limit fills ($1.24 optimistic) and zero slippage at base ($2.49). In live trading, fill rates on limit orders are <100% (adverse selection), and market orders may experience >0.5 tick slippage during volatile periods. The break-even RT cost from CPCV data provides the margin of safety.

---

## Deliverables

```
.kit/results/cpcv-corrected-costs/
  metrics.json                     # All SC statuses + CPCV aggregate metrics
  analysis.md                      # CPCV results, per-group analysis, verdict
  run_experiment.py                # Adapted pipeline with CPCV splitting
  spec.md                          # Local copy of spec
  cpcv_per_split.csv               # 45 splits: expectancy at 3 cost levels, trade rate, hold frac, dir-bar PnL, hold-bar PnL
  per_group_analysis.csv           # Per-group (1-10) mean expectancy when in test set
  cost_sensitivity.csv             # 3 cost scenarios: aggregate CPCV metrics
  wf_reproduction.csv              # 3-fold WF results at both old and corrected costs
  holdout_results.csv              # Holdout: expectancy at 3 costs, trade rate, dir accuracy
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences: is the pipeline statistically validated?)
2. **MVE gate result** — 3-fold WF reproduction at old and corrected costs
3. **CPCV aggregate results** — mean, std, CI, t-stat, p-value at corrected-base
4. **PBO and fraction positive** — at all 3 cost levels
5. **Per-group analysis** — which temporal groups drive profitability?
6. **Per-quarter expectancy** — Q1-Q4 breakdown
7. **Worst/best 5 splits** — identify failure modes and success patterns
8. **Cost sensitivity table** — 3 scenarios x aggregate metrics + break-even RT
9. **CPCV vs WF comparison** — does CPCV agree with WF?
10. **Holdout results** — one-shot OOS at all 3 cost levels
11. **Feature importance** — top 10 pooled, Stage 1 vs Stage 2
12. **Explicit SC-1 through SC-6 pass/fail**
13. **Outcome verdict (A/B/C)**

---

## Exit Criteria

- [x] 3-fold WF reproduces PR #35 at old-base ($0.90 +/- $0.05) — $0.9013, delta $0.0013
- [x] 3-fold WF computed at corrected-base ($2.49) — $2.15 mean
- [x] CPCV 45 splits computed with purge (500 bars) and embargo (4,600 bars)
- [x] Per-split expectancy at 3 cost levels recorded — cpcv_per_split.csv
- [x] CPCV mean, std, CI, t-stat, p-value computed at corrected-base — $1.81, $1.18, [$1.46,$2.16], 10.29, 1.35e-13
- [x] PBO computed at all 3 cost levels — opt 2.2%, base 6.7%, pess 73.3%
- [x] Per-group analysis completed (which groups drive profitability) — groups 6-9 strongest
- [x] Per-quarter expectancy mapped — Q1 $1.49, Q2 $1.39, Q3 $2.18, Q4 $2.93
- [x] Worst/best 5 splits identified — worst: splits 32,3,17; best: splits 44,38,40
- [x] Cost sensitivity table (3 scenarios) with break-even RT — BE RT $4.30
- [x] CPCV vs WF comparison table — CPCV $0.34 lower than WF (expected)
- [x] Holdout evaluation completed (one-shot, all 3 cost levels) — $1.46 base
- [x] Feature importance reported (Stage 1 and Stage 2 separately) — message_rate dominates S1
- [x] All metrics in metrics.json
- [x] analysis.md with all required sections and verdict
- [x] SC-1 through SC-6 explicitly evaluated — ALL PASS

---

## Key References

- **PnL realized return script:** `.kit/results/pnl-realized-return/run_experiment.py` — pipeline code to adapt
- **Threshold sweep script:** `.kit/results/threshold-sweep/run_experiment.py` — threshold sweep reference (for P(dir) distribution)
- **PnL realized return results:** `.kit/results/pnl-realized-return/metrics.json` — 3-fold WF baseline at old costs
- **Class-weighted results:** `.kit/results/class-weighted-stage1/metrics.json` — exhausted weight parameter
- **Threshold sweep results:** `.kit/results/threshold-sweep/metrics.json` — P(dir) distribution at w=1.0
- **e2e-cnn-classification:** `.kit/results/e2e-cnn-classification/` — CPCV reference implementation (different pipeline, 10:5 geometry)
- **Parquet data:** `.kit/results/label-geometry-1h/geom_19_7/` (152-col, 3600s horizon, bidirectional labels)
- **PnL constants:** tick_value=$1.25, tick_size=0.25 (MES)
- **Corrected costs:** Optimistic $1.24, Base $2.49, Pessimistic $4.99 (AMP volume-tiered, $0.62/side)
- **Break-even RT:** $4.64 (from PR #35 — model gross edge per trade, invariant to cost scenario)
