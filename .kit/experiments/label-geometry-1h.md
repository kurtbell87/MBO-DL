# Experiment: Label Geometry 1h — Model Training with Corrected Time Horizon

**Date:** 2026-02-26
**Priority:** P0 — re-run of label-geometry-phase1 with root cause fixed
**Parent:** Label Geometry Phase 1 (REFUTED — 300s time horizon too short, PR #31), Time Horizon CLI (PR #32)
**Depends on:**
1. Time horizon CLI flags (`--max-time-horizon`, `--volume-horizon`) — DONE (PR #32)
2. bar_feature_export `--target`/`--stop` CLI flags — DONE (PR #28)
3. Bidirectional TB labels (152-col schema) — DONE (PR #26, #27)
4. Phase 0 oracle sweep (123 geometries) — DONE (`.kit/results/label-design-sensitivity/oracle/`)
5. XGBoost tuned params — DONE (XGB tuning experiment)

**All prerequisites DONE. Experiment is FULLY UNBLOCKED.**

---

## Context

Label Geometry Phase 1 (PR #31) was REFUTED because 90.7-98.9% of bars were labeled "hold" at ALL geometries. Root cause: `max_time_horizon_s` was hardcoded to 300 seconds (5 minutes). MES price rarely traverses 10+ tick barriers in 5 minutes outside opening volatility. The trading strategy holds positions for seconds to up to 1 hour.

PR #32 fixed this by:
- Adding `--max-time-horizon <seconds>` CLI flag (default 3600 = 1 hour)
- Adding `--volume-horizon <contracts>` CLI flag (default 50000 = effectively unlimited)
- Changing struct defaults: `max_time_horizon_s` 300→3600, `volume_horizon` 500→50000

This experiment re-runs the identical geometry sweep with `--max-time-horizon 3600`, which should produce meaningful label distributions where price has sufficient time to reach barriers.

**Why this is a new spec (not updating label-geometry-phase1.md):**
- Phase 1 is committed with PR #31, has logged results, and RESEARCH_LOG entry
- The root cause fix (PR #32) changes the export CLI invocation
- Results go to a new directory to preserve the audit trail

**Lesson from Phase 1 analysis (CRITICAL):** Phase 1's SC-3 ("CPCV accuracy > BEV WR + 2pp") was trivially satisfied by predicting hold on a 90.7% hold dataset (89.98% overall accuracy). The directional accuracy — among bars where both prediction and true label are nonzero — was only 53.65%, just 0.37pp above breakeven. This spec redefines SC-3 to use **directional accuracy**, the metric that actually tests the hypothesis.

---

## Hypothesis

XGBoost trained at high-ratio geometries (15:3, 19:7, 20:3) with a 1-hour time horizon achieves CPCV **directional accuracy** > breakeven WR + 2pp at at least one geometry, producing positive per-trade expectancy after base costs ($3.74 RT).

**Mechanism:** At high target:stop ratios, the breakeven WR drops dramatically (33.3% at 15:3 vs 53.3% at 10:5). With a 1-hour forward-looking window (vs the prior 5-minute cap), the triple barrier labels should reflect realistic hold durations. If the model's directional signal (~45% accuracy on long-perspective labels) transfers to the corrected bidirectional labels, the favorable payoff structure converts that signal into positive expectancy.

**Key uncertainty:** Label distributions under 1-hour horizon have never been measured. The hold fraction should drop substantially from 90%+, but the exact distribution is unknown. If holds remain dominant (>80%) even at 3600s, the geometry hypothesis is genuinely refuted (not just untestable).

---

## Independent Variables

### 4 Geometries (same as Phase 1 — selected by breakeven WR diversity)

| Geometry | Target:Stop | Ratio | BEV WR | Model @ 45% Acc (theoretical) | Oracle (Phase 0, 300s)* |
|----------|------------|-------|--------|-------------------------------|-------------------------|
| 10:5 (control) | 10 ticks : 5 ticks | 2:1 | 53.3% | -$1.56/trade | $2.747/trade |
| 15:3 | 15 ticks : 3 ticks | 5:1 | 33.3% | +$2.63/trade | $1.744/trade |
| 19:7 | 19 ticks : 7 ticks | 2.71:1 | 38.4% | +$2.14/trade | $3.878/trade |
| 20:3 | 20 ticks : 3 ticks | 6.67:1 | 29.6% | +$5.45/trade | ~$1.920/trade |

*Phase 0 oracle numbers were computed at 300s time horizon (old default). With 3600s, more trades will reach barriers → expect HIGHER oracle WR and potentially different net expectancy. Step 0 of the protocol measures the actual 3600s oracle as the reference baseline.

### 1 Time Horizon Variable

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| `--max-time-horizon` | 300 (5 min) | 3600 (1 hour) | Match intended hold duration |
| `--volume-horizon` | 500 | 50000 | Effectively unlimited for MES |

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Model | XGBoost with tuned params (LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014) | Best available from XGB tuning experiment |
| Feature set | 20 non-spatial features (same as tuning experiment) | Isolate label design effect |
| Bar type | time_5s | Locked since R1/R6 |
| Label type | Bidirectional triple barrier (compute_bidirectional_tb_label) | Default labels with corrected time horizon |
| Time horizon | 3600 seconds (1 hour) | Matches intended trading strategy hold time |
| Volume horizon | 50000 contracts | Effectively unlimited — time horizon is the binding constraint |
| n_estimators | 2000 (upper bound, early stopping determines actual) | Same as tuned XGB protocol |
| Seed | 42 | Reproducibility |
| Data source | Raw .dbn.zst for C++ label/feature computation; C++-produced Parquet for Python model training | C++ is canonical source (CLAUDE.md rule #8) |

---

## Baselines

| Baseline | Source | Value | Notes |
|----------|--------|-------|-------|
| 10:5 CPCV accuracy (tuned, long-perspective) | xgb-tuning | 0.4504 | Long-perspective labels — reference only |
| 10:5 CPCV expectancy (tuned, long-perspective) | xgb-tuning | -$0.001/trade | Long-perspective labels — reference only |
| 10:5 WF expectancy (tuned, long-perspective) | xgb-tuning | -$0.140/trade | Deployment-realistic estimate |
| 10:5 oracle net exp (bidirectional, 19d, 300s) | label-sensitivity | $2.747/trade | From Phase 0 oracle sweep |
| 10:5 oracle WR (bidirectional, 19d, 300s) | label-sensitivity | 64.29% | Reproduces R7 exactly |
| Phase 1 hold rates (all, 300s horizon) | label-geom-p1 | 90.7-98.9% | 10:5=90.7%, 15:3=97.1%, 19:7=98.6%, 20:3=98.9% |
| Phase 1 directional accuracy (10:5, holdout) | label-geom-p1 | 53.65% | Only 0.37pp above BEV WR 53.28% |
| Phase 1 volatility dominance (10:5) | label-geom-p1 | 76% of top-10 gain | Model learned "will barrier trigger?" not direction |

**The 10:5 control trained IN THIS EXPERIMENT (with 3600s horizon) is the proper baseline — not historical numbers.** Prior baselines are reference points only.

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

1. **directional_accuracy_per_geometry**: CPCV mean directional accuracy for each of the 4 geometries. Directional accuracy = correct directional predictions / total bars where BOTH prediction ≠ 0 AND true label ≠ 0. **This is the metric that tests the hypothesis — NOT overall 3-class accuracy.** Phase 1 showed overall accuracy is trivially inflated by hold-prediction on hold-dominated data.
2. **cpcv_expectancy_per_geometry**: CPCV mean per-trade expectancy ($) under base costs for each geometry
3. **walkforward_expectancy_per_geometry**: Walk-forward mean per-trade expectancy ($) for each geometry

### Secondary

| Metric | Description |
|--------|-------------|
| overall_accuracy_per_geometry | Standard 3-class CPCV accuracy (reported for reference, NOT used in SC-3) |
| breakeven_margin_per_geometry | (directional accuracy - breakeven WR) for each geometry |
| class_distribution_per_geometry | -1/0/+1 fractions for each of the 4 geometries |
| hold_rate_comparison | Hold fraction at 3600s vs Phase 1's 90.7-98.9% at 300s — the KEY diagnostic |
| per_class_recall_per_geometry | Long/flat/short recall for each geometry |
| directional_prediction_rate | Fraction of bars where model predicts ≠ 0, per geometry |
| per_direction_oracle | Long and short oracle WR/expectancy for 4 geometries at 3600s |
| both_triggered_rate | Fraction of bars where both long and short races trigger |
| time_of_day_breakdown | Oracle/model metrics for Bands A (09:30-10:00), B (10:00-15:00), C (15:00-15:30) |
| profit_factor_per_geometry | PF for each geometry |
| feature_importance_shift | Top 10 features by gain: each geometry vs baseline (10:5). Note whether volatility dominance persists (Phase 1: 76% of top-10) or decreases (indicating model learned direction). |
| cost_sensitivity | Expectancy under optimistic ($2.49), base ($3.74), pessimistic ($6.25) RT costs |
| long_recall_vs_short | Long (+1) vs short (-1) recall asymmetry per geometry |
| holdout_expectancy | Holdout expectancy for best geometry + baseline |
| label0_trade_fraction | Fraction of directional predictions hitting hold-labeled bars, per geometry. Flag if >25%. |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: Baseline (10:5) CPCV overall accuracy > 0.40 | Yes | Pipeline is broken |
| SC-S2: No single feature > 60% gain share | Yes | Degenerate model |
| SC-S3: Re-export produces 152 columns | Yes | Export pipeline broken |
| SC-S4: Label distribution check (no >95% single class) | Yes | Time horizon still insufficient at that geometry |
| SC-S5: Hold rate at 10:5 < 80% (vs 90.7% at 300s) | Yes | 3600s horizon not sufficient — deeper issue |
| SC-S6: Oracle net exp > $0 at 3600s for all 4 geometries | Yes | Label computation bug — perfect foresight must profit |

---

## Minimum Viable Experiment

Before the full protocol, validate the tool chain end-to-end with the new time horizon:

1. **Oracle CLI gate.** Run `oracle_expectancy --target 10 --stop 5 --max-time-horizon 3600 --volume-horizon 50000 --output /tmp/test_oracle_3600s.json` on 1 .dbn.zst file. Assert:
   - JSON output is valid and contains: total_trades, win_rate, expectancy, profit_factor
   - Oracle net expectancy > $0 (perfect foresight with more time must profit)
   - Trade count > 0
   - **ABORT if oracle_expectancy fails, produces empty output, or shows non-positive oracle expectancy.**

2. **Export CLI gate.** Run `bar_feature_export --bar-type time --bar-param 5 --target 15 --stop 3 --max-time-horizon 3600 --volume-horizon 50000 --output /tmp/test_export_1h.parquet` on 1 .dbn.zst file. Assert:
   - Parquet output has 152 columns (bidirectional schema)
   - tb_label column exists with values in {-1, 0, +1}
   - Row count in [3000, 6000]
   - **Hold rate < 80%** (key validation: 3600s horizon should produce more directional labels than Phase 1's 90.7% at 300s)
   - **ABORT if hold rate > 80%.** This means 3600s is still insufficient — deeper investigation needed before spending budget on full export.

3. **Training pipeline gate.** Load the 1-day test Parquet. Extract 20 features + tb_label. Train XGBoost with tuned params on an 80/20 split. Assert:
   - No NaN in features or predictions
   - Accuracy > 0.33 (above random)
   - Per-class predictions include all 3 classes

Pass all 3 gates -> proceed to full protocol.

---

## Full Protocol

### Step 0: Oracle Re-Validation at 3600s Horizon

Phase 0 oracle data (`.kit/results/label-design-sensitivity/oracle/`) was computed at 300s time horizon. With 3600s, more trades will reach barriers, changing oracle metrics. Establish updated oracle baselines before training.

For each of 4 geometries, run `oracle_expectancy --target T --stop S --max-time-horizon 3600 --volume-horizon 50000` on a **20-day stratified subsample** (5 per quarter, same days as Phase 0 for consistency):
- Output: `.kit/results/label-geometry-1h/oracle/oracle_T_S_3600s.json`
- 4 geometries × 20 days = 80 C++ runs. With 8-way parallelism: ~2 min
- Record per-geometry: oracle WR, net expectancy (after $3.74 RT cost), trade count, hold rate

**Key validation:** Oracle net exp > $0 at ALL 4 geometries (perfect foresight with more time should profit at least as well as 300s). If any geometry shows oracle net exp < $0, STOP and investigate — likely a label computation bug.

**Log the hold rate comparison** — this is the primary diagnostic for whether 3600s fixes the degenerate distributions:

| Geometry | Hold rate (300s, Phase 1) | Hold rate (3600s, Step 0) | Δ |
|----------|--------------------------|---------------------------|---|
| 10:5 | 90.7% | ? | ? |
| 15:3 | 97.1% | ? | ? |
| 19:7 | 98.6% | ? | ? |
| 20:3 | 98.9% | ? | ? |

### Step 1: Re-Export at 4 Geometries with 1-Hour Horizon

Re-export 251 RTH days at all 4 geometries via `bar_feature_export --target T --stop S --max-time-horizon 3600 --volume-horizon 50000`:
- 4 geometries × 251 days = 1,004 export runs
- Output: `.kit/results/label-geometry-1h/geom_T_S/*.parquet`
- Parallelize at 8+ workers
- Each file ~1MB (zstd-compressed Parquet), total ~1 GB across all 4 geometries
- **Log hold rate per geometry** — compare to Phase 1's 90.7-98.9%
- **Note:** 3600s forward scan is up to 12× longer than 300s (720 bars vs 60 bars lookahead per bar). Per-file export time will increase — estimate ~5-15s vs prior ~2s. Many bars will hit barriers before 3600s, so the actual increase is less than 12×. With 8-way parallelism: ~10-20 min total (vs prior ~4 min at 300s).

### Step 2: Train XGBoost CPCV

For each of 4 geometries:
- Load all 251-day Parquet for that geometry
- CPCV (N=10, k=2, 45 splits) with tuned XGB params
- 4 × 45 = 180 fits (~36 min estimated)
- **Report BOTH overall accuracy AND directional accuracy per geometry per split**
- Directional accuracy = (correct directional predictions) / (total bars where both pred ≠ 0 AND label ≠ 0)
- If any geometry has >95% hold after full export, SKIP training for that geometry (same as Phase 1)

### Step 3: Walk-Forward Evaluation

For each of 4 geometries:
- 3 expanding-window folds
- 4 × 3 = 12 fits (~2 min estimated)
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

- [ ] **SC-1**: Re-export completed for all 4 geometries (4 × 251 = 1,004 files)
- [ ] **SC-2**: CPCV + walk-forward evaluation completed for all 4 geometries
- [ ] **SC-3**: At least one geometry CPCV **directional accuracy** > BEV WR + 2pp. Directional accuracy = win rate among trades where both prediction and label are directional (both ≠ 0). This corrects Phase 1's flawed SC-3 that was trivially satisfied by hold-prediction.
- [ ] **SC-4**: At least one geometry CPCV expectancy > $0.00 (base costs)
- [ ] **SC-5**: Best geometry holdout expectancy > -$0.10
- [ ] **SC-6**: Walk-forward expectancy reported for all 4 geometries
- [ ] **SC-7**: Per-direction + time-of-day analysis reported
- [ ] **SC-8**: Hold rate at 10:5 < 80% (confirms time horizon fix worked)

---

## Abort Criteria

- **MVE gate failure:** STOP. Diagnose before proceeding.
- **Baseline (10:5) CPCV overall accuracy < 0.40:** STOP. Pipeline broken.
- **Hold rate > 80% at 10:5 with 3600s horizon:** STOP. Time horizon is not the root cause — deeper investigation needed. Triggers Outcome D.
- **Degenerate labels:** Any geometry produces >95% one class → skip that geometry, log warning.
- **Per-fit time > 120s:** Investigate. Expected ~12-20s. Slight increase over Phase 1 is acceptable if class distribution changed (different tree structures). Only abort if systematic (>50% of fits exceed 120s).
- **NaN loss:** Any XGB fit produces NaN → skip, log warning. If >10% NaN, ABORT.
- **Wall-clock > 4 hours:** Abort remaining, report partial results. (~3× estimated 80 min, per 3-5× guidance in research-engineering-practices.md.)
- **NO oracle net exp gate.** Phase 0 is already done. We proceed directly to training.

---

## Decision Rules

```
OUTCOME A — SC-3 AND SC-4 pass:
  -> CONFIRMED. A viable geometry exists.
  -> Record: best geometry, directional accuracy margin over breakeven, expectancy, per-class recall.
  -> Next: Regime-conditional experiment + multi-year data validation.

OUTCOME B — SC-3/SC-4 fail, directional accuracy stable across geometries:
  -> PARTIAL. Payoff structure insufficient at current directional accuracy.
  -> The model has a fixed directional signal that doesn't change with geometry.
  -> Next: 2-class formulation (short/no-short), class-weighted loss, or feature engineering.

OUTCOME C — Directional accuracy drops >10pp at all non-baseline geometries:
  -> REFUTED. Label distribution change breaks the model.
  -> The wider-target classification problem is fundamentally harder.
  -> Next: Per-direction asymmetric strategy (different geometry for longs vs shorts).

OUTCOME D — Hold rate still >80% at all geometries even with 3600s:
  -> REFUTED. Time horizon is not the root cause of label degeneracy.
  -> Next: Long-perspective labels (--legacy-labels), longer bar types, or 2-class formulation.
```

---

## Resource Budget

**Tier:** Standard

- Max wall-clock time: 4 hours
- Max training runs: ~194 (180 CPCV + 12 walk-forward + 2 holdout)
- Max export runs: 1,004 (4 geometries × 251 days)
- Max oracle runs: 80 (4 geometries × 20 days)
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
| MVE (oracle + export + train gates) | 3 gates | — | 3 | ~5 min |
| Step 0: Oracle re-validation (4 geom × 20d) | 80 C++ runs | ~10s | 80 | ~2 min (8-way parallel) |
| Re-export: 4 geometries × 251 days | 1,004 C++ runs | ~5-15s* | 1,004 | ~15 min (8-way parallel) |
| CPCV training: 4 × 45 splits | 180 XGB fits | ~12s | 180 | ~36 min |
| Walk-forward: 4 × 3 folds | 12 XGB fits | ~12s | 12 | ~2 min |
| Holdout: 2 configs | 2 XGB fits | ~15s | 2 | ~1 min |
| Per-direction + time-of-day analysis | 4 geometries | ~1 min | 4 | ~5 min |
| Cost sensitivity + reporting | — | — | — | ~10 min |
| **Total** | | | | **~76 min** |

*Export per-file time increases with 3600s forward scan (up to 720 bars lookahead vs 60 at 300s). Many bars will hit barriers before 3600s, so actual increase is less than 12×. Conservative estimate: 5-15s/file vs prior ~2s.

---

## Key References

- **Phase 1 results (REFUTED):** `.kit/results/label-geometry-phase1/` (90.7-98.9% hold at 300s)
- **Phase 1 analysis (SC-3 flaw):** `.kit/results/label-geometry-phase1/analysis.md` — directional accuracy lesson, volatility dominance finding
- **Phase 0 oracle data:** `.kit/results/label-design-sensitivity/oracle/` (123 JSON files)
- **Phase 1 run_experiment.py:** `.kit/results/label-geometry-phase1/run_experiment.py` (closest starting point — add `--max-time-horizon 3600 --volume-horizon 50000`, switch SC-3 to directional accuracy, update results dir)
- **Walk-forward pattern:** `.kit/results/xgb-hyperparam-tuning/run_experiment.py`
- **Synthesis-v2 analysis:** `.kit/results/synthesis-v2/analysis.md` (GO verdict, geometry selection rationale)
- **Tuned XGB params:** LR=0.0134, L2=6.586, depth=6, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014

---

## Confounds to Watch For

1. **Accuracy does NOT transfer across geometries.** Each geometry is a different classification problem. Compare directional accuracy to per-geometry breakeven WR, not a fixed number.
2. **Narrow stops (2-3 ticks) get swept by noise.** At 15:3, stop=$0.75 (3 ticks × $0.25) is within the typical bid-ask spread — may generate many stop-outs from bid-ask bounce rather than genuine adverse moves. At 20:3, the same issue is even more extreme. If narrow-stop geometries show dramatically higher trade counts with worse directional accuracy, this is the cause.
3. **Hold-majority predictions.** Even with balanced labels, the model may still favor hold predictions if the directional signal is weak. Monitor per-class recall and directional prediction rate. Phase 1 showed 99.86% hold recall at 90.7% hold rate — if hold rate drops to ~50%, the model should shift behavior, but verify.
4. **Walk-forward divergence.** CPCV may again show near-breakeven while WF shows negative (as in XGB tuning: -$0.001 vs -$0.140). Both must be reported; WF is primary for deployment.
5. **Time-of-day effects may dominate.** Phase 1 showed opening range has 2.8× the directional rate vs close. If opening-range is significantly better across geometries, the optimal strategy may be "trade only the open" regardless of geometry.
6. **Label distribution shift from 300s→3600s.** With a 12× wider time horizon, many formerly-hold bars will now reach barriers. The class balance could shift dramatically — from ~91% hold to potentially majority-directional. If hold rate drops below 20%, the classification problem fundamentally changes (directional-dominated). Report the full distribution.
7. **Volatility dominance diagnostic.** Phase 1 showed volatility features captured 76% of top-10 gain — the model learned "will any barrier trigger?" not "which direction?" If 3600s produces balanced labels and the model shifts to directional features (spread, OFI, trade imbalance rising in importance), that's a positive signal. If volatility still dominates despite balanced labels, the model may still be doing barrier-trigger detection at a subtler level.
8. **PnL model simplification for label=0 predictions.** The PnL model assigns $0 when the model predicts directional but the true label is 0. At 3600s, fewer bars will be hold-labeled → this confound should be less severe than in Phase 1 (where 26.7% of directional predictions hit hold bars). Still monitor via label0_trade_fraction.
9. **Phase 0 oracle mismatch.** Oracle numbers in the IV table were computed at 300s horizon. With 3600s, more trades reach barriers → oracle WR and expectancy will change. Step 0 measures the actual 3600s oracle as the reference. BEV WR is purely a function of geometry (target:stop ratio), not time horizon, so the geometry selection remains valid.
10. **Export time increase.** 3600s forward scan scans up to 720 bars ahead per bar (vs 60 at 300s). If export phase takes 15-25 min, this is expected — not a reason to abort. Only investigate if per-file time exceeds 60s, suggesting a bug rather than legitimate scanning.

---

## Deliverables

```
.kit/results/label-geometry-1h/
  metrics.json                    # All SC statuses + per-geometry metrics
  analysis.md                     # Comparative analysis, verdict, SC pass/fail
  run_experiment.py               # Experiment script (created by RUN phase)
  spec.md                         # Local copy of spec
  oracle/                         # Step 0: oracle_T_S_3600s.json per geometry (aggregated from 20 days)
  oracle_comparison.csv           # 300s vs 3600s oracle metrics + hold rates
  geom_10_5/                      # Re-exported Parquet (251 files)
  geom_15_3/                      # Re-exported Parquet (251 files)
  geom_19_7/                      # Re-exported Parquet (251 files)
  geom_20_3/                      # Re-exported Parquet (251 files)
  cpcv_results.csv                # CPCV metrics per geometry (overall + directional accuracy)
  walkforward_results.csv         # Walk-forward metrics per geometry
  holdout_results.json            # Best geometry + baseline holdout
  per_direction_oracle.csv        # Long/short oracle metrics at 3600s
  time_of_day.csv                 # Band A/B/C metrics
  cost_sensitivity.csv            # 3 cost scenarios x 4 geometries
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **Hold rate comparison table** — 3600s vs Phase 1's 90.7-98.9% at 300s (KEY DIAGNOSTIC — confirms root cause fix)
3. **Oracle comparison** — 300s (Phase 0) vs 3600s (Step 0) oracle WR and net expectancy per geometry
4. **CPCV results table** (4 geometries × directional accuracy, overall accuracy, expectancy, breakeven margin)
5. **Walk-forward results table** (4 geometries × expectancy) — CRITICAL comparison to CPCV
6. **Breakeven margin analysis** (directional accuracy - breakeven_WR per geometry)
7. **Class distribution** (-1/0/+1 fractions per geometry) — explicit comparison to Phase 1
8. **Per-class recall** — did long/short asymmetry persist? Did hold-dominance break?
9. **Directional prediction rate** — what fraction of bars does the model choose to trade? Compare to Phase 1's 0.72%.
10. **Feature importance shift** — did volatility dominance decrease from Phase 1's 76%? This indicates whether the model learned direction or still detects barrier-trigger likelihood.
11. **Per-direction oracle analysis** — optimal long vs short geometry
12. **Time-of-day breakdown** — Bands A, B, C
13. **Cost sensitivity table** (3 scenarios × 4 geometries)
14. **Key diagnostic:** Does directional accuracy track geometry, or is it geometry-invariant?
15. **Holdout evaluation** (reported last, clearly separated)
16. **Explicit SC-1 through SC-8 pass/fail**
17. **Outcome verdict** (A/B/C/D per decision rules)

---

## Exit Criteria

- [ ] MVE gates passed (oracle CLI, export CLI with `--max-time-horizon 3600`, training pipeline)
- [ ] Oracle re-validation at 3600s completed for all 4 geometries (Step 0)
- [ ] Re-export completed for all 4 geometries (1,004 files) with `--max-time-horizon 3600 --volume-horizon 50000`
- [ ] Hold rate at all 4 geometries reported and compared to Phase 1's 90.7-98.9%
- [ ] CPCV completed for all 4 geometries (180 fits)
- [ ] Walk-forward completed for all 4 geometries (12 fits)
- [ ] Holdout evaluated for best geometry + baseline
- [ ] Per-direction oracle metrics reported for all 4 geometries
- [ ] Time-of-day analysis computed for all 4 geometries
- [ ] Cost sensitivity (3 scenarios) reported for all 4 geometries
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with comparison tables and verdict
- [ ] Decision rule applied (Outcome A/B/C/D)
- [ ] SC-1 through SC-8 explicitly evaluated
