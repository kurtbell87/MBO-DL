# Experiment: Label Design Sensitivity — Triple Barrier Geometry

**Date:** 2026-02-25
**Priority:** P1 — orthogonal to model tuning; geometry determines the game the model plays
**Parent:** Oracle Expectancy (R7), XGBoost Hyperparameter Tuning (Outcome C)
**Depends on:**
1. oracle-expectancy-params TDD (`--target`/`--stop` CLI flags) — DONE
2. bidirectional-label-export TDD (independent long/short race evaluation) — DONE (PR #26)
3. bidirectional-export-wiring TDD (wired into bar_feature_export, 152-col schema) — DONE (PR #27)
4. Full-year Parquet re-export with bidirectional labels — DONE (312/312, S3)
5. bar-feature-export-geometry TDD (`--target`/`--stop` on bar_feature_export) — DONE

**All prerequisites DONE. Experiment is FULLY UNBLOCKED.**

---

## Context

The 10:5 geometry (target=10 ticks, stop=5 ticks) is at the friction floor:

- CPCV expectancy: -$0.001/trade (tuned), -$0.066 (default) — both net zero or negative
- Holdout expectancy: -$0.132 — no out-of-sample edge
- Breakeven RT cost ($3.74) aligns with actual base RT — no margin
- Long recall 0.149 vs short recall 0.634 — model learned barrier asymmetry, not direction
- XGB tuning showed 0.33pp accuracy plateau across 64 configs — feature set is the binding constraint on accuracy
- **Breakeven WR at 10:5 is ~53.3%.** Model achieves ~45%. The gap is 8pp. No feasible accuracy intervention closes 8pp.

**The only remaining lever is changing the economic structure of the label.** Wider targets and/or narrower stops lower the breakeven WR, potentially bringing it within reach of the model's fixed ~45% directional signal.

**Example breakeven WRs (base cost $3.74 RT):**
| Geometry | Breakeven WR | Model accuracy needed |
|----------|-------------|----------------------|
| 10:5 (current) | 53.3% | 55.3% (+2pp margin) |
| 15:5 | 40.0% | 42.0% |
| 20:5 | 32.0% | 34.0% |
| 15:3 | 33.3% | 35.3% |
| 10:3 | 41.5% | 43.5% |

**Formula:** `breakeven_WR = (stop × $1.25 + RT_cost) / ((target + stop) × $1.25)`

---

## Hypothesis

Widening the triple barrier target/stop ratio from the current 2:1 (10:5) to ≥3:1 will reduce the breakeven win rate to ≤42%, enabling the XGBoost model to produce positive per-trade expectancy (>$0.00) after base costs ($3.74 RT). Specifically:

1. At least one geometry in the {target: 5-20, stop: 2-10} space has an oracle net expectancy ceiling ≥$5.00/trade (sufficient margin above costs for a realistic model to capture some fraction).
2. At least one geometry achieves GBT CPCV per-trade expectancy >$0.00, with the model's accuracy on that geometry's labels exceeding the breakeven WR by ≥2pp.

**Mechanism:** The current 10:5 geometry requires 53.3% directional accuracy to break even — 8pp above the model's ~45%. Wider targets increase profit per correct trade, lowering breakeven WR. At 15:5, breakeven drops to 40.0%, providing 5pp of margin if the model's directional signal (~45%) persists under the new label distribution.

**Key uncertainty:** Accuracy on new labels is NOT guaranteed to remain at ~45%. The label distribution shifts substantially (more holds at wider targets), changing the classification problem. This experiment measures that empirically.

**Critical note on prior baselines:** ALL prior XGB results used flawed long-perspective-only labels. The bidirectional re-export (2026-02-25) corrects this. The 10:5 baseline is re-measured in this experiment as a control — not compared to historical numbers.

---

## Independent Variables

### Phase 0: Full Geometry Sweep (Oracle Only — No Model Training)

**Heatmap grid:**
- Target: {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} ticks (16 values)
- Stop: {2, 3, 4, 5, 6, 7, 8, 9, 10} ticks (9 values)
- Total: 144 geometry combinations

For each geometry, run `oracle_expectancy --target T --stop S` on a **fixed 20-day stratified subsample** (5 days per quarter, same 20 days for all 144 geometries to ensure internal consistency).

Per-geometry oracle output:
- Oracle win rate (WR), profit factor (PF), net expectancy ($/trade after base costs)
- Trade count
- Exit reason breakdown (target, stop, take_profit, expiry, session_end)

Derived metrics (computed by analysis script from oracle output):
- Breakeven WR = (stop × $1.25 + $3.74) / ((target + stop) × $1.25)
- Oracle margin = oracle WR − breakeven WR
- geometry_score = oracle_net_expectancy × sqrt(trade_count / max_trade_count)

### Phase 1: Model Training on Top Geometries (Data-Driven Selection)

Select the **top-3 geometries** from Phase 0 by `geometry_score`. This metric balances per-trade edge with sufficient trade frequency — a geometry with $10/trade but 100 trades/year is less useful than $5/trade with 10,000 trades.

Plus include **baseline (10:5)** as control = **4 total geometries** for model training.

For each geometry:
1. Re-export full-year Parquet using `bar_feature_export --target T --stop S` (C++ on raw MBO data)
2. Train XGBoost with tuned params on re-exported labels
3. CPCV evaluation (N=10, k=2, 45 splits)

**Per-direction analysis (Phase 1 only, from re-exported Parquet):**
From the bidirectional label columns in re-exported Parquet, compute:
- Long oracle: WR and expectancy for bars where `tb_long_triggered=1`
- Short oracle: WR and expectancy for bars where `tb_short_triggered=1`
- Both-triggered rate and its correlation with time-of-day

**Time-of-day conditioning (Phase 1 only, from re-exported Parquet):**
- Band A: First 30 min RTH (09:30-10:00 ET) — opening range
- Band B: Mid-session (10:00-15:00 ET) — steady state
- Band C: Last 30 min (15:00-15:30 ET) — close

Python loads C++-produced Parquet and performs ONLY aggregation/filtering on pre-computed labels (no label recomputation).

---

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Model | XGBoost with tuned params (max_depth=6, lr=0.0134, mcw=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014, reg_lambda=6.586) | Best available from XGB tuning experiment |
| Feature set | 20 non-spatial features (same as tuning experiment) | Isolate label design effect |
| Bar type | time_5s | Locked since R1/R6 |
| CV scheme | CPCV (N=10, k=2, 45 splits) | Same as prior experiments |
| Dev/holdout split | Days 1-201 / 202-251 | Same as prior experiments |
| Early stopping | 50 rounds, val = last 20% of training days | Same as tuned XGB protocol |
| n_estimators | 2000 (upper bound, early stopping determines actual) | Same as tuned XGB protocol |
| Seed | 42 | Reproducibility |
| Data source | Raw .dbn.zst for C++ label/feature computation; C++-produced Parquet for Python model training | C++ is canonical source for all labels/features (CLAUDE.md rule) |
| Label type | Bidirectional triple barrier (compute_bidirectional_tb_label) | Corrected labels; NOT long-perspective-only |

---

## Metrics (ALL must be reported)

### Primary

1. **oracle_viable_count**: Number of geometries (out of 144) with oracle net expectancy > $5.00/trade
2. **best_geometry_cpcv_expectancy**: CPCV mean per-trade expectancy ($) under base costs for the best trained geometry

### Secondary

| Metric | Description |
|--------|-------------|
| oracle_peak_expectancy | Max oracle net expectancy across all 144 geometries |
| oracle_peak_geometry | (target, stop) pair at peak |
| class_distribution_per_geometry | -1/0/+1 fractions for each of the 4 trained geometries |
| cpcv_accuracy_per_geometry | CPCV mean accuracy for each of the 4 trained geometries |
| cpcv_expectancy_per_geometry | CPCV mean expectancy for each of the 4 trained geometries |
| breakeven_margin_per_geometry | (model accuracy - breakeven WR) for each trained geometry |
| per_class_recall_per_geometry | Long/flat/short recall for each trained geometry |
| per_direction_oracle | Long and short oracle WR/expectancy for 4 trained geometries |
| both_triggered_rate | Fraction of bars where both long and short races trigger |
| time_of_day_breakdown | Oracle/model metrics for Bands A, B, C for 4 trained geometries |
| profit_factor_per_geometry | PF for each trained geometry |
| per_quarter_expectancy_best | Per-quarter (Q1-Q4) expectancy for the best geometry |
| feature_importance_shift | Top 10 features by gain: best geometry vs baseline (10:5) |
| cost_sensitivity | Expectancy under optimistic ($2.49), base ($3.74), pessimistic ($6.25) RT costs |
| long_recall_vs_short | Long (+1) vs short (-1) recall asymmetry per geometry |

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| SC-S1: Oracle expectancy > 0 for all 144 geometries | Yes | Perfect foresight must profit — label computation bug |
| SC-S2: Higher target → fewer directional labels | Monotonic trend (at fixed stop) | Label generation bug |
| SC-S3: Narrower stop → more directional labels | Monotonic trend (at fixed target) | Label generation bug |
| SC-S4: Baseline (10:5) CPCV accuracy > 0.40 | Yes | Pipeline is broken (NOTE: bidirectional labels may shift accuracy — 0.40 is a floor, not a match to prior 0.450 which used long-perspective labels) |
| SC-S5: No single feature > 60% gain share in any geometry | Yes | Degenerate model — single feature monopoly |

---

## Baselines

| Baseline | Source | Value | Notes |
|----------|--------|-------|-------|
| GBT accuracy (10:5, CPCV, tuned) | XGB tuning | 0.450 | Long-perspective labels — NOT directly comparable |
| GBT expectancy (10:5, CPCV, tuned) | XGB tuning | -$0.001 | Long-perspective labels — expect change |
| GBT holdout exp (10:5, tuned) | XGB tuning | -$0.132 | Long-perspective labels |
| Oracle exp (10:5, long-perspective) | R7 | $4.00/trade | FLAWED — will be lower with bidirectional labels |
| Oracle WR (10:5, long-perspective) | R7 | 64.3% | FLAWED — same caveat |

**The 10:5 control in THIS experiment (with bidirectional labels) is the proper baseline — not historical numbers.** Any comparison to prior results is confounded by the label correction.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Oracle heatmap computed for all 144 geometries on 20-day subsample
- [ ] **SC-2**: At least one geometry has oracle net expectancy > $5.00/trade
- [ ] **SC-3**: At least one trained geometry achieves CPCV mean accuracy > breakeven_WR + 2pp
- [ ] **SC-4**: At least one trained geometry achieves CPCV mean per-trade expectancy > $0.00
- [ ] **SC-5**: Best geometry holdout expectancy > -$0.10 (less negative than prior baseline -$0.132)
- [ ] **SC-6**: Per-direction oracle metrics and time-of-day analysis reported for all 4 trained geometries
- [ ] No regression on sanity checks

---

## Decision Rules

```
OUTCOME A — SC-3 AND SC-4 pass:
  -> CONFIRMED. A viable geometry exists.
  -> Record: best geometry, accuracy margin over breakeven, expectancy, per-class recall.
  -> Next: Deploy best geometry. If H1/H2 split persists, combine with regime-conditional.

OUTCOME B — SC-2 passes but SC-3/SC-4 fail:
  -> PARTIAL. Oracle ceiling adequate, model can't capture it.
  -> Key diagnostic: Is accuracy geometry-invariant (~45% on all geometries)?
  -> If yes: model has fixed directional signal, feature set is the bottleneck.
     -> Next: Feature engineering, 2-class formulation (short/no-short), or class-weighted loss.
  -> If no (accuracy drops at wider targets): the classification problem gets harder with geometry change.
     -> Next: Per-direction asymmetric strategy (different geometry for longs vs shorts).

OUTCOME C — SC-2 fails (no geometry has oracle ceiling > $5.00):
  -> REFUTED. MES 5-second bars lack sufficient directional signal for ANY geometry.
  -> This is a hard stop on the current approach.
  -> Next: Consider longer bar intervals, different instruments, or accept no-edge verdict.

OUTCOME D — SC-4 passes but ONLY for time Band A (opening range):
  -> REGIME-CONDITIONAL. Edge exists but is time-localized.
  -> Next: Regime-conditional strategy trading only the opening range.
```

---

## Minimum Viable Experiment

Before the full protocol, validate all tool chain components end-to-end:

1. **Oracle CLI gate.** Run `oracle_expectancy --target 15 --stop 3 --output /tmp/test_oracle.json` on 1 .dbn.zst file. Assert:
   - JSON output is valid and contains: total_trades, win_rate, expectancy, profit_factor
   - Oracle net expectancy > $0 (perfect foresight must profit)
   - Trade count > 0

   **ABORT if oracle_expectancy fails, produces empty output, or shows non-positive oracle expectancy.**

2. **Export CLI gate.** Run `bar_feature_export --bar-type time --bar-param 5 --target 15 --stop 3 --output /tmp/test_export.parquet` on the same day. Assert:
   - Parquet output has 152 columns (bidirectional schema: 149 original + tb_both_triggered, tb_long_triggered, tb_short_triggered)
   - tb_label column exists with values in {-1, 0, +1}
   - Label distribution differs from default (10:5) export for the same day (wider target → fewer directional labels)
   - Row count in [3000, 6000]

   **ABORT if export fails or schema doesn't match 152-column bidirectional format.**

3. **Training pipeline gate.** Load the 1-day test Parquet. Extract 20 features + tb_label. Train XGBoost with tuned params on an 80/20 split. Assert:
   - No NaN in features or predictions
   - Accuracy > 0.33 (above random)
   - Per-class predictions include all 3 classes

   **ABORT if training crashes, produces NaN, or accuracy < 0.33.**

Pass all 3 gates → proceed to full protocol.

---

## Full Protocol

### Phase 0: Oracle Ceiling Heatmap

**Execution target:** Requires raw .dbn.zst data — EC2 (EBS-mounted MBO data) or local if data available.

1. **Select 20 stratified days.** 5 per quarter from the 251 RTH trading days, spread evenly within each quarter. Document which 20 days are selected. Use the **same 20 days for all 144 geometries** to ensure the heatmap is internally consistent.

2. **Run oracle sweep.** For each (target, stop) in the 16x9 grid (144 combinations):
   - `oracle_expectancy --target T --stop S --output .kit/results/label-design-sensitivity/oracle/oracle_T_S.json`
   - Each run processes the 20-day subsample (~5s per run)
   - Parallelize across geometries (each run is independent)

3. **Aggregate oracle results (Python).** Load all 144 JSON files. For each geometry compute:
   - Net expectancy = oracle expectancy − base RT cost per trade ($3.74)
   - Breakeven WR = (stop × $1.25 + $3.74) / ((target + stop) × $1.25)
   - Oracle margin = oracle WR − breakeven WR
   - geometry_score = oracle_net_expectancy × sqrt(trade_count / max_trade_count)
   - Class distribution estimate from trade counts vs total bars

4. **Select top-3 geometries** by geometry_score. Add baseline (10:5) = 4 geometries for Phase 1.

5. **Gate:** If no geometry has oracle net expectancy > $5.00/trade → STOP. Report Outcome C.

### Phase 1: GBT Training on Top Geometries

For each of 4 geometries (top-3 + baseline 10:5):

**Step 1 — Re-export Parquet (C++, requires raw MBO data):**
- `bar_feature_export --bar-type time --bar-param 5 --target T --stop S` on all 251 RTH days
- Output: `.kit/results/label-design-sensitivity/geom_T_S/*.parquet`
- ~2s/day × 251 days × 4 geometries = ~33 min (parallelizable across days)

**Step 2 — Train XGBoost (local, Python on pre-computed Parquet):**
- Load Parquet for this geometry. Extract 20 features + tb_label.
- CPCV (N=10, k=2, 45 splits):
  - 10 sequential groups of ~20 days from 201 dev days
  - Purge: 500 bars at each train/test boundary
  - Embargo: 4,600 bars (~1 day) after each test-group boundary
  - Early stopping: last 20% of training days as validation for mlogloss
  - Feature normalization: z-score using training fold stats only; NaN → 0.0 after normalization
- Per split record: accuracy, per-class predictions, confusion matrix, PnL (3 cost scenarios), feature importance (gain), actual n_estimators

**Step 3 — Per-direction and time-of-day analysis (Python, from Parquet labels):**
- For each geometry, from the bidirectional label columns in the full-year Parquet:
  - Long oracle metrics: WR, expectancy, trade count for bars where `tb_long_triggered=1`
  - Short oracle metrics: WR, expectancy, trade count for bars where `tb_short_triggered=1`
  - Both-triggered rate: fraction where `tb_both_triggered=1`
  - Time-of-day breakdown: Band A (09:30-10:00), B (10:00-15:00), C (15:00-15:30) — oracle metrics per band
  - Asymmetric analysis: optimal long geometry vs optimal short geometry from per-direction data

### Phase 2: Holdout Evaluation

For the **best geometry** by CPCV expectancy AND the **baseline (10:5)**:
1. Train on full dev set (201 days), internal 80/20 val split for early stopping
2. Evaluate on holdout (days 202-251, 50 days)
3. Record: accuracy, expectancy (3 cost scenarios), per-quarter breakdown, confusion matrix, per-class recall
4. **Key diagnostic:** Is long recall still pathologically low? If per-class recall is similarly asymmetric as 10:5 baseline, the model is learning barrier asymmetry regardless of geometry.

### Phase 3: Comparative Analysis

1. **Oracle heatmap (full session)** — 16x9 grid color-coded by net expectancy
2. **Oracle feasibility contour** — boundaries where oracle ceiling > $5, $3, $1
3. **Trade count heatmap** — where in geometry space are there sufficient trades?
4. **Class distribution shift** — how -1/0/+1 fractions change across the 144-geometry space
5. **GBT comparison table** — 4 geometries × all Phase 1 metrics
6. **Key diagnostic: Does accuracy track oracle ceiling, or is it geometry-invariant?**
   - If invariant (~same accuracy everywhere): model has fixed directional signal → geometry just shifts breakeven → the right geometry is wherever breakeven WR falls below the model's accuracy floor
   - If tracks ceiling: model captures more signal at certain geometries → pick the peak
7. **Breakeven margin analysis** — (accuracy − breakeven_WR) for each trained geometry
8. **Per-class recall comparison** — did long/short asymmetry persist across geometries?
9. **Per-direction oracle analysis** — do optimal long and short geometries differ? If yes, compute composite asymmetric expectancy (post-hoc, no additional training needed)
10. **Both-triggered analysis** — correlation with time-of-day. Flag `tb_both_triggered` as candidate trade filter if it identifies regime zones
11. **Cost sensitivity table** — 3 cost scenarios × 4 geometries
12. **Holdout results** (reported last, clearly separated from dev results)
13. **Explicit SC-1 through SC-6 pass/fail**

### Transaction Cost Scenarios

| Scenario | Commission/side | Spread (ticks) | Slippage (ticks) | Total RT cost |
|----------|----------------|----------------|-------------------|---------------|
| Optimistic | $0.62 | 1 | 0 | $2.49 |
| Base | $0.62 | 1 | 0.5 | $3.74 |
| Pessimistic | $1.00 | 1 | 1 | $6.25 |

### PnL Model (geometry-dependent — tick values change per geometry!)

```
tick_value = $1.25 per tick (MES)

For geometry (target=T, stop=S):
  Correct directional call (pred sign = label sign, both nonzero):
    PnL = +(T × $1.25) - RT_cost
  Wrong directional call (pred sign != label sign, both nonzero):
    PnL = -(S × $1.25) - RT_cost
  Predict 0 (hold): $0 (no trade)
  True label=0, model predicted ±1: $0 (conservative simplification)
```

Report label=0 trade fraction per geometry. If >25% of directional predictions hit label=0, flag as unreliable.

---

## Resource Budget

**Tier:** Standard

- Max wall-clock time: 2.5 hours
- Max training runs: ~185 (4 geometries × 45-split CPCV + 2 holdout)
- Max oracle runs: 144 (Phase 0) + 3 (MVE)
- Max export runs: ~1,004 (4 geometries × 251 days)
- Max seeds per configuration: 1 (seed=42; variance assessed across temporal folds)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 185
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 2.0
```

### Wall-Time Estimation

| Phase | Work | Per-unit | Units | Subtotal |
|-------|------|----------|-------|----------|
| MVE (oracle + export + train gates) | 3 gates | — | 3 | ~3 min |
| Phase 0: oracle sweep (20-day subsample) | 144 C++ runs | ~5s | 144 | ~12 min |
| Phase 1 export: 4 geometries × 251 days | 1,004 C++ runs | ~2s | 1,004 | ~33 min (parallel) |
| Phase 1 training: 4 × 45-split CPCV | 180 XGB fits | ~12s | 180 | ~36 min |
| Phase 1 per-direction + time-of-day analysis | 4 geometries | ~1 min | 4 | ~5 min |
| Phase 2: holdout (2 configs) | 2 XGB fits | ~15s | 2 | ~1 min |
| Phase 3: analysis + heatmaps + reporting | — | — | — | ~10 min |
| **Total** | | | | **~100 min** |

**Per-fit estimate:** ~925K training rows (dev set), 20 features, XGBoost tree_method='hist', nthread=-1 on Apple Silicon. Early stopping averages ~385 trees at lr=0.0134. Conservative: 12s/fit.

**Execution targets:**
- Phases requiring raw .dbn.zst: EC2 c5.2xlarge (~$0.50/hr) with EBS-mounted MBO data, OR local if available
- GBT training + analysis: local (Apple Silicon, CPU-only, Parquet <1GB)
- Estimated EC2 cost: ~45 min × ~$0.34/hr = ~$0.26

---

## Abort Criteria

- **MVE gate failure:** STOP. Diagnose before proceeding. Most likely: CLI flag wiring issue.
- **Phase 0: No geometry with oracle net expectancy > $5.00:** STOP at Phase 0. Report Outcome C. Do NOT proceed to Phase 1 — model training on all geometries will fail if the oracle ceiling is insufficient.
- **Degenerate labels:** Any geometry produces >95% one class → skip that geometry, log warning. If >50% of 144 geometries are degenerate, investigate label computation pipeline.
- **Baseline reproduction failure:** Baseline (10:5) CPCV accuracy on bidirectional labels < 0.40 → STOP. Pipeline is broken or bidirectional labels fundamentally change the problem to an unrecognizable degree. Investigate before spending budget on other geometries.
- **Per-fit time > 60s:** Investigate. Expected ~12s. If 5× slower, likely data loading or parallelism issue.
- **NaN loss:** Any XGB fit produces NaN → skip that config, log warning. If >10% of fits produce NaN, ABORT.
- **Wall-clock > 3 hours:** Abort remaining phases, report completed work. Save all partial results.

---

## Confounds to Watch For

1. **Win rate does not transfer across geometries.** A model achieving 45% accuracy on 10:5 labels will NOT achieve 45% on 15:3 labels. Each geometry is a different classification problem with different class distributions. All accuracy comparisons must use per-geometry breakeven thresholds, not a fixed number.

2. **Barrier asymmetry → class distribution bias.** With highly asymmetric barriers (e.g., 20:2), nearly all bars hit the narrow stop, producing pathological class imbalance (>90% holds). Monitor per-class recall for extreme asymmetry (long recall < 0.15 was already observed at 10:5).

3. **Very narrow stops (2-3 ticks) get swept by noise.** 2-3 ticks is $0.50-$0.75 — normal MES spread and microstructure noise. Expect high trade counts but low signal quality. The oracle will show high WR (perfect foresight always works), but the model's accuracy may degrade at these thresholds because the signal-to-noise ratio drops.

4. **Oracle ceiling is necessary but not sufficient.** A $10/trade oracle ceiling doesn't mean the model can capture any of it. The oracle uses perfect foresight; the model sees only entry-time features. Phase 1 GBT training measures the actual capture rate.

5. **Time-of-day effects may dominate geometry effects.** If opening-range oracle ceiling is 3× mid-session, the optimal strategy may be "trade only the open" regardless of geometry. The time-of-day analysis in Phase 1 detects this.

6. **20-day oracle subsample may not represent the full year.** If 5 stratified days per quarter miss regime variation, the oracle heatmap may be misleading. Phase 1 re-exports use the full 251-day year, correcting any subsample bias at the model-training stage.

7. **Bidirectional label impact on baseline.** The 10:5 baseline under bidirectional labels will differ from all prior results (which used long-perspective labels). Prior baselines (CPCV 0.450 accuracy, -$0.001 expectancy) are NOT directly comparable. The 10:5 control measured in THIS experiment is the only valid baseline.

8. **geometry_score selection criterion may be suboptimal.** The sqrt(trade_count) scaling is heuristic. Consider whether the top-3 geometries cluster in one region of the space (e.g., all wide target, narrow stop). If they do, the selection has low diversity. The READ agent should check whether adding a diverse 4th geometry would change conclusions.

---

## Deliverables

```
.kit/results/label-design-sensitivity/
  metrics.json                    # All phases: oracle heatmap summary + GBT metrics
  analysis.md                     # Comparative analysis, heatmaps, verdict, SC pass/fail
  oracle/                         # Phase 0: 144 JSON files (oracle_T_S.json)
  oracle_heatmap_full.csv         # 144-row: target, stop, oracle metrics (full session)
  geom_T_S/                       # Phase 1: Parquet re-exports per geometry (4 dirs)
  gbt_results.csv                 # GBT CPCV metrics per geometry (4 rows)
  class_distributions.csv         # Label distribution per geometry (144 rows from Phase 0)
  per_direction_oracle.csv        # Long/short oracle metrics for 4 trained geometries
  time_of_day.csv                 # Band A/B/C metrics for 4 trained geometries
  holdout_results.json            # Best geometry + baseline holdout evaluation
  asymmetric_analysis.csv         # Composite long-short geometry analysis
  cost_sensitivity.csv            # 3 cost scenarios × 4 geometries
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **Oracle heatmap (full session)** — 16x9 grid, color-coded by net expectancy
3. **Oracle feasibility contour** — where oracle ceiling > $5, $3, $1
4. **Trade count heatmap** — where in geometry space are there sufficient trades?
5. **Class distribution shift** — how -1/0/+1 fractions change across geometry space
6. **Top-3 geometry selection rationale** (geometry_score ranking)
7. **GBT CPCV results table** (4 geometries × all metrics)
8. **Breakeven margin analysis** (accuracy − breakeven_WR per geometry)
9. **Per-class recall comparison** — did long/short asymmetry persist?
10. **Per-direction oracle analysis** — optimal long vs short geometry
11. **Time-of-day breakdown** — oracle and model metrics for Bands A, B, C
12. **Key diagnostic:** Does accuracy track oracle ceiling, or is it geometry-invariant?
13. **Holdout evaluation** (reported last, clearly separated from dev results)
14. **Cost sensitivity table** (3 scenarios × 4 geometries)
15. Explicit pass/fail for SC-1 through SC-6

---

## Exit Criteria

- [ ] MVE gates passed (oracle CLI, export CLI, training pipeline)
- [ ] Oracle heatmap computed for all 144 geometries (full session)
- [ ] Top-3 geometries selected with rationale
- [ ] GBT CPCV completed for top-3 + baseline (4 geometries)
- [ ] Per-direction oracle metrics reported for all 4 trained geometries
- [ ] Time-of-day analysis computed for all 4 trained geometries
- [ ] Best geometry evaluated on holdout
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with heatmaps, comparison tables, and verdict
- [ ] Decision rule applied (Outcome A/B/C/D)
- [ ] Per-class recall reported — long/short asymmetry diagnosed
