# Experiment: Label Design Sensitivity — Triple Barrier Geometry

**Date:** 2026-02-24
**Priority:** P1 — orthogonal to model tuning; 15:3 ratio drops breakeven to ~42.5%
**Parent:** Oracle Expectancy (R7), E2E CNN Classification (Outcome D)

---

## Hypothesis

Alternative triple barrier geometries (wider target and/or narrower stop) will produce a label distribution where GBT classification accuracy exceeds the breakeven win rate by at least 2 percentage points, yielding positive per-trade expectancy under base costs.

**Key insight:** Current geometry (target=10, stop=5 ticks) requires 53.3% win rate to break even at $3.74 RT. A 15:3 geometry drops breakeven to ~42.5%. If GBT achieves ~45% accuracy on 15:3 labels (plausible given current 44.9% on 10:5), expectancy flips positive.

## Independent Variables

| Config | Target (ticks) | Stop (ticks) | Ratio | Implied Breakeven WR | Notes |
|--------|---------------|-------------|-------|---------------------|-------|
| **Baseline** | 10 | 5 | 2:1 | 53.3% | Current geometry |
| **Wide-target** | 15 | 5 | 3:1 | 45.0% | Same stop, more room to run |
| **Narrow-stop** | 10 | 3 | 3.3:1 | 42.5% | Tighter risk, same target |
| **Balanced** | 15 | 3 | 5:1 | 37.5% | Maximum ratio tested |
| **Aggressive** | 10 | 2 | 5:1 | 37.5% | Minimum stop (noise risk) |

**Implied breakeven formula:** `WR_breakeven = (stop + RT_cost_ticks) / (target + stop)` where `RT_cost_ticks = $3.74 / $1.25 = 2.99 ticks`.

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Model | XGBoost with default params (or tuned if xgb-hyperparam-tuning runs first) | Isolate label effect from model effect |
| Feature set | 16 hand-crafted GBT features | Same as baseline |
| Bar type | time_5s | Locked since R1/R6 |
| CV scheme | CPCV (N=10, k=2, 45 splits) | Same as prior experiments |
| Dev/holdout split | Days 1-201 / 202-251 | Same as prior experiments |
| Data | Full-year Parquet (255MB, 1.16M bars) | Same dataset |

## Metrics (ALL must be reported)

### Primary
- **Per-trade expectancy** ($) under base costs ($3.74 RT) for each geometry
- **Accuracy margin over breakeven** (accuracy - breakeven_WR) for each geometry

### Secondary
- CPCV mean accuracy per geometry
- Per-class recall (long/flat/short) per geometry
- Class distribution (% of bars labeled -1, 0, +1) per geometry
- Oracle expectancy per geometry (upper bound)
- Profit factor per geometry
- Per-quarter expectancy per geometry

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|----------------|
| Wider target → fewer +1/-1 labels, more 0 | Yes | Label generation bug |
| Narrower stop → more +1/-1 labels, fewer 0 | Yes | Label generation bug |
| Oracle expectancy increases with higher ratio | Yes | Cost model error |
| Baseline reproduces prior results ($4.00/trade oracle) | Yes | Data or labeling bug |

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| GBT accuracy (10:5 labels, CPCV) | E2E CNN experiment | 0.449 |
| GBT expectancy (10:5 labels, CPCV) | E2E CNN experiment | -$0.064 |
| Oracle expectancy (10:5 labels) | R7 | $4.00/trade |
| Breakeven WR (10:5 labels) | Cost analysis | 53.3% |

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: At least one geometry achieves CPCV mean accuracy > breakeven_WR + 2pp
- [ ] **SC-2**: At least one geometry achieves CPCV mean per-trade expectancy > $0.00
- [ ] **SC-3**: Best geometry's holdout expectancy > -$0.10 (less negative than baseline -$0.204)
- [ ] **SC-4**: Oracle expectancy computed for all 5 geometries (validates upper bound exists)
- [ ] **SC-5**: Class distributions reported for all geometries (no degenerate labels)
- [ ] Sanity checks pass for all geometries

## Decision Rules

```
OUTCOME A — SC-1 AND SC-2 pass:
  -> CONFIRMED. Label geometry closes the gap.
  -> Next: Combine best geometry with tuned XGBoost params.

OUTCOME B — SC-1 passes but SC-2 fails:
  -> PARTIAL. Accuracy exceeds breakeven but costs still dominate.
  -> Next: Combine with XGBoost tuning (P1 experiment).

OUTCOME C — No geometry passes SC-1:
  -> REFUTED. Label geometry alone cannot close the gap.
  -> Next: Feature engineering or accept GBT as regime-conditional only.

OUTCOME D — 15:3 or 15:5 passes SC-2 but 10:2 does not:
  -> Wide target is key lever (not narrow stop).
  -> Next: Test wider targets (20, 25 ticks) as follow-up.
```

## Minimum Viable Experiment

1. **Baseline reproduction:** Re-export labels with (10, 5) using `oracle_expectancy` tool. Assert oracle expectancy within 5% of $4.00/trade.
2. **Single alternative:** Export labels with (15, 3). Assert class distribution shifts as expected (more 0 labels due to wider target? or fewer?). Train GBT. Assert CPCV completes.
3. **Breakeven calculation:** Verify implied breakeven WR formula against oracle results.
4. Pass all gates -> proceed to full protocol.

## Full Protocol

### Phase 1: Label Re-Export (C++ tool)

For each of 5 geometries:
1. Run `./build/oracle_expectancy` with appropriate target/stop parameters on full-year data.
   - **NOTE:** This requires the C++ tool to accept parameterized target/stop values. Check if `oracle_expectancy` supports `--target` and `--stop` flags. If not, a TDD sub-cycle is needed to add parameterization.
2. Export tb_label column to CSV or merge into Parquet.
3. Compute and record oracle expectancy, class distribution, win rate.

### Phase 2: GBT Training Per Geometry

For each of 5 geometries:
1. Load Parquet features + re-exported tb_label.
2. Train XGBoost with default params (or tuned params if available from xgb-hyperparam-tuning).
3. CPCV (N=10, k=2, 45 splits) on dev set.
4. Record: accuracy, expectancy, per-class recall, profit factor.

### Phase 3: Holdout Evaluation

For the best geometry (by CPCV expectancy):
1. Train on full dev set.
2. Evaluate on holdout (50 days).
3. Record: accuracy, expectancy, per-quarter breakdown.

### Phase 4: Comparative Analysis

1. Build comparison table: geometry x metric.
2. Plot accuracy margin over breakeven for each geometry.
3. Identify which lever matters more: target width or stop width.

## Resource Budget

**Tier:** Standard (1-2 hours)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 5
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 1.5
```

**Breakdown:**
- Label re-export: ~10 min per geometry (C++ tool on full-year data, 5 geometries = ~50 min)
- GBT training: ~3 min per geometry CPCV (5 geometries = ~15 min)
- Total: ~65 min serial, less with parallelism on label export

**Note on C++ tool dependency:** If `oracle_expectancy` does not support parameterized target/stop, a TDD sub-cycle (~30 min) is needed first. Check the tool's `--help` output before starting.

## Abort Criteria

- Any geometry produces degenerate labels (>95% one class): skip that geometry, log warning.
- C++ tool cannot accept parameterized target/stop: PAUSE, create TDD spec for parameterization.
- Wall-clock exceeds 3 hours: save partial results, evaluate available geometries.

## Confounds to Watch For

1. **Class imbalance shift:** Wider targets produce fewer +1/-1 labels → model may default to predicting 0. Monitor per-class recall.
2. **Stop-loss noise:** Very narrow stops (2 ticks) may trigger on noise, not signal. Check if 10:2 geometry has abnormally high trade count.
3. **Oracle expectancy ceiling:** Wider targets have higher oracle expectancy (more reward per trade) but fewer trades. Net PnL may decrease even if per-trade improves.
4. **Interaction with XGBoost tuning:** Label geometry and hyperparameters may interact. This experiment isolates label effect; combination tested later.

## Deliverables

```
.kit/results/label-design-sensitivity/
  metrics.json            # All 5 geometries: oracle + GBT metrics
  analysis.md             # Comparative analysis, verdict, SC pass/fail
  oracle_expectancy.csv   # Oracle metrics per geometry
  gbt_results.csv         # GBT CPCV metrics per geometry
  class_distributions.csv # Label distribution per geometry
  holdout_results.json    # Best geometry holdout evaluation
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. Oracle expectancy table (all 5 geometries)
3. Class distribution table (all 5 geometries)
4. GBT CPCV results table (accuracy, expectancy, PF, per-class recall)
5. Breakeven margin analysis (accuracy - breakeven_WR)
6. Holdout evaluation for best geometry
7. Per-quarter breakdown for best geometry
8. Key finding: which lever matters more (target width vs stop width)?
9. Explicit pass/fail for each SC-1 through SC-5

## Exit Criteria

- [ ] MVE gates passed (baseline oracle reproduction, single alternative, breakeven formula)
- [ ] Labels exported for all 5 geometries
- [ ] Oracle expectancy computed for all 5 geometries
- [ ] GBT CPCV completed for all 5 geometries
- [ ] Best geometry evaluated on holdout
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with comparative tables and verdict
- [ ] Decision rule applied (Outcome A/B/C/D)
