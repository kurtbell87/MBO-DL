# Experiment: Label Design Sensitivity — Triple Barrier Geometry

**Date:** 2026-02-25
**Priority:** P1 — orthogonal to model tuning; geometry determines the game the model plays
**Parent:** Oracle Expectancy (R7), XGBoost Hyperparameter Tuning (Outcome C)
**Depends on:**
1. oracle-expectancy-params TDD (adds `--target`/`--stop` CLI flags) — DONE
2. bidirectional-label-export TDD (fixes long-perspective-only labeling flaw) — DONE (PR #26)
3. bidirectional-export-wiring TDD (wires into bar_feature_export) — DONE (PR #27)
4. Full-year Parquet re-export with bidirectional labels — DONE (312/312 files, S3)
5. **bar-feature-export-geometry TDD** (adds `--target`/`--stop` to `bar_feature_export`) — **REQUIRED for Phase 1 re-export**

---

## Context

The 10:5 geometry (target=10 ticks, stop=5 ticks) is a **dead strategy**. Key evidence:

- CPCV expectancy: -$0.001/trade (tuned), -$0.066 (default) — both net zero or negative
- Holdout expectancy: -$0.132 — no out-of-sample edge
- Breakeven RT cost ($3.739) aligns almost exactly with actual RT ($3.74) — optimizer found the friction floor, not alpha
- Long recall 0.149 vs short recall 0.634 — model learned barrier asymmetry, not directional signal
- The 45% win rate may be an artifact of the 2:1 asymmetric barrier, not genuine directional prediction

**Critical unknown:** Is the ~45% win rate evidence of genuine directional signal, or a geometry artifact? This experiment answers that by sweeping the geometry space and finding where (if anywhere) the oracle ceiling provides sufficient margin for a realistic model to be profitable.

## Hypothesis

There exists a triple barrier geometry (target, stop) where:
1. The oracle expectancy ceiling provides sufficient margin above RT costs, AND
2. GBT can capture enough of that margin to yield positive per-trade expectancy after costs.

The geometry is NOT pre-specified. Phase 0 identifies candidate geometries from the data.

## Independent Variables

### Phase 0: Full Geometry Sweep (Oracle Only — No Model Training)

**Heatmap grid:**
- Target: {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} ticks (16 values)
- Stop: {2, 3, 4, 5, 6, 7, 8, 9, 10} ticks (9 values)
- Total: 144 geometry combinations

For each geometry, compute oracle (perfect foresight) metrics from the Parquet bar prices.

**Report long and short metrics separately** (not just combined). This enables asymmetric strategy construction post-hoc — e.g., if long oracle peaks at (12, 5) and short oracle peaks at (8, 4), you can composite them without re-running barriers.

Per-direction metrics (long and short independently):
- Oracle expectancy ($/trade) net of base RT cost ($3.74)
- Oracle win rate (WR)
- Oracle profit factor (PF)
- Trigger rate (fraction of bars where this direction's race triggers)

Combined metrics:
- Trade count (how many bars get non-zero bidirectional labels)
- Class distribution (fraction -1, 0, +1, both-triggered)
- Breakeven WR = (stop_ticks × $1.25 + $3.74) / ((target_ticks + stop_ticks) × $1.25)
- Both-triggered rate (fraction of bars where both races trigger — volatility regime indicator)

**Time-of-day conditioning:**
- Band A: First 30 min RTH (09:30–10:00 ET) — opening range
- Band B: Mid-session (10:00–15:00 ET) — steady state
- Band C: Last 30 min (15:00–15:30 ET) — close
- Band D: Full session (all bars)

**Label computation uses C++ tools on raw MBO data (MANDATORY — Python NEVER computes labels):**
- Phase 0 oracle sweep: `oracle_expectancy --target T --stop S --output /work/results/oracle_T_S.json` for each geometry (runs on EC2 with EBS-mounted .dbn.zst data, ~5s per geometry, 144 total)
- Phase 1 GBT training: `bar_feature_export --bar-type time --bar-param 5 --target T --stop S --output <path>.parquet` for each top geometry + baseline (requires TDD: `bar-feature-export-geometry.md`)
- Python loads pre-computed Parquet for model training ONLY — no label recomputation

### Phase 1: Model Training on Top Geometries (Data-Driven Selection)

Select the **top-3 geometries** from Phase 0 by the criterion:
```
geometry_score = oracle_net_expectancy × sqrt(trade_count / max_trade_count)
```
This balances per-trade edge with sufficient trade frequency. A geometry with $10/trade but only 100 trades/year is less useful than $5/trade with 10,000 trades.

Plus include **baseline (10:5)** as control = 4 total geometries for model training.

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Model | XGBoost with tuned params from xgb-hyperparam-tuning | Best available model config |
| Feature set | 20 features (same as tuning experiment) | Isolate label effect |
| Bar type | time_5s | Locked since R1/R6 |
| CV scheme | CPCV (N=10, k=2, 45 splits) | Same as prior experiments |
| Dev/holdout split | Days 1-201 / 202-251 | Same as prior experiments |
| Data | Raw MBO .dbn.zst (49GB, 312 files on EBS) for C++ label computation; C++-produced Parquet for Python model training | C++ is canonical source for all labels/features |

## Metrics (ALL must be reported)

### Phase 0 Outputs
- **Oracle heatmap:** 16×9 grid of oracle net expectancy (target × stop)
- **Optimal geometry region:** contour of where oracle net expectancy > $2.00/trade
- **Time-of-day heatmaps:** separate 16×9 grids for Bands A, B, C
- **Trade count heatmap:** 16×9 grid of non-zero label count per geometry
- **Class distribution shift:** how the -1/0/+1 split changes across the geometry space

### Phase 1 Outputs (per geometry)
- CPCV mean accuracy
- CPCV mean per-trade expectancy ($) under base costs
- Accuracy margin over breakeven WR
- Per-class recall (long/flat/short)
- Profit factor
- Per-quarter expectancy
- Walk-forward accuracy (optional)

### Phase 2 Outputs (best geometry)
- Holdout accuracy and expectancy
- Per-quarter breakdown
- Comparison table: best geometry vs baseline (10:5)

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|----------------|
| Baseline (10:5) oracle under bidirectional labels | Report value (will be LOWER than prior $4.00 — that figure used flawed long-perspective labels) | If higher than $4.00, labeling bug |
| Higher target → fewer directional labels | Monotonic trend | Label generation bug |
| Narrower stop → more directional labels | Monotonic trend | Label generation bug |
| Oracle expectancy always > 0 for all geometries | Yes | Perfect foresight should always profit |
| Time Band A oracle > Band B oracle (for most geometries) | Plausible | Opening range typically more predictable |

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| GBT accuracy (10:5 labels, CPCV, tuned) | XGB tuning | 0.450 |
| GBT expectancy (10:5 labels, CPCV, tuned) | XGB tuning | -$0.001 |
| GBT holdout expectancy (10:5 labels, tuned) | XGB tuning | -$0.132 |
| Oracle expectancy (10:5 labels, long-perspective — FLAWED) | R7 | $4.00/trade (will be lower with bidirectional labels) |
| Breakeven WR (10:5 labels) | Cost analysis | 53.3% |

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Oracle heatmap computed for all 144 geometries on full session
- [ ] **SC-2**: At least one geometry has oracle net expectancy > $5.00/trade (sufficient ceiling)
- [ ] **SC-3**: At least one geometry achieves CPCV mean accuracy > breakeven_WR + 2pp
- [ ] **SC-4**: At least one geometry achieves CPCV mean per-trade expectancy > $0.00
- [ ] **SC-5**: Best geometry holdout expectancy > -$0.10 (less negative than baseline -$0.132)
- [ ] **SC-6**: Time-of-day heatmaps computed for 3 bands
- [ ] Sanity checks pass

## Decision Rules

```
OUTCOME A — SC-3 AND SC-4 pass:
  -> CONFIRMED. A viable geometry exists.
  -> Next: Deploy best geometry; combine with regime-conditional if H1/H2 split persists.

OUTCOME B — SC-2 passes but SC-3/SC-4 fail:
  -> PARTIAL. Oracle ceiling is adequate but model can't capture it.
  -> Next: Feature engineering (model is the bottleneck, not geometry).

OUTCOME C — SC-2 fails (no geometry has oracle ceiling > $5.00):
  -> REFUTED. MES 5-second bars lack sufficient directional signal for any geometry.
  -> Next: Consider longer bar intervals, different instruments, or accept no-edge verdict.

OUTCOME D — SC-4 passes but only for time Band A (opening range):
  -> REGIME-CONDITIONAL. Edge exists but is time-localized.
  -> Next: Regime-conditional strategy on opening range only.
```

## Full Protocol

### Phase 0: Oracle Ceiling Heatmap (EC2 — C++ `oracle_expectancy`)

1. For each (target, stop) in the 16×9 grid (144 combinations):
   a. Run `oracle_expectancy --target T --stop S --output /work/results/oracle_T_S.json` on raw .dbn.zst files (EBS-mounted at `/data/GLBX-*/`). Uses 20 stratified days per geometry.
   b. Oracle output includes: WR, PF, net expectancy, trade count, class distribution, per-direction metrics.
2. Python orchestration script on EC2 loops over geometries, calls `oracle_expectancy` for each. ~144 × 5s = ~12 min.
3. Time-of-day filtering: if `oracle_expectancy` supports time bands, use them; otherwise apply time filters in the Python analysis phase on the oracle JSON outputs.
4. Upload all 144 JSON result files to S3.
5. Python (local) loads the 144 JSON files, produces heatmaps, identifies top-3 geometries by `geometry_score`.
6. **Gate:** If no geometry has oracle net expectancy > $5.00/trade, STOP. Report Outcome C.

### Phase 1: GBT Training on Top Geometries (EC2 re-export + local training)

For each of 4 geometries (top-3 from Phase 0 + baseline 10:5):
1. **Re-export with C++ (EC2):** Run `bar_feature_export --bar-type time --bar-param 5 --target T --stop S --output /work/results/geom_T_S/` on all 251 RTH days. Raw .dbn.zst on EBS. ~4 × 251 = 1,004 runs × ~2s = ~35 min. Upload Parquet to S3.
2. **Download Parquet (local):** Pull C++-produced Parquet from S3.
3. **Train XGBoost (local, Python):** Load pre-computed Parquet (Python ONLY loads data, never recomputes labels). Train with tuned params (max_depth=6, lr=0.0134, min_child_weight=20, subsample=0.561, colsample=0.748, reg_alpha=0.0014, reg_lambda=6.586).
4. CPCV (N=10, k=2, 45 splits) on dev set.
5. Record all Phase 1 metrics.

### Phase 2: Holdout Evaluation

For the best geometry by CPCV expectancy:
1. Train on full dev set.
2. Evaluate on holdout (50 days).
3. Record accuracy, expectancy, per-quarter breakdown.
4. **Key diagnostic:** Is long recall still pathologically low? If per-class recall is similarly asymmetric as baseline, the model is still learning barrier geometry, not direction.

### Phase 3: Comparative Analysis

1. Oracle heatmap visualization (full session + 3 time bands), with **long and short metrics shown separately**.
2. Geometry comparison table: all 4 trained geometries × all metrics.
3. Answer: does the model's directional accuracy track the oracle ceiling, or flatten out regardless of geometry?
4. If accuracy is geometry-invariant (~45% everywhere), the model has a fixed amount of directional signal and geometry just shifts the breakeven bar — the core finding.
5. **Asymmetric strategy analysis:** From the per-direction oracle heatmaps, identify if the optimal long geometry differs from the optimal short geometry. If long oracle peaks at (T_l, S_l) and short oracle peaks at (T_s, S_s), report the composite asymmetric expectancy. This doesn't require additional model training — it's a post-hoc analysis on Phase 0 data.
6. **Both-triggered analysis:** Report both-triggered rate across the geometry space. If correlated with known volatility regimes (FOMC, open/close), flag `tb_both_triggered` as a candidate trade filter — cutting worst trades is a cost-free way to improve expectancy without improving the model.

## Resource Budget

**Tier:** Standard (1-3 hours)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: xgboost
sequential_fits: 4
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 2.0
ec2_phases: [0, 1-export]  # Oracle sweep + Parquet re-export need raw MBO data on EBS
local_phases: [1-train, 2, 3]  # GBT training + analysis on pre-computed Parquet
```

**Breakdown:**
- Phase 0 (oracle sweep, EC2): ~12 min — 144 `oracle_expectancy` C++ runs on raw MBO data (~5s each)
- Phase 1 re-export (EC2): ~35 min — 1,004 `bar_feature_export` C++ runs on raw MBO data (~2s each)
- Phase 1 GBT training (local): ~12 min per geometry CPCV × 4 = ~48 min
- Phase 2 (holdout, local): ~5 min
- Phase 3 (analysis, local): ~5 min
- Total: ~50 min EC2 (c5.2xlarge, ~$0.50) + ~60 min local

## Abort Criteria

- Any geometry produces degenerate labels (>95% one class): skip that geometry, log warning.
- Phase 0 oracle heatmap shows no geometry with net expectancy > $5.00/trade: STOP at Phase 0, report Outcome C.
- Do NOT abort on wall-clock. Let the run complete.

## Critical Prerequisite: Bidirectional Labels

**ALL prior XGB results used flawed long-perspective-only labels.** The -1 label (price dropped 5 ticks) was treated as a short signal with a 10-tick profit target, but the label only validated a 5-tick move. This inflated short-side P/L.

**Expected impact of corrected labels:**
- Many bars previously labeled -1 (short) will become 0 (HOLD) — a 5-tick drop doesn't qualify as a 10-tick short signal
- The class distribution will shift toward more HOLD labels
- The tuned XGB expectancy (-$0.001) was likely an accounting error — expect meaningfully more negative with correct labels
- Short recall will drop (fewer genuine short signals exist)
- "Both-triggered" frequency provides a free volatility regime indicator

**The full-year Parquet has been re-exported with bidirectional labels (2026-02-25).** 312/312 files, 152-column schema, S3: `s3://kenoma-labs-research/results/bidirectional-reexport/`. This experiment is UNBLOCKED.

## Confounds to Watch For

1. **Win rate won't transfer across geometries.** A model achieving 45% on 10:5 labels will NOT achieve 45% on 15:3 labels. Each geometry is a different classification problem. The model must be retrained per geometry.
2. **Barrier asymmetry → class distribution bias.** With asymmetric barriers, the model may again learn to predict the majority directional class. Monitor per-class recall for pathological asymmetry (e.g., long recall < 0.15).
3. **Very narrow stops (2-3 ticks) get swept by noise.** 2-3 ticks is 0.50-0.75 ES points — normal MES volatility can easily hit this within a single 5-second bar. Expect high trade counts but low signal quality at narrow stops.
4. **Oracle ceiling is necessary but not sufficient.** A $10/trade oracle ceiling doesn't mean the model can capture $5 of it. The oracle uses perfect foresight; the model sees only features at entry time.
5. **Time-of-day effects may dominate geometry effects.** If opening-range oracle ceiling is 3× mid-session, the right answer may be "trade only the open" rather than "change the geometry."

## Deliverables

```
.kit/results/label-design-sensitivity/
  metrics.json                    # All phases: oracle heatmap + GBT metrics
  analysis.md                     # Comparative analysis, verdict, SC pass/fail
  oracle_heatmap_full.csv         # 144-row table: target, stop, combined + per-direction oracle metrics (full session)
  oracle_heatmap_band_a.csv       # Same for opening range
  oracle_heatmap_band_b.csv       # Same for mid-session
  oracle_heatmap_band_c.csv       # Same for close
  asymmetric_analysis.csv         # Best long geometry × best short geometry composite
  gbt_results.csv                 # GBT CPCV metrics per geometry (4 rows)
  class_distributions.csv         # Label distribution per geometry (144 rows)
  holdout_results.json            # Best geometry holdout evaluation
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. **Oracle heatmap (full session)** — 16×9 grid, color-coded by net expectancy
3. **Oracle heatmap (time bands)** — 3 additional grids for Bands A, B, C
4. Trade count heatmap — where in the geometry space do we get enough trades?
5. Top-3 geometry selection rationale
6. GBT CPCV results table (4 geometries × all metrics)
7. Per-class recall comparison — did the pathological long/short asymmetry persist?
8. Breakeven margin analysis (accuracy - breakeven_WR) for each geometry
9. Holdout evaluation for best geometry
10. **Key diagnostic:** Does accuracy track oracle ceiling, or is it geometry-invariant?
11. Explicit pass/fail for SC-1 through SC-6

## Exit Criteria

- [ ] Oracle heatmap computed for all 144 geometries (full session)
- [ ] Time-of-day oracle heatmaps computed for Bands A, B, C
- [ ] Top-3 geometries selected with rationale
- [ ] GBT CPCV completed for top-3 + baseline (4 geometries)
- [ ] Best geometry evaluated on holdout
- [ ] All metrics reported in metrics.json
- [ ] analysis.md written with heatmaps, comparison tables, and verdict
- [ ] Decision rule applied (Outcome A/B/C/D)
- [ ] Per-class recall reported — long/short asymmetry diagnosed
