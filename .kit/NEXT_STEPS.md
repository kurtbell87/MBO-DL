# Next Steps — Updated 2026-02-26

## FIRST: Create a New Branch

**BEFORE you do ANY work — create a worktree and branch.**

```bash
git worktree add ../MBO-DL-<slug> -b <branch-type>/<name> main
cd ../MBO-DL-<slug>
source .orchestration-kit.env
```

**All work happens on a branch. Never touch main directly.**

---

## Current State

- **30+ phases complete** (15 engineering + 21 research). All on `main`, clean.
- **Label Geometry 1h — INCONCLUSIVE** (2026-02-26). Time horizon fix WORKS (hold 90.7%→32.6% at 10:5). But model hits ~50-51% directional accuracy ceiling. At high-ratio geometries, model becomes hold predictor (<0.3% trade rate). Binding constraint: feature-label correlation, not geometry. PR #33.
- **Time Horizon CLI — COMPLETE** (2026-02-26). `--max-time-horizon` and `--volume-horizon` flags added. Defaults 300→3600s, 500→50000. PR #32.
- **Label Geometry Phase 1 — REFUTED** (2026-02-26). 90.7-98.9% hold at 300s cap. Root cause fixed in PR #32. PR #31.
- **Synthesis-v2 — GO verdict** (2026-02-26). 55-60% prior. Breakeven WR is the correct metric. PR #30 merged.
- **CNN line CLOSED** for classification. GBT beats CNN by 5.9pp accuracy.
- **XGBoost tuning DONE** — 0.33pp plateau. Feature set is the binding constraint.
- GBT-only on full-year CPCV: accuracy 0.449 (long-perspective), 0.384 (bidirectional).
- **Critical finding:** Directional accuracy is ~50-51% regardless of label type, geometry, or hyperparameters. This is the hard ceiling of 20 microstructure features on MES time_5s bars.

## Priority Experiments

### 1. 2-Class Directional vs Hold at 19:7 (HIGHEST PRIORITY)

Train binary XGBoost: "will this bar be directional?" on 19:7 labels (47.4% directional, near-balanced). If model predicts barrier-reachability with >60% accuracy, a second-stage direction model on predicted-directional bars could exploit the favorable 2.71:1 payoff. Decouples what the model CAN learn (volatility/barrier reachability) from what it CANNOT (direction at wider barriers).

**Rationale:** Feature importance at 19:7 concentrates on volatility features (high_low_range_50 at 43.5% of top-10 gain). The model already detects barrier-reachability — it just can't distinguish direction. A 2-class formulation formalizes this.

**Spec:** Not yet created
**Branch:** `experiment/2class-directional`
**Compute:** Local

### 2. Class-Weighted XGBoost at 19:7 (ALTERNATIVE TO #1)

Force the model to trade by up-weighting directional classes 3:1 vs hold. Accept lower overall accuracy for higher directional prediction rate. Question: does forced directional accuracy stay above 38.4% (19:7 BEV WR)?

**Spec:** Not yet created
**Branch:** `experiment/class-weighted-19-7`
**Compute:** Local

### 3. Long-Perspective Labels at Varied Geometries (ORIGINAL QUESTION)

The original geometry hypothesis can still be tested on long-perspective labels (`--legacy-labels --max-time-horizon 3600`). Prior ~45% accuracy was on long-perspective labels with balanced classes. Test if this transfers to high-ratio geometries.

**Spec:** Not yet created
**Branch:** `experiment/label-geometry-legacy`
**Compute:** Local

### 4. Feature Engineering for Wider Barriers (HIGHEST EFFORT)

Current 20 features capture ~5s microstructure. Add rolling VWAP slope, cumulative order flow (50-500 bars), intraday trend indicators, volatility regime markers. Addresses root cause (feature ceiling) but requires TDD cycle + new export.

**Spec:** Not yet created
**Branch:** `feat/wider-barrier-features`
**Compute:** Local

### 5. Regime-Conditional Trading (INDEPENDENT)

GBT profitable in H1 2022 (+$0.003, +$0.029), negative in H2. Test Q1-Q2-only strategy.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| Label Geometry 1h | INCONCLUSIVE (Outcome B) | Dir acc 50.7%, model refuses to trade at high-ratio geom (<0.3% rate). Feature ceiling. |
| Time Horizon CLI TDD | DONE (PR #32) | `--max-time-horizon`/`--volume-horizon` flags. Defaults 300→3600s, 500→50000. |
| Label Geometry Phase 1 | REFUTED | 90.7-98.9% hold — root cause: 300s time cap (now fixed). |
| Synthesis-v2 | GO (55-60% prior) | Breakeven WR is the correct metric. Label geometry is the remaining lever. |
| Label Design Sensitivity P0 | REFUTED (Outcome C) | 123 geometries mapped. $5.00 gate miscalibrated. |
| XGBoost Tuning | REFUTED (Outcome C) | 0.33pp plateau across 64 configs. |
| Bidirectional Re-Export | PASS (312/312) | 152-col schema, S3 backed. |
| bar-feature-export-geometry TDD | DONE (PR #28) | --target/--stop CLI flags, 47 tests. |

---

## Key Constraints

- **Directional accuracy ceiling**: ~50-51% regardless of geometry, labels, hyperparameters. This is THE binding constraint.
- **GBT baseline**: accuracy 0.449 (long-persp) / 0.384 (bidirectional), expectancy -$0.064 / -$0.49
- **Breakeven**: 53.3% at 10:5, 38.4% at 19:7, 33.3% at 15:3, 29.6% at 20:3
- **Oracle at 3600s**: $3.22-$9.44/trade — perfect foresight IS profitable.
- **Model behavior at wider barriers**: becomes hold predictor (<0.3% dir prediction rate). Feature importance concentrates on volatility (barrier reachability), not direction.
- **10:5 is only reliable geometry**: 90.4% dir prediction rate, ~1.05M directional pairs. All others have <3,280 trades.
- **Compute preference**: Local for CPU-only (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB).
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-26. Label geometry 1h INCONCLUSIVE. Feature-label ceiling is binding. Next: 2-class formulation or class-weighted training at 19:7.
