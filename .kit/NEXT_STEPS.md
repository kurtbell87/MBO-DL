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

- **30+ phases complete** (14 engineering + 20 research). All on `main`, clean.
- **Label Geometry Phase 1 — REFUTED** (2026-02-26). Bidirectional labels + time_5s = 90.7-98.9% hold at ALL geometries. Model degenerates to hold-predictor. Geometry hypothesis untestable on these labels. PR #31.
- **Synthesis-v2 — GO verdict** (2026-02-26). 55-60% prior. Breakeven WR is the correct metric. PR #30 merged.
- **Label Design Sensitivity Phase 0 DONE** (2026-02-26). 123 geometries mapped. PR #29 merged.
- **CNN line CLOSED** for classification. GBT beats CNN by 5.9pp accuracy.
- **XGBoost tuning DONE** — 0.33pp plateau. Feature set is the binding constraint.
- GBT-only on full-year CPCV: accuracy 0.449, expectancy -$0.064 base.
- **Q1-Q2 marginally profitable** (+$0.003, +$0.029) under base costs.
- **Critical finding:** The ~45% accuracy baseline was on long-perspective labels with balanced classes. Bidirectional labels produce degenerate distributions on 5s bars.

## Priority Experiments

### 1. Geometry Sweep on Long-Perspective Labels (HIGHEST PRIORITY)

Re-run the geometry hypothesis using long-perspective labels (`--legacy-labels`) which produce balanced class distributions. This isolates the geometry effect from the label-type confound discovered in label-geometry-phase1.

**Prerequisite (30-min verification):** Export 1 day at 10:5 and 15:3 with `--legacy-labels`. Confirm class distribution is balanced (not >90% hold). If balanced, proceed.

**Spec:** Not yet created (adapt from label-geometry-phase1.md, change to `--legacy-labels`)
**Branch:** `experiment/label-geometry-legacy`
**Compute:** Local

### 2. Label Distribution Characterization (ALTERNATIVE TO #1)

Before a full sweep, export 1 day at 4-6 geometries under BOTH label types and tabulate class distributions. 30-min investigation that prevents another wasted experiment.

### 3. Regime-Conditional Trading (AFTER #1)

GBT profitable in H1 2022 (+$0.003, +$0.029), negative in H2. Test Q1-Q2-only strategy.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

### 4. Tick_100 Multi-Seed Replication (INDEPENDENT, LOW PRIORITY)

R3b-genuine showed tick_100 R2=0.124 (+39%) but p=0.21. Needs 3 seeds/fold.

**Spec:** Not yet created
**Branch:** `experiment/tick100-replication`
**Compute:** Local or RunPod

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| Label Geometry Phase 1 | REFUTED | Bidirectional labels + 5s bars = 90.7-98.9% hold. Geometry hypothesis untestable. |
| Synthesis-v2 | GO (55-60% prior) | Breakeven WR is the correct metric. Label geometry is the remaining lever. |
| Label Design Sensitivity P0 | REFUTED (Outcome C) | 123 geometries mapped. $5.00 gate miscalibrated. |
| XGBoost Tuning | REFUTED (Outcome C) | 0.33pp plateau across 64 configs. |
| Bidirectional Re-Export | PASS (312/312) | 152-col schema, S3 backed. |
| bar-feature-export-geometry TDD | DONE (PR #28) | --target/--stop CLI flags, 47 tests. |

---

## Key Constraints

- **GBT baseline**: accuracy 0.449, expectancy -$0.064 (base costs $3.74 RT)
- **Breakeven**: 53.3% win rate at current label geometry (10:5 TB). Tuning cannot close this gap.
- **Key insight from synthesis-v2**: At 15:3, breakeven drops to 33.3% — 12pp below model's 45%.
- **Critical from label-geom-p1**: Bidirectional labels + 5s bars = 91-99% hold. Must use long-perspective labels for geometry testing.
- **Bidirectional data**: `s3://kenoma-labs-research/results/bidirectional-reexport/` (312 files, 152 columns)
- **Full-year data**: `.kit/results/full-year-export/` (S3 artifact store)
- **Compute preference**: Local for CPU-only (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB).
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-26. Label geometry phase 1 REFUTED. Next: verify long-perspective label distributions at varied geometries.
