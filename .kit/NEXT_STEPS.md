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

- **30+ phases complete** (14 engineering + 19 research). All on `main`, clean.
- **Synthesis-v2 — GO verdict** (2026-02-26). Breakeven WR is the correct metric, not oracle ceiling. 55-60% prior at high-ratio geometries. PR #30 merged.
- **Label Design Sensitivity Phase 0 DONE** (2026-02-26). 123 geometries mapped. Phase 1 training aborted by miscalibrated $5.00 gate. PR #29 merged.
- **CNN line CLOSED** for classification (Outcome D, 2026-02-22). GBT beats CNN by 5.9pp accuracy.
- **XGBoost tuning DONE** (2026-02-24) — REFUTED (Outcome C). Accuracy is a 0.33pp plateau. Feature set is the binding constraint.
- GBT-only on full-year CPCV: accuracy 0.449, expectancy -$0.064 base. Breakeven RT=$3.74 exactly.
- **Q1-Q2 marginally profitable** (+$0.003, +$0.029) under base costs.
- **Bidirectional re-export COMPLETE** (2026-02-25). 312/312 files, 152-column schema.
- Parallel batch dispatch shipped. Cloud pipeline verified E2E.

## Priority Experiments

### 1. Label Geometry Phase 1 Training (HIGHEST PRIORITY — IN PROGRESS)

Train XGBoost at 4 breakeven-favorable geometries (10:5, 15:3, 19:7, 20:3). Test whether model accuracy transfers to high-ratio label distributions where favorable payoff structure enables positive expectancy.

**Decision tree:**
- Outcome A (accuracy > BEV WR + 2pp, expectancy > $0): Viable -> regime-conditional + multi-year
- Outcome B (accuracy stable, expectancy fails): Payoff insufficient -> 2-class/class-weight
- Outcome C (accuracy drops >10pp): Label distribution breaks model -> per-direction asymmetric

**Spec:** `.kit/experiments/label-geometry-phase1.md`
**Branch:** `experiment/label-geometry-phase1`
**Compute:** Local (CPU-only, ~82 min estimated)

### 2. Regime-Conditional Trading (AFTER #1)

GBT profitable in H1 2022 (+$0.003, +$0.029), negative in H2. Test Q1-Q2-only strategy. Limited by single year of data.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

### 3. Tick_100 Multi-Seed Replication (INDEPENDENT, LOW PRIORITY)

R3b-genuine showed tick_100 R2=0.124 (+39%) but p=0.21. Needs 3 seeds/fold.

**Spec:** Not yet created
**Branch:** `experiment/tick100-replication`
**Compute:** Local or RunPod (CNN training)

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| Synthesis-v2 | GO (55-60% prior) | Breakeven WR is the correct metric. Label geometry is the remaining lever. |
| Label Design Sensitivity P0 | REFUTED (Outcome C) | 123 geometries mapped. $5.00 gate miscalibrated. Phase 1 never attempted. |
| XGBoost Tuning | REFUTED (Outcome C) | 0.33pp plateau across 64 configs. Feature set is binding constraint. |
| Bidirectional Re-Export | PASS (312/312) | 152-col schema, S3 backed. |
| bar-feature-export-geometry TDD | DONE (PR #28) | --target/--stop CLI flags, 47 tests. |

---

## Key Constraints

- **GBT baseline**: accuracy 0.449, expectancy -$0.064 (base costs $3.74 RT)
- **Breakeven**: 53.3% win rate at current label geometry (10:5 TB). Tuning cannot close this gap.
- **Key insight from synthesis-v2**: At 15:3, breakeven drops to 33.3% — 12pp below model's 45%.
- **Bidirectional data**: `s3://kenoma-labs-research/results/bidirectional-reexport/` (312 files, 152 columns)
- **Full-year data**: `.kit/results/full-year-export/` (S3 artifact store)
- **Compute preference**: Local for CPU-only (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB).
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-26. Label geometry phase 1 experiment in progress.
