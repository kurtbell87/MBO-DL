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

- **30+ phases complete** (15 engineering + 20 research). All on `main`, clean.
- **Time Horizon CLI — COMPLETE** (2026-02-26). `--max-time-horizon` and `--volume-horizon` flags added to both tools. Defaults: 300→3600s, 500→50000. PR #32. **Fixes root cause of degenerate hold rates.**
- **Label Geometry Phase 1 — REFUTED** (2026-02-26). 90.7-98.9% hold caused by 5-minute time horizon cap. Root cause now fixed (PR #32). PR #31.
- **Synthesis-v2 — GO verdict** (2026-02-26). 55-60% prior. Breakeven WR is the correct metric. PR #30 merged.
- **Label Design Sensitivity Phase 0 DONE** (2026-02-26). 123 geometries mapped. PR #29 merged.
- **CNN line CLOSED** for classification. GBT beats CNN by 5.9pp accuracy.
- **XGBoost tuning DONE** — 0.33pp plateau. Feature set is the binding constraint.
- GBT-only on full-year CPCV: accuracy 0.449, expectancy -$0.064 base.
- **Q1-Q2 marginally profitable** (+$0.003, +$0.029) under base costs.
- **Root cause found:** The 90.7-98.9% hold rate was NOT a fundamental data problem — it was a 300-second time horizon cap. MES price doesn't traverse 10-tick barriers in 5 minutes on most bars. With the 3600s cap, labels should reflect realistic hold times (seconds to 1 hour).

## Priority Experiments

### 1. Geometry Sweep with 1-Hour Time Horizon (HIGHEST PRIORITY)

Re-run the geometry hypothesis with `--max-time-horizon 3600`. This fixes the root cause that made phase 1 untestable. Test BOTH label types:
- Bidirectional labels (default): `--target T --stop S --max-time-horizon 3600`
- Long-perspective labels: `--legacy-labels --target T --stop S --max-time-horizon 3600`

**Prerequisite (30-min verification):** Export 1 day at 10:5 with `--max-time-horizon 3600` (both label types). Confirm class distribution is NOT >90% hold. If balanced, proceed with full sweep.

**Spec:** Not yet created (adapt from label-geometry-phase1.md, add `--max-time-horizon 3600`)
**Branch:** `experiment/label-geometry-1h`
**Compute:** Local

### 2. Regime-Conditional Trading (AFTER #1)

GBT profitable in H1 2022 (+$0.003, +$0.029), negative in H2. Test Q1-Q2-only strategy.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

### 3. Tick_100 Multi-Seed Replication (INDEPENDENT, LOW PRIORITY)

R3b-genuine showed tick_100 R2=0.124 (+39%) but p=0.21. Needs 3 seeds/fold.

**Spec:** Not yet created
**Branch:** `experiment/tick100-replication`
**Compute:** Local or RunPod

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| Time Horizon CLI TDD | DONE (PR #32) | `--max-time-horizon`/`--volume-horizon` flags. Defaults 300→3600s, 500→50000. |
| Label Geometry Phase 1 | REFUTED | 90.7-98.9% hold — root cause: 300s time cap (now fixed). |
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
- **Root cause from label-geom-p1**: 300s time horizon was too short — fixed in PR #32. Re-run with `--max-time-horizon 3600`.
- **Bidirectional data**: `s3://kenoma-labs-research/results/bidirectional-reexport/` (312 files, 152 columns)
- **Full-year data**: `.kit/results/full-year-export/` (S3 artifact store)
- **Compute preference**: Local for CPU-only (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB).
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-26. Time horizon fix COMPLETE (PR #32). Next: re-run geometry sweep with `--max-time-horizon 3600`.
