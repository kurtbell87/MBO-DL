# Next Steps — Updated 2026-02-25

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

- **30+ phases complete** (13 engineering + 17 research). All on `main`, clean.
- **CNN line CLOSED** for classification (Outcome D, 2026-02-22). GBT beats CNN by 5.9pp accuracy.
- **XGBoost tuning DONE** (2026-02-24) — REFUTED (Outcome C). Accuracy is a 0.33pp plateau (64 configs). Best tuned 0.4504 vs default 0.4489. Feature set is the binding constraint, not hyperparameters.
- GBT-only on full-year CPCV: accuracy 0.449, expectancy -$0.064 base. Breakeven RT=$3.74 exactly.
- **Q1-Q2 marginally profitable** (+$0.003, +$0.029) under base costs.
- **Bidirectional re-export COMPLETE** (2026-02-25). 312/312 files, 152-column schema, S3: `s3://kenoma-labs-research/results/bidirectional-reexport/`.
- Breakeven requires alternative label geometry (tuning exhausted) OR regime-conditional strategy.
- Parallel batch dispatch shipped. Cloud pipeline verified E2E.

## Priority Experiments

### 1. Label Design Sensitivity (HIGHEST PRIORITY — FULLY UNBLOCKED)

Test alternative triple barrier geometries using bidirectional labels. At 15:3 ratio, breakeven win rate drops to ~42.5% — well below current 51.3%. XGBoost tuning showed accuracy is plateaued at ~45% — label geometry is the remaining lever.

**POLICY (2026-02-25):** Python NEVER computes labels. C++ binaries (`oracle_expectancy`, `bar_feature_export`) operate on raw .dbn.zst MBO data. Python only loads pre-computed Parquet for model training. Experiment spec rewritten to enforce this.

**All prerequisites DONE:**
- Oracle CLI params (`--target/--stop`) — DONE
- Bidirectional TB labels — DONE (PR #26)
- Bidirectional export wiring — DONE (PR #27)
- Full-year re-export (312 files) — DONE (S3)
- `bar_feature_export --target/--stop` flags — **DONE** (PR #28, TDD cycle complete, 47 tests pass)

**Reexport diagnosis:** EC2 reexport produced 14KB header-only files (script/Docker path issue, not code bug). Local binary produces 17MB / 152 cols / 87,970 rows correctly. Rebuild Docker image from current main (includes --target/--stop) and re-run.

**Spec:** `.kit/experiments/label-design-sensitivity.md`
**Branch:** `experiment/label-design-sensitivity`
**Compute:** EC2 (oracle sweep + re-export) + Local (GBT training)

### 2. Regime-Conditional Trading (AFTER #1)

GBT profitable in H1 2022 (+$0.003, +$0.029), negative in H2. Test Q1-Q2-only strategy. Limited by single year of data — cannot validate regime prediction without 2023+ data.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

### 3. Tick_100 Multi-Seed Replication (INDEPENDENT, LOW PRIORITY)

R3b-genuine showed tick_100 R²=0.124 (+39%) but p=0.21 — driven by fold 5 outlier. Needs 3 seeds/fold and adjacent thresholds to confirm.

**Spec:** Not yet created
**Branch:** `experiment/tick100-replication`
**Compute:** Local or RunPod (CNN training)

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| XGBoost Tuning | REFUTED (Outcome C) | 0.33pp plateau across 64 configs. Feature set is binding constraint. |
| Bidirectional Re-Export | PASS (312/312) | 152-col schema, S3 backed. Unblocks label design sensitivity. |
| Bidirectional Export Wiring | DONE (PR #27) | `bar_feature_export` defaults to bidirectional labels. |
| Bidirectional TB Labels | DONE (PR #26) | `compute_bidirectional_tb_label()` — independent long/short races. |
| Oracle Expectancy Params | DONE | CLI `--target/--stop/--take-profit/--output/--help` flags. |
| bar-feature-export-geometry TDD | DONE (PR #28) | `--target/--stop` CLI flags, 47 tests, all passing. |

---

## Key Constraints

- **GBT baseline**: accuracy 0.449, expectancy -$0.064 (base costs $3.74 RT)
- **Breakeven**: 53.3% win rate at current label geometry (10:5 TB). Tuning cannot close this gap.
- **Bidirectional data**: `s3://kenoma-labs-research/results/bidirectional-reexport/` (312 files, 152 columns)
- **Full-year data**: `.kit/results/full-year-export/` (S3 artifact store — `artifact-store hydrate` if needed)
- **Compute preference**: Local for CPU-only (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB).
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-25. All branches clean. No worktrees active.
