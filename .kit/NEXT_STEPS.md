# Next Steps — Updated 2026-02-24

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

- **27+ phases complete** (10 engineering + 17 research). All on `main`, clean.
- **CNN line CLOSED** for classification (Outcome D, 2026-02-22). GBT beats CNN by 5.9pp accuracy.
- GBT-only on full-year CPCV: accuracy 0.449, expectancy -$0.064 base.
- **Q1-Q2 marginally profitable** (+$0.003, +$0.029) with default hyperparameters — never tuned.
- Breakeven requires +2.0pp win rate (51.3% → 53.3%) OR alternative label geometry.
- Full-year dataset production-ready: 1.16M bars, 251 days, 255MB Parquet.
- Parallel batch dispatch shipped. Cloud pipeline verified E2E.

## Priority Experiments

### 1. XGBoost Hyperparameter Tuning (HIGHEST PRIORITY)

Default XGBoost params inherited from 9B (broken pipeline era) — never optimized. The 2pp win rate gap is small enough that tuning could close it. Q1-Q2 already positive at default params.

**Spec:** `.kit/experiments/xgb-hyperparam-tuning.md`
**Branch:** `experiment/xgb-hyperparam-tuning`
**Compute:** Local (CPU-only, 255MB data, ~2-3 hours on Apple Silicon)

### 2. Label Design Sensitivity (PARALLEL WITH #1)

Test alternative triple barrier parameters. At 15:3 ratio, breakeven win rate drops to ~42.5% — well below current 51.3%. Orthogonal to model architecture.

**Spec:** `.kit/experiments/label-design-sensitivity.md`
**Branch:** `experiment/label-design-sensitivity`
**Compute:** Local (CPU-only, requires re-export from C++ oracle_expectancy tool)

### 3. Regime-Conditional Trading (AFTER #1 AND #2)

GBT profitable in H1 2022 (+$0.003, +$0.029), negative in H2. Test Q1-Q2-only strategy. Limited by single year of data — cannot validate regime prediction without 2023+ data.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

### 4. Tick_100 Multi-Seed Replication (INDEPENDENT, LOW PRIORITY)

R3b-genuine showed tick_100 R²=0.124 (+39%) but p=0.21 — driven by fold 5 outlier. Needs 3 seeds/fold and adjacent thresholds to confirm.

**Spec:** Not yet created
**Branch:** `experiment/tick100-replication`
**Compute:** Local or RunPod (CNN training)

---

## Key Constraints

- **GBT baseline**: accuracy 0.449, expectancy -$0.064 (base costs $3.74 RT)
- **Breakeven**: 53.3% win rate at current label geometry (10:5 TB)
- **Full-year data**: `.kit/results/full-year-export/` (S3 artifact store — `artifact-store hydrate` if needed)
- **Compute preference**: Local for CPU-only (<1GB data). RunPod for GPU. EC2 spot only for large data (>10GB).
- **Orchestrator protocol**: YOU are the orchestrator. You NEVER write code. Delegate via kit phases.

---

Written: 2026-02-24. All branches clean. No worktrees active.
