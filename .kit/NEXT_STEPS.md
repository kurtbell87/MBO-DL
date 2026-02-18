# Next Steps: Model Architecture Build Spec

## What happened last session

R4d re-run completed successfully. The original RUN agent only tested 2 of 5 needed operating points (dollar_5M, tick_3000). This session:

1. **Fixed preflight bug** — `experiment.sh` invoked `tools/cloud/preflight.py` directly (broken relative imports). Changed to `tools/preflight` wrapper. Fix is in `orchestration-kit/research-kit/experiment.sh` line 482/490.
2. **Amended R4d spec** — Arm 2 now explicitly enumerates 5 fixed operating points with DONE/MISSING status so the RUN agent can't skip them. Phase 1 marked COMPLETE (skip calibration).
3. **Re-ran RUN-READ-LOG cycle** — All 3 phases exited 0. Results for dollar_10M, dollar_50M, tick_500 now exist alongside dollar_5M and tick_3000.
4. **Updated all breadcrumbs** — SC (5/6 pass), exit criteria (11/11), RESEARCH_LOG.md, CLAUDE.md Current State, this file.

## Current state

**All research phases are complete.** R4d confirmed with full coverage: 0/38 dual threshold passes across 5 operating points spanning 7s–300s. Cumulative R4 chain: 0/168+ passes across 7 bar types, 0.14s–300s. R4 line permanently closed.

The TRAJECTORY.md §13 audit stands: 21/21 engineering PASS, 13/13 research PASS.

## What needs to happen now

**Write the model architecture build spec** — the CNN+GBT Hybrid. This is the next engineering phase. All research inputs are locked:

- **Architecture:** CNN + GBT Hybrid (R6 synthesis)
- **Spatial encoder:** CNN on structured (20,2) book input (R3: R²=0.132)
- **Temporal encoder:** None (R4 chain: 0/168+ passes)
- **Message encoder:** None (R2: features sufficient)
- **Bar type:** time_5s (R1: subordination refuted)
- **Labels:** Triple barrier (R7b: $4.00/trade expectancy, PF=3.30, WR=64.3%)
- **Horizons:** h=1 and h=5 (R6 synthesis)
- **Features:** Static book features only

### Open questions to address in the build spec

1. **CNN at h=1** — R3's R²=0.132 was at h=5. Does it hold at h=1? This could be tested as part of the build or as a quick R5 experiment.
2. **Transaction cost sensitivity** — How robust is oracle expectancy to spread widening, commission increases, adverse fills? Could be a standalone experiment or baked into the build spec's evaluation criteria.
3. **CNN+GBT integration pipeline** — How do CNN embeddings feed into GBT? Concatenation with hand-crafted features? End-to-end? Two-stage?

### Key files

- Synthesis results: `.kit/results/synthesis/analysis.md`, `.kit/results/synthesis/metrics.json`
- Oracle expectancy: `.kit/results/oracle-expectancy/metrics.json`
- R4d final analysis: `.kit/results/temporal-predictability-dollar-tick-actionable/analysis.md`
- Calibration table: `.kit/results/temporal-predictability-dollar-tick-actionable/calibration/calibration_table.json`
- TRAJECTORY.md: project-level phase sequence and exit criteria

## Branch

`experiment/temporal-predictability-dollar-tick-actionable` — changes not yet committed. Uncommitted changes:
- `orchestration-kit/research-kit/experiment.sh` (preflight fix)
- `.kit/experiments/temporal-predictability-dollar-tick-actionable.md` (spec amendments + exit criteria)
- `.kit/NEXT_STEPS.md` (this file)
- `CLAUDE.md` (current state update)
- `.kit/RESEARCH_LOG.md` (updated by LOG phase)
- `.kit/results/temporal-predictability-dollar-tick-actionable/` (new operating point results)

Consider committing these changes before starting the build spec work.
