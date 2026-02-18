# Next Steps: R4d Re-run — Missing Operating Points

## What happened

R4d SURVEY-FRAME-RUN-READ-LOG cycle completed, but the RUN agent only tested **2 of 8 planned operating points** (dollar_5M at 7s and tick_3000 at 300s). It skipped the middle of the timescale range, leaving a massive gap between 7s and 300s.

## What was fixed this session

Mandatory Compute Profile enforcement added to orchestration-kit (3 files changed, committed on orchestration-kit main):
- `research-kit/.claude/prompts/frame.md` — Compute Profile now mandatory in spec structure
- `tools/cloud/spec_parser.py` — raises ValueError if block missing (was silent zeros)
- `research-kit/experiment.sh` — RUN phase aborts on missing profile (was `2>/dev/null || true`)

Compute Profile block added to R4d spec. Preflight now correctly recommends EC2 c7a.8xlarge.

## What needs to happen now

Re-run R4d's RUN phase to cover the **missing operating points** that have adequate data:

| Bar type | Threshold | Bars (19d) | Median duration | Status |
|----------|-----------|-----------|-----------------|--------|
| dollar | $5M | 47,865 | 7.0s | **DONE** |
| dollar | $10M | 23,648 | 13.9s | **MISSING — must run** |
| dollar | $50M | 3,994 | 69.3s | **MISSING — must run** |
| tick | 500 | 7,923 | 50.0s | **MISSING — must run** |
| tick | 3,000 | 513 | 300.0s | **DONE** (underpowered) |

The longer timescales are genuinely data-limited (dollar_250M=55 bars, dollar_1B/tick_10K/tick_25K=0 exported bars). These can be documented as infeasible.

## How to run

The calibration is already done. Feature CSVs for dollar_5M and tick_3000 exist. Need to:

1. Export features for dollar_10M, dollar_50M, tick_500 via `bar_feature_export`
2. Run R4 temporal analysis protocol on each
3. Update the cross-timescale analysis with all 5 operating points
4. Re-evaluate decision framework

```bash
source .orchestration-kit.env
.kit/experiment.sh run .kit/experiments/temporal-predictability-dollar-tick-actionable.md
```

The spec already has the Compute Profile block, so preflight will inject the cloud advisory. The RUN agent needs to be told (or the spec amended) to cover the missing operating points.

**Alternative:** Amend the spec's Phase 3 to explicitly list the 5 required operating points so the RUN agent can't skip them. The spec currently says "select thresholds whose median is closest to each target" but doesn't enforce running all of them.

## Key files

- Spec: `.kit/experiments/temporal-predictability-dollar-tick-actionable.md`
- Existing results: `.kit/results/temporal-predictability-dollar-tick-actionable/`
- Calibration: `.kit/results/temporal-predictability-dollar-tick-actionable/calibration/calibration_table.json`
- Analysis script: `research/R4d_temporal_predictability_dollar_tick_actionable.py`
- Research log entry: `.kit/RESEARCH_LOG.md` (top entry, will need updating after re-run)

## Current branch

`experiment/temporal-predictability-dollar-tick-actionable` — pushed to origin.
