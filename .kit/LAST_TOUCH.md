# Last Touch — Cold-Start Briefing

**Read this file first. Then read `CLAUDE.md` for the full protocol. Do not read source files.**

---

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, launch kit phases (MCP tools or bash), check exit codes via dashboard
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list of things you must never do

---

## TL;DR — Where We Are and What To Do

We've built a complete MES microstructure research platform over 24 phases. The CNN spatial signal on order book snapshots is **real and reproducible** (R²=0.089, 3 independent reproductions). But the prior pipeline (CNN regression → frozen embeddings → XGBoost classification) loses too much information at the handoff — expectancy is -$0.37/trade, just $0.37 short of breakeven.

**Your job: launch the end-to-end CNN classification experiment.**

The spec is ready: `.kit/experiments/e2e-cnn-classification.md`
A survey has already been completed: `.kit/experiments/survey-e2e-cnn-classification.md`

This experiment trains the CNN directly on 3-class triple barrier labels (CrossEntropyLoss) instead of going through a regression intermediary. It runs on the full-year dataset (251 days, 1.16M bars — 13× more data than prior experiments) with a rigorous CPCV validation protocol that generates 9 independent backtest paths and computes Probability of Backtest Overfitting.

### NEW: EC2 Mandatory Execution (2026-02-21)

**All compute-heavy work runs on EC2. The research kit has been modified to enforce this.**

`experiment.sh` now has:
- `COMPUTE_TARGET=ec2` set in `.orchestration-kit.env`
- A mandatory compute directive injected into the RUN sub-agent prompt when EC2 is active
- A `sync_results()` function that pulls results from cloud-run/S3 between RUN and READ phases
- This works automatically with `experiment.sh full` and `experiment.sh cycle`

The survey phase is already done. You can skip it and run:
```bash
source .orchestration-kit.env
# Use MCP tool:
kit.research_cycle spec_path=".kit/experiments/e2e-cnn-classification.md"
# Or bash fallback:
.kit/experiment.sh cycle .kit/experiments/e2e-cnn-classification.md
```

This will run: FRAME → RUN (on EC2) → sync results → READ → LOG

If you want the full cycle including survey (survey is already done, so it'll be fast):
```bash
kit.research_full question="Can end-to-end CNN classification on tb_label close the viability gap?" spec_path=".kit/experiments/e2e-cnn-classification.md"
```

### IMPORTANT: Hydrate Artifacts First

If Parquet files show as broken symlinks, run before launching:
```bash
orchestration-kit/tools/artifact-store hydrate
```

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 → integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() → z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**25 phases complete (10 engineering + 12 research + 1 data export + 1 infra + 1 kit modification). Branch: `main`. Working tree: clean.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export. 1003+ unit tests, 22 integration tests, 28 Parquet tests.
- **Full-year dataset**: 251 Parquet files (time_5s bars, 1,160,150 bars, 149 columns, zstd compression). Stored in S3 artifact store — run `orchestration-kit/tools/artifact-store hydrate` after clone/worktree to restore.
- **Cloud pipeline**: Docker image in ECR (`651323680805.dkr.ecr.us-east-1.amazonaws.com/mbo-dl`), EBS snapshot with 49GB MBO data (`snap-0efa355754c9a329d`), IAM profile (`cloud-run-ec2`). Verified end-to-end on 2026-02-21.
- **EC2 mandatory execution**: `experiment.sh` modified to mandate cloud-run for RUN phases when `COMPUTE_TARGET=ec2` (2026-02-21).

### Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
| R1 | Subordination refuted | 0/3 significant | Time bars are the baseline |
| R2 | Features sufficient | R²=0.0067 | Book snapshot is sufficient statistic |
| R3 | CNN best encoder | R²=0.132 (leaked) / 0.084 (proper) | Spatial structure matters |
| R4/R4b/R4c/R4d | No temporal signal | 0/168+ passes | Drop SSM/temporal encoder permanently |
| R6 | Synthesis | CONDITIONAL GO → GO | CNN + GBT Hybrid architecture |
| R7 | Oracle expectancy | $4.00/trade, PF=3.30 | Edge exists at oracle level |
| 9B→9D | Normalization root cause | R²: -0.002 → 0.089 | TICK_SIZE + per-day z-score required |
| **9E** | **Pipeline bottleneck** | **exp=-$0.37/trade** | **Regression→classification gap is the limit** |
| FYE | Full-year export | 251 days, 1.16M bars | 13× more data, unblocks next experiment |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all 12+ experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/e2e-cnn-classification.md`** — the next experiment spec (576 lines, fully specified)
6. **`.kit/experiments/survey-e2e-cnn-classification.md`** — survey already completed

---

Updated: 2026-02-21. Next action: launch E2E CNN classification experiment.
