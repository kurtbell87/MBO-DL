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

### Just Completed: Oracle Expectancy Parameterization (experiment/xgb-hyperparam-tuning)

Parameterized the `oracle_expectancy` CLI tool with `--target`, `--stop`, `--take-profit`, `--output`, and `--help` flags. This enables scripted label design sensitivity sweeps without recompiling. 49 new tests all pass. Spec: `.kit/docs/oracle-expectancy-params.md`.

**What was done this cycle (2026-02-25):**
- Modified `tools/oracle_expectancy.cpp` — CLI arg parsing (--target, --stop, --take-profit, --output, --help)
- Modified `src/backtest/oracle_expectancy_report.hpp` — JSON output support
- Modified `CMakeLists.txt` — added oracle_expectancy_params_test target
- Created `tests/oracle_expectancy_params_test.cpp` — 49 tests (binary, help, invalid args, backward compat, modified geometry, JSON output, flag order)
- Modified `.kit/experiments/label-design-sensitivity.md` — updated with CLI usage

**Also created (not part of TDD cycle):**
- `.kit/docs/bidirectional-label-export.md` — new spec (untracked)

**Key files:**
- Spec: `.kit/docs/oracle-expectancy-params.md`
- Tool: `tools/oracle_expectancy.cpp`
- Header: `src/backtest/oracle_expectancy_report.hpp`
- Tests: `tests/oracle_expectancy_params_test.cpp`
- Label sensitivity spec: `.kit/experiments/label-design-sensitivity.md`

**Next steps:**
1. Commit oracle-expectancy-params changes
2. Run label design sensitivity experiment: `oracle_expectancy --target 15 --stop 3 --output results.json`
3. XGBoost hyperparameter tuning (P1)

### Background: CNN Line Closed

The CNN spatial signal on order book snapshots is **real and reproducible** (R²=0.089, 3 independent reproductions). But end-to-end CNN classification (Outcome D) showed GBT-only beats CNN by 5.9pp accuracy. CNN line is permanently closed for classification.

**Priority research tasks:**
1. **XGBoost hyperparameter tuning** — default params never optimized. GBT shows Q1-Q2 positive expectancy. Batch dispatch enables parallel sweeps.
2. **Label design sensitivity** — infrastructure now ready. `oracle_expectancy --target 15 --stop 3 --output results.json`. Spec: `.kit/experiments/label-design-sensitivity.md`.
3. **Regime-conditional trading** — Q1-Q2 only strategy.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 → integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() → z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**28+ phases complete (11 engineering + 12 research + 1 data export + 1 infra + 3 kit modifications). Branch: `experiment/xgb-hyperparam-tuning`. 1144 unit tests registered, 49 new oracle_expectancy_params tests all pass.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export. 1003+ unit tests, 22 integration tests, 28 Parquet tests.
- **Full-year dataset**: 251 Parquet files (time_5s bars, 1,160,150 bars, 149 columns, zstd compression). Stored in S3 artifact store.
- **Cloud pipeline**: Docker image in ECR, EBS snapshot with 49GB MBO data, IAM profile. Verified E2E.
- **EC2 mandatory execution**: `experiment.sh` mandates cloud-run for RUN phases when `COMPUTE_TARGET=ec2`.
- **Parallel batch dispatch (COMPLETE)**: `cloud-run batch run` launches N experiments in parallel on separate EC2 instances. MCP tool `kit.research_batch` for orchestrator use. `experiment.sh batch` for shell-level dispatch.

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
| **10** | **E2E CNN classification** | **GBT wins by 5.9pp** | **CNN line closed for classification** |
| FYE | Full-year export | 251 days, 1.16M bars | 13× more data available |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all 12+ experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/docs/oracle-expectancy-params.md`** — latest completed TDD spec

---

Updated: 2026-02-25. Next action: commit oracle-expectancy-params, then label design sensitivity experiment or XGBoost hyperparameter tuning.
