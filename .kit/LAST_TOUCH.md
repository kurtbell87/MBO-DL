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

### Just Completed: Bidirectional Export Wiring (tdd/bidirectional-label-export)

TDD cycle wired `compute_bidirectional_tb_label()` into `bar_feature_export.cpp`. Parquet schema expanded 149 → 152 columns. `--legacy-labels` flag for backward compatibility. Tests T1-T6 pass. TDD phase exited 0. Spec: `.kit/docs/bidirectional-export-wiring.md`.

**What was done this cycle (2026-02-25):**
- Modified `tools/bar_feature_export.cpp` — replaced `compute_tb_label()` with `compute_bidirectional_tb_label()`, added 3 new Parquet columns, added `--legacy-labels` CLI flag
- Modified `CMakeLists.txt` — added bidirectional_export_test target
- Modified `tests/parquet_export_test.cpp` — updated for new 152-column schema
- Created `tests/bidirectional_export_test.cpp` — export wiring tests T1-T6

**Key files:**
- Spec: `.kit/docs/bidirectional-export-wiring.md`
- Tool: `tools/bar_feature_export.cpp`
- Tests: `tests/bidirectional_export_test.cpp`, `tests/parquet_export_test.cpp`
- CMake: `CMakeLists.txt`

**Exit criteria (from spec):**
- [x] `bar_feature_export` defaults to bidirectional labels
- [x] 3 new Parquet columns present in output schema
- [x] `--legacy-labels` flag produces old-style 149-column output
- [x] All tests T1-T6 pass
- [x] No regression on existing bar_feature_export_test tests
- [x] No regression on existing triple_barrier_test tests

### Prior Completed: Bidirectional TB Labels + Oracle Expectancy Params

- `compute_bidirectional_tb_label()` in `triple_barrier.hpp` with independent long/short race evaluation. Tests T1-T10 pass. Spec: `.kit/docs/bidirectional-label-export.md`.
- `oracle_expectancy` CLI parameterized (`--target/--stop/--take-profit/--output/--help`). 49 tests. Spec: `.kit/docs/oracle-expectancy-params.md`.

### Next Steps

1. **Re-export full-year data with bidirectional labels** — run `bar_feature_export` on 251 days with new 152-column schema on EC2. This produces updated Parquet for downstream experiments.
2. **Label design sensitivity experiment** — test wider target (15 ticks) / narrower stop (3 ticks). Requires bidirectional full-year export. Spec: `.kit/experiments/label-design-sensitivity.md`.
3. **XGBoost hyperparameter tuning** — default params never optimized. GBT shows Q1-Q2 positive expectancy.
4. **Regime-conditional trading** — Q1-Q2 only strategy.

### Background: CNN Line Closed

The CNN spatial signal on order book snapshots is **real and reproducible** (R²=0.089, 3 independent reproductions). But end-to-end CNN classification (Outcome D) showed GBT-only beats CNN by 5.9pp accuracy. CNN line is permanently closed for classification.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 → integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() → z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**30+ phases complete (13 engineering + 17 research). Branch: `tdd/bidirectional-label-export`. 1144+ unit tests registered. Bidirectional export wiring COMPLETE.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Parquet schema**: 152 columns (149 original + `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`). `--legacy-labels` flag for 149-column backward compat.
- **Full-year dataset**: 251 Parquet files (time_5s bars, 1,160,150 bars, zstd compression). Stored in S3 artifact store. **Needs re-export with 152-column schema.**
- **Cloud pipeline**: Docker image in ECR, EBS snapshot with 49GB MBO data, IAM profile. Verified E2E.
- **Parallel batch dispatch**: `cloud-run batch run` launches N experiments in parallel on separate EC2 instances.

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
5. **`.kit/docs/bidirectional-export-wiring.md`** — last completed TDD spec

---

Updated: 2026-02-25. Next action: re-export full-year data with 152-column bidirectional schema, then label design sensitivity experiment.
