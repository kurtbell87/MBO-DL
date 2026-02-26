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

### Just Completed (2026-02-26)

1. **Label Design Sensitivity — REFUTED (Outcome C)** — Oracle heatmap sweep across 123 valid geometries (16×9 grid). **Peak oracle net expectancy: $4.13/trade at (16:10) — below $5.00 threshold.** Abort triggered at Phase 0 per pre-committed criteria. No model training performed. **Critical caveat:** the $5.00 abort threshold may have been miscalibrated — multiple geometries have breakeven WRs of 29-38%, well below model's ~45% accuracy. Phase 1 training would have been informative. PR #29.

2. **bar_feature_export --target/--stop CLI flags (TDD)** — `bar_feature_export` now accepts `--target <ticks>` and `--stop <ticks>` to vary triple barrier geometry per-export. 47 tests pass. PR #28.

3. **Policy fix** — Added CLAUDE.md rule #8: Python NEVER computes labels/features. Experiment spec rewritten to use C++ tools exclusively. Reexport diagnosis: EC2 14KB files were script/Docker path issue, not code bug (local binary produces 17MB correctly).

4. **Bidirectional Full-Year Re-Export (EC2)** — 312/312 files exported with bidirectional labels (152-column schema).

### Next: Decide Follow-Up

**The label-design-sensitivity experiment's Outcome C verdict has a methodological caveat.** The $5.00/trade oracle ceiling threshold was meant to ensure sufficient margin, but the payoff structure changes are what matter:

| Geometry | Breakeven WR | Oracle Margin | Model Accuracy Needed |
|----------|-------------|---------------|----------------------|
| 10:5 (current) | 53.3% | +11.0pp | 55.3% (model achieves ~45%) |
| 15:3 | 33.3% | — | 35.3% (model may achieve this) |
| 20:5 | 32.0% | — | 34.0% (model may achieve this) |

**Option A:** Re-run with relaxed criteria (oracle > $3/trade), proceed to Phase 1 training on top geometries.
**Option B:** Accept Outcome C and pivot to regime-conditional trading (Q1-Q2 only).
**Option C:** Accept no-edge verdict for MES 5-second bars.

### After That

1. **Regime-conditional trading** — Q1-Q2 only strategy.
2. **Tick_100 multi-seed replication** — low priority.

### Background: Key Verdicts

- **CNN line CLOSED** for classification (GBT beats CNN by 5.9pp accuracy).
- **XGBoost tuning EXHAUSTED** — 0.33pp plateau. Feature set is the constraint.
- **Oracle edge EXISTS** — $4.00/trade at 10:5 geometry (long-perspective; will be lower bidirectional).
- **GBT Q1-Q2 marginally profitable** (+$0.003, +$0.029) under base costs.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 → integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() → z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**30+ phases complete (14 engineering + 17 research). Branch: `main`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Parquet schema**: 152 columns (149 original + `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`). `--legacy-labels` flag for 149-column backward compat.
- **Bidirectional dataset**: 312 Parquet files (152-col), S3 backed. Bidirectional re-export DONE.
- **Full-year dataset**: 251 Parquet files (149-col, time_5s bars, 1,160,150 bars, zstd compression). S3 artifact store.
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
| **XGB Tune** | **Accuracy plateau** | **0.33pp span, 64 configs** | **Feature set is binding constraint** |
| FYE | Full-year export | 251 days, 1.16M bars | 13× more data available |
| Bidir-Export | Re-export complete | 312/312 files, 152-col | Label design sensitivity unblocked |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all 12+ experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/label-design-sensitivity.md`** — next experiment spec

---

Updated: 2026-02-25. Next action: label design sensitivity experiment (all prerequisites done).
