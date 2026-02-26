# Last Touch — Cold-Start Briefing

**Read this file first. Then read `CLAUDE.md` for the full protocol. Do not read source files.**

---

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, launch kit phases (MCP tools or bash), check exit codes via dashboard
- If you need code written -> delegate to a kit phase
- If you need something verified -> the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` ABSOLUTE RULES for the full list of things you must never do

---

## TL;DR — Where We Are and What To Do

### Just Completed (2026-02-26)

1. **Synthesis-v2 — GO Verdict (PR #30)** — Comprehensive synthesis of 23 experiments. Strategic pivot: breakeven WR (not oracle ceiling) is the correct viability metric. At 15:3 geometry, BEV WR = 33.3% — 12pp below model's 45% accuracy. 8 closed research lines. GBT-only on 20 features is canonical architecture. 55-60% prior of positive expectancy at favorable geometries.

2. **Label Design Sensitivity Phase 0 — REFUTED (Outcome C, PR #29)** — Oracle heatmap sweep across 123 valid geometries (16x9 grid). Peak oracle net expectancy: $4.13/trade at (16:10) — below $5.00 threshold. Abort triggered at Phase 0. **Critical caveat:** the $5.00 abort threshold was miscalibrated — multiple geometries have breakeven WRs of 29-38%, well below model's ~45% accuracy. Phase 1 training was never attempted.

3. **bar_feature_export --target/--stop CLI flags (TDD, PR #28)** — 47 tests pass.

4. **Bidirectional Full-Year Re-Export (EC2)** — 312/312 files, 152-column schema.

### Next: Label Geometry Phase 1 Training (IN PROGRESS)

**Experiment: `.kit/experiments/label-geometry-phase1.md`**
**Branch: `experiment/label-geometry-phase1`**

This is the single highest-priority experiment from synthesis-v2. Train XGBoost at 4 geometries selected by breakeven WR diversity:

| Geometry | Ratio | BEV WR | Model @ 45% |
|----------|-------|--------|-------------|
| 10:5 (ctrl) | 2:1 | 53.3% | -$1.56 |
| 15:3 | 5:1 | 33.3% | +$2.63 |
| 19:7 | 2.71:1 | 38.4% | +$2.14 |
| 20:3 | 6.67:1 | 29.6% | +$5.45 |

**Key question:** Does the model's ~45% directional accuracy transfer to high-ratio geometries? If yes, the favorable payoff structure converts that signal into positive expectancy.

### After That

1. **Regime-conditional trading** — Q1-Q2 only strategy (if label geometry doesn't pan out).
2. **Tick_100 multi-seed replication** — low priority.

### Background: Key Verdicts

- **CNN line CLOSED** for classification (GBT beats CNN by 5.9pp accuracy).
- **XGBoost tuning EXHAUSTED** — 0.33pp plateau. Feature set is the constraint.
- **Oracle edge EXISTS** — $4.00/trade at 10:5 geometry.
- **GBT Q1-Q2 marginally profitable** (+$0.003, +$0.029) under base costs.
- **Synthesis-v2 GO** — 55-60% prior. Label geometry is the remaining lever.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 -> integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() -> z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**30+ phases complete (14 engineering + 19 research). Branch: `experiment/label-geometry-phase1`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Parquet schema**: 152 columns (149 original + `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`). `--legacy-labels` flag for 149-column backward compat.
- **Bidirectional dataset**: 312 Parquet files (152-col), S3 backed.
- **Full-year dataset**: 251 Parquet files (149-col, time_5s bars, 1,160,150 bars, zstd compression). S3 artifact store.
- **Cloud pipeline**: Docker image in ECR, EBS snapshot with 49GB MBO data, IAM profile. Verified E2E.
- **Parallel batch dispatch**: `cloud-run batch run` launches N experiments in parallel.

### Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
| R1 | Subordination refuted | 0/3 significant | Time bars are the baseline |
| R2 | Features sufficient | R2=0.0067 | Book snapshot is sufficient statistic |
| R3 | CNN best encoder | R2=0.132 (leaked) / 0.084 (proper) | Spatial structure matters |
| R4/R4b/R4c/R4d | No temporal signal | 0/168+ passes | Drop SSM/temporal encoder permanently |
| R6 | Synthesis | CONDITIONAL GO -> GO | CNN + GBT Hybrid architecture |
| R7 | Oracle expectancy | $4.00/trade, PF=3.30 | Edge exists at oracle level |
| 9B-9D | Normalization root cause | R2: -0.002 -> 0.089 | TICK_SIZE + per-day z-score required |
| **9E** | **Pipeline bottleneck** | **exp=-$0.37/trade** | **Regression->classification gap** |
| **10** | **E2E CNN classification** | **GBT wins by 5.9pp** | **CNN line closed** |
| **XGB Tune** | **Accuracy plateau** | **0.33pp span, 64 configs** | **Feature set is binding constraint** |
| **Label-Sens P0** | **Oracle heatmap** | **123 geometries, peak $4.13** | **Phase 1 training needed** |
| **Synthesis-v2** | **GO verdict** | **55-60% prior** | **Label geometry is the lever** |
| FYE | Full-year export | 251 days, 1.16M bars | 13x more data available |
| Bidir-Export | Re-export complete | 312/312 files, 152-col | Label design sensitivity unblocked |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/label-geometry-phase1.md`** — current experiment spec

---

Updated: 2026-02-26. Next action: label-geometry-phase1 experiment (running).
