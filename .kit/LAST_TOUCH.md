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

1. **Label Geometry 1h — INCONCLUSIVE (practically Outcome B, PR #33)** — Time horizon fix WORKS (hold rates dropped 29-58pp). But model cannot exploit high-ratio geometries: at 10:5 (only geometry with real trade volume), directional accuracy 50.7% is 2.6pp BELOW 53.3% breakeven, exp=-$0.49/trade. At high-ratio geometries (15:3, 19:7, 20:3), model becomes near-total hold predictor (<0.3% directional prediction rate). Reported positive expectancy at those geometries is small-sample artifact (30-3,280 trades, Infinity profit factors). **Binding constraint: feature-label correlation ceiling (~50-51% directional accuracy), not geometry, not hyperparameters, not label design.**

2. **Time Horizon CLI Flags — COMPLETE (PR #32)** — `--max-time-horizon` and `--volume-horizon` CLI flags. Defaults: 300→3600s, 500→50000.

3. **Label Geometry Phase 1 — REFUTED (PR #31)** — 90.7-98.9% hold at 300s cap. Root cause identified and fixed in PR #32.

4. **Synthesis-v2 — GO Verdict (PR #30)** — 55-60% prior at high-ratio geometries.

### Next: Break Through the Feature-Label Ceiling

The model's directional accuracy is ~50-51% regardless of label type, geometry, or hyperparameters. This is the hard ceiling of 20 microstructure features on MES time_5s bars. Options:

- **Option A (HIGHEST PRIORITY): 2-Class Formulation** — Train binary "directional vs hold" at 19:7 (47.4% directional). If model predicts WHICH bars will be directional with >60% accuracy, a second-stage direction model on predicted-directional bars could exploit the 19:7 payoff. Decouples barrier-reachability from direction.
- **Option B: Class-Weighted XGBoost at 19:7** — Force directional predictions with 3:1 weight. Accept lower accuracy for higher trade rate. Question: does forced directional accuracy stay above 38.4% BEV WR?
- **Option C: Long-Perspective Labels at Varied Geometries** — Test if directional accuracy transfers when labels avoid the hold-prediction trap.
- **Option D: Feature Engineering for Wider Barriers** — Add rolling VWAP slope, cumulative order flow over 50-500 bars, volatility regime markers. Highest effort, addresses root cause.

### After That

1. **Regime-conditional trading** — Q1-Q2 only strategy.
2. **Tick_100 multi-seed replication** — low priority.

### Background: Key Verdicts

- **CNN line CLOSED** for classification (GBT beats CNN by 5.9pp accuracy).
- **XGBoost tuning EXHAUSTED** — 0.33pp plateau. Feature set is the constraint.
- **Oracle edge EXISTS** — $3.22-$9.44/trade at 3600s across all 4 geometries.
- **Directional accuracy ceiling** — ~50-51% regardless of geometry/labels/hyperparameters.
- **Time horizon fix CONFIRMED** — hold rates dropped 90.7%→32.6% at 10:5.
- **Bidirectional labels harder** — 3-class accuracy 38.4% vs 44.9% on long-perspective.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 -> integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() -> z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**30+ phases complete (15 engineering + 21 research). Branch: `experiment/label-geometry-1h`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Parquet schema**: 152 columns (149 original + `tb_both_triggered`, `tb_long_triggered`, `tb_short_triggered`). `--legacy-labels` flag for 149-column backward compat.
- **CLI flags**: `--target`, `--stop`, `--max-time-horizon`, `--volume-horizon`, `--legacy-labels` on both tools.
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
| 9E | Pipeline bottleneck | exp=-$0.37/trade | Regression->classification gap |
| 10 | E2E CNN classification | GBT wins by 5.9pp | CNN line closed |
| XGB Tune | Accuracy plateau | 0.33pp span, 64 configs | Feature set is binding constraint |
| Label-Sens P0 | Oracle heatmap | 123 geometries, peak $4.13 | Phase 1 training needed |
| Synthesis-v2 | GO verdict | 55-60% prior | Label geometry is the lever |
| Geom P1 | REFUTED — degenerate labels | 90.7-98.9% hold | 300s cap = untestable |
| **Geom 1h** | **INCONCLUSIVE — hold predictor** | **Dir acc 50.7%, <0.3% trade rate** | **Feature ceiling, not geometry** |
| FYE | Full-year export | 251 days, 1.16M bars | 13x more data available |
| Bidir-Export | Re-export complete | 312/312 files, 152-col | Label design sensitivity unblocked |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/label-geometry-1h.md`** — completed experiment spec (INCONCLUSIVE)
6. **`.kit/results/label-geometry-1h/analysis.md`** — full analysis

---

Updated: 2026-02-26. Label geometry 1h INCONCLUSIVE. Feature-label correlation ceiling (~50-51% dir acc) is the binding constraint. Next: 2-class formulation or class-weighted training.
