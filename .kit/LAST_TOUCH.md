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

1. **2-Class Directional — CONFIRMED with PnL caveat (PR #34, OPEN)** — Two-stage pipeline (Stage 1: reachability, Stage 2: direction) liberates trade volume: 0.28%→85.2% at 19:7 (301x increase). Stage 1 accuracy 58.6% (6pp above majority baseline). Stage 2 direction accuracy ~50% (essentially random at 19:7). Reported expectancy $3.78/trade BUT corrected estimate ~$0.44/trade (PnL model assigns full barrier payoffs to hold-bar trades instead of $0). Corrected 95% CI: [$0.04, $0.84]. Feature importance: Stage 2 at 19:7 does NOT learn directional features — same volatility features as Stage 1. **Key insight: reachability is solvable, direction at 19:7 is not. The favorable payoff ratio (2.71:1) is the economic driver, not prediction skill.**

2. **Label Geometry 1h — INCONCLUSIVE (PR #33, merged)** — Time horizon fix WORKS (hold rates dropped 29-58pp). But directional accuracy ceiling ~50-51%. Model becomes hold predictor at high-ratio geometries (<0.3% trade rate).

3. **Time Horizon CLI Flags — COMPLETE (PR #32, merged)** — `--max-time-horizon` and `--volume-horizon` CLI flags.

4. **Synthesis-v2 — GO Verdict (PR #30, merged)** — 55-60% prior at high-ratio geometries.

### Next: Validate 2-Class Economics

The 2-class formulation works structurally (trade rate liberated, reachability learnable). The open question is whether the corrected economics hold up:

- **Option A (HIGHEST PRIORITY): Corrected PnL Validation** — Re-run walk-forward with PnL model that assigns $0 to hold-bar trades (or actual realized PnL at horizon expiry). Same data, same models — just fix the PnL computation. Resolves dominant uncertainty.
- **Option B: CPCV at 19:7 with Corrected PnL** — 3 WF folds give CI [$0.04, $0.84]. Need 45 CPCV splits for reliable go/no-go.
- **Option C: Stage 1 Threshold Optimization** — Raise P(directional) threshold from 0.5 to 0.6-0.8. Trades fewer bars but with higher fraction hitting directional barriers. Reduces sensitivity to hold-bar PnL assumption.
- **Option D: Intermediate Geometry (14:6, 15:5)** — 19:7 has random direction prediction, 10:5 has marginal signal. Sweet spot may be in between: favorable BEV + partial directional signal.

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

**30+ phases complete (15 engineering + 23 research). Branch: `experiment/2class-directional`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

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
| **2-Class Dir** | **CONFIRMED (PnL caveat)** | **Trade rate 0.28%→85.2%, corrected exp ~$0.44** | **Reachability solvable, direction random at 19:7** |
| FYE | Full-year export | 251 days, 1.16M bars | 13x more data available |
| Bidir-Export | Re-export complete | 312/312 files, 152-col | Label design sensitivity unblocked |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/2class-directional.md`** — completed experiment spec (CONFIRMED w/ caveat)
6. **`.kit/results/2class-directional/analysis.md`** — full analysis with PnL model critique
7. **`.kit/experiments/label-geometry-1h.md`** — completed experiment spec (INCONCLUSIVE)
8. **`.kit/results/label-geometry-1h/analysis.md`** — full analysis

---

Updated: 2026-02-26. 2-class directional CONFIRMED (trade rate liberated, corrected exp ~$0.44). PR #34 open. Next: corrected PnL validation or CPCV at 19:7.
