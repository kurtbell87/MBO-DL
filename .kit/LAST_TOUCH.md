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

### Just Completed (2026-02-27)

1. **Trade-Level Risk Metrics — REFUTED (modified Outcome A), PR #39** — Sequential 1-contract execution simulation on top of the validated CPCV pipeline. The edge is real ($2.50/trade) but account sizing is 10x worse than hypothesized ($48K, not $5K).

   | Metric | Value |
   |--------|-------|
   | Sequential expectancy/trade | **$2.50** (vs $1.81 bar-level — higher due to hold-skip) |
   | Sequential trades/day | **162.2** (not ~62 — barrier races resolve 2.7x faster under sequential selection) |
   | Daily PnL (1 MES) | **$412.77** |
   | Annual PnL (1 MES) | **$103,605** |
   | Win rate | **49.93%** (coin flip — edge is entirely payoff asymmetry) |
   | Min account (all 45 paths survive) | **$48,000** |
   | Min account (95% paths survive) | **$26,600** |
   | Worst max drawdown | **$47,894** |
   | Median max drawdown | **$12,917** |
   | Calmar ratio | **2.16** |
   | Annualized Sharpe | **2.27** |
   | Concurrent positions (mean) | **35.9** (scaling ceiling) |

   **Key insights:**
   - Sequential execution selects shorter barrier races (28 bars avg vs 75 overall) due to volatility-timing bias from Stage 1 reachability filter.
   - Hold-skip rate at entry points is 66.1% (not 43%) because barrier resolution lands in calmer periods.
   - Timeout trades dilute pure barrier-hit expectancy by ~50% ($5/trade theoretical -> $2.50 observed).
   - The $48K account minimum is driven by the worst-path drawdown across 45 CPCV paths.
   - Strategy captures 5.7% of bar-level daily PnL on 1 contract. Full capture needs ~36 MES (~7 ES).

2. **CPCV Validation — CONFIRMED (Outcome A), PR #38** — $1.81/trade at corrected-base costs ($2.49 RT). PBO 6.7%. 95% CI [$1.46, $2.16]. t=10.29, p<1e-13. First statistically validated pipeline.

### Next: Optimize or Scale

The edge is validated under sequential execution. Two paths forward:

1. **Timeout filtering** — Remove entries that are unlikely to resolve before day end. Could increase per-trade expectancy from $2.50 toward $5.00 (the barrier-hit theoretical max) and reduce drawdowns.
2. **Multi-contract scaling** — Run N sequential executors. ~36 MES captures full bar-level edge at $1.73M capital.
3. **Paper trading** — Deploy at $48K+ account. Validates real-world fill rates/slippage.

### Background: Key Verdicts

- **Trade-Level Risk REFUTED** (modified A) — Edge real ($2.50), account sizing fails ($48K vs $5K target)
- **CPCV CONFIRMED (Outcome A)** — $1.81/trade, PBO 6.7%, p < 1e-13. First validated pipeline.
- **CNN line CLOSED** for classification (GBT beats CNN by 5.9pp accuracy).
- **Edge is structural, not predictive** — 19:7 payoff asymmetry x coin-flip accuracy > 34.6% BEV.
- **Cost correction was the key unlock** — $3.74 -> $2.49 base RT recovered $1.25/trade.
- **Sequential expectancy > bar-level** — $2.50 > $1.81 (hold-skip removes negative-EV hold bars).

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 -> integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() -> z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**31+ phases complete (15 engineering + 27 research). Branch: `experiment/trade-level-risk-metrics`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Full-year dataset**: 251 Parquet files (152-col, time_5s bars, 1,160,150 bars, zstd compression).
- **Validated pipeline**: Two-stage XGBoost (reachability + direction) at 19:7 geometry.
- **Sequential simulator**: 1-contract execution with trade logging, equity curves, account sizing.

### Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
| **Trade-Level Risk** | **Edge real, account too large** | **$2.50/trade, $48K min** | **Viable for medium accounts; timeout filtering next** |
| **CPCV Corrected** | **CONFIRMED (Outcome A)** | **$1.81/trade, PBO 6.7%, p<1e-13** | **First validated pipeline. Deploy.** |
| R1 | Subordination refuted | 0/3 significant | Time bars are the baseline |
| R2 | Features sufficient | R2=0.0067 | Book snapshot is sufficient statistic |
| R4/R4b/R4c/R4d | No temporal signal | 0/168+ passes | Drop SSM/temporal encoder permanently |
| 9E | Pipeline bottleneck | exp=-$0.37/trade | Regression->classification gap |
| 10 | E2E CNN classification | GBT wins by 5.9pp | CNN line closed |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/trade-level-risk-metrics.md`** — latest experiment spec
6. **`.kit/results/trade-level-risk-metrics/analysis.md`** — full sequential risk analysis

---

Updated: 2026-02-27. Trade-level risk metrics REFUTED (modified A): $2.50/trade, $48K min account. PR #39. Next: timeout filtering or paper trading at $48K+.
