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

1. **Volume-Flow Conditioned Entry — REFUTED (Outcome B), PR #41** — Two-stage diagnostic-first experiment: all 5 volume/activity features show massive 20pp bar-level timeout variation by quartile, but sequential simulation reduces this to 1.76pp max — a **91% signal evaporation**. The sequential execution process (66.1% hold-skip) already self-selects for high-activity bars, making volume gating redundant. The stacked config (cutoff=270 + message_rate_p25) is marginally better than cutoff=270 alone (+$0.08/trade, +0.10 Calmar, -$1K min account) but the volume gate's contribution is noise-level.

   **D5 Cross-Table (key finding):** The entire diagnostic signal concentrates in one corner: v50_Q1 × tc_Q1 = 71.5% timeout. Outside this low-vol/low-activity corner, timeout fraction is flat at 41-45%. Volume gating is really "avoid extreme quiet periods" gating — but sequential execution already avoids quiet periods through hold-skip mechanics.

   | Metric | Unfiltered | Cutoff=270 | Volume-Gated | Stacked |
   |--------|-----------|------------|-------------|---------|
   | Exp/trade | $2.50 | $3.02 | $2.51 | $3.10 |
   | Trades/day | 162.2 | 116.8 | 158.9 | 114.7 |
   | Daily PnL | $413 | $337 | $405 | $338 |
   | DD worst | $47,894 | $33,984 | $47,434 | $32,830 |
   | Min account | $48,000 | $34,000 | $47,500 | $33,000 |
   | Timeout | 41.33% | 41.30% | 41.20% | 41.23% |
   | Calmar | 2.16 | 2.49 | 2.14 | 2.59 |
   | Annual PnL | $103,605 | $84,530 | $101,579 | $84,941 |

   **Key insights:**
   - Timeout fraction (~41.3%) is a structural constant — confirmed across TWO independent experiments (time-of-day PR #40, volume-flow PR #41)
   - Entry-time filtering is exhausted as an intervention class. Both time-of-day and volume/activity gating fail to reduce timeouts.
   - The diagnostic paradox (20pp bar-level → 1.76pp simulation) is explained by sequential selection bias
   - SC-2 fails ($2.51 < $3.50), SC-3 fails ($47.5K > $30K), SC-5 fails (0.12pp < 5pp)

2. **Timeout-Filtered Sequential — REFUTED (Outcome B), PR #40** — Timeout fraction invariant at 41.3% across all 7 cutoff levels. Best cutoff=270: $3.02/trade, $34K min acct, Calmar 2.49.

3. **Trade-Level Risk Metrics — REFUTED (modified A), PR #39** — Sequential 1-contract: $2.50/trade, 162 trades/day, $412.77/day, $48K min account.

4. **CPCV Validation — CONFIRMED (Outcome A), PR #38** — $1.81/trade at corrected-base costs. PBO 6.7%. p < 1e-13.

### Next: Change Barrier Geometry or Accept and Deploy

Entry-time filtering is now exhaustively refuted. Two options remain:

1. **Barrier geometry re-parameterization (HIGHEST)** — Reduce volume horizon from 50,000 to 10,000-25,000, OR reduce time horizon from 3,600s to 600-1,200s. This is the ONLY intervention that can mechanically change the timeout rate. The 41.3% is a property of the race between barriers and horizons.
2. **Accept cutoff=270 and paper trade (HIGH)** — $3.02/trade, $34K min account, Calmar 2.49, 249% ROC. The operational question is whether backtested numbers hold live, not whether entry filters can squeeze more.
3. **Regime-conditional trading (MODERATE)** — Q1-Q2 only, a different class of intervention (regime gating, not entry-time gating).

### Background: Key Verdicts

- **Volume-flow gating REFUTED** — 20pp diagnostic signal evaporates 91% due to sequential selection bias
- **Timeout filtering REFUTED** — Timeouts are volume-driven, not time-driven. Time cutoff is wrong proxy.
- **Trade-Level Risk REFUTED** (modified A) — Edge real ($2.50), account sizing fails ($48K vs $5K target)
- **CPCV CONFIRMED (Outcome A)** — $1.81/trade, PBO 6.7%, p < 1e-13. First validated pipeline.
- **CNN line CLOSED** for classification (GBT beats CNN by 5.9pp accuracy).
- **Edge is structural, not predictive** — 49.93% win rate, 19:7 payoff asymmetry, breakeven 34.6%.
- **Cost correction was the key unlock** — $3.74 -> $2.49 base RT recovered $1.25/trade.
- **Entry-time filtering exhausted** — Neither time-of-day nor volume/activity can change the 41.3% timeout rate.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 -> integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() -> z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**33+ phases complete (15 engineering + 29 research). Branch: `experiment/volume-flow-entry`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Full-year dataset**: 251 Parquet files (152-col, time_5s bars, 1,160,150 bars, zstd compression).
- **Validated pipeline**: Two-stage XGBoost (reachability + direction) at 19:7 geometry.
- **Sequential simulator**: 1-contract execution with trade logging, equity curves, account sizing, timeout tracking.
- **Time-filtered simulator**: 7-level cutoff sweep. Timeout fraction invariance established.
- **Volume-flow gated simulator**: 5 features × 3 gate levels + stacked configs. 91% signal evaporation confirmed.

### Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
| **Volume-Flow** | **REFUTED (Outcome B)** | **20pp diagnostic → 1.76pp simulation (91% evaporation)** | **Entry-time gating exhausted. Sequential self-selects.** |
| **Timeout Filter** | **REFUTED (Outcome B)** | **Timeout invariant at 41.3%** | **Time-of-day is wrong proxy for timeouts** |
| **Trade-Level Risk** | **REFUTED (modified A)** | **$2.50/trade, $48K min** | **Edge real, medium account needed** |
| **CPCV Corrected** | **CONFIRMED (Outcome A)** | **$1.81/trade, PBO 6.7%, p<1e-13** | **First validated pipeline. Deploy.** |
| R1 | Subordination refuted | 0/3 significant | Time bars are the baseline |
| R2 | Features sufficient | R2=0.0067 | Book snapshot is sufficient statistic |
| R4/R4b/R4c/R4d | No temporal signal | 0/168+ passes | Drop SSM/temporal encoder permanently |
| 10 | E2E CNN classification | GBT wins by 5.9pp | CNN line closed |

---

## State Files (read order)

1. **This file** — you're here
2. **`CLAUDE.md`** — full protocol, absolute rules, current state, institutional memory
3. **`.kit/RESEARCH_LOG.md`** — cumulative findings from all experiments
4. **`.kit/QUESTIONS.md`** — open and answered research questions
5. **`.kit/experiments/volume-flow-conditioned-entry.md`** — REFUTED experiment spec
6. **`.kit/results/volume-flow-conditioned-entry/analysis.md`** — full volume-flow analysis

---

Updated: 2026-02-27. Volume-flow conditioned entry REFUTED (Outcome B): 20pp bar-level diagnostic signal evaporates 91% in sequential simulation due to hold-skip self-selection. Entry-time filtering exhausted as intervention class. PR #41.
