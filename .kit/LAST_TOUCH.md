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

1. **Timeout-Filtered Sequential — REFUTED (Outcome B, mechanism falsified), PR #40** — Time-of-day cutoff sweep across 7 levels (390→270 minutes) on the sequential 1-contract simulator. **Timeout fraction is invariant at ~41.3% across ALL cutoff levels** — timeouts are driven by the volume horizon (50,000 contracts), NOT clock time. The hypothesis's core mechanism is empirically falsified.

   | Metric | Unfiltered (390) | Best Filtered (270) | Delta |
   |--------|-----------------|---------------------|-------|
   | Expectancy/trade | $2.50 | **$3.02** | +20.6% |
   | Trades/day | 162.2 | 116.8 | -28.0% |
   | Daily PnL | $412.77 | $336.77 | -18.4% |
   | Annual PnL | $103,605 | $84,530 | -18.4% |
   | DD worst | $47,894 | $33,984 | -29.0% |
   | DD median | $12,917 | $8,687 | -32.7% |
   | Min account (all) | $48,000 | **$34,000** | -29.2% |
   | Min account (95%) | $26,500 | $25,500 | -3.8% |
   | Timeout fraction | 41.33% | **41.30%** | **-0.03pp** |
   | Calmar | 2.16 | **2.49** | +15.0% |
   | ROC (all-path) | 216% | **249%** | +33pp |

   **Key insights:**
   - Timeout fraction is invariant (~41.3% ± 0.1pp) — time-of-day is NOT the timeout mechanism
   - Volume horizon (50,000 contracts) is the binding constraint for timeouts
   - Cutoff=270 improvement comes from hold-skip restructuring (66.1%→34.4%), NOT timeout avoidance
   - Expectancy is U-shaped across cutoffs (drops 390→330, then rises 300→270) — late-day trades in 5–5.75h window are actually BETTER than average
   - SC-2 fails ($3.02 < $3.50), SC-3 fails ($34K > $30K), SC-S4 fails (non-monotonic expectancy)
   - ROC is HIGHER with filtering (249% vs 216%) despite lower absolute PnL — capital efficiency improves
   - Splits 18 & 32 (outliers): DD drops ~29-30%, but both remain deeply negative-expectancy

2. **Trade-Level Risk Metrics — REFUTED (modified A), PR #39** — Sequential 1-contract: $2.50/trade, 162 trades/day, $412.77/day, $48K min account.

3. **CPCV Validation — CONFIRMED (Outcome A), PR #38** — $1.81/trade at corrected-base costs. PBO 6.7%. p < 1e-13.

### Next: Volume-Based or Volatility-Based Filtering

The time-of-day approach is proven wrong. Three viable paths:

1. **Volume-flow conditioned entry** (HIGHEST) — Target the actual timeout mechanism. Condition entry on recent volume flow quantile. High-volume periods → faster volume horizon consumption → fewer timeouts.
2. **Volatility-conditional entry** (HIGH) — volatility_50 is the dominant XGBoost feature (49.7% gain). Low-volatility entries may have longer barrier races and higher timeouts.
3. **Accept $34K/$25.5K and paper trade** (MODERATE) — Cutoff=270 at 249% ROC is already operationally viable. Min account (95%) = $25.5K.
4. **Pure time horizon re-export** — Use `--max-time-horizon 1800` (30 min) instead of volume horizon. If clock-based timeouts behave differently, they may be more filterable.

### Background: Key Verdicts

- **Timeout filtering REFUTED** — Timeouts are volume-driven, not time-driven. Time cutoff is wrong proxy.
- **Trade-Level Risk REFUTED** (modified A) — Edge real ($2.50), account sizing fails ($48K vs $5K target)
- **CPCV CONFIRMED (Outcome A)** — $1.81/trade, PBO 6.7%, p < 1e-13. First validated pipeline.
- **CNN line CLOSED** for classification (GBT beats CNN by 5.9pp accuracy).
- **Edge is structural, not predictive** — 19:7 payoff asymmetry x coin-flip accuracy > 34.6% BEV.
- **Cost correction was the key unlock** — $3.74 -> $2.49 base RT recovered $1.25/trade.

---

## Normalization Protocol (CRITICAL — institutional memory)

Three prior experiments (9B, 9C, R3b) failed because of normalization errors. The correct protocol, verified in 9D and 9E:

1. **Book prices (channel 0)**: Divide by TICK_SIZE=0.25 -> integer tick offsets. Do NOT z-score.
2. **Book sizes (channel 1)**: log1p() -> z-score PER DAY (not per-fold, not globally).
3. **Non-spatial features**: z-score using train-fold stats only.

If you see the sub-agent z-scoring channel 0 or using per-fold z-scoring on sizes, that's the bug that cost us 3 experiment cycles.

---

## Project Status

**32+ phases complete (15 engineering + 28 research). Branch: `experiment/timeout-filtered-sequential`. 1144+ unit tests registered. COMPUTE_TARGET=local.**

### What's Built
- **C++20 data pipeline**: Bar construction, order book replay, multi-day backtest, feature computation/export, oracle expectancy, Parquet export, bidirectional TB labels. 1144+ unit tests, 22 integration tests.
- **Full-year dataset**: 251 Parquet files (152-col, time_5s bars, 1,160,150 bars, zstd compression).
- **Validated pipeline**: Two-stage XGBoost (reachability + direction) at 19:7 geometry.
- **Sequential simulator**: 1-contract execution with trade logging, equity curves, account sizing, timeout tracking.
- **Time-filtered simulator**: 7-level cutoff sweep. Timeout fraction invariance established.

### Key Research Results

| Experiment | Finding | Key Number | Implication |
|-----------|---------|------------|-------------|
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
5. **`.kit/experiments/timeout-filtered-sequential.md`** — REFUTED experiment spec
6. **`.kit/results/timeout-filtered-sequential/analysis.md`** — full timeout filter analysis

---

Updated: 2026-02-27. Timeout filtering REFUTED (Outcome B): timeout fraction invariant at 41.3%, timeouts are volume-driven not time-driven. Best cutoff=270: $3.02/trade, $34K min acct, 249% ROC. PR #40.
