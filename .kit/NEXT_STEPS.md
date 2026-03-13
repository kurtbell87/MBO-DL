# Next Steps — Updated 2026-02-27

## FIRST: Create a New Branch

**BEFORE you do ANY work — create a worktree and branch.**

```bash
git worktree add ../MBO-DL-<slug> -b <branch-type>/<name> main
cd ../MBO-DL-<slug>
source .orchestration-kit.env
```

**All work happens on a branch. Never touch main directly.**

---

## Current State

- **33+ phases complete** (15 engineering + 29 research). Branch: `experiment/volume-flow-entry`.
- **Volume-Flow Conditioned Entry — REFUTED (Outcome B)** (2026-02-27). 20pp bar-level diagnostic signal evaporates 91% in sequential simulation. Timeout fraction invariant at ~41.3%. Entry-time filtering exhausted as intervention class. Best stacked config: $3.10/trade, $33K min acct, Calmar 2.59. PR #41.
- **Timeout-Filtered Sequential — REFUTED (Outcome B, mechanism falsified)** (2026-02-27). Timeout fraction invariant at 41.3% across all 7 cutoff levels. Best cutoff=270: $3.02/trade, $34K min acct, Calmar 2.49. PR #40.
- **Trade-Level Risk Metrics — REFUTED (modified A)** (2026-02-27). Sequential 1-contract: $2.50/trade, 162 trades/day, $412.77/day, $103,605/year. Min account $48K (all paths) / $26.6K (95% paths). PR #39.
- **CPCV Validation — CONFIRMED (Outcome A)** (2026-02-27). $1.81/trade at corrected-base. PBO 6.7%. PR #38.
- **Edge is structural** — 49.93% win rate, 19:7 payoff asymmetry, breakeven 34.6%.

## Critical Finding: Entry-Time Filtering Is Exhausted

Two experiments confirm timeout fraction (~41.3%) is invariant to all observable entry-time features:
- **PR #40 (time-of-day):** 0.21pp range across 7 cutoff levels
- **PR #41 (volume/activity):** 1.76pp max across 15 gate configs (20pp bar-level signal evaporates 91%)

The sequential execution process (66.1% hold-skip) already self-selects for high-activity bars. Any entry gate duplicates what hold-skip already does.

**The next intervention must change the barrier parameters, not the entry timing.**

## Priority: Change Barrier Geometry or Accept and Deploy

### 1. Barrier Geometry Re-Parameterization (HIGHEST PRIORITY)

The 41.3% timeout rate is a structural constant of the 50,000-contract volume horizon at 19:7 geometry. Only changing the race parameters can change it.

**Option A: Reduce volume horizon (50,000 → 10,000-25,000)**
- Faster race resolution → fewer volume-horizon timeouts
- BUT: more time-horizon timeouts if barriers don't hit in shorter volume window
- Net effect unknown — this is the experiment

**Option B: Reduce time horizon (3,600s → 600-1,200s)**
- Directly caps race duration
- Changes which barrier events are reachable
- Requires full data re-export (EC2, ~10 min, ~$0.10)

**Spec:** Not yet created
**Branch:** `experiment/barrier-geometry-sweep`
**Compute:** Option A: local (same Parquet). Option B: EC2 (re-export) + local.

### 2. Paper Trading at Cutoff=270 (HIGH PRIORITY)

Cutoff=270 (or stacked cutoff=270 + message_rate_p25) is the best risk-adjusted config:
- Stacked: $3.10/trade, 115 trades/day, $338/day, $33K min acct, Calmar 2.59
- Cutoff=270: $3.02/trade, 117 trades/day, $337/day, $34K min acct, Calmar 2.49

Rithmic R|API+ integration for live /MES paper trading. 1 contract.

**Validates:** Real-world fill rates, slippage, latency.
**Spec:** Not yet created
**Branch:** `feat/rithmic-paper-trading`
**Compute:** Local

### 3. Regime-Conditional Trading (MODERATE PRIORITY)

Q1-Q2 only strategy. GBT is marginally profitable in Q1 (+$0.003/trade) and Q2 (+$0.029/trade) under base costs, negative in Q3-Q4. A seasonal filter is a different class of intervention (regime gating, not entry-time gating) that hasn't been tested with sequential execution.

**Spec:** Not yet created
**Branch:** `experiment/regime-conditional`
**Compute:** Local

### 4. Multi-Contract Scaling (MODERATE PRIORITY)

At cutoff=270 (or stacked):
- 1 contract: $34K capital, $337/day, 249% ROC
- N contracts (independent sequential executors): N × $34K capital, ~N × $337/day, ~249% ROC

**Hypothesis:** N staggered sequential executors produce N-proportional daily PnL with sqrt(N) drawdown.
**Spec:** Not yet created
**Branch:** `experiment/multi-contract-scaling`
**Compute:** Local

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| **Volume-Flow** | **REFUTED (Outcome B)** | **20pp diagnostic → 1.76pp simulation (91% evaporation). Entry-time gating exhausted.** |
| **Timeout Filter** | **REFUTED (Outcome B)** | **Timeout fraction invariant at 41.3%. Volume-driven, not time-driven.** |
| **Trade-Level Risk** | **REFUTED (modified A)** | **$2.50/trade seq, $48K min account. Edge real but larger account needed.** |
| **CPCV Corrected** | **CONFIRMED (Outcome A)** | **$1.81/trade, PBO 6.7%, p<1e-13. 42/45 splits positive.** |

---

## Key Numbers to Remember

### Stacked (Cutoff=270 + message_rate_p25) — Best Risk-Adjusted
- **Expectancy/trade:** $3.10 (+24% vs unfiltered)
- **Trades/day:** 114.7 (-29%)
- **Daily PnL:** $338 (-18%)
- **Annual PnL:** $84,941
- **Win rate:** 50.30%
- **Timeout fraction:** 41.23% (invariant)
- **Worst drawdown:** $32,830 (-31%)
- **Median drawdown:** $8,684 (-33%)
- **Calmar ratio:** 2.59 (best ever)
- **Min account (all paths):** $33,000
- **Min account (95% paths):** $25,500
- **ROC (all-path):** 258% annual

### Cutoff=270 Only — Simpler, Nearly Identical
- **Expectancy/trade:** $3.02
- **Trades/day:** 116.8
- **Daily PnL:** $337
- **Calmar:** 2.49
- **Min account:** $34,000 / $25,500

### Unfiltered Sequential (Baseline)
- **Expectancy/trade:** $2.50
- **Trades/day:** 162.2
- **Daily PnL:** $413
- **Annual PnL:** $103,605
- **Min account:** $48,000 / $26,500
- **ROC:** 216% annual

### Bar-Level (CPCV)
- **Expectancy/trade:** $1.81 (95% CI [$1.46, $2.16])
- **PBO:** 6.7%
- **Break-even RT:** $4.30

---

Written: 2026-02-27. Volume-flow conditioned entry REFUTED (Outcome B): 20pp bar-level signal evaporates 91% due to sequential self-selection. Entry-time filtering exhausted. Best stacked: $3.10/trade, $33K min acct, Calmar 2.59. PR #41.
