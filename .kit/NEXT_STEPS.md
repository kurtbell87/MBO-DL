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

- **32+ phases complete** (15 engineering + 28 research). Branch: `experiment/timeout-filtered-sequential`.
- **Timeout-Filtered Sequential — REFUTED (Outcome B, mechanism falsified)** (2026-02-27). Timeout fraction invariant at 41.3% across all 7 cutoff levels. Timeouts are volume-driven, not time-driven. Best cutoff=270: $3.02/trade, $34K min acct, 249% ROC. PR #40.
- **Trade-Level Risk Metrics — REFUTED (modified A)** (2026-02-27). Sequential 1-contract: $2.50/trade, 162 trades/day, $412.77/day, $103,605/year. Min account $48K (all paths) / $26.6K (95% paths). PR #39.
- **CPCV Validation — CONFIRMED (Outcome A)** (2026-02-27). $1.81/trade at corrected-base. PBO 6.7%. PR #38.
- **Edge is structural** — 49.93% win rate, 19:7 payoff asymmetry, breakeven 34.6%.

## Critical New Finding: Timeouts Are Volume-Driven

The timeout-filtered experiment **falsified** the time-of-day mechanism. Key facts:
- Timeout fraction = 41.3% +/- 0.1pp regardless of entry time cutoff
- The 50,000-contract volume horizon is the binding constraint
- Even entries 4.5h into RTH have ~2 hours remaining — far more than the 2.3-minute avg barrier race
- Any future timeout reduction must target the volume horizon mechanism, not clock time

## Priority: Reduce Timeouts or Accept and Deploy

### 1. Volume-Flow Conditioned Entry (HIGHEST PRIORITY)

Since timeouts are driven by the volume horizon, condition entry on recent volume flow. High-volume periods -> faster volume horizon consumption -> barrier resolution before stochastic expiry.

**Hypothesis:** Entries during top-quartile volume flow have lower timeout rates.
**IV:** Rolling N-bar volume quantile at entry time.
**Spec:** Not yet created
**Branch:** `experiment/volume-flow-entry`
**Compute:** Local

### 2. Volatility-Conditional Entry Filter (HIGH PRIORITY)

volatility_50 is the dominant XGBoost feature (49.7% gain share) and directly relates to barrier reachability. Low-volatility entries may have longer barrier races and higher timeout rates.

**Hypothesis:** Conditioning entry on volatility_50 > threshold reduces timeout fraction.
**Spec:** Not yet created
**Branch:** `experiment/volatility-entry-filter`
**Compute:** Local

### 3. Paper Trading at Cutoff=270 (HIGH PRIORITY)

Cutoff=270 at 249% ROC is already operationally viable:
- $34K account (all-path) or $25.5K (95%-path)
- $3.02/trade, 117 trades/day, $337/day
- Calmar 2.49, Sharpe 2.20

Rithmic R|API+ integration for live /MES paper trading. 1 contract.

**Validates:** Real-world fill rates, slippage, latency.
**Spec:** Not yet created
**Branch:** `feat/rithmic-paper-trading`
**Compute:** Local

### 4. Pure Time Horizon Re-Export (MODERATE PRIORITY)

Re-export with `--max-time-horizon 1800` (30 min) instead of the volume horizon. If clock-based timeouts concentrate differently than volume-based ones, they may be more filterable by time-of-day.

**Spec:** Not yet created
**Branch:** `experiment/time-horizon-reexport`
**Compute:** EC2 (full re-export of 312 files)

### 5. Accept $34K and Multi-Contract Scaling (MODERATE PRIORITY)

At cutoff=270:
- 1 contract: $34K capital, $337/day, 249% ROC
- N contracts (independent sequential executors): N x $34K capital, ~N x $337/day, ~249% ROC

**Hypothesis:** N staggered sequential executors produce N-proportional daily PnL with sqrt(N) drawdown.
**Spec:** Not yet created
**Branch:** `experiment/multi-contract-scaling`
**Compute:** Local

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| **Timeout Filter** | **REFUTED (Outcome B)** | **Timeout fraction invariant at 41.3%. Volume-driven, not time-driven. Best cutoff=270: $3.02/trade, $34K min acct.** |
| **Trade-Level Risk** | **REFUTED (modified A)** | **$2.50/trade seq, $48K min account. Edge real but larger account needed.** |
| **CPCV Corrected** | **CONFIRMED (Outcome A)** | **$1.81/trade, PBO 6.7%, p<1e-13. 42/45 splits positive.** |

---

## Key Numbers to Remember

### Filtered Sequential (Cutoff=270) — Best Risk-Adjusted
- **Expectancy/trade:** $3.02 (20.6% better than unfiltered)
- **Trades/day:** 116.8 (28% fewer)
- **Daily PnL:** $336.77 (18.4% lower absolute)
- **Annual PnL:** $84,530
- **Win rate:** 50.27%
- **Timeout fraction:** 41.30% (invariant -- same as unfiltered!)
- **Hold-skip rate:** 34.4% (vs 66.1% unfiltered)
- **Worst drawdown:** $33,984 (29% better)
- **Median drawdown:** $8,687 (33% better)
- **Calmar ratio:** 2.49 (best of all cutoffs)
- **Min account (all paths):** $34,000
- **Min account (95% paths):** $25,500
- **ROC (all-path):** 249% annual

### Unfiltered Sequential (Cutoff=390) — Baseline
- **Expectancy/trade:** $2.50
- **Trades/day:** 162.2
- **Daily PnL:** $412.77
- **Annual PnL:** $103,605
- **Worst drawdown:** $47,894
- **Min account (all paths):** $48,000
- **Min account (95% paths):** $26,500
- **ROC (all-path):** 216% annual

### Bar-Level (CPCV)
- **Expectancy/trade:** $1.81 (95% CI [$1.46, $2.16])
- **PBO:** 6.7%
- **Break-even RT:** $4.30

---

Written: 2026-02-27. Timeout filtering REFUTED (Outcome B): timeout fraction invariant at 41.3%, volume-driven not time-driven. Best cutoff=270: $3.02/trade, $34K min acct, 249% ROC. PR #40.
