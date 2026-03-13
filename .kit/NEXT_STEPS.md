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

- **31+ phases complete** (15 engineering + 27 research). Branch: `experiment/trade-level-risk-metrics`.
- **Trade-Level Risk Metrics — REFUTED (modified A)** (2026-02-27). Sequential 1-contract: $2.50/trade, 162 trades/day, $412.77/day, $103,605/year. Min account $48K (all paths) / $26.6K (95% paths). PR #39.
- **CPCV Validation — CONFIRMED (Outcome A)** (2026-02-27). $1.81/trade at corrected-base. PBO 6.7%. PR #38.
- **Edge is structural** — 49.93% win rate, 19:7 payoff asymmetry, breakeven 34.6%.

## Priority: Optimize Sequential Execution

### 1. Timeout-Filtered Sequential Execution (HIGHEST PRIORITY)

Pure barrier-hit expectancy is ~$5/trade but observed sequential is $2.50/trade — 50% dilution from timeout trades. Filter entries to bars where barrier resolution is likely before day end.

**Hypothesis:** Avoiding entries where `minutes_since_open > 360 - expected_hold_minutes` increases per-trade expectancy toward $5/trade and reduces drawdowns.

**Impact:** If successful, reduces min_account by ~40-60% (fewer but better trades, smaller drawdowns).

**Spec:** Not yet created
**Branch:** `experiment/timeout-filtered-sequential`
**Compute:** Local

### 2. Paper Trading Infrastructure (HIGH PRIORITY)

Rithmic R|API+ integration for live /MES paper trading. 1 contract. Account >= $48K (or $26.6K at 95% confidence).

**Validates:** Real-world fill rates, slippage, latency.
**Spec:** Not yet created
**Branch:** `feat/rithmic-paper-trading`
**Compute:** Local

### 3. Multi-Contract Scaling Analysis (MODERATE PRIORITY)

Sequential 1-contract captures 5.7% of bar-level daily PnL. Concurrent mean = 35.9 suggests ~36 MES (~7 ES) captures full edge at $1.73M capital.

**Hypothesis:** N staggered sequential executors produce N-proportional daily PnL with sqrt(N) drawdown.
**Spec:** Not yet created
**Branch:** `experiment/multi-contract-scaling`
**Compute:** Local

### 4. Hold-Bar Exit Optimization (MODERATE PRIORITY)

43% of bar-level predictions are hold bars with unbounded returns. Sequential mode already skips these, but if we ENTER on holds with a tighter stop, we could increase trade count without timeout dilution.

**Spec:** Not yet created
**Branch:** `experiment/hold-bar-exits`
**Compute:** Local

### 5. Multi-Year Validation (STRONGEST POSSIBLE TEST)

2022 results are regime-specific. Testing on 2023/2024 MES data determines if structural edge persists.
**Requires:** Additional MBO data purchase from Databento.

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| **Trade-Level Risk** | **REFUTED (modified A)** | **$2.50/trade seq, $48K min account. Edge real but larger account needed than hypothesized.** |
| **CPCV Corrected** | **CONFIRMED (Outcome A)** | **$1.81/trade, PBO 6.7%, p<1e-13. 42/45 splits positive.** |

---

## Key Numbers to Remember

### Sequential 1-Contract Execution
- **Expectancy/trade:** $2.50 (higher than bar-level $1.81 — hold-skip removes drag)
- **Trades/day:** 162.2 (not ~62 — barrier races resolve 2.7x faster under sequential selection)
- **Daily PnL:** $412.77 +/- $2,885.39
- **Annual PnL:** $103,605
- **Win rate:** 49.93% (coin flip)
- **Avg bars held:** 28 (2.3 min — not 75 as estimated)
- **Hold-skip rate:** 66.1% (not 43% — volatility-timing selection bias)
- **Worst drawdown:** $47,894 (across 45 paths)
- **Median drawdown:** $12,917
- **Calmar ratio:** 2.16
- **Annualized Sharpe:** 2.27
- **Min account (all paths):** $48,000
- **Min account (95% paths):** $26,600

### Bar-Level (CPCV)
- **Expectancy/trade:** $1.81 (95% CI [$1.46, $2.16])
- **PBO:** 6.7%
- **Break-even RT:** $4.30
- **Concurrent positions:** 35.9 mean, 106 p95, 345 max

### Scaling
- 1 contract captures 5.7% of bar-level daily PnL
- ~36 MES (~7 ES) for full bar-level capture
- Full-scale capital: $1.73M (all-path) / $958K (95%-path)

---

Written: 2026-02-27. Trade-level risk REFUTED (modified A): $2.50/trade, $48K min. PR #39. Next: timeout filtering or paper trading.
