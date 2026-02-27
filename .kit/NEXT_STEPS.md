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

- **30+ phases complete** (15 engineering + 26 research). Branch: `experiment/cpcv-corrected-costs`.
- **CPCV Validation — CONFIRMED (Outcome A)** (2026-02-27). $1.81/trade at corrected-base costs ($2.49 RT). PBO 6.7%. 95% CI [$1.46, $2.16]. t=10.29, p<1e-13. First statistically validated pipeline. PR #38.
- **Threshold Sweep — REFUTED** (PR #36). **Class-Weighted — REFUTED** (PR #37). All parameter-level interventions exhausted.
- **CNN line CLOSED**. **XGBoost tuning DONE**.
- **Critical finding:** Edge is structural (19:7 payoff asymmetry × ~50% accuracy > 34.6% BEV), NOT predictive. Cost correction ($3.74→$2.49) was the key unlock.

## Priority: Deployment Path (Outcome A Triggers)

### 1. Paper Trading Infrastructure (HIGHEST PRIORITY)

Rithmic R|API+ integration for live /MES paper trading. 1 contract. Validates:
- Real-world fill rates and slippage vs assumed $2.49 RT
- Latency (signal→order→fill)
- Execution quality in different market regimes

R|API+ is installed at `~/.local/rapi/13.6.0.0/` but NOT integrated into any source code. CMake target `RApiPlus::RApiPlus` available. SDK docs at `/Users/brandonbell/Downloads/13.6.0.0/`.

**Spec:** Not yet created
**Branch:** `feat/rithmic-paper-trading`
**Compute:** Local

### 2. Hold-Bar Exit Optimization (HIGH VALUE)

43% of traded bars are hold bars with unbounded returns (±63 ticks). Hold-bar PnL swings (-$9.39 to +$3.48 per CPCV split) drive the majority of per-split variance. Test stop-loss rules on hold bars (e.g., exit at -7 ticks) to bound downside while preserving directional-bar payoff asymmetry.

**Rationale:** Could significantly reduce per-split variance without reducing mean expectancy. Dir-bar PnL std = $0.22; hold-bar PnL std = $2.87. Hold bars are 13x more variable.

**Spec:** Not yet created
**Branch:** `experiment/hold-bar-exits`
**Compute:** Local

### 3. Multi-Year Validation (STRONGEST POSSIBLE TEST)

2022 results are regime-specific (rising rates, elevated volatility). Testing on 2023/2024 MES data determines if the structural edge persists. This is the strongest validation short of live trading.

**Requires:** Additional MBO data purchase from Databento.

**Spec:** Not yet created
**Branch:** `experiment/multi-year-validation`
**Compute:** Local or EC2

### 4. Regime-Conditional Position Sizing (INDEPENDENT)

Per-group analysis: groups 6-9 (Jul-Oct) average $2.37/trade vs groups 0-5 (Jan-Jun) $1.44. If message_rate or volatility_50 can predict regime quality, position sizing could increase exposure in high-edge regimes.

**Spec:** Not yet created
**Branch:** `experiment/regime-sizing`
**Compute:** Local

---

## Completed (This Cycle)

| Experiment | Verdict | Key Finding |
|-----------|---------|-------------|
| **CPCV Corrected Costs** | **CONFIRMED (Outcome A)** | **$1.81/trade, PBO 6.7%, p<1e-13. 42/45 splits positive. All 6 SC pass. First validated pipeline.** |
| Threshold Sweep | REFUTED (PR #36) | Probability compression structural — threshold doesn't help. |
| Class-Weighted Stage 1 | REFUTED (PR #37) | Weighting doesn't improve discrimination. |
| PnL Realized Return | REFUTED SC-2, +$0.90/trade (PR #35) | Dir bars +$2.10, hold bars -$1.19. Fold instability. |
| 2-Class Directional | CONFIRMED (PnL caveat, PR #34) | Trade rate 0.28%→85.2%. Dir-bar edge $3.77. |

---

## Key Numbers to Remember

- **CPCV mean expectancy:** $1.81/trade at corrected-base ($2.49 RT)
- **95% CI:** [$1.46, $2.16] — entirely above zero
- **PBO:** 6.7% (3/45 splits negative)
- **Holdout:** $1.46/trade (consistent with CPCV lower CI bound)
- **Break-even RT:** $4.30 ($1.81 margin above base costs)
- **Pooled dir accuracy:** 50.16% (coin flip — edge is payoff asymmetry)
- **Dir-bar PnL:** $5.06 ± $0.22 (stable)
- **Hold-bar PnL:** -$2.50 ± $2.87 (volatile — the variance driver)
- **Regime dispersion:** Late-year $2.37 vs early-year $1.44 (1.65x)
- **Cost sensitivity:** Optimistic $3.06, Base $1.81, Pessimistic -$0.69
- **Wall-clock:** 2.6 min for 98 XGB fits

---

Written: 2026-02-27. CPCV CONFIRMED (Outcome A): $1.81/trade, PBO 6.7%, p<1e-13. PR #38. Next: paper trading (Rithmic R|API+).
