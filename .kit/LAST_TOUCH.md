# Last Touch — Cold-Start Briefing

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, run kit phases, check exit codes
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list

## Project Status

**22 phases complete (9 engineering + 13 research). CNN signal confirmed on time_5s (R²=0.089) and tentatively on tick_100 (R²=0.124, low confidence). NOT economically viable under base costs. Branch: `experiment/r3b-genuine-tick-bars`.**

## What was completed this cycle

- **R3b Genuine Tick Bars** — `.kit/experiments/r3b-genuine-tick-bars.md`
- **Research kit cycle** — RUN phase exit 0, READ phase exit 128 (killed, results already written)
- **Tick bar fix VALIDATED**: All 8 thresholds show bars_per_day_cv > 0 (range 0.189–0.467). p10 != p90 at all thresholds. Genuine trade-event bars confirmed.
- **Calibration complete** for all 8 thresholds (tick_25 through tick_5000):
  - tick_25: 16,836 bars/day, 1.4s median
  - tick_100: 4,171 bars/day, 5.7s median (near-match to time_5s)
  - tick_500: 794 bars/day, 28.5s median
  - tick_5000: 34 bars/day (dropped — below 100 bars/day threshold)
- **CNN trained** on 3 thresholds (tick_25: 5/5 folds, tick_100: 5/5 folds, tick_500: 3/5 folds)
  - tick_2000 not reached (wall clock budget exceeded)
- **Key results:**
  - tick_100 mean R2 = 0.124 (+39% vs time_5s baseline of 0.089) — BETTER nominally
  - BUT: paired t-test p=0.21 (not statistically significant)
  - Fold 5 outlier: tick_100 R2=0.259 (test > train, regime-dependent?)
  - Excluding fold 5: tick_100 R2=0.091 (COMPARABLE, not BETTER)
  - tick_25: R2=0.064 (WORSE — sub-second bars degrade signal)
  - tick_500: R2=0.050 (WORSE, incomplete — 3/5 folds, data starvation on fold 3)
  - Inverted-U curve shape: peak at tick_100
  - Fold 3 diagnostic: tick_25 eliminates fold 3 deficit (+0.004 vs -0.049), tick_100 does not (-0.058)
- **Protocol deviation noted**: Run agent fixed bar_feature_export.cpp in-run (StreamingBookBuilder.emit_snapshot trade_count field). Needs formal TDD cycle.
- **Verdict: CONFIRMED (low confidence)** — tick_100 nominally passes BETTER threshold but evidence is statistically weak

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export, tick bar fix (9 TDD phases)
- **Research results**: 13 complete research phases. CNN spatial signal confirmed (proper-validation R2=0.089 on time_5s, tentatively R2=0.124 on tick_100). End-to-end Hybrid pipeline not viable under base costs.
- **Architecture decision**: CNN + GBT Hybrid — signal is REAL but pipeline design is the bottleneck
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across 7 bar types, 0.14s-300s

## What to do next

### 1. End-to-End CNN Classification (HIGHEST PRIORITY — unchanged)
Train CNN directly on tb_label (3-class cross-entropy) instead of regression->frozen embedding->XGBoost. Eliminates the regression-to-classification bottleneck. Run on time_5s first.

### 2. XGBoost Hyperparameter Tuning (LOW-HANGING FRUIT — unchanged)
Grid search to close the 2pp win rate gap. Current hyperparameters inherited from 9B (broken pipeline era).

### 3. Label Design Sensitivity (ARCHITECTURAL — unchanged)
Test wider target (15 ticks) / narrower stop (3 ticks). At 15:3 ratio, breakeven win rate drops to ~42.5%.

### 4. Tick_100 Replication with Multi-Seed (if pursuing event bars)
Run tick_50, tick_100, tick_200, tick_500, tick_2000 each with 3 seeds per fold (75 CNN fits). Budget ~6h.

### 5. TDD: bar_feature_export StreamingBookBuilder Fix
Formalize the in-run fix with a proper TDD cycle. Run agent patched StreamingBookBuilder.emit_snapshot to populate trade_count.

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R2=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R2=0.132 (leaked), proper R2=0.084 |
| R4-R4d | No temporal signal | 0/168+ dual threshold passes |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO -> GO |
| Oracle | TB passes all 6 criteria | $4.00/trade, PF=3.30 |
| 9B-9C | CNN normalization wrong | R2=-0.002 |
| 9D | R3 reproduced, root cause resolved | R2=0.1317 (leaked) / 0.084 (proper) |
| R3b-orig | Tick bars are time bars | Peak R2=0.057, all < baseline |
| 9E | CNN viable, pipeline not | R2=0.089, exp=-$0.37/trade |
| **R3b-genuine** | **tick_100 tentatively BETTER** | **R2=0.124, p=0.21 (low confidence)** |

---

Updated: 2026-02-19 (R3b Genuine Tick Bars — CONFIRMED low confidence. tick_100 R2=0.124 vs time_5s R2=0.089. Fold-5-dependent, not statistically significant.)
