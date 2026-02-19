# Last Touch — Cold-Start Briefing

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, run kit phases, check exit codes
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list

## Project Status

**17 phases complete (8 engineering + 9 research). R3 CNN REPRODUCED (R²=0.1317). Root cause of 9B/9C failure RESOLVED: normalization + validation leakage.** Branch: `tdd/hybrid-model`.

## What was completed this cycle

- **R3 Reproduction & Pipeline Comparison** — `.kit/experiments/r3-reproduction-pipeline-comparison.md`
- **Research kit phases** — frame→run→read all exit 0
- **Step 1 (R3 Reproduction): CONFIRMED.** Mean R²=0.1317 (Δ=-0.0003 from R3 original). Per-fold correlation with R3 = 0.9997. Near-perfect reproduction.
- **Step 2 (Pipeline Comparison): REFUTED.** Data is byte-identical (identity rate=1.0, max diff=0.0). There was NEVER a "Python vs C++" pipeline difference — R3 loaded from the same C++ export as 9B/9C.
- **Root cause RESOLVED:** The 9B/9C R²=0.002 failure was caused by:
  1. **Missing TICK_SIZE normalization** — prices must be divided by 0.25 to get tick offsets (scale ±22.5), not used as raw index points (scale ±5.6)
  2. **Per-fold z-scoring instead of per-day z-scoring** on sizes
  3. **R3's R²=0.132 includes ~36% inflation** from test-as-validation leakage. Proper-validation R²≈0.084.
- **CNN spatial signal is REAL** (R²≈0.084 with proper validation) — still 12× higher than flattened MLP (R²=0.007). CNN+GBT architecture is viable.
- **No C++ export fix needed.** The fix is in the Python training pipeline normalization.
- **Outcome C** (R3 Reproduces + Pipelines Equivalent) — the "UNEXPECTED" outcome from the spec.

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export (8 TDD phases)
- **Research results**: 11 complete research phases. CNN spatial signal confirmed (proper-validation R²≈0.084). Root cause of reproduction failures fully resolved.
- **Architecture decision**: CNN + GBT Hybrid — **NOW GROUNDED.** CNN spatial signal is real. True R²≈0.084 (not 0.132). R6 recommendation validated qualitatively, quantitative edge is 36% smaller than assumed.
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across 7 bar types, 0.14s–300s

## Phase Sequence

| # | Spec | Kit | Status |
|---|------|-----|--------|
| 1 | `.kit/docs/bar-construction.md` | TDD | **Done** |
| 2 | `.kit/docs/oracle-replay.md` | TDD | **Done** |
| 3 | `.kit/docs/multi-day-backtest.md` | TDD | **Done** |
| R1 | `.kit/experiments/subordination-test.md` | Research | **Done** (REFUTED) |
| 4 | `.kit/docs/feature-computation.md` | TDD | **Done** |
| 5 | `.kit/docs/feature-analysis.md` | TDD | **Done** |
| R2 | `.kit/experiments/info-decomposition.md` | Research | **Done** (FEATURES SUFFICIENT) |
| R3 | `.kit/experiments/book-encoder-bias.md` | Research | **Done** (CNN WINS) |
| R4 | `.kit/experiments/temporal-predictability.md` | Research | **Done** (NO SIGNAL) |
| 6 | `.kit/experiments/synthesis.md` | Research | **Done** (CONDITIONAL GO) |
| 7 | `.kit/docs/oracle-expectancy.md` | TDD | **Done** |
| 7b | `tools/oracle_expectancy.cpp` | Research | **Done** (GO) |
| 8 | `.kit/docs/bar-feature-export.md` | TDD | **Done** |
| R4b | `.kit/experiments/temporal-predictability-event-bars.md` | Research | **Done** (NO SIGNAL — robust) |
| R4c | `.kit/experiments/temporal-predictability-completion.md` | Research | **Done** (CONFIRMED — all nulls) |
| R4d | `.kit/experiments/temporal-predictability-dollar-tick-actionable.md` | Research | **Done** (CONFIRMED) |
| 9A | `.kit/docs/hybrid-model.md` | TDD | **Done** (C++ TB label export) |
| 9B | `.kit/experiments/hybrid-model-training.md` | Research | **Done (REFUTED)** — normalization wrong |
| 9C | `.kit/experiments/cnn-reproduction-diagnostic.md` | Research | **Done (REFUTED)** — deviations not root cause |
| **9D** | **`.kit/experiments/r3-reproduction-pipeline-comparison.md`** | **Research** | **Done (CONFIRMED Step 1 / REFUTED Step 2)** — R3 reproduced, root cause resolved |

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

### CNN+GBT Integration with Corrected Pipeline (HIGHEST PRIORITY)

Root cause is fully resolved. The fix is straightforward:
1. **TICK_SIZE normalization**: Divide book price offsets by 0.25 to get integer tick offsets
2. **Per-day z-scoring**: Z-score log1p(size) per day, not per fold
3. **Proper validation**: Use 80/20 train/val split, not test-as-validation

Re-attempt Phase 9B hybrid model training with these corrections. Expected CNN R²≈0.084 (proper validation).

### Multi-Seed Robustness Study (MEDIUM PRIORITY)

Run 5-fold CV with 5 seeds (25 total runs) using corrected pipeline + proper validation. Confirm R²≈0.084 is robust. Fold 3 (Oct 2022) showed R²=-0.047 under proper validation — determine if this is seed-specific or regime-specific.

### GBT-Only Baseline Refinement (INDEPENDENT PATH)

Oracle shows $4.00/trade available. Phase B GBT-only at -$0.38/trade. Can improved feature engineering close the gap without CNN? Independent of CNN question.

### Mental Model Update

- **Before:** "R3's CNN signal cannot be reproduced. Data pipeline is the primary suspect."
- **After:** "R3's CNN signal is REAL and fully reproduced (R²=0.1317). The 9B/9C failure was caused by missing TICK_SIZE normalization + per-day z-scoring. R3's reported R²=0.132 is inflated ~36% by validation leakage; true R²≈0.084. The C++ data export is correct. CNN+GBT path is viable."

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132 (leaked), proper R²≈0.084 |
| R4 | No temporal signal (time_5s) | All 36 AR configs negative R² |
| R4b vol100 | No temporal signal (volume bars) | All 36 AR configs negative R² |
| R4b dollar25k | Marginal signal, redundant | AR R²=0.0006 at h=1, augmentation fails |
| R4c | Tick + extended horizons null | 0/54+ passes across 4 bar types |
| R4d | Dollar + tick actionable null | 0/38 passes, 7s–300s coverage |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO -> **GO** |
| Oracle Expectancy | TB passes all 6 criteria | $4.00/trade, PF=3.30 |
| 9B Hybrid Training | CNN normalization wrong | R²=-0.002 (missing TICK_SIZE div) |
| 9C CNN Repro Diag | Deviations not root cause | R²=0.002, same failure mode |
| **9D R3 Reproduction** | **R3 REPRODUCED, root cause resolved** | **R²=0.1317 (leaked) / 0.084 (proper). Data byte-identical.** |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

---

Updated: 2026-02-19 (R3 Reproduction CONFIRMED — root cause: TICK_SIZE normalization + per-day z-scoring + validation leakage)
