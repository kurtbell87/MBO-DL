# Last Touch — Cold-Start Briefing

## CRITICAL: Orchestrator Protocol

**You are the ORCHESTRATOR. You NEVER write code. You NEVER read source files. You NEVER run verification commands.**

- ALL code is written by kit sub-agents (`.kit/tdd.sh`, `.kit/experiment.sh`, `.kit/math.sh`)
- You ONLY: read/write state `.md` files, run kit phases, check exit codes
- If you need code written → delegate to a kit phase
- If you need something verified → the kit phase already verified it (trust exit 0)
- See `CLAUDE.md` §ABSOLUTE RULES for the full list

## Project Status

**19 phases complete (9 engineering + 10 research). Tick bar construction defect FIXED. All bar types now genuine.** Branch: `main`.

## What was completed this cycle

- **Tick Bar Fix TDD** — `.kit/docs/tick-bar-fix.md`
- **TDD phases** — red→green→refactor all exit 0
- **Root fix:** `book_builder.hpp` now emits `trade_count` (uint32) per snapshot — counts action='T' MBO events since previous snapshot emission.
- **Bar construction fix:** `tick_bar_builder.hpp` accumulates `trade_count` across snapshots and closes a tick bar when cumulative trades >= threshold. Remainder carries over to next bar.
- **Regression:** Time, dollar, and volume bar construction unchanged. Existing tests pass.
- **New tests:** `tests/tick_bar_fix_test.cpp` — validates trade counting, variable duration, no-trade gaps, daily variance, trade reconciliation, threshold proportionality.
- **Files changed:** `src/book_builder.hpp`, `src/bars/bar_builder_base.hpp`, `src/bars/tick_bar_builder.hpp`, `src/features/bar_features.hpp`, `CMakeLists.txt`, `tests/tick_bar_fix_test.cpp`

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction (time/tick/dollar/volume — all genuine), oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export (9 TDD phases)
- **Research results**: 10 complete research phases. CNN spatial signal confirmed (proper-validation R²≈0.084). Root cause of reproduction failures fully resolved.
- **Architecture decision**: CNN + GBT Hybrid — **NOW GROUNDED.** CNN spatial signal is real. True R²≈0.084 (not 0.132). R6 recommendation validated qualitatively, quantitative edge is 36% smaller than assumed.
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across 7 bar types, 0.14s–300s
- **Bar construction**: ALL bar types now genuine event bars. Tick bars fixed 2026-02-19.

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
| 9D | `.kit/experiments/r3-reproduction-pipeline-comparison.md` | Research | **Done (CONFIRMED Step 1 / REFUTED Step 2)** — R3 reproduced, root cause resolved |
| R3b | `.kit/experiments/r3b-event-bar-cnn.md` | Research | **Done (INCONCLUSIVE)** — bar construction defect |
| **TB-Fix** | **`.kit/docs/tick-bar-fix.md`** | **TDD** | **Done** — tick bars count trades, not snapshots |

## Test summary

- **1003/1004 unit tests** pass (baseline) + new tick_bar_fix tests. TDD phases exited 0.
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

### R3b Rerun with Genuine Tick Bars (UNBLOCKED)

Tick bars are now genuine event bars. Rerun R3b experiment (`.kit/experiments/r3b-event-bar-cnn.md`) to test whether CNN spatial R² on activity-normalized event bars exceeds the time_5s baseline of 0.084.

### CNN+GBT Integration with Corrected Pipeline (HIGHEST PRIORITY)

Root cause is fully resolved. The fix is straightforward:
1. **TICK_SIZE normalization**: Divide book price offsets by 0.25 to get integer tick offsets
2. **Per-day z-scoring**: Z-score log1p(size) per day, not per fold
3. **Proper validation**: Use 80/20 train/val split, not test-as-validation

Re-attempt Phase 9B hybrid model training with these corrections. Expected CNN R²≈0.084 (proper validation).

### Multi-Seed Robustness Study (MEDIUM PRIORITY)

Run 5-fold CV with 5 seeds (25 total runs) using corrected pipeline + proper validation. Confirm R²≈0.084 is robust.

## Key files changed this cycle

| File | Change |
|------|--------|
| `src/book_builder.hpp` | Added `trade_count` field to BookSnapshot |
| `src/bars/bar_builder_base.hpp` | Base class updates for trade-count bar construction |
| `src/bars/tick_bar_builder.hpp` | Tick bars now accumulate `trade_count`, not snapshot count |
| `src/features/bar_features.hpp` | Updated for new tick bar boundary logic |
| `CMakeLists.txt` | Added `tick_bar_fix_test` target |
| `tests/tick_bar_fix_test.cpp` | New: trade counting, variable duration, reconciliation tests |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

---

Updated: 2026-02-19 (Tick Bar Fix TDD — complete. Tick bars now count action='T' trade events, not snapshots.)
