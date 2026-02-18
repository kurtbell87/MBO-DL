# Last Touch — Cold-Start Briefing

## Project Status

**14 phases complete (8 engineering + 6 research). Phase 9 (hybrid-model) Phase A in progress.** C++ triple barrier labeling added to `bar_feature_export`. Branch: `feature/hybrid-model`.

## What was completed this cycle

- **Hybrid model Phase A — C++ TB label export**: Extended `bar_feature_export.cpp` with triple barrier label columns (`tb_label`, `tb_exit_type`, `tb_bars_held`).
  - `src/backtest/triple_barrier.hpp` — Added `label_bar()` for position-independent TB labeling per bar.
  - `tools/bar_feature_export.cpp` — Added TB label computation and CSV columns to export pipeline.
  - `tests/hybrid_model_tb_label_test.cpp` — **New.** Tests for TB label computation (spec tests 1–6).
  - `tests/test_export_helpers.hpp` — **New.** Export test helpers.
  - `tests/bar_feature_export_test.cpp` — Updated for new TB columns.
  - `tests/test_bar_helpers.hpp` — Extended for TB label testing.
  - `CMakeLists.txt` — Added hybrid_model_tb_label_test target.

## What exists

A C++20 MES microstructure model suite with:
- **Infrastructure**: Bar construction, oracle replay, multi-day backtest, feature computation/export, feature analysis, oracle expectancy report, bar feature export (8 TDD phases)
- **Research results**: Subordination test, info decomposition, book encoder bias, temporal predictability (time_5s + event bars + dollar/tick actionable), synthesis, oracle expectancy (8 complete research phases)
- **Architecture decision**: CNN + GBT Hybrid — Conv1d on raw (20,2) book -> 16-dim embedding -> concat with ~20 non-spatial features -> XGBoost
- **Labeling decision**: Triple barrier (preferred over first-to-hit)
- **Temporal verdict**: NO TEMPORAL SIGNAL — confirmed across 7 bar types, 0.14s–300s. Drop SSM/temporal encoder.

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
| **9** | **`.kit/docs/hybrid-model.md`** | **TDD** | **Phase A (C++ TB labels) in progress** |

## Test summary

- **1003 unit tests** pass, 1 disabled, 1 skipped, 1004 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~14 min. Integration: ~20 min.

## What to do next

### Immediate: Complete Phase A TDD cycle

Finish the C++ TB label export TDD cycle (red→green→refactor→ship) for `.kit/docs/hybrid-model.md`:

1. **Verify** current test state — do the new TB label tests pass?
2. **Complete** remaining C++ tests (spec tests 7–10: label distribution, no NaN, CSV schema, backward compat).
3. **Ship** Phase A — ensure all 1003+ unit tests still pass.

### Then: Python pipeline (Phases B–D)

1. Python CNN encoder + training (spec tests 11–15)
2. Python data loading + normalization (spec tests 16–19)
3. Python XGBoost training (spec tests 20–21)
4. Python evaluation pipeline (spec tests 22–26)
5. Run full 5-fold CV, collect results
6. Write analysis document to `.kit/results/hybrid-model/analysis.md`

## Key research results

| Experiment | Finding | Key Number |
|-----------|---------|------------|
| R1 | Subordination refuted for MES | 0/3 primary tests significant |
| R2 | Hand-crafted features sufficient | Best R²=0.0067 (1-bar MLP) |
| R3 | CNN best book encoder | R²=0.132, CNN>Attention p=0.042 |
| R4 | No temporal signal (time_5s) | All 36 AR configs negative R² |
| R4b vol100 | No temporal signal (volume bars) | All 36 AR configs negative R² |
| R4b dollar25k | Marginal signal, redundant | AR R²=0.0006 at h=1, augmentation fails |
| R4c | Tick + extended horizons null | 0/54+ passes across 4 bar types |
| R4d | Dollar + tick actionable null | 0/38 passes, 7s–300s coverage |
| Synthesis | CNN + GBT Hybrid | CONDITIONAL GO -> **GO** |
| Oracle Expectancy | TB passes all 6 criteria | $4.00/trade, PF=3.30 |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~14 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
./build/oracle_expectancy                                                  # oracle expectancy extraction
./build/bar_feature_export --bar-type <type> --bar-param <param> --output <csv>  # feature export
```

## Key files this cycle

| File | Change |
|------|--------|
| `src/backtest/triple_barrier.hpp` | Added `label_bar()` function |
| `tools/bar_feature_export.cpp` | TB label columns in CSV export |
| `tests/hybrid_model_tb_label_test.cpp` | **New** — TB label unit tests |
| `tests/test_export_helpers.hpp` | **New** — export test helpers |
| `tests/bar_feature_export_test.cpp` | Updated for TB columns |
| `tests/test_bar_helpers.hpp` | Extended for TB testing |
| `CMakeLists.txt` | hybrid_model_tb_label_test target |

---

Updated: 2026-02-18 (hybrid-model Phase A — C++ TB label export)
