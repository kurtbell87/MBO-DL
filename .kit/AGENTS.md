# Agents — MBO-DL Session State

## Current State (updated 2026-02-18, hybrid-model Phase A in progress)

**Build:** Green.
**Tests:** 1003/1004 unit tests pass (1 disabled, 1 skipped), 22 integration tests (labeled, excluded).
**Branch:** `feature/hybrid-model`

### Completed This Cycle (hybrid-model Phase A — C++ TB label export)

- `src/backtest/triple_barrier.hpp` — Modified (added `label_bar()` function for position-independent TB labeling).
- `tools/bar_feature_export.cpp` — Modified (added `tb_label`, `tb_exit_type`, `tb_bars_held` columns to CSV export).
- `tests/bar_feature_export_test.cpp` — Modified (updated for new TB label columns).
- `tests/test_bar_helpers.hpp` — Modified (extended helpers for TB label testing).
- `tests/hybrid_model_tb_label_test.cpp` — **New.** TB label computation tests (label correctness, exit type, bars held, volume accumulation, time cap, min_return filter).
- `tests/test_export_helpers.hpp` — **New.** Export test helpers.
- `CMakeLists.txt` — Modified (hybrid_model_tb_label_test target).

### Phase Sequence

| # | Spec | Status |
|---|------|--------|
| 1 | bar-construction | **Done** |
| 2 | oracle-replay | **Done** |
| 3 | multi-day-backtest | **Done** |
| R1 | subordination-test | **Done** (REFUTED) |
| 4 | feature-computation | **Done** |
| 5 | feature-analysis | **Done** |
| R2 | info-decomposition | **Done** (FEATURES SUFFICIENT) |
| R3 | book-encoder-bias | **Done** (CNN WINS) |
| R4 | temporal-predictability | **Done** (NO SIGNAL) |
| 6 | synthesis | **Done** (CONDITIONAL GO) |
| 7 | oracle-expectancy | **Done** |
| 7b | oracle_expectancy tool | **Done** (GO) |
| 8 | bar-feature-export | **Done** |
| R4b | temporal-predictability-event-bars | **Done** (NO SIGNAL — robust) |
| R4c | temporal-predictability-completion | **Done** (CONFIRMED — all nulls) |
| R4d | temporal-predictability-dollar-tick-actionable | **Done** (CONFIRMED) |
| **9** | **hybrid-model** | **Phase A (C++ TB labels) in progress** |

### Next Actions

1. Complete Phase A TDD cycle: red → green → refactor → ship for C++ TB label export.
2. Phase B: Python CNN encoder training pipeline (`scripts/hybrid_model/`).
3. Phase C: Python XGBoost classification + evaluation.
4. Phase D: 5-fold expanding window CV + ablation comparisons + analysis document.
