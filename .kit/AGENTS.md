# Agents — MBO-DL Session State

## Current State (updated 2026-02-17, oracle-expectancy TDD cycle)

**Build:** Green.
**Tests:** 953/954 unit tests pass (1 disabled), 22 integration tests (labeled, excluded).
**Branch:** `main`

### Completed This Cycle (oracle-expectancy)

- `src/backtest/oracle_expectancy_report.hpp` — **New.** `OracleExpectancyReport` struct, `to_json` serializer, `aggregate_day_results` aggregation with per-quarter splits
- `tests/oracle_expectancy_test.cpp` — **New.** Unit tests for report layer (JSON output, aggregation, quarter splits, edge cases)
- `src/backtest/multi_day_runner.hpp` — Modified (support for oracle expectancy aggregation)
- `src/backtest/oracle_replay.hpp` — Modified (support for oracle expectancy aggregation)
- `src/serialization.hpp` — Modified
- `tests/bar_features_test.cpp` — Modified
- `tests/test_bar_helpers.hpp` — Modified
- `CMakeLists.txt` — Modified (oracle_expectancy_test target)

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

### Next Actions

1. Build `tools/oracle_expectancy.cpp` standalone executable to run on real MES data.
2. Resolve remaining open questions: CNN at h=1, transaction cost model.
3. Proceed to model architecture build spec.
