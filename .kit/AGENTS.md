# Agents — MBO-DL Session State

## Current State (updated 2026-02-17, bar-feature-export TDD cycle)

**Build:** Green.
**Tests:** 1003/1004 unit tests pass (1 disabled, 1 skipped), 22 integration tests (labeled, excluded).
**Branch:** `main`

### Completed This Cycle (bar-feature-export)

- `tools/bar_feature_export.cpp` — **New.** CLI tool for exporting bar-level feature CSVs with parameterized bar type/threshold. Pipeline: StreamingBookBuilder → BarFactory → BarFeatureComputer → CSV.
- `tests/bar_feature_export_test.cpp` — **New.** Unit tests for bar_feature_export (CLI arg parsing, CSV header, metadata columns, warmup exclusion, NaN exclusion).
- `CMakeLists.txt` — Modified (bar_feature_export target + test target with WORKING_DIRECTORY)
- `src/features/bar_features.hpp` — Modified (weighted_imbalance → static)
- `src/backtest/multi_day_runner.hpp` — Modified (push_back loop → vector insert)
- `src/backtest/oracle_expectancy_report.hpp` — Modified (dead code removal: exit_reason_name, push_back → insert)
- `src/analysis/statistical_tests.hpp` — Modified (dead code removal: t_cdf, unused include)

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

### Next Actions

1. Proceed to model architecture build spec.
2. Resolve remaining open questions: CNN at h=1, transaction cost model, CNN+GBT integration pipeline.
