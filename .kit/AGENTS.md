# Agents — MBO-DL Session State

## Current State (updated 2026-02-17, Phase 5 feature-analysis TDD cycle)

**Build:** Green.
**Tests:** 886/887 unit tests pass (1 disabled), 22 integration tests (labeled, excluded).
**Branch:** `main`

### Completed This Cycle (Phase 5)

- `src/analysis/mutual_information.hpp` — MI(feature, return_sign) in bits, quantile binning, bootstrapped null
- `src/analysis/spearman.hpp` — Spearman rank correlation with p-value and 95% CI
- `src/analysis/gbt_importance.hpp` — XGBoost feature importance with stability selection (20 runs, 80% subsamples)
- `src/analysis/conditional_returns.hpp` — Quintile-bucketed mean returns, monotonicity Q5-Q1, t-statistic
- `src/analysis/decay_analysis.hpp` — Correlation decay curves across horizons, signal classification
- `src/analysis/bar_comparison.hpp` — Jarque-Bera, ARCH LM, ACF, Ljung-Box, AR R² tests per bar type
- `src/analysis/multiple_comparison.hpp` — Holm-Bonferroni correction across metric families
- `src/analysis/power_analysis.hpp` — Per-stratum power analysis (detectable effect size at α=0.05, power=0.80)
- `src/analysis/statistical_tests.hpp` — Core statistical test primitives
- `src/analysis/analysis_result.hpp` — Unified result struct (point estimate, CI, raw/corrected p-value, significance flag)
- `src/features/bar_features.hpp` — Modified (added `featureNames()` accessor)
- `tests/feature_mi_test.cpp` — MI + Spearman + decay + Holm-Bonferroni tests
- `tests/conditional_returns_test.cpp` — Conditional returns + warmup exclusion tests
- `tests/bar_comparison_test.cpp` — Bar type comparison + statistical tests
- `tests/gbt_importance_test.cpp` — GBT stability selection tests
- Modified: `CMakeLists.txt` (4 new test targets)

### Phase Sequence

| # | Spec | Status |
|---|------|--------|
| 1 | bar-construction | **Done** |
| 2 | oracle-replay | **Done** |
| 3 | multi-day-backtest | **Done** |
| R1 | subordination-test | **Unblocked** |
| 4 | feature-computation | **Done** |
| 5 | feature-analysis | **In Progress** |
| R2–R3 | Research phases | **Unblocked** |
| R4 | temporal-predictability | Blocked by R1 |
| 6 | synthesis | Blocked by all |

### Next Actions

1. Ship Phase 5 (commit breadcrumbs + changed files).
2. Start Phase R1 (subordination-test), R2 (info-decomposition), or R3 (book-encoder-bias).
