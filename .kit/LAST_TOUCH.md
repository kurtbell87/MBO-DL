# Last Touch — Cold-Start Briefing

## Project Status

**Phase 5 (feature-analysis) TDD cycle in progress.** Statistical analysis infrastructure for Track A features: MI analysis, Spearman correlation, GBT stability-selected importance, conditional returns, decay curves, bar type comparison, Holm-Bonferroni multiple comparison correction, and power analysis. 160 new unit tests added (886/887 total pass). New `src/analysis/` directory with 10 headers.

## What was completed this cycle

- `src/analysis/mutual_information.hpp` — MI(feature, return_sign) in bits, quantile binning (5/10 bins), bootstrapped null (≥1000 shuffles)
- `src/analysis/spearman.hpp` — Spearman rank correlation with p-value and 95% CI
- `src/analysis/gbt_importance.hpp` — XGBoost feature importance, stability selection (20 runs, 80% subsamples, top-20 in >60% threshold)
- `src/analysis/conditional_returns.hpp` — Quintile-bucketed mean returns, monotonicity (Q5-Q1), t-statistic
- `src/analysis/decay_analysis.hpp` — Correlation decay curves at horizons 1,2,5,10,20,50,100; signal classification (short-horizon vs regime indicator)
- `src/analysis/bar_comparison.hpp` — Jarque-Bera, ARCH LM, ACF, Ljung-Box, AR R² tests per bar type config
- `src/analysis/multiple_comparison.hpp` — Holm-Bonferroni correction across 1,800 tests per metric family
- `src/analysis/power_analysis.hpp` — Per-stratum power analysis (detectable effect size at α=0.05, power=0.80)
- `src/analysis/statistical_tests.hpp` — Core statistical test primitives
- `src/analysis/analysis_result.hpp` — Unified result struct (point estimate, CI, raw/corrected p-value, significance flag)
- `src/features/bar_features.hpp` — Modified (added `featureNames()` accessor for analysis pipeline)
- `tests/feature_mi_test.cpp` — MI, Spearman, decay analysis, Holm-Bonferroni tests
- `tests/conditional_returns_test.cpp` — Conditional returns, warmup exclusion tests
- `tests/bar_comparison_test.cpp` — Bar type comparison, statistical test tests
- `tests/gbt_importance_test.cpp` — GBT stability selection tests
- Modified: `CMakeLists.txt` (4 new test targets)

## What exists

A C++20 MES microstructure model suite that reads raw Databento MBO (L3) order data from `.dbn.zst` files. The overfit harness (MLP, CNN, GBT) is validated at N=32 and N=128 on real data. Serialization (checkpoint + ONNX) is shipped. Bar construction (Phase 1), oracle replay (Phase 2), multi-day backtest infrastructure (Phase 3), feature computation/export (Phase 4), and feature analysis statistical infrastructure (Phase 5) are complete.

## Phase Sequence

| # | Spec | Kit | Status |
|---|------|-----|--------|
| 1 | `.kit/docs/bar-construction.md` | TDD | **Done** |
| 2 | `.kit/docs/oracle-replay.md` | TDD | **Done** |
| 3 | `.kit/docs/multi-day-backtest.md` | TDD | **Done** |
| R1 | `.kit/experiments/subordination-test.md` | Research | **Unblocked** |
| 4 | `.kit/docs/feature-computation.md` | TDD | **Done** |
| 5 | `.kit/docs/feature-analysis.md` | TDD | **In Progress** |
| R2 | `.kit/experiments/info-decomposition.md` | Research | **Unblocked** |
| R3 | `.kit/experiments/book-encoder-bias.md` | Research | **Unblocked** |
| R4 | `.kit/experiments/temporal-predictability.md` | Research | Blocked by R1 |
| 6 | `.kit/experiments/synthesis.md` | Research | Blocked by all |

## Test summary

- **886 unit tests** pass, 1 disabled (`BookBuilderIntegrationTest.ProcessSingleDayFile`), 887 total
- **22 integration tests** (14 N=32 + 8 N=128) — labeled `integration`, excluded from default ctest
- Unit test time: ~6 min. Integration: ~20 min.

## What to do next

1. Ship Phase 5 (feature-analysis): commit all changed files + breadcrumbs.
2. Or start Phase R1 (subordination-test): `source .master-kit.env && ./.kit/experiment.sh survey .kit/experiments/subordination-test.md`
3. Or start Phase R2 (info-decomposition): `source .master-kit.env && ./.kit/experiment.sh survey .kit/experiments/info-decomposition.md`

## Key files (Phase 5)

| File | Purpose |
|------|---------|
| `src/analysis/mutual_information.hpp` | MI(feature, return_sign), bootstrapped null |
| `src/analysis/spearman.hpp` | Spearman rank correlation with CI |
| `src/analysis/gbt_importance.hpp` | XGBoost stability-selected feature importance |
| `src/analysis/conditional_returns.hpp` | Quintile returns, monotonicity |
| `src/analysis/decay_analysis.hpp` | Correlation decay curves, signal classification |
| `src/analysis/bar_comparison.hpp` | Bar type statistical comparison suite |
| `src/analysis/multiple_comparison.hpp` | Holm-Bonferroni correction |
| `src/analysis/power_analysis.hpp` | Per-stratum power analysis |
| `src/analysis/analysis_result.hpp` | Unified result struct |
| `src/analysis/statistical_tests.hpp` | Core statistical test primitives |
| `src/features/bar_features.hpp` | Track A features (modified: `featureNames()`) |

## Build commands

```bash
cmake --build build -j12                                                  # build
cd build && ctest --output-on-failure --label-exclude integration         # unit tests (~6 min)
cd build && ctest --output-on-failure --label-regex integration           # integration tests (~20 min)
```

---

Updated: 2026-02-17
