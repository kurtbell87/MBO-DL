# src/analysis/ — Feature Analysis Infrastructure

Phase 5 (feature-analysis) statistical analysis library. Analyzes predictive power of Track A features across feature × return horizon × bar type combinations.

## Headers

| File | Purpose |
|------|---------|
| `analysis_result.hpp` | Unified result struct: point estimate, 95% CI, raw/corrected p-value, significance flag |
| `mutual_information.hpp` | MI(feature, return_sign) in bits; quantile binning; bootstrapped null (≥1000 shuffles) |
| `spearman.hpp` | Spearman rank correlation with p-value and 95% CI |
| `gbt_importance.hpp` | XGBoost feature importance with stability selection (20 runs, 80% subsamples) |
| `conditional_returns.hpp` | Quintile-bucketed mean returns; monotonicity (Q5−Q1); t-statistic |
| `decay_analysis.hpp` | Correlation decay curves at horizons 1,2,5,10,20,50,100; signal classification |
| `bar_comparison.hpp` | Jarque-Bera, ARCH LM, ACF, Ljung-Box, AR R² per bar type config |
| `multiple_comparison.hpp` | Holm-Bonferroni correction across metric families (~1,800 tests) |
| `power_analysis.hpp` | Per-stratum power analysis (detectable effect size at α=0.05, power=0.80) |
| `statistical_tests.hpp` | Core statistical test primitives |

## Spec

`.kit/docs/feature-analysis.md` — TRAJECTORY.md §8.4, §8.5, §8.7.
