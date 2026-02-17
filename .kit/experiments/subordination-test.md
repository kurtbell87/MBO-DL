# Phase R1: Subordination Hypothesis Test [Research]

**Spec**: TRAJECTORY.md §2.1 (theory), §8.5 (bar comparison metrics), §8.7 (multiple comparisons)
**Depends on**: Phase 1 (bar-construction) — needs bar builders to produce return series.
**Can run in parallel with**: Phases 2–3 (doesn't need the oracle).
**Unlocks**: Phase R4 (temporal-predictability) — provides bar type recommendation.

---

## Hypothesis

**Clark (1973), Ane & Geman (2000)**: Price returns sampled at event-driven boundaries (volume, tick) are closer to IID Gaussian than returns sampled at fixed time intervals. If the subordination model holds for MES, volume bars should produce:
- Lower Jarque-Bera statistic (more normal returns)
- Lower ARCH(1) coefficient (less conditional heteroskedasticity)
- Faster decay of |return| autocorrelation (less volatility clustering)

---

## Method

### Bar Configurations to Test

```
Volume bars:  V ∈ {50, 100, 200}
Tick bars:    K ∈ {25, 50, 100}
Time bars:    interval ∈ {1s, 5s, 60s}
Dollar bars:  D ∈ {25000, 50000, 100000}   (optional, lower priority)

Total: ~10 configurations
```

### Data

Use 10+ trading days from the 2022 MES dataset (non-rollover days, representative sample across months). For each day × bar config, produce bar sequence and compute 1-bar return series.

### Metrics (per bar configuration)

1. **Jarque-Bera statistic** on 1-bar returns (normality). Lower = more Gaussian.
2. **ARCH(1) coefficient** from ARCH LM test on 1-bar returns. Lower = less heteroskedastic.
3. **Autocorrelation of |returns|** at lags 1, 5, 10. Faster decay = better normalization.
4. **Bar count statistics**: mean, std, coefficient of variation per day.

### Statistical Framework

- Compute metrics per day, then aggregate (mean ± std across days).
- Rank bar types by each metric.
- Apply **Holm-Bonferroni correction** for bar type comparisons (10 configs per metric).
- Report whether volume/tick bars significantly outperform time bars (p < 0.05 corrected).

---

## Implementation

```
research/R1_subordination_test.py

Dependencies: scipy (jarque_bera, chi2), statsmodels (ARCH LM), numpy, polars
Input: Bar sequences from C++ bar builders (via CSV/Parquet export or direct bar builder invocation)
```

---

## Deliverable

```
Table: bar_type × param → (JB_stat, ARCH_coeff, |r| autocorrelation, bar_count_mean, bar_count_CV)
       with p-values and Holm-Bonferroni corrected significance

Finding: Does the subordination model hold for MES? Which bar type best conditions out
         the stochastic time change?

Decision: Recommended primary bar type and parameter for subsequent phases.
```

---

## Exit Criteria

- [ ] Jarque-Bera, ARCH, volatility clustering compared across all bar type configs
- [ ] Holm-Bonferroni corrected p-values reported for all comparisons
- [ ] Bar count stability (CV) reported per config
- [ ] Primary bar type recommended with empirical justification
- [ ] Results written to `.kit/results/R1_subordination/`
