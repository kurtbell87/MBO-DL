# Phase 5: Feature Analysis [Engineering + Research]

**Spec**: TRAJECTORY.md §8.4 (predictiveness), §8.5 (bar comparison), §8.7 (power analysis)
**Depends on**: Phase 4 (feature-computation) — needs exported features + returns.
**Unlocks**: Phase 6 (synthesis) — provides feature ranking and bar type signal quality.

---

## Objective

Analyze the predictive power of Track A features across all feature × return horizon × bar type combinations. Produce MI analysis, GBT feature importance with stability selection, decay curves, and bar type signal quality comparison. Apply rigorous multiple comparison correction throughout.

---

## Analysis Protocol

### 1. Mutual Information (§8.4)

For each feature × return_horizon × bar_type:
- Discretize feature into 5 or 10 quantile bins.
- Compute MI(feature, return_sign) in bits.
- Baseline: bootstrapped null (shuffle feature, recompute MI, take 95th percentile). Report excess MI over null.
- Apply Holm-Bonferroni correction across all 1,800 feature-horizon-bartype tests.

### 2. Spearman Rank Correlation

For each feature × return_horizon × bar_type:
- Compute Spearman corr(feature, return_n) with p-value.
- Apply Holm-Bonferroni correction (1,800 tests).

### 3. GBT Feature Importance (Stability-Selected)

- Train XGBoost regressor: all Track A features → return_n.
- 5-fold expanding-window time-series CV.
- **Stability selection**: 20 runs with different random seeds and 80% subsamples.
- Report features appearing in top-20 in >60% of runs.
- Not p-value-based — more robust than single-run importance.

### 4. Conditional Returns

- Bucket each feature into quintiles.
- Compute mean return_n per quintile.
- Report monotonicity: Q5 mean - Q1 mean, t-statistic.

### 5. Decay Analysis

For each predictive feature:
- Compute correlation with return_n for n = 1, 2, 5, 10, 20, 50, 100 bars.
- Plot decay curve.
- Sharp decay → short-horizon signal. Slow decay → regime indicator.

---

## Bar Type Comparison (§8.5)

For each bar type configuration (~10 configs):

| Metric | What It Tests | Source |
|--------|--------------|--------|
| Jarque-Bera on 1-bar returns | Normality (§2.1) | scipy.stats |
| ARCH LM test | Heteroskedasticity (§2.1) | statsmodels |
| ACF of \|return_1\| at lags 1, 5, 10 | Volatility clustering (§2.1) | statsmodels |
| Ljung-Box at lags 1, 5, 10 | Returns autocorrelation | statsmodels |
| AR R² for return_h from last 10 returns | Temporal predictability (§2.4) | sklearn/xgboost |
| Sum of excess MI across Track A | Aggregate feature info | computed |
| CV of daily bar counts | Bar count stability | numpy |

Apply Holm-Bonferroni within each metric family (10 tests per metric).

---

## Power Analysis (§8.7)

```
Dimensionality:
  ~45 features × 4 horizons × ~10 bar configs = 1,800 tests per metric

Minimum sample sizes:
  Spearman r=0.05 at α=0.05, power=0.80: n ≈ 2,500 bars
  Full dataset: 50,000–250,000 bars → adequate for small effects
  Per-fold: 10,000–50,000 → adequate
  Per-stratum: 5,000–20,000 → detectable r ≈ 0.03–0.04

Report per-stratum power alongside results.
```

---

## Reporting Standard

All tables report:
- Point estimate
- 95% CI
- Raw p-value
- Corrected p-value (Holm-Bonferroni)
- Whether result survives correction

Results significant before but not after correction: flagged as "suggestive, insufficient evidence after correction."

---

## Outputs

```
analysis/
  feature_analysis.py       # MI, correlation, GBT importance, decay curves
  bar_comparison.py         # Normality, autocorrelation, heteroskedasticity tests

Deliverables:
  - Feature × horizon heatmap (MI or correlation) per bar type
    with significance markers (★ = survives correction, ○ = suggestive)
  - Top-20 features ranked by excess MI for each bar type (stability-selected)
  - Decay curves for top features
  - Bar type comparison table (normality, autocorrelation, heteroskedasticity, aggregate MI)
    with Holm-Bonferroni corrected p-values
  - Per-stratum power analysis (detectable effect size at α=0.05, power=0.80)
  - Recommendations for GBT model input features and bar type selection
```

---

## Validation Gate

```
Assert: MI analysis runs on is_warmup == false bars only
Assert: Bootstrapped null uses >= 1000 shuffles per feature
Assert: GBT stability selection uses 20 independent runs
Assert: Holm-Bonferroni correction applied within each metric family
Assert: Time-series CV uses expanding window (no shuffling, no future leakage)
Assert: Per-stratum power analysis reported for all stratifications
Assert: All tables include raw p-value, corrected p-value, and correction flag
```
