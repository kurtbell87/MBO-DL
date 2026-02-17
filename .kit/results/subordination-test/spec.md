# Phase R1: Subordination Hypothesis Test [Research]

**Spec**: TRAJECTORY.md §2.1 (theory), §2.4 (entropy rate), §8.5 (bar comparison metrics), §8.7 (multiple comparisons)
**Depends on**: Phase 1 (bar-construction) — bar builders for volume, tick, dollar, time.
**Can run in parallel with**: Phases 2–5 (no oracle or feature dependency).
**Unlocks**: Phase R4 (temporal-predictability) — provides bar type recommendation.

---

## Hypothesis

**Clark (1973), Ané & Geman (2000)**: Price returns sampled at event-driven boundaries (volume, tick, dollar) are closer to IID Gaussian than returns sampled at fixed time intervals. If the subordination model $P(t) = W(N(t))$ holds for MES, conditioning on fixed increments of the directing process $N$ removes the stochastic time change, yielding:

$$\tilde{r}_k = W(kV + V) - W(kV) \sim \mathcal{N}(0, \sigma^2 V)$$

**Testable predictions** (three primary, one secondary):

1. **Normality**: Jarque-Bera statistic on volume/tick/dollar bar returns < Jarque-Bera on time bar returns at matched sample sizes.
2. **Homoskedasticity**: ARCH(1) coefficient on event-bar returns < ARCH(1) on time-bar returns.
3. **Volatility clustering**: Autocorrelation of |return| decays faster for event bars than time bars.
4. **Temporal structure** (secondary): Autoregressive R² for predicting $r_{k+1}$ is higher for the bar type that best normalizes the return distribution, indicating the temporal encoder has cleaner signal.

**Null hypothesis**: The subordination model is a poor fit for MES microstructure (plausible — MES is a derivative whose information arrival is driven by ES, not its own order flow). In this case, event-driven and time-driven bars produce statistically indistinguishable return distributions.

---

## Bar Configurations

| Bar Type  | Parameter | Values | Expected Daily Bars (RTH) |
|-----------|-----------|--------|---------------------------|
| Volume    | V (contracts) | 50, 100, 200 | ~1000–4000, ~500–2000, ~250–1000 |
| Tick      | K (trades)    | 25, 50, 100  | ~800–3000, ~400–1500, ~200–750 |
| Dollar    | D ($ volume)  | 25000, 50000, 100000 | varies by price level |
| Time      | interval (s)  | 1, 5, 60     | ~23400, ~4680, ~390 |

**Total**: 12 configurations (3 event-driven types × 3 params + 1 time type × 3 params).

**Matching constraint**: For fair comparison, pair each event-bar config with the time-bar interval that produces the closest median daily bar count. This controls for sample-size effects on test statistics.

---

## Data Selection

**Source**: `DATA/GLBX-20260207-L953CAPU5B/` — 312 daily `.dbn.zst` files, MES MBO 2022.

**Day selection**: 20 non-rollover trading days, stratified across calendar quarters and volatility regimes:
- 5 days per quarter (Q1–Q4 2022)
- Within each quarter, select: 2 high-activity days, 2 normal-activity days, 1 low-activity day (ranked by total session volume)
- Exclude rollover windows via `RolloverCalendar::is_excluded()`
- Use RTH session only (09:30–16:00 ET) to avoid thin overnight books

**Rationale**: 20 days gives sufficient statistical power for cross-day aggregation while keeping total compute under budget. Stratification across quarters captures seasonal variation (2022 had distinct regimes: Jan–Mar drawdown, Jun–Oct bear, Nov–Dec recovery).

---

## Protocol

### Step 1: Bar Construction (per day × config)

For each of the 20 days and 12 bar configs:
1. Load `.dbn.zst` via `databento::DbnFileStore` → `BookBuilder` → `BookSnapshot` stream
2. Feed snapshots through the appropriate `BarBuilder` (VolumeBarBuilder, TickBarBuilder, DollarBarBuilder, TimeBarBuilder)
3. Compute 1-bar log-returns: $r_k = \ln(\text{close\_mid}_{k+1}) - \ln(\text{close\_mid}_k)$
4. Record bar count per day
5. Discard first 20 bars per day (EWMA warm-up per §8.6 policy)

**Output**: 240 return series (20 days × 12 configs), each stored as `std::vector<float>`.

### Step 2: Per-Configuration Metrics

For each of the 12 configurations, pool returns across all 20 days (but also compute per-day for cross-day variability):

| # | Metric | Implementation | Interpretation |
|---|--------|----------------|----------------|
| 1 | **Jarque-Bera** | `jarque_bera_test()` from `statistical_tests.hpp` | Lower stat = more Gaussian |
| 2 | **ARCH LM(1)** | `arch_lm_test()` from `statistical_tests.hpp` | Lower stat = less heteroskedastic |
| 3 | **ACF of \|return\| at lags 1, 5, 10** | `compute_acf()` on absolute returns | Faster decay = less vol clustering |
| 4 | **Ljung-Box at lags 1, 5, 10** | `ljung_box_test()` on raw returns | Tests serial dependence structure |
| 5 | **AR(10) R²** | `compute_ar_r2()` with p=10 | Higher = more temporal structure for SSM |
| 6 | **Bar count CV** | `compute_bar_count_cv()` on daily counts | Lower = more stable bar production |

**Per-day metrics**: Compute JB, ARCH(1), and ACF(1) independently for each of the 20 days to get cross-day mean ± std. This enables paired statistical tests.

### Step 3: Pairwise Comparisons

Compare each event-bar config against its matched time-bar config:

1. **Paired test**: For each metric, form 20-element vectors (one value per day) for the event-bar config and its matched time-bar config. Use a **Wilcoxon signed-rank test** (non-parametric, robust to non-normal metric distributions) on the paired differences.
2. **Direction**: For JB, ARCH, and ACF — test whether event-bar values are *lower* (one-sided). For AR R² — test whether event-bar values are *higher* (one-sided).
3. **Multiple comparison correction**: 12 configs × 6 metric families = 72 tests total. Apply **Holm-Bonferroni correction** within each metric family (12 tests per family) via `holm_bonferroni_correct()` from `multiple_comparison.hpp`. Report both raw and corrected p-values.
4. **Effect size**: Report the median paired difference and the Hodges-Lehmann estimator (median of pairwise averages) as a robust effect size measure.

### Step 4: Aggregate Ranking

Produce a composite ranking across all metrics:

1. For each metric, rank configs 1–12 (best to worst).
2. Compute the mean rank across all 6 metrics for each config.
3. The config with the lowest mean rank is the recommended primary bar type.
4. If multiple configs are within 1 rank of each other, prefer the one with lower bar-count CV (more stable).

### Step 5: Robustness Checks

1. **Quarter-level split**: Repeat the primary comparison (best event bar vs. best time bar) separately for each quarter. If the effect reverses in any quarter, flag as regime-dependent.
2. **Sample-size sensitivity**: For the top-ranked event bar, vary the threshold parameter ±50% and check whether the ordering is stable.
3. **KS test on return distribution**: Kolmogorov-Smirnov 2-sample test between event-bar returns and a fitted Gaussian (same mean, variance). Report the KS statistic as a supplementary normality measure.

---

## Implementation

```
Language: C++ (all infrastructure exists)
Entry point: Stand-alone executable or integration test

Key source files:
  src/bars/volume_bar_builder.hpp
  src/bars/tick_bar_builder.hpp
  src/bars/dollar_bar_builder.hpp
  src/bars/time_bar_builder.hpp
  src/analysis/bar_comparison.hpp     — BarComparisonAnalyzer, BarTypeData, BarComparisonResult
  src/analysis/statistical_tests.hpp  — jarque_bera_test, arch_lm_test, compute_acf, ljung_box_test, compute_ar_r2, compute_bar_count_cv
  src/analysis/multiple_comparison.hpp — holm_bonferroni_correct
  src/book_builder.hpp               — BookBuilder, BookSnapshot
  src/backtest/rollover.hpp           — RolloverCalendar::is_excluded()

Data loading:
  databento::DbnFileStore for .dbn.zst ingestion → BookBuilder → BarBuilder pipeline

Output format:
  JSON results file per configuration, plus aggregate summary table.
  All written to .kit/results/R1_subordination/
```

---

## Compute Budget

| Item | Estimate |
|------|----------|
| Days × configs | 20 × 12 = 240 bar-construction runs |
| Bar construction per day | ~5–15s (I/O bound, .dbn.zst decompression) |
| Statistical tests per config | <1s (in-memory, ~500–5000 returns) |
| Total wall-clock | ~1–2 hours single-threaded, <30 min with day-level parallelism |
| GPU | None required (all CPU) |
| Runs | 1 (deterministic — no stochastic training) |

**Within budget**: 0 GPU-hours, 1 run.

---

## Deliverables

### Table 1: Bar Type Comparison Matrix

```
Config      | JB stat | ARCH(1) | ACF|r|(1) | ACF|r|(5) | ACF|r|(10) | AR R² | Bar CV | Mean Rank
------------|---------|---------|-----------|-----------|------------|-------|--------|----------
vol_50      |         |         |           |           |            |       |        |
vol_100     |         |         |           |           |            |       |        |
vol_200     |         |         |           |           |            |       |        |
tick_25     |         |         |           |           |            |       |        |
tick_50     |         |         |           |           |            |       |        |
tick_100    |         |         |           |           |            |       |        |
dollar_25k  |         |         |           |           |            |       |        |
dollar_50k  |         |         |           |           |            |       |        |
dollar_100k |         |         |           |           |            |       |        |
time_1s     |         |         |           |           |            |       |        |
time_5s     |         |         |           |           |            |       |        |
time_60s    |         |         |           |           |            |       |        |
```

### Table 2: Pairwise Tests (Event Bar vs. Matched Time Bar)

```
Comparison       | Metric  | Median Diff | Wilcoxon p (raw) | Wilcoxon p (corrected) | Significant?
-----------------|---------|-------------|------------------|------------------------|-------------
vol_100 vs time  | JB      |             |                  |                        |
vol_100 vs time  | ARCH    |             |                  |                        |
...
```

### Summary Finding

One of:
- **CONFIRMED**: Event-driven bars (type X, param Y) produce significantly more Gaussian, homoskedastic returns than time bars across all 3 primary metrics (corrected p < 0.05). Subordination model fits MES. **Recommendation**: Use bar type X with param Y as primary bar type for all subsequent phases.
- **PARTIALLY CONFIRMED**: Event bars win on N/3 primary metrics. The subordination model is a partial fit. **Recommendation**: Use the best-performing bar type, noting which properties it fails to normalize.
- **REFUTED**: No event-bar config significantly outperforms time bars on any primary metric. MES information arrival is not well-captured by volume/tick/dollar counting. **Recommendation**: Use time bars (simplest) or investigate information-driven bars (§4.6) in a follow-up experiment.

---

## Exit Criteria

- [ ] All 12 bar configs constructed across 20 stratified days
- [ ] Jarque-Bera, ARCH LM, ACF of |return| computed for all configs
- [ ] Ljung-Box and AR R² computed for all configs
- [ ] Bar count CV reported per config
- [ ] Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
- [ ] Effect sizes (Hodges-Lehmann) reported for all pairwise comparisons
- [ ] Quarter-level robustness check completed
- [ ] Primary bar type recommended with empirical justification
- [ ] Results written to `.kit/results/R1_subordination/`
- [ ] Summary entry appended to `.kit/RESEARCH_LOG.md`
