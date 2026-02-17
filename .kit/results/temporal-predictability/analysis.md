# R4: Temporal Predictability — Analysis

**Experiment**: R4-temporal-predictability
**Date**: 2026-02-17
**Finding**: **NO TEMPORAL SIGNAL**

---

## Summary

MES 5-second bar returns are a martingale difference sequence. All 36 Tier 1 autoregressive configurations produce negative R². All 16 Tier 2 temporal augmentation gaps fail the dual threshold (relative >20% of baseline AND Holm-Bonferroni corrected p < 0.05). Temporal-Only features have zero standalone predictive power. This converges with R2's finding (Δ_temporal = −0.006): temporal lookback adds no value regardless of representation — raw book snapshots (R2) or low-dimensional derived features (R4).

**Architecture recommendation**: Drop SSM / temporal encoder. Static current-bar features with GBT is sufficient.

---

## Data

- **Source**: `.kit/results/info-decomposition/features.csv` (reused from R2)
- **Total bars loaded**: 87,970 across 19 trading days
- **Bar type**: `time_5s` (fixed per R1)
- **After warmup exclusion**: ~84,000 bars
- **Cross-validation**: 5-fold expanding-window time-series CV, no shuffling, no day leakage

---

## Tier 1: Pure Return Autoregression

### Table 1: R² (mean ± std across 5 folds)

| Lookback | Model  | return_1              | return_5              | return_20             | return_100            |
|----------|--------|-----------------------|-----------------------|-----------------------|-----------------------|
| AR-10    | Linear | −0.0010 ± 0.0004     | −0.0024 ± 0.0015     | −0.0026 ± 0.0026     | −0.0122 ± 0.0169     |
| AR-10    | Ridge  | −0.0010 ± 0.0004     | −0.0024 ± 0.0015     | −0.0026 ± 0.0026     | −0.0122 ± 0.0169     |
| AR-10    | GBT    | −0.0002 ± 0.0004     | −0.0012 ± 0.0011     | −0.0026 ± 0.0026     | −0.0143 ± 0.0154     |
| AR-50    | Linear | −0.0028 ± 0.0015     | −0.0047 ± 0.0037     | −0.0042 ± 0.0030     | −0.0182 ± 0.0163     |
| AR-50    | Ridge  | −0.0028 ± 0.0015     | −0.0047 ± 0.0037     | −0.0042 ± 0.0030     | −0.0182 ± 0.0163     |
| AR-50    | GBT    | −0.0003 ± 0.0005     | −0.0017 ± 0.0018     | −0.0028 ± 0.0037     | −0.0114 ± 0.0152     |
| AR-100   | Linear | −0.0052 ± 0.0032     | −0.0074 ± 0.0060     | −0.0072 ± 0.0036     | −0.0330 ± 0.0232     |
| AR-100   | Ridge  | −0.0052 ± 0.0031     | −0.0073 ± 0.0060     | −0.0072 ± 0.0035     | −0.0329 ± 0.0230     |
| AR-100   | GBT    | −0.0004 ± 0.0005     | −0.0005 ± 0.0007     | −0.0025 ± 0.0031     | −0.0092 ± 0.0125     |

**All 36 cells are negative.** Every corrected p-value = 1.0. No configuration achieves R² > 0 at any horizon.

### Key Comparisons

**1. AR-10 Linear vs. constant model (H0: R² ≤ 0)**

R² is negative at all four horizons (h1: −0.0010, h5: −0.0024, h20: −0.0026, h100: −0.0122). All p-values = 1.0. The constant (mean) model outperforms linear AR at every horizon. There is no linear serial dependence in MES 5s returns.

**2. GBT vs. Linear (nonlinearity test)**

GBT consistently produces less negative R² than Linear/Ridge, especially at higher lookback depths:

| Lookback | Horizon | Δ(GBT−Linear) | Raw p   | Cohen's d | Corrected p |
|----------|---------|----------------|---------|-----------|-------------|
| AR-10    | h1      | +0.0007        | 0.070   | 1.10      | 0.488       |
| AR-10    | h5      | +0.0012        | 0.049   | 1.25      | 0.488       |
| AR-50    | h1      | +0.0025        | 0.031   | 1.46      | 0.373       |
| AR-100   | h1      | +0.0049        | 0.043   | 1.31      | 0.474       |
| AR-100   | h100    | +0.0238        | 0.109   | 0.92      | 0.488       |

GBT mitigates overfitting (less negative R²) but **no GBT vs. Linear comparison survives Holm-Bonferroni correction** (all corrected p > 0.37). The improvement is in damage mitigation, not signal extraction — GBT's regularization (early stopping, subsampling) prevents the worst overfit, but both models are fitting noise. The key evidence: even GBT's best R² (AR-10 h1: −0.0002) is still negative.

**3. Lookback depth**

More lookback consistently hurts. Linear R² degrades monotonically:

| Horizon | AR-10 Linear | AR-50 Linear | AR-100 Linear |
|---------|-------------|-------------|---------------|
| h1      | −0.0010     | −0.0028     | −0.0052       |
| h5      | −0.0024     | −0.0047     | −0.0074       |
| h100    | −0.0122     | −0.0182     | −0.0330       |

Each doubling of lookback roughly doubles the R² penalty for linear models. This is classic overfitting to noise — additional lagged returns add dimensionality with no compensating signal. GBT dampens this effect (its R² is more stable across lookback depths) but still cannot extract positive R² at any configuration.

### Tier 1 Verdict

**RULE 1 fails.** AR R² ≤ 0 at all horizons. Returns are a martingale difference sequence at all tested horizons (5s to 500s). No temporal structure exists in MES returns at this timescale.

---

## Tier 2: Temporal Feature Augmentation

### Table 2: R² (mean ± std, GBT)

| Config          | return_1              | return_5              | return_20             | return_100            |
|-----------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Static-Book     | +0.0046 ± 0.0071     | −0.0009 ± 0.0013     | −0.0030 ± 0.0036     | −0.0135 ± 0.0147     |
| Static-HC       | +0.0032 ± 0.0054     | −0.0189 ± 0.0362     | −0.0039 ± 0.0019     | −0.0462 ± 0.0485     |
| Book+Temporal   | +0.0025 ± 0.0081     | −0.0005 ± 0.0010     | −0.0027 ± 0.0043     | −0.0131 ± 0.0179     |
| HC+Temporal     | +0.0025 ± 0.0077     | −0.0116 ± 0.0229     | −0.0337 ± 0.0528     | −0.0080 ± 0.0119     |
| Temporal-Only   | +0.0000 ± 0.0002     | −0.0005 ± 0.0008     | −0.0042 ± 0.0055     | −0.0127 ± 0.0186     |

Key observations:
- Only h=1 produces weakly positive R² for any config. This matches R2's finding that signal exists only at the 1-bar horizon.
- Static-Book GBT at h1 (R² = 0.0046) is the best performer, consistent with R2.
- Adding temporal features to book (Book+Temporal) **reduces** h1 R² from 0.0046 to 0.0025.
- Temporal-Only at h1 has R² ≈ 0 (8.8e-7) — effectively zero.
- HC+Temporal shows catastrophic degradation at h20 (R² = −0.034), driven by a single fold with R² = −0.138, indicating severe overfitting.

### Table 3: Information Gaps

| Gap                 | Horizon | Δ_R²    | 95% CI                | Raw p  | Corrected p | Cohen's d | Passes? |
|---------------------|---------|---------|-----------------------|--------|-------------|-----------|---------|
| Δ_temporal_book     | 1       | −0.0021 | [−0.0058, +0.0015]   | 0.183  | 0.733       | −0.72     | **No**  |
| Δ_temporal_book     | 5       | +0.0004 | [−0.0004, +0.0013]   | 0.232  | 0.733       | +0.63     | **No**  |
| Δ_temporal_book     | 20      | +0.0003 | [−0.0015, +0.0020]   | 0.678  | 1.000       | +0.20     | **No**  |
| Δ_temporal_book     | 100     | +0.0004 | [−0.0086, +0.0093]   | 0.916  | 1.000       | +0.05     | **No**  |
| Δ_temporal_hc       | 1       | −0.0007 | [−0.0101, +0.0087]   | 0.840  | 1.000       | −0.10     | **No**  |
| Δ_temporal_hc       | 5       | +0.0073 | [−0.0113, +0.0258]   | 0.313  | 0.938       | +0.49     | **No**  |
| Δ_temporal_hc       | 20      | −0.0298 | [−0.1031, +0.0435]   | 0.625  | 1.000       | −0.50     | **No**  |
| Δ_temporal_hc       | 100     | +0.0383 | [−0.0339, +0.1104]   | 0.063  | 0.250       | +0.66     | **No**  |
| Δ_temporal_only     | 1       | +0.0000 | [−0.0003, +0.0003]   | 0.497  | 1.000       | +0.00     | **No**  |
| Δ_temporal_only     | 5       | −0.0005 | [−0.0017, +0.0006]   | 1.000  | 1.000       | −0.57     | **No**  |
| Δ_temporal_only     | 20      | −0.0042 | [−0.0119, +0.0035]   | 1.000  | 1.000       | −0.68     | **No**  |
| Δ_temporal_only     | 100     | −0.0127 | [−0.0385, +0.0131]   | 1.000  | 1.000       | −0.61     | **No**  |
| Δ_static_comparison | 1       | +0.0014 | [−0.0075, +0.0104]   | 0.678  | 1.000       | +0.20     | **No**  |
| Δ_static_comparison | 5       | +0.0179 | [−0.0329, +0.0688]   | 1.000  | 1.000       | +0.44     | **No**  |
| Δ_static_comparison | 20      | +0.0008 | [−0.0026, +0.0042]   | 0.526  | 1.000       | +0.31     | **No**  |
| Δ_static_comparison | 100     | +0.0328 | [−0.0421, +0.1076]   | 0.291  | 1.000       | +0.54     | **No**  |

**0/16 gaps pass the dual threshold.** No corrected p-value approaches 0.05. The tightest result is Δ_temporal_hc at h100 (corrected p = 0.250), but the 95% CI spans [−0.034, +0.110] — massive uncertainty with no reliable signal.

### Threshold Evaluation Detail

Every Δ_temporal_book gap was evaluated against the dual threshold (relative: Δ > 20% of Static-Book GBT R²; statistical: corrected p < 0.05):

| Horizon | Δ_temporal_book | Baseline R² | 20% Threshold | Passes Relative? | Passes Statistical? | Passes Dual? |
|---------|-----------------|-------------|---------------|------------------|---------------------|--------------|
| h1      | −0.0021         | +0.0046     | +0.0009       | No (negative)    | No (p=0.733)        | **No**       |
| h5      | +0.0004         | −0.0009     | +0.0002       | Yes              | No (p=0.733)        | **No**       |
| h20     | +0.0003         | −0.0030     | +0.0006       | No               | No (p=1.000)        | **No**       |
| h100    | +0.0004         | −0.0135     | +0.0027       | No               | No (p=1.000)        | **No**       |

At h1 (the only horizon with positive baseline R²), adding temporal features **hurts** by −0.0021 R² units — a 45% relative degradation from baseline. At h5, the relative threshold is met trivially (baseline is negative, so any positive delta exceeds 20% of a negative number), but statistical significance is absent.

### Tier 2 Verdict

**RULE 2 fails.** Δ_temporal_book is negative at h1 (the only horizon with positive baseline R²) and negligibly positive at longer horizons. Temporal features do not improve the static feature set.

**RULE 3 fails.** Temporal-Only R² ≈ 0 at h1 (8.8e-7) and negative at all other horizons. No standalone temporal signal exists.

---

## Feature Importance (GBT, Fold 5)

### Table 4a: Book+Temporal, h=1

| Rank | Feature       | Importance | Category |
|------|---------------|------------|----------|
| 1    | book_snap_19  | 0.0368     | static   |
| 2    | book_snap_21  | 0.0330     | static   |
| 3    | book_snap_35  | 0.0315     | static   |
| 4    | book_snap_5   | 0.0304     | static   |
| 5    | book_snap_27  | 0.0293     | static   |
| 6    | book_snap_31  | 0.0283     | static   |
| 7    | book_snap_33  | 0.0269     | static   |
| 8    | book_snap_39  | 0.0267     | static   |
| 9    | lag_return_3  | 0.0267     | temporal |
| 10   | book_snap_25  | 0.0267     | static   |

**Temporal importance fraction**: 47.5%

### Table 4b: Book+Temporal, h=5

| Rank | Feature        | Importance | Category |
|------|----------------|------------|----------|
| 1    | book_snap_33   | 0.0421     | static   |
| 2    | rolling_vol_20 | 0.0357     | temporal |
| 3    | book_snap_25   | 0.0356     | static   |
| 4    | vol_ratio      | 0.0325     | temporal |
| 5    | book_snap_3    | 0.0323     | static   |
| 6    | book_snap_27   | 0.0322     | static   |
| 7    | lag_return_10  | 0.0312     | temporal |
| 8    | rolling_vol_5  | 0.0306     | temporal |
| 9    | momentum_20    | 0.0302     | temporal |
| 10   | momentum_5     | 0.0301     | temporal |

**Temporal importance fraction**: 50.8%

### Table 4c: Book+Temporal, h=20

| Rank | Feature         | Importance | Category |
|------|-----------------|------------|----------|
| 1    | book_snap_27    | 0.0573     | static   |
| 2    | momentum_100    | 0.0482     | temporal |
| 3    | book_snap_33    | 0.0480     | static   |
| 4    | vol_ratio       | 0.0453     | temporal |
| 5    | momentum_20     | 0.0430     | temporal |
| 6    | rolling_vol_100 | 0.0425     | temporal |
| 7    | book_snap_13    | 0.0412     | static   |
| 8    | book_snap_11    | 0.0391     | static   |
| 9    | book_snap_31    | 0.0376     | static   |
| 10   | book_snap_29    | 0.0362     | static   |

**Temporal importance fraction**: 42.6%

### Table 4d: Book+Temporal, h=100

| Rank | Feature         | Importance | Category |
|------|-----------------|------------|----------|
| 1    | momentum_100    | 0.1113     | temporal |
| 2    | rolling_vol_100 | 0.0724     | temporal |
| 3    | book_snap_1     | 0.0530     | static   |
| 4    | book_snap_23    | 0.0489     | static   |
| 5    | rolling_vol_20  | 0.0426     | temporal |
| 6    | book_snap_17    | 0.0410     | static   |
| 7    | book_snap_5     | 0.0402     | static   |
| 8    | book_snap_35    | 0.0398     | static   |
| 9    | book_snap_11    | 0.0396     | static   |
| 10   | book_snap_13    | 0.0391     | static   |

**Temporal importance fraction**: 34.9%

### HC+Temporal Feature Importance Summary

| Horizon | Temporal fraction | Top temporal feature  |
|---------|-------------------|-----------------------|
| h1      | 29.6%             | lag_return_3 (rank 7) |
| h5      | 31.5%             | momentum_5 (rank 7)   |
| h20     | 19.1%             | rolling_vol_20 (rank 4) |
| h100    | 13.3%             | momentum_100 (rank 4) |

### Feature Importance Interpretation

Temporal features receive substantial GBT gain share (30-50% for Book+Temporal, 13-32% for HC+Temporal) **despite providing zero marginal R² improvement**. This is the classic overfitting signature identified in the spec: the model allocates importance to temporal noise, splitting on it frequently, but these splits do not generalize to the test fold. The high importance of momentum_100 at h100 (11.1% of total gain — rank 1) is particularly revealing: the model aggressively fits to 100-bar momentum in-sample, but this feature produces R² = −0.013 out-of-sample.

The pattern across horizons is instructive: temporal importance share rises for Book+Temporal at shorter horizons (47.5% at h1, 50.8% at h5) where the signal-to-noise ratio is highest, but this does not translate to R² improvement. The model treats temporal features as additional noise dimensions to overfit.

---

## Decision Rule Evaluation

### RULE 1 — Pure return AR

**Result: FAILS.** All 36 Tier 1 cells have R² < 0. All corrected p = 1.0. No autoregressive structure detected at any lookback depth (10, 50, 100 bars), any model (Linear, Ridge, GBT), or any horizon (1, 5, 20, 100 bars).

Interpretation: Returns are martingale at horizons 5s-500s. Tier 2 was run as confirmation.

### RULE 2 — Temporal augmentation

**Result: FAILS.** Δ_temporal_book is negative at h1 (−0.0021) and negligible at other horizons. No horizon passes either the relative threshold or the statistical threshold. Best corrected p = 0.733.

Interpretation: Current-bar features are sufficient. No temporal lookback needed. Drop temporal encoder / SSM.

### RULE 3 — Temporal-only signal

**Result: FAILS.** Temporal-Only R² = 8.8e-7 at h1 (not distinguishable from zero; corrected p = 1.0). Negative at all longer horizons (h5: −0.0005, h20: −0.0042, h100: −0.0127).

Interpretation: No temporal signal of any kind. Martingale confirmed.

### RULE 4 — Reconciliation with R2

**Result: CONVERGING EVIDENCE.** R2 found Δ_temporal = −0.006 (raw book concatenation hurt). R4 finds Δ_temporal_book in [−0.002, +0.0004] with low-dimensional derived features. Both approaches agree: temporal lookback does not help.

The two confounds from R2 are now resolved:
1. **Dimensionality curse** (R2 confound): R2 inflated input from 45 to 845 dimensions by appending 20 raw book vectors. R4 uses only 21 temporal features. Result is the same — no improvement.
2. **Representation mismatch** (R2 confound): R2 used raw book snapshots as the temporal representation. R4 uses purpose-built features (lagged returns, rolling volatility, momentum, mean reversion, signed volatility) — exactly the features an SSM or temporal encoder would extract. Result is the same — no improvement.

Interpretation: Strong recommendation to drop SSM from architecture. The temporal information gap is zero regardless of representation.

---

## Cross-Experiment Convergence

| Experiment | Test | Result | Implication |
|------------|------|--------|-------------|
| R1 | Event bars vs. time bars | Time bars are baseline | No temporal resampling benefit |
| R2 | Δ_temporal (raw book concat) | −0.006, p=0.25 | Temporal lookback hurts with raw features |
| R2 | Δ_spatial, Δ_msg | ≤ 0, p ≥ 0.96 | No spatial/message gap either |
| R4 | Tier 1 AR (36 configs) | All R² < 0 | Returns are martingale |
| R4 | Δ_temporal_book (derived) | ≤ +0.0004, p ≥ 0.73 | Temporal lookback neutral with derived features |
| R4 | Temporal-Only | R² ≈ 0 | No standalone temporal signal |

Three independent experiments, using different temporal representations (raw book lags, lagged returns, derived features) and different models (MLP, Linear, Ridge, GBT), converge on the same conclusion: **MES returns at the 5-second bar scale contain no exploitable temporal structure.**

---

## Architecture Recommendation

**Drop the SSM / temporal encoder.** The model architecture should be:

- **Input**: Current-bar features (40 raw book or 62 hand-crafted — both perform comparably)
- **Model**: GBT (XGBoost) on static features
- **Lookback**: None (single-bar prediction)
- **Horizon**: h=1 only (the sole horizon with weakly positive R²; R² ~ 0.005)

This is the simplest viable architecture. R2 showed no spatial or message encoder is justified. R4 shows no temporal encoder is justified. The remaining open question is R3 (book-encoder-bias): whether a CNN on raw book images outperforms flattened features — but this is a feature representation question, not an architecture expansion.

---

## Exit Criteria Checklist

- [x] R2 feature export loaded and validated (bar count matches R2: 87,970)
- [x] Tier 1 AR R² computed for all lookback x model x horizon combinations (36 cells)
- [x] Tier 2 augmentation R² computed for all config x horizon combinations
- [x] All temporal features correctly constructed with no lookahead leakage
- [x] Lookback exclusion applied (no cross-day lookback)
- [x] Paired statistical tests with Holm-Bonferroni correction for all information gaps
- [x] 95% CIs reported for all gaps
- [x] Feature importance extracted from GBT for interpretability
- [x] Decision rules evaluated with explicit outcome
- [x] Reconciliation with R2 Δ_temporal documented
- [x] Results written to `.kit/results/temporal-predictability/`
- [x] `analysis.md` contains explicit decision rule outcome and architecture recommendation
- [x] Summary entry ready for `.kit/RESEARCH_LOG.md`
