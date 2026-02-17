# Phase R4: Temporal Predictability [Research]

**Spec**: TRAJECTORY.md §2.4 (entropy rate theory)
**Depends on**: Phase 1 (bar-construction), Phase 4 (feature-computation), Phase R1 (bar type settled), Phase R2 (temporal gap measured).
**Unlocks**: Phase 6 (synthesis) — provides temporal encoder go/no-go decision.
**GPU budget**: 0 hours (CPU only — no neural nets). **Max runs**: 1 (deterministic).

---

## Motivation

R1 settled the bar type question: time bars are the baseline (subordination refuted).
R2 measured the temporal information gap and found it **negative** (Δ_temporal = −0.006, corrected p = 0.25): concatenating 20 lagged book snapshots to an MLP *hurt* prediction. However, R2's temporal test was confounded by two factors:

1. **Dimensionality curse**: Config (d) inflated input from 45 → 845 dimensions by appending 20 × 40 raw book vectors. With R² < 0.007, the signal-to-noise ratio cannot support 845-dim regression.
2. **Representation mismatch**: R2 tested temporal lookback using raw book snapshots — the crudest possible temporal representation. An SSM or GBT operating on *derived temporal features* (lagged returns, volatility regime indicators, momentum signals) might extract structure that raw book concatenation cannot.

R4 resolves this by testing temporal predictability with **low-dimensional, purpose-built temporal representations** — the question an actual temporal encoder would face.

---

## Hypothesis

MES 5-second bar returns contain exploitable autoregressive structure beyond the current-bar feature set. Specifically:

**H1 (Return AR)**: Lagged returns (up to 100 bars) predict future returns better than a constant (mean) model, after proper cross-validation.

**H2 (Feature AR)**: Augmenting the current-bar feature set with temporal features (lagged returns, rolling volatility, momentum) improves prediction over the static feature set alone.

**H3 (Nonlinearity)**: Temporal structure, if present, is nonlinear — GBT on temporal features outperforms linear AR.

**Null hypothesis**: MES returns at the 5-second scale are a martingale difference sequence. No temporal representation improves on current-bar features. The SSM/temporal encoder adds no value.

---

## Bar Type

**Primary**: `time_5s` (5-second time bars). Fixed per R1 — no bar type comparison needed.

---

## Data

### Source

Reuse the R2 feature export: `.kit/results/info-decomposition/features.csv`
- 87,970 bars across 19 trading days (same days as R1 and R2)
- Columns used: `day`, `is_warmup`, `bar_index`, `return_1`, `return_5`, `return_20`, `return_100`, plus the 62 hand-crafted features (Track A) and 40 raw book features (Track B.1)

### Warmup

Discard `is_warmup = true` bars (first 50 per day). Additionally discard the first `L` bars per day for configs requiring lookback of depth `L` (no cross-day lookback).

### Expected sample size

~84,000 bars (after warmup exclusion), ~83,900 after lookback exclusion (L=100 costs 100 bars × 19 days = 1,900 bars).

---

## Protocol

### Tier 1: Pure Return Autoregression

Test whether lagged returns alone predict future returns.

| Config | Input | Dim | Description |
|--------|-------|-----|-------------|
| AR-10 | return_{t-1}, ..., return_{t-10} | 10 | Short-range linear memory |
| AR-50 | return_{t-1}, ..., return_{t-50} | 50 | Medium-range memory |
| AR-100 | return_{t-1}, ..., return_{t-100} | 100 | Long-range memory |

**Models per config**:
- **Linear**: `sklearn.LinearRegression` (captures linear serial dependence)
- **Ridge**: `sklearn.Ridge` (α via nested 3-fold CV on train set; regularized baseline)
- **GBT**: `xgboost.XGBRegressor` (max_depth=4, n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=20 on validation fold)

**Target horizons**: `return_h` for h ∈ {1, 5, 20, 100} (matching R2 horizons).

**Total Tier 1 cells**: 3 lookback depths × 3 models × 4 horizons = 36 configurations.

### Tier 2: Temporal Feature Augmentation

Test whether temporal features improve the best static feature set from R2.

**Temporal feature construction** (computed from lagged returns only — no lookahead):

| Feature | Formula | Dim |
|---------|---------|-----|
| `lag_returns` | return_{t-1}, ..., return_{t-10} | 10 |
| `rolling_vol_5` | std(return_{t-5..t-1}) | 1 |
| `rolling_vol_20` | std(return_{t-20..t-1}) | 1 |
| `rolling_vol_100` | std(return_{t-100..t-1}) | 1 |
| `vol_ratio` | rolling_vol_5 / rolling_vol_20 | 1 |
| `momentum_5` | sum(return_{t-5..t-1}) | 1 |
| `momentum_20` | sum(return_{t-20..t-1}) | 1 |
| `momentum_100` | sum(return_{t-100..t-1}) | 1 |
| `mean_reversion_20` | return_{t-1} − mean(return_{t-20..t-1}) | 1 |
| `abs_return_lag1` | |return_{t-1}| | 1 |
| `signed_vol` | sign(return_{t-1}) × rolling_vol_5 | 1 |

**Total temporal features**: 21 dimensions. Combined with static features: 21 + 40 = 61 (book + temporal) or 21 + 62 = 83 (hand-crafted + temporal).

| Config | Input | Dim | Description |
|--------|-------|-----|-------------|
| Static-Book | Raw book snapshot (Track B.1) | 40 | R2 Config (b) — best static performer |
| Static-HC | Hand-crafted features (Track A) | 62 | R2 Config (a) |
| Book+Temporal | Raw book + 21 temporal features | 61 | Temporal augmentation of best static |
| HC+Temporal | Hand-crafted + 21 temporal features | 83 | Temporal augmentation of hand-crafted |
| Temporal-Only | 21 temporal features alone | 21 | Temporal signal without book context |

**Model**: GBT only (best candidate from Tier 1 for nonlinear temporal patterns; linear included as sanity check for Static-Book and Book+Temporal only).

**Target horizons**: `return_h` for h ∈ {1, 5, 20, 100}.

**Total Tier 2 cells**: 5 configs × 4 horizons × 1–2 models = 24 configurations.

### Cross-Validation

5-fold expanding-window time-series CV (identical to R2):

| Fold | Train days | Test days |
|------|-----------|-----------|
| 1 | days 1–4 | days 5–8 |
| 2 | days 1–8 | days 9–11 |
| 3 | days 1–11 | days 12–14 |
| 4 | days 1–14 | days 15–17 |
| 5 | days 1–17 | days 18–19 |

No shuffling. No leakage across days.

**Standardization**: Z-score normalize all inputs using training fold statistics only. Apply same transform to test fold.

---

## Analysis

### Tier 1 Analysis: Is there any temporal structure in returns?

**Primary table**: lookback_depth × model × horizon → R² (mean ± std across 5 folds).

**Key comparisons**:
1. **AR-10 Linear vs. constant model**: Is R² > 0 at any horizon? (one-sided t-test, H0: R² ≤ 0)
2. **GBT vs. Linear within each lookback**: Is temporal structure nonlinear? (paired t-test on per-fold R²)
3. **Lookback depth**: Does AR-50 or AR-100 beat AR-10? Or does extra lookback hurt (overfitting)?

### Tier 2 Analysis: Does temporal augmentation help?

**Information gaps** (matching R2 notation):

| Gap | Formula | Question |
|-----|---------|----------|
| Δ_temporal_book | R²(Book+Temporal) − R²(Static-Book) | Do temporal features add value to raw book? |
| Δ_temporal_hc | R²(HC+Temporal) − R²(Static-HC) | Do temporal features add value to hand-crafted? |
| Δ_temporal_only | R²(Temporal-Only) − 0 | Is there standalone temporal signal? |
| Δ_static_comparison | R²(Static-Book) − R²(Static-HC) | Sanity: replicate R2 book vs. hand-crafted comparison |

### Statistical Framework

- **Per gap, per horizon**: Paired t-test on 5-fold R² differences. If Shapiro-Wilk p < 0.05 on differences, use Wilcoxon signed-rank.
- **Multiple comparison correction**: Holm-Bonferroni within each gap family.
  - Tier 1: 3 lookback × 3 models × 4 horizons = 36 tests (correct within model family: 12 tests per model)
  - Tier 2: 4 gaps × 4 horizons = 16 tests (correct within gap family: 4 tests per gap)
- **Threshold policy** (matching R2): Both conditions must hold for temporal encoder to be justified:
  1. **Relative**: Δ_temporal > 20% of baseline R² (where baseline = Static-Book GBT R²)
  2. **Statistical**: Holm-Bonferroni corrected p < 0.05
- Report: point estimate, 95% CI (from t-distribution with df=4), raw p, corrected p.

### Feature Importance (Tier 2 only)

For the GBT model on Book+Temporal and HC+Temporal configs:
- Extract `feature_importances_` (gain-based) from the XGBoost model trained on the largest fold (fold 5).
- Report top-10 features and what fraction of total importance comes from temporal vs. static features.
- This provides interpretability: if temporal features dominate importance but R² doesn't improve, the model is overfitting to temporal noise.

---

## Decision Rules

```
RULE 1 — Pure return AR:
  If AR-10 GBT R² > 0 at h > 1 (corrected p < 0.05):
    → Genuine temporal structure exists in returns.
    → Proceed to Tier 2 with confidence.

  If AR R² ≤ 0 at all h > 1:
    → Returns are martingale at horizons > 5s.
    → Tier 2 is unlikely to help but proceed as confirmation.

RULE 2 — Temporal augmentation:
  If Δ_temporal_book > 20% of baseline R² AND corrected p < 0.05:
    → Temporal features carry exploitable signal beyond current bar.
    → DECISION: Temporal encoder (SSM or GBT with temporal features) justified.
    → Report whether signal is linear (Ridge ≈ GBT) or nonlinear (GBT >> Ridge).

  If Δ_temporal_book ≤ 0 or fails threshold:
    → Current-bar features are sufficient. No temporal lookback needed.
    → DECISION: Drop temporal encoder / SSM. Static feature model is optimal.

RULE 3 — Temporal-only signal:
  If R²(Temporal-Only) > 0 at corrected p < 0.05:
    → Temporal features have standalone predictive power.
    → Even if augmentation doesn't help (Rule 2 fails), the temporal
      signal exists but is redundant with static features.

  If R²(Temporal-Only) ≤ 0:
    → No temporal signal of any kind. Martingale confirmed.

RULE 4 — Reconciliation with R2:
  If R4 finds temporal value (Rule 2 passes) but R2 Δ_temporal was negative:
    → Low-dimensional temporal features succeed where high-dimensional
      book concatenation failed. The SSM should operate on derived features,
      not raw book snapshots.

  If R4 confirms R2 (no temporal value):
    → Converging evidence: temporal encoder adds no value for MES at 5s bars.
    → Strong recommendation to drop SSM from architecture.

"≈" means: pairwise corrected p ≥ 0.05 (no significant difference).
">>" means: pairwise corrected p < 0.05 AND Cohen's d > 0.5.
```

---

## Implementation

```
Language: Python (CPU only — no GPU required)
Entry point: research/R4_temporal_predictability.py

Dependencies:
  - scikit-learn (LinearRegression, Ridge, r2_score)
  - xgboost (XGBRegressor)
  - polars (CSV loading, feature construction)
  - scipy (shapiro, ttest_rel, wilcoxon)
  - numpy

Input:
  - .kit/results/info-decomposition/features.csv (reuse R2 export)

Output:
  - .kit/results/temporal-predictability/
```

---

## Compute Budget

| Item | Estimate |
|------|----------|
| Data loading + temporal feature construction | ~1 min |
| Tier 1: Linear + Ridge (36 configs × 5 folds) | ~2 min (CPU, trivial) |
| Tier 1: GBT (12 configs × 5 folds = 60 GBT fits) | ~10 min (CPU, 200 trees each) |
| Tier 2: GBT (24 configs × 5 folds = 120 GBT fits) | ~20 min (CPU) |
| Tier 2: Linear sanity checks (8 configs × 5 folds) | ~1 min |
| Analysis + statistical tests | ~1 min |
| **Total wall-clock** | **~35 min** |
| **GPU hours** | **0** |
| **Runs** | **1** (deterministic — GBT with fixed seed) |

**Within budget**: 0 GPU-hours, 1 run.

---

## Deliverables

### Table 1: Tier 1 — Pure Return AR

```
Lookback | Model  | return_1       | return_5       | return_20      | return_100
---------|--------|----------------|----------------|----------------|---------------
AR-10    | Linear |                |                |                |
AR-10    | Ridge  |                |                |                |
AR-10    | GBT    |                |                |                |
AR-50    | Linear |                |                |                |
AR-50    | Ridge  |                |                |                |
AR-50    | GBT    |                |                |                |
AR-100   | Linear |                |                |                |
AR-100   | Ridge  |                |                |                |
AR-100   | GBT    |                |                |                |

Each cell: mean_R² ± std_R² (5-fold). Bold if R² > 0 at corrected p < 0.05.
```

### Table 2: Tier 2 — Temporal Feature Augmentation (GBT)

```
Config          | return_1       | return_5       | return_20      | return_100
----------------|----------------|----------------|----------------|---------------
Static-Book     |                |                |                |
Static-HC       |                |                |                |
Book+Temporal   |                |                |                |
HC+Temporal     |                |                |                |
Temporal-Only   |                |                |                |

Each cell: mean_R² ± std_R² (5-fold).
```

### Table 3: Information Gaps (GBT, Tier 2)

```
Gap                  | Horizon | Δ_R²  | 95% CI    | Raw p | Corrected p | Passes?
---------------------|---------|-------|-----------|-------|-------------|--------
Δ_temporal_book      | 1       |       |           |       |             |
Δ_temporal_book      | 5       |       |           |       |             |
Δ_temporal_book      | 20      |       |           |       |             |
Δ_temporal_book      | 100     |       |           |       |             |
Δ_temporal_hc        | 1       |       |           |       |             |
...                  |         |       |           |       |             |
Δ_temporal_only      | 100     |       |           |       |             |
```

### Table 4: Feature Importance (GBT, Fold 5, Book+Temporal)

```
Rank | Feature          | Importance | Category
-----|------------------|------------|----------
1    |                  |            | static / temporal
2    |                  |            |
...  |                  |            |
10   |                  |            |

Temporal feature share: X% of total importance
```

### Summary Finding

One of:
- **TEMPORAL SIGNAL**: Temporal features pass the dual threshold at ≥1 horizon. SSM/temporal encoder justified. Report whether linear or nonlinear.
- **MARGINAL SIGNAL**: Positive R² in Tier 1 but temporal augmentation fails threshold in Tier 2. Temporal structure exists but is redundant with static features.
- **NO TEMPORAL SIGNAL**: R² ≤ 0 across all AR configs and horizons. Returns are a martingale difference sequence at the 5s scale. Drop SSM. Converges with R2 Δ_temporal finding.

---

## Exit Criteria

- [ ] R2 feature export loaded and validated (bar count matches R2: ~87,970)
- [ ] Tier 1 AR R² computed for all lookback × model × horizon combinations (36 cells)
- [ ] Tier 2 augmentation R² computed for all config × horizon combinations
- [ ] All temporal features correctly constructed with no lookahead leakage
- [ ] Lookback exclusion applied (no cross-day lookback)
- [ ] Paired statistical tests with Holm-Bonferroni correction for all information gaps
- [ ] 95% CIs reported for all gaps
- [ ] Feature importance extracted from GBT for interpretability
- [ ] Decision rules evaluated with explicit outcome
- [ ] Reconciliation with R2 Δ_temporal documented
- [ ] Results written to `.kit/results/temporal-predictability/`
- [ ] `analysis.md` contains explicit decision rule outcome and architecture recommendation
- [ ] Summary entry ready for `.kit/RESEARCH_LOG.md`
