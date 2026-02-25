# Experiment: Regime-Stratified Information Decomposition

**Date:** 2026-02-24
**Priority:** P2 — contingent follow-up if P1 experiments (XGBoost tuning, label design) don't close the gap alone
**Parent:** R2 (info-decomposition), E2E CNN Classification (Outcome D)

---

## Background

R2 established that message summaries add no value to book snapshots **in aggregate** across 19 days (Δ_msg = -0.0009, corrected p > 0.05, 0/40 dual threshold passes). However, R2 tested on a single 19-day pilot window and did not stratify by volatility regime.

**The open question:** Does order flow dynamics (message sequences) carry information the static book snapshot misses **specifically during high-volatility regimes** — FOMC days, CPI releases, aggressive directional selling — even though the aggregate signal washes out?

Intuition: In calm markets, the book snapshot is a sufficient statistic because nothing interesting happens between snapshots. In fast markets, the *sequence* of events — large aggressive orders eating levels, cancellation cascades, iceberg detection — may carry predictive information that the static snapshot destroys by averaging.

The full-year dataset (1.16M bars, 251 days) is 13.2× larger than R2's pilot and spans regimes from the calm January start to peak VIX in June to the October rally. This provides the statistical power to test the interaction.

## Hypothesis

The information gap Δ_msg (message summary features vs book-only) is **positive and significant** (>20% relative improvement, Holm-Bonferroni p < 0.05) in the top volatility tercile, while remaining null in the bottom two terciles. In other words: message features interact with volatility regime.

**Null hypothesis:** Δ_msg is indistinguishable from zero across all volatility terciles. Book snapshot remains sufficient in all regimes.

## Independent Variables

| Variable | Levels | Notes |
|----------|--------|-------|
| **Volatility regime** | 3 terciles: Low / Medium / High | Defined by lagged `volatility_50` (see Controls) |
| **Feature configuration** | 2: Book-only (40 cols) vs Book+Message (40 + 33 = 73 cols) | Direct test of Δ_msg |
| **Return horizon** | 2: `fwd_return_1` (5s), `fwd_return_5` (25s) | R2 showed only h=1 had positive R²; include h=5 as falsification check |

**Total comparisons:** 3 terciles × 2 horizons × 1 gap test = 6 primary tests.

**Why not Tier 2 (LSTM/Transformer)?** R2's Tier 2 was severely compromised by 500-event truncation (85.39% of bars clipped; median bar has 1,319 events). The 33-column message summary (Tier 1) captures the full bar via time-decile binning without truncation. Re-running LSTM/Transformer at full sequence length would require GPU and architectural changes — out of scope. If Tier 1 message summaries show regime-dependent signal, a follow-up Tier 2 experiment with increased truncation budget is warranted.

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Model | MLP (hidden=64, ReLU, 1 hidden layer) | Same as R2 Tier 1 best architecture |
| Optimizer | Adam, lr=0.001, epochs=50, early stopping patience=10 | Same as R2 |
| CV scheme | 5-fold expanding-window time-series CV **within each tercile** | Preserves temporal ordering per regime |
| Regime assignment | **Lagged `volatility_50` from previous bar** | Avoids look-ahead bias; regime known at prediction time |
| Tercile boundaries | Computed on training fold only, applied to test fold | Prevents data leakage in boundary definition |
| Data | Full-year Parquet (1.16M bars, 251 days) | 13.2× more data than R2's 19-day pilot |
| Seed | 42 | Reproducibility |
| Warmup exclusion | First 50 bars per day | Same as R2 and full-year export |
| Feature normalization | Z-score per training fold (train stats applied to test) | Same as R2 |

**Regime definition detail:**

```
For each bar i:
  regime_vol[i] = volatility_50[i-1]   # lagged one bar to avoid look-ahead

Per training fold:
  q33 = np.percentile(regime_vol[train_idx], 33.3)
  q67 = np.percentile(regime_vol[train_idx], 66.7)

  low_vol  = regime_vol <= q33
  med_vol  = (regime_vol > q33) & (regime_vol <= q67)
  high_vol = regime_vol > q67
```

This ensures ~387K bars per tercile (well-powered) with no look-ahead and no leakage of tercile boundaries from test to train.

## Metrics (ALL must be reported)

### Primary
- **Δ_msg per tercile** = R²(book+msg) - R²(book-only) at h=1, within each volatility tercile
- **Interaction p-value**: test whether Δ_msg(high_vol) > Δ_msg(low_vol) via paired difference across CV folds

### Secondary
- R²(book-only) per tercile (characterize how predictable each regime is independently)
- R²(book+msg) per tercile
- Number of bars per tercile per fold (verify balanced splits)
- Mean `volatility_50` per tercile (confirm separation is meaningful)
- Tercile boundary values (q33, q67) per fold
- Same metrics at h=5 (falsification: R2 showed h=5 is unpredictable; should remain so)

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|----------------|
| Bars per tercile roughly equal (~387K each) | Yes | Tercile computation bug |
| R²(book-only, h=1, high_vol) > R²(book-only, h=1, low_vol) | Plausible (more variance to explain) | Not necessarily a bug; may be regime-dependent |
| All h=5 R² near zero across all terciles | Yes | Overfitting or data leakage |
| Tercile boundaries stable across folds (CV < 0.2) | Yes | Highly non-stationary volatility |
| Lagged volatility uncorrelated with forward return | Yes | Confound: volatility predicts returns directly |

**Critical confound check:** If `volatility_50` itself predicts `fwd_return_1` (i.e., high vol → directional returns), then any Δ_msg finding could be spurious — the message features may just proxy for volatility. Report the correlation between `volatility_50[i-1]` and `|fwd_return_1[i]|` before interpreting any positive Δ_msg. If correlation > 0.3, add `volatility_50` as a control feature to both configs (book-only gets 41 cols, book+msg gets 74 cols) to isolate the message contribution above and beyond volatility level.

## Baselines

| Baseline | Source | Value | Notes |
|----------|--------|-------|-------|
| R2 Δ_msg (aggregate, 19 days) | R2 analysis.md | -0.0009 | Null result |
| R2 R²(book MLP, h=1) | R2 analysis.md | 0.0067 | Best R2 result |
| R2 R²(book+msg, h=1) | R2 analysis.md | 0.0058 | Message *hurt* |
| Dual threshold | Project convention | >20% relative + Holm-Bonferroni p < 0.05 | Applied per tercile |

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1**: Δ_msg(high_vol, h=1) > 0.0013 (20% of R2 book-only R²=0.0067) AND Holm-Bonferroni corrected p < 0.05
- [ ] **SC-2**: Interaction test significant — Δ_msg(high_vol) > Δ_msg(low_vol), paired t-test p < 0.05
- [ ] **SC-3**: Confound check passes — either vol-return correlation < 0.3, OR Δ_msg remains positive after adding volatility as control feature
- [ ] **SC-4**: h=5 falsification holds — no Δ_msg at h=5 passes dual threshold in any tercile
- [ ] **SC-5**: All sanity checks pass

## Decision Rules

```
OUTCOME A — SC-1 AND SC-2 AND SC-3 pass:
  → CONFIRMED. Message features carry regime-dependent information.
  → Next: Re-open message encoder line for high-vol regime only.
  → Tier 2 follow-up: LSTM/Transformer with increased truncation budget (2000+ events)
     on high-vol tercile. Consider GPU (RunPod) for Tier 2.
  → Integrate regime-conditional message features into GBT pipeline.

OUTCOME B — SC-1 passes but SC-2 or SC-3 fails:
  → INCONCLUSIVE. Message features may help in high vol, but either the
     interaction is not significant or the effect is confounded with volatility.
  → Next: Larger sample (2023 data) or finer regime definition (deciles).

OUTCOME C — SC-1 fails across all terciles:
  → REFUTED. Book snapshot is sufficient in ALL volatility regimes.
  → Message encoder line remains permanently closed.
  → R2's aggregate null result generalizes to regime-stratified analysis.
  → This closes the question definitively on full-year data.

OUTCOME D — SC-4 fails (h=5 shows signal):
  → SUSPECT. h=5 should be null (R2 showed negative R² at longer horizons).
  → Indicates possible data leakage or overfitting in the pipeline.
  → Investigate before trusting any positive h=1 results.
```

## Minimum Viable Experiment

1. **Data loading gate:** Load full-year Parquet. Assert ≥1,160,000 bars, columns `book_snap_0..39`, `msg_summary_0..32`, `volatility_50`, `fwd_return_1`, `fwd_return_5` all present.
2. **Regime assignment gate:** Compute lagged `volatility_50`, assign terciles on full dataset. Assert each tercile has 350K-420K bars. Assert tercile boundaries are not degenerate (q33 ≠ q67).
3. **Confound check gate:** Compute correlation between `volatility_50[i-1]` and `|fwd_return_1[i]|`. If > 0.3, add volatility control to both configs before proceeding.
4. **Single tercile pilot:** Train book-only and book+msg MLPs on high_vol tercile at h=1, 5-fold CV. Assert training completes, R² is finite, fold variance < 0.02.
5. Pass all gates → proceed to full protocol.

## Full Protocol

### Phase 1: Data Preparation

1. Load full-year Parquet from `.kit/results/full-year-export/`.
   - If symlinks (S3 artifact store), run `orchestration-kit/tools/artifact-store hydrate` first.
2. Extract columns: `book_snap_0..39` (40 cols), `msg_summary_0..32` (33 cols), `volatility_50`, `fwd_return_1`, `fwd_return_5`, `day`.
3. Exclude first 50 bars per day (warmup).
4. Compute `regime_vol[i] = volatility_50[i-1]`. Drop first bar of each day (no lagged value).
5. Run confound check: `corr(regime_vol, |fwd_return_1|)`. Log result.

### Phase 2: Per-Tercile Evaluation

For each horizon in [h=1, h=5]:
  For each tercile in [low_vol, med_vol, high_vol]:
    1. Select bars in this tercile.
    2. 5-fold expanding-window time-series CV (split by day within tercile).
    3. For each fold:
       a. Compute tercile boundaries from training fold (for boundary stability check).
       b. Z-score features using training fold stats.
       c. Train MLP(book-only, 40 dims) → record R²_book.
       d. Train MLP(book+msg, 73 dims) → record R²_msg.
       e. Compute Δ_msg = R²_msg - R²_book.
    4. Report: mean(Δ_msg), std(Δ_msg), paired t-test p-value for Δ_msg > 0.
    5. Apply Holm-Bonferroni correction across 6 primary tests.

### Phase 3: Interaction Test

1. Collect Δ_msg per fold for high_vol and low_vol terciles (5 paired observations each).
2. Paired t-test: H1: mean(Δ_msg_high) > mean(Δ_msg_low).
3. Report: t-statistic, p-value, effect size (Cohen's d).

### Phase 4: Reporting

1. Build 3×2 table: tercile × horizon, cells = mean Δ_msg ± std.
2. Build regime characterization table: tercile → mean vol, bar count, R²_book, R²_msg, Δ_msg, p-value.
3. Evaluate all 5 success criteria.
4. Apply decision rule (Outcome A/B/C/D).

## Resource Budget

**Tier:** Quick (≤30 min)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 1160150
model_type: sklearn-mlp
sequential_fits: 60
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 0.5
```

**Breakdown:** 3 terciles × 2 horizons × 2 configs × 5 folds = 60 MLP fits. Each fit: ~387K rows × 73 features × 50 epochs ≈ 5-10 seconds. Total: ~10 min serial, ~3 min with parallelism.

## Abort Criteria

- Confound correlation > 0.5 AND adding volatility control eliminates all positive Δ_msg: ABORT, declare result confounded.
- All 6 primary tests have |Δ_msg| < 0.0005: ABORT early, declare REFUTED — effect is negligible even if significant.
- SC-4 fails (h=5 shows signal in any tercile): STOP, investigate data leakage before trusting h=1 results.

## Confounds to Watch For

1. **Volatility-return correlation:** If high vol predicts larger |returns|, message features may proxy for volatility level. Mitigated by confound check gate and optional volatility control feature.
2. **Non-stationarity of tercile boundaries:** If vol trends strongly (e.g., rising through 2022), terciles may capture time-of-year effects, not volatility effects. Report tercile boundaries per fold to check stability.
3. **Sample size imbalance across folds within tercile:** Expanding-window CV means early folds have fewer training samples. Report fold sizes.
4. **Multiple comparisons:** 6 primary tests (3 terciles × 2 horizons) corrected with Holm-Bonferroni. Report both uncorrected and corrected p-values.
5. **Feature dimensionality:** book+msg (73 dims) vs book-only (40 dims) — the 33 extra features could overfit, especially on shorter folds. Monitor train-test gap. If train R² >> test R², regularize (dropout or L2).

## Deliverables

```
.kit/results/regime-stratified-info-decomposition/
  metrics.json              # All Δ_msg values, p-values, interaction test
  analysis.md               # Full analysis with tables, verdict, SC pass/fail
  confound_check.json       # vol-return correlation, control feature results
  regime_characterization.csv # Per-tercile summary statistics
  per_fold_results.csv      # All 60 fits: tercile, horizon, config, fold, R²
```

### Required Outputs in analysis.md

1. Executive summary (2-3 sentences)
2. Confound check result (vol-return correlation, was control needed?)
3. Regime characterization table (tercile → mean vol, bar count, q33/q67 boundaries)
4. 3×2 Δ_msg table (tercile × horizon) with mean ± std and corrected p-values
5. Interaction test result (t-stat, p, Cohen's d)
6. R² comparison: book-only vs book+msg per tercile (absolute values, not just gaps)
7. h=5 falsification check
8. Tercile boundary stability across folds
9. Explicit pass/fail for each SC-1 through SC-5
10. Decision rule outcome (A/B/C/D)

## Exit Criteria

- [ ] MVE gates passed (data loading, regime assignment, confound check, single-tercile pilot)
- [ ] Per-tercile evaluation complete for all 3 terciles × 2 horizons
- [ ] Interaction test complete
- [ ] Confound check reported
- [ ] h=5 falsification reported
- [ ] All metrics in metrics.json
- [ ] analysis.md written with verdict and SC pass/fail
- [ ] Decision rule applied (Outcome A/B/C/D)
