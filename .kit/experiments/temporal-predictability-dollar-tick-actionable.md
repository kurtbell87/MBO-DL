# Experiment: R4d — Temporal Predictability at Actionable Dollar & Tick Timescales

**Date:** 2026-02-18
**Depends on:** R4c (calibration gap), R4b (dollar_25k sub-actionable reference), R4 (time_5s baseline)
**Prior results location:** `.kit/results/temporal-predictability-completion/analysis.md`

---

## Hypothesis

**H1 (Dollar bars at actionable timescales — null confirmation):** When dollar bar thresholds are empirically calibrated to produce bars with ≥5s median duration, Tier 1 AR R² will be negative (mean across 5 CV folds) at all tested horizons, and temporal augmentation (Δ_temporal_book) will fail the dual threshold (corrected p > 0.05 OR relative Δ < 20% of baseline). Specifically: the monotonic decline in AR R² with increasing dollar threshold (R1: $25k=0.011→$50k=0.0097→$100k=0.0071, all in-sample; R4b CV: $25k=0.0006) continues to zero or negative R² at thresholds producing ≥5s bars.

**H2 (Tick bars at 5-30min timescales — null confirmation):** When tick bar thresholds are calibrated to produce bars at 5min and 15min median duration, temporal features fail the dual threshold at all horizons. R4c's null at tick_100 (~10s) and tick_250 (~25s) extends to longer timescales without reversal.

Both hypotheses predict the null. The prior is overwhelmingly negative: 0/154+ dual threshold passes across R4→R4b→R4c. This experiment exists for **methodological completeness** — R4c's dollar bar calibration was extrapolated, not empirical. The calibration itself is the primary deliverable.

---

## Independent Variables

### Arm 1: Empirical Calibration Sweep (primary deliverable)

Construct bars at each threshold on **1 representative day** (20220103) empirically. Measure actual bar durations from timestamps.

**Dollar thresholds** (6 points, log-spaced):

| Threshold | Volume-math estimate | Purpose |
|-----------|---------------------|---------|
| $1M | ~0.9s (R4c extrapolation) | Anchor to R4c |
| $5M | ~3-5s | Actionable floor |
| $10M | ~5-10s | 5s target |
| $50M | ~25-50s | 30s target |
| $250M | ~2-5min | 5min target |
| $1B | ~10-15min | 15min target |

**Tick thresholds** (4 points, extending R4c):

| Threshold | Estimate | Purpose |
|-----------|----------|---------|
| 500 | ~50s | Bridge from R4c (250) |
| 3,000 | ~5min | 5min target |
| 10,000 | ~15min | 15min target |
| 25,000 | ~40min | Boundary / feasibility check |

**Output:** `calibration_table.json` with empirical per-threshold stats: `{bar_type, threshold, total_bars_day1, median_duration_s, mean_duration_s, p10_duration_s, p90_duration_s, bars_per_session}`.

### Arm 2: Temporal Analysis at 3 Selected Operating Points

From the calibration table, select **3 total** operating points for the R4 protocol — the minimum needed to span the untested range:

1. **Dollar bar nearest 5s median** — the critical gap (R4c never tested dollar bars above $1M)
2. **Dollar bar nearest 5min median** — tests regime-scale on dollar bars
3. **Tick bar nearest 5min median** — extends R4c tick range from 25s to 5min

For each selected threshold, run the standard R4 protocol:
- Feature export via `bar_feature_export --bar-type {dollar|tick} --bar-param {threshold}`
- `R4_temporal_predictability.py` with identical settings to R4/R4b/R4c
- Horizons: h={1, 5, 20, 100} — except restrict to h={1, 5, 20} if <50 bars/session
- Feature sets: Static-Book, Book+Temporal, Temporal-Only (3 configs — drop HC variants to save time; Book configs are what matter for dual threshold)
- Models: GBT only (R4 showed GBT ≥ Ridge ≥ Linear; GBT is the strongest candidate)

**Short-circuit:** If Arm 1 calibration shows no dollar threshold produces ≥5s median bars (contradicting volume math), dollar operating points are skipped. Document and close.

---

## Controls (Fixed across all arms)

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | 19 RTH days (2022) from R1/R2/R4 chain | Consistency |
| Calibration data | Day 20220103 only | Single representative day sufficient for threshold→duration mapping; cross-day variation is small relative to threshold spacing |
| CV protocol | 5-fold expanding-window time-series CV | Identical to R4/R4b/R4c |
| CV fold splits | days 1-4/5-8, 1-8/9-11, 1-11/12-14, 1-14/15-17, 1-17/18-19 | Identical to R4 |
| Standardization | Z-score per training fold | No test-set leakage |
| GBT hyperparameters | max_depth=4, n_estimators=200, lr=0.05, subsample=0.8, colsample_bytree=0.8, early_stopping=20, seed=42 | Identical to R4/R4b/R4c |
| Warmup | First 50 bars/day excluded | Same as R4 |
| Statistical corrections | Holm-Bonferroni within each operating point | Same as R4/R4b |
| Dual threshold | Δ > 20% of baseline R² AND corrected p < 0.05 | Same as R4/R4b/R4c |
| Temporal features | 21 dimensions (lag_return_{1..10}, rolling_vol_{5,20,100}, vol_ratio, momentum_{5,20,100}, mean_reversion_20, abs_return_lag1, signed_vol) | Identical to R4/R4b/R4c |
| Software | Python 3.x, scikit-learn, xgboost, polars, scipy, numpy | Same environment |
| Dollar bar multiplier | 5.0 (MES) — `--bar-param` is raw notional, builder accumulates `price × size × 5` | Consistent with R4b parameterization |

---

## Metrics (ALL must be reported)

### Primary

1. **Empirical calibration table** — threshold → {median_duration_s, bars_per_session, total_bars} for all 10 thresholds. This is the primary deliverable regardless of temporal analysis results.
2. **Tier 1 AR R²** (OOS, 5-fold CV mean ± std) for each of the 3 selected operating points × applicable horizons.
3. **Tier 2 Δ_temporal_book** — R²(Book+Temporal) − R²(Static-Book), with paired t-test, 95% CI, corrected p, for each operating point × horizon.

### Secondary

- **Temporal-Only R²** — standalone temporal predictive power per operating point × horizon.
- **Feature importance** — GBT gain-based, temporal vs. static share (fold 5 only).
- **Signal linearity** — GBT R² vs. Ridge R² comparison, but ONLY if any Tier 1 R² is positive. Do not compute Ridge fits for null-confirming operating points (saves compute).
- **Calibration validation** — empirical duration within 2× of volume-math estimate for each threshold.

### Sanity Checks

- Calibration anchor: empirical $1M duration within ±50% of R4c's extrapolated 0.9s.
- Bar counts: any empirically exported operating point bar count within ±20% of calibration-day extrapolation to 19 days.
- Static-Book R² at h=1: same order of magnitude as time_5s (R4: +0.0046) for operating points at comparable timescales.
- Temporal-Only R² ≤ Book+Temporal R² (information subset property).
- No operating point with <100 total bars across 19 days included in Tier 1/Tier 2 analysis.

---

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| R4 time_5s Tier 1 best (h=1) | `.kit/results/temporal-predictability/metrics.json` | AR-10 GBT: −0.0002 |
| R4 time_5s Δ_temporal_book (h=1) | Same | −0.0021 (p=0.733) |
| R4b dollar_25k Tier 1 (h=1, CV) | `.kit/results/temporal-predictability-event-bars/dollar_25k/metrics.json` | +0.000633 |
| R4b dollar_25k Δ_temporal_book (h=1) | Same | +0.0132 (p=0.250) |
| R4c tick_100 Tier 1 (h=1) | `.kit/results/temporal-predictability-completion/arm2_calibration/tick_100/metrics.json` | −0.000297 |
| R4c tick_250 Tier 1 (h=1) | `.kit/results/temporal-predictability-completion/arm2_calibration/tick_250/metrics.json` | −0.000762 |
| R4c calibration table | `.kit/results/temporal-predictability-completion/arm2_calibration/calibration_table.json` | All extrapolated — R4d replaces this |
| R1 dollar AR R² trend (in-sample) | `.kit/results/subordination-test/metrics.json` | $25k=0.011, $50k=0.0097, $100k=0.0071 |

No baselines need reproduction — all from prior experiments in this chain.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC1:** Empirical calibration table produced for all 10 thresholds with measured (not extrapolated) durations from actual bar construction on day 20220103.
- [ ] **SC2:** At least 2 dollar thresholds produce bars with empirical median duration ≥5s, confirming dollar bars ARE achievable at actionable timescales. (If this fails, the R4c "sub-actionable" conclusion was correct and Arm 2 dollar points are skipped.)
- [ ] **SC3:** Full R4 protocol completed for all 3 selected operating points (or fewer if calibration precludes selection).
- [ ] **SC4:** 0/N dual threshold passes across all operating points × horizons (where N = total tests run). Any pass is a genuine surprise requiring characterization.
- [ ] **SC5:** Timescale response data produced: AR R² (best config) at h=1 vs. bar duration, unifying R4c tick results with R4d new points.
- [ ] **SC6:** Results reproducible across 5 CV folds — all folds agree on sign of Tier 1 AR R² (i.e., p(R²>0) reported per operating point).

---

## Minimum Viable Experiment

**Run Arm 1 calibration only.** This takes ~30-60 minutes (10 thresholds × 1 day of bar construction) and is independently valuable:

1. Build bars at each of the 10 thresholds on day 20220103 via `bar_feature_export`.
2. Parse output CSVs for timestamp columns. Compute median/mean/p10/p90 duration and bar count.
3. Produce `calibration_table.json`.
4. Validate: $1M empirical duration within ±50% of R4c's 0.9s extrapolation.

**MVE decision gate:**
- If ≥2 dollar thresholds produce ≥5s median duration → proceed to Arm 2 (select 3 operating points, run R4 protocol).
- If 0-1 dollar thresholds reach ≥5s → R4c's "sub-actionable" conclusion is largely correct. Run Arm 2 with 1 dollar point (the longest-duration threshold) + 1 tick point (5min). Reduced scope.
- If NO threshold of any type produces ≥5s → calibration table is the final deliverable. Document that R4c was correct. Close R4d.

---

## Full Protocol

### Phase 1: Empirical Calibration (~1 hr)

1. For each of the 10 thresholds (6 dollar, 4 tick):
   ```bash
   ./build/bar_feature_export --bar-type {dollar|tick} --bar-param {threshold} \
     --output .kit/results/R4d/calibration/{type}_{threshold}_day1.csv
   ```
   Run on day 20220103 only. Parse the CSV to extract `bar_ts` (bar timestamp) and `bar_index` columns.

2. For each threshold, compute:
   - Total bars on day 1
   - Inter-bar durations: `bar_ts[i+1] - bar_ts[i]` for all consecutive bars
   - Median, mean, p10, p90 of durations (in seconds)
   - Extrapolated bars per session (use day 1 count directly — cross-day variation assessed in confounds)

3. Save `calibration_table.json`. Compare empirical $1M to R4c's extrapolated 0.9s.

4. **Gate: Select operating points** per MVE decision gate above.

### Phase 2: Feature Export for Selected Operating Points (~2-6 hr)

5. For each selected operating point (up to 3), run full 19-day feature export:
   ```bash
   ./build/bar_feature_export --bar-type {dollar|tick} --bar-param {threshold} \
     --output .kit/results/R4d/{type}_{threshold}/features.csv
   ```

6. Validate bar counts against calibration-day extrapolation (within ±20%).

7. For operating points with <50 bars/session: restrict horizon list to h={1, 5, 20}. For <26 bars/session: restrict to h={1, 5}. Document restrictions.

### Phase 3: Temporal Analysis (~1-2 hr)

8. For each selected operating point, run R4 protocol:
   ```bash
   python research/R4_temporal_predictability.py \
     --input-csv .kit/results/R4d/{type}_{threshold}/features.csv \
     --output-dir .kit/results/R4d/{type}_{threshold} \
     --bar-label {type}_{threshold}
   ```

9. Extract per operating point:
   - Tier 1: AR R² (best across AR-{10,50,100} lookbacks) per horizon
   - Tier 2: Δ_temporal_book per horizon, with paired test + Holm-Bonferroni
   - Temporal-Only R² per horizon
   - Feature importance (fold 5)

10. Evaluate dual threshold for each operating point × horizon.

### Phase 4: Cross-Timescale Synthesis (~30 min)

11. Produce timescale response data combining:
    - R4: time_5s at h=1 (AR R²=−0.0002)
    - R4c: tick_50 (~5s), tick_100 (~10s), tick_250 (~25s) — all negative
    - R4d: new operating points from Arm 2

    Table: `{bar_type, threshold, median_duration_s, tier1_ar_r2_h1, delta_temporal_book_h1, dual_pass}`

12. Write `.kit/results/R4d/analysis.md`:
    - Calibration table (standalone deliverable)
    - Per-operating-point results
    - Cross-timescale AR R² trend
    - Decision framework evaluation
    - Final recommendation

13. Prepare summary entry for `.kit/RESEARCH_LOG.md`.

---

## Resource Budget

**Tier: Quick**

This is a gate check with a strong null prior (0/154+ passes). The calibration sweep is the primary deliverable; the temporal analysis is confirmatory.

| Phase | Est. Wall-Clock |
|-------|----------------|
| Phase 1: Calibration (10 thresholds × 1 day) | 30-60 min |
| Phase 2: Feature export (up to 3 operating points × 19 days) | 3-6 hr |
| Phase 3: Temporal analysis (3 operating points, GBT only, 3 feature sets) | 1-2 hr |
| Phase 4: Cross-timescale synthesis | 30 min |
| **Total** | **5-9 hr** |

- **Max GPU-hours:** 0 (CPU only)
- **Max wall-clock time:** 10 hours
- **Max training runs:** 3 (one per operating point)
- **Max seeds per configuration:** 1 (deterministic GBT, seed=42, 5-fold CV provides variance estimates)

**Compute savings vs. original draft:** Reduced from ~17 hr to ~5-9 hr by: (a) 3 operating points instead of 8, (b) GBT-only instead of GBT+Linear+Ridge, (c) 3 feature configs instead of 5 (drop HC variants), (d) single calibration day instead of all 19.

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 500000
model_type: xgboost
sequential_fits: 45
parallelizable: false
memory_gb: 8
gpu_type: none
estimated_wall_hours: 8
```

---

## Abort Criteria

- **Calibration $1M anchor fails sanity:** Empirical $1M duration deviates >5× from R4c's 0.9s → investigate bar construction. Likely a multiplier misconfiguration.
- **No threshold produces ≥5s bars:** All dollar and tick thresholds at tested ranges produce sub-5s bars → document calibration table, confirm R4c conclusion, close R4d without Arm 2.
- **Any operating point has <100 total bars across 19 days:** Skip that operating point. Document as insufficient sample.
- **Feature export for any threshold takes >3 hours:** The budget allows ~2 hr per export. If one exceeds 3 hr, kill it and subsample to 10 representative days.
- **Any GBT fit exceeds 10 minutes:** Subsample to 50k rows for that operating point.
- **Wall clock exceeds 10 hours:** Stop remaining exports/analyses. Report completed operating points.
- **Any Tier 2 Δ_temporal_book has raw p < 0.01:** Do NOT abort. This is unexpected but possible. Complete that operating point fully. Add Ridge model for linearity check. Report as potential signal requiring replication.
- **Tier 1 AR R² > +0.01 at any operating point:** Unexpected strong signal. Pause, verify forward return computation (no lookahead), check bar construction. If verified, expand to full R4 protocol (all models, all feature configs) for that operating point.

---

## Confounds to Watch For

1. **Single-day calibration bias.** Day 20220103 may not be representative (it's the first RTH day of 2022 — could have low volume from holiday effect). If calibration durations are suspiciously long (suggesting low volume), cross-check bar count against R1's 19-day average. If >2× discrepancy, run calibration on a second day (20220301, mid-quarter) and use the average.

2. **Dollar bar multiplier confusion.** `DollarBarBuilder` accumulates `price × size × multiplier` (multiplier=5.0 for MES). The `--bar-param` threshold is compared against this accumulated value. If thresholds are specified in "contract-notional" instead of "multiplied-notional," all durations will be 5× shorter than expected. Verify by checking that the $1M calibration matches R4c's ~0.9s anchor.

3. **Sparse-bar horizon truncation.** At 5min+ timescales, bars per session are ~50-80. With h=100, most bars lose their forward return to end-of-session NaN. The spec restricts horizons to h={1,5,20} for sparse operating points. If the RUN agent runs h=100 anyway and finds "signal," it's likely an artifact of extreme data truncation (only the first few bars of each session survive).

4. **Extrapolation from 1-day to 19-day bar counts.** Volume varies across days (FOMC, OpEx, quiet days). A single calibration day undercounts volatility in bar-count estimates. This is acceptable for threshold selection (we need order-of-magnitude accuracy, not precision) but must be documented.

5. **Declining AR R² with threshold (expected, not a confound).** If R² decreases monotonically with dollar threshold size, this confirms the R1 trend and is the EXPECTED null result. It would be a confound only if someone misinterpreted "R² decreases" as "something is wrong." The experiment is designed for null confirmation — declining R² IS the hypothesis.

6. **Multiple testing across operating points.** 3 operating points × up to 4 horizons × 1 dual threshold test = up to 12 tests. Holm-Bonferroni is applied within each operating point. Across operating points, we expect 0 passes. If exactly 1 marginal pass occurs, treat with extreme skepticism — the family-wise error rate across the full experiment is ~0.05 × 12 ≈ 0.6 if uncorrected.

---

## Decision Framework (post-experiment)

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| All operating points fail Rule 2 (expected) | Temporal signal absent across 5s-15min on event-driven bars. Combined with R4c: 5s-83min on time bars. R4 line closed permanently with maximum confidence. | Proceed to CNN+GBT build. |
| Rule 2 passes at ≥5min on dollar bars | Regime-scale temporal structure specific to dollar-volume sampling. Most interesting possible outcome. | **Architecture review.** Design targeted temporal encoder for dollar bars at that timescale. High priority follow-up. |
| Rule 2 passes at 5s on dollar bars only | Dollar bars at 5s reveal signal time bars miss. | Characterize (linear vs. nonlinear). If linear, add AR features. If nonlinear, consider lightweight SSM. |
| Calibration confirms all dollar bars sub-actionable (<5s) | R4c was correct despite extrapolation. No dollar operating points tested. | Close dollar bar temporal question permanently. Calibration table documents this. |

---

## Output Structure

```
.kit/results/R4d/
├── calibration/
│   ├── calibration_table.json          # empirical: all 10 thresholds
│   ├── dollar_1M_day1.csv              # anchor validation
│   └── selected_thresholds.json        # operating points + rationale
├── {type}_{threshold}/                 # one dir per selected operating point
│   ├── features.csv                    # 19-day C++ export
│   ├── metrics.json                    # R4 protocol output
│   └── analysis.md                     # per-operating-point results
└── analysis.md                         # cross-timescale synthesis + decision
```

---

## Exit Criteria

- [ ] Phase 1: Empirical calibration table for all 10 thresholds (6 dollar, 4 tick) with measured durations
- [ ] Phase 1: $1M anchor validated against R4c extrapolation (within ±50%)
- [ ] Phase 1: Operating points selected with documented rationale (or documented that none exist)
- [ ] Phase 2: Feature export completed for all selected operating points
- [ ] Phase 2: Bar counts validated against calibration extrapolation (within ±20%)
- [ ] Phase 3: Tier 1 AR R² and Tier 2 Δ_temporal_book reported for each operating point × horizon
- [ ] Phase 3: Dual threshold evaluated — pass/fail for each operating point × horizon
- [ ] Phase 4: Timescale response data unifying R4c tick results with R4d new points
- [ ] Phase 4: Cross-timescale analysis.md with decision framework evaluation
- [ ] Summary entry ready for `.kit/RESEARCH_LOG.md`
- [ ] `CLAUDE.md` Current State updated
