# Experiment: R4c — Temporal Predictability Completion

**Date:** 2026-02-18
**Depends on:** R4 (time_5s temporal analysis), R4b (volume_100, dollar_25k temporal analysis), R1 (subordination/bar stats)
**Unlocks:** Final closure of temporal encoder investigation. Enables full commitment to CNN+GBT static architecture.
**GPU budget:** 0 hours (CPU only). **Max runs:** 10.

---

## Hypothesis

Three falsifiable null hypotheses, one per arm:

**H1 (Extended horizons):** Temporal features fail the dual threshold (Rule 2) at horizons h=200, h=500, h=1000 on time_5s bars (~17min, ~42min, ~83min). The monotonic R² degradation observed from h=1 to h=100 in R4 continues at longer horizons. Specifically: all Tier 2 augmentation deltas (Δ_temporal_book) will have corrected p > 0.05 at every extended horizon, and all Tier 1 AR R² values will be negative.

**H2 (Tick bars):** Tick_50 bar returns show no more temporal structure than time_5s or volume_100. Temporal features fail Rule 2 at all horizons {1, 5, 20, 100}. Prior: R1 tick_50 AR R²=0.00034 — identical to time_5s (0.00034).

**H3 (Event bars at actionable timescales):** When dollar and tick bar thresholds are calibrated to produce ≥5s median bar duration, temporal features still fail Rule 2. The R4b finding that dollar_25k temporal signal is redundant with static features is not an artifact of testing at sub-second timescales — it holds at execution-relevant timescales.

All three hypotheses predict the null. We design to efficiently confirm or reject them.

---

## Independent Variables

### Arm 3: Extended Horizons on time_5s (FIRST — highest info value, lowest cost)

- **Bar type:** time_5s (existing data, no export needed)
- **New horizons:** {200, 500, 1000} bars (~17min, ~42min, ~83min)
- **Forward returns:** Computed in Python from the `close` column: `fwd_return_h = close[i+h] / close[i] - 1`. No C++ changes.
- **Feature sets:** Static-Book (40 dim), Book+Temporal (61 dim), Temporal-Only (21 dim)
- **Models:** GBT only (R4 showed Linear/Ridge ≤ GBT everywhere; GBT is the strongest candidate)
- **Existing h={1,5,20,100} results from R4 serve as baseline — do NOT re-run**

### Arm 1: Tick Bars (SECOND — conditional on Arm 3 not passing Rule 2)

- **Bar type:** tick_50 (50 trades/bar). Feature export via `bar_feature_export --bar-type tick --bar-param 50`
- **Horizons:** {1, 5, 20, 100} bars
- **Feature sets:** Same 5 configs as R4/R4b (Static-Book, Static-HC, Book+Temporal, HC+Temporal, Temporal-Only)
- **Models:** GBT + Linear (matching R4b)

### Arm 2: Event Bar Threshold Calibration + Temporal Analysis at Actionable Timescales (THIRD — conditional)

**Step 2a — Calibration sweep (cheap, no feature export):**

Construct bars at each threshold across 19 RTH days. Report only timing statistics:
```
threshold → {median_duration_s, mean_duration_s, p10_duration_s, p90_duration_s, bars_per_day}
```

Sweep ranges:
- Dollar: $50k, $100k, $250k, $500k, $1M (5 thresholds — skip $25k, already tested in R4b)
- Tick: 100, 250, 500, 1000, 2500 (5 thresholds — skip 50, already in Arm 1)

Implementation: Run `bar_feature_export` at each threshold on a single representative day (20220103) with `--output /dev/null`-equivalent to get bar counts and duration stats. If the tool doesn't support stats-only mode, parse the CSV metadata columns (bar_index, bar_ts) for just the first day and extrapolate. The full 19-day sweep is unnecessary for calibration — cross-day variation is small for threshold selection.

**Step 2b — Select actionable thresholds:**

Pick up to 2 thresholds per bar type (dollar, tick) whose median bar duration falls closest to 5s and 30s. If no threshold produces ≥5s bars for a bar type, that bar type is entirely sub-actionable — skip it in 2c and document.

**Step 2c — Temporal analysis at selected thresholds (up to 4 configs):**

Run the full R4 protocol (Tier 1 + Tier 2) for each selected threshold. Same feature pipeline, same CV, same statistical framework.

---

## Controls (Fixed across all arms)

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | 19 RTH days from R1/R2/R4/R4b | Consistency with prior experiments |
| CV protocol | 5-fold expanding-window time-series CV | Same as R4/R4b; no shuffling, no day leakage |
| CV fold splits | days 1-4/5-8, 1-8/9-11, 1-11/12-14, 1-14/15-17, 1-17/18-19 | Identical to R4 |
| Standardization | Z-score per training fold | No test-set leakage |
| GBT hyperparameters | max_depth=4, n_estimators=200, lr=0.05, subsample=0.8, colsample_bytree=0.8, early_stopping=20, seed=42 | Identical to R4/R4b |
| Warmup | First 50 bars/day excluded + lookback depth bars | Same as R4 |
| Statistical corrections | Holm-Bonferroni within each gap family | Same as R4/R4b |
| Dual threshold | Δ > 20% of baseline R² AND corrected p < 0.05 | Same as R4/R4b |
| Temporal features | 21 dimensions (lag_return_{1..10}, rolling_vol_{5,20,100}, vol_ratio, momentum_{5,20,100}, mean_reversion_20, abs_return_lag1, signed_vol) | Identical to R4/R4b |
| Software | Python 3.x, scikit-learn, xgboost, polars, scipy, numpy | Same environment as R4/R4b |

---

## Metrics (ALL must be reported)

### Primary

1. **Tier 1 AR R²** (out-of-sample, cross-validated): per arm × horizon × model. Mean ± std across 5 folds.
2. **Tier 2 Δ_temporal_book**: R²(Book+Temporal) − R²(Static-Book), per arm × horizon. With paired test, 95% CI, corrected p.

### Secondary

- **Temporal-Only R²**: Standalone temporal predictive power per arm × horizon.
- **Feature importance**: GBT gain-based importance, temporal vs. static share (fold 5).
- **Signal linearity**: GBT R² vs. Linear R² for any positive-R² configs.
- **Data loss at extended horizons**: Number of bars lost to end-of-session truncation per horizon.

### Sanity Checks

- R4 time_5s h={1,5,20,100} R² values reproducible within ±0.001 if re-loaded (not re-run).
- Arm 1 tick_50 bar count within ±10% of R1's 88,521 (19 days).
- Forward returns computed in Python (Arm 3) match C++ fwd_return_{1,5,20,100} within floating-point tolerance where both exist.
- Temporal-Only R² ≤ Book+Temporal R² everywhere (information subset).
- Extended horizon R² monotonically worsens (or stays flat) relative to h=100.

---

## Baselines

| Baseline | Source | Value |
|----------|--------|-------|
| R4 time_5s Tier 1 best (h=1) | `.kit/results/temporal-predictability/metrics.json` | AR-10 GBT: −0.0002 |
| R4 time_5s Tier 2 Static-Book (h=1) | Same | +0.0046 |
| R4 time_5s Δ_temporal_book (h=1) | Same | −0.0021 (p=0.733) |
| R4 time_5s Tier 1 best (h=100) | Same | AR-100 GBT: −0.0092 |
| R4b dollar_25k Tier 1 best (h=1) | `.kit/results/temporal-predictability-event-bars/dollar_25k/metrics.json` | Linear: +0.000633 |
| R4b dollar_25k Δ_temporal_book (h=1) | Same | +0.0132 (p=0.250) |
| R1 tick_50 AR R² | `.kit/results/subordination-test/metrics.json` | 0.00034 |
| R4 Tier 1 R² trend (h=1→100) | R4 analysis | −0.0002 → −0.0092 (monotonic degradation) |

No baselines need reproduction — all come from prior experiments in this chain.

---

## Success Criteria (immutable once RUN begins)

- [ ] **SC1:** Arm 3 all extended horizons produce Tier 1 AR R² < 0 (mean across folds), confirming monotonic degradation trend from R4.
- [ ] **SC2:** Arm 3 Δ_temporal_book has corrected p > 0.05 at every extended horizon {200, 500, 1000}.
- [ ] **SC3:** Arm 1 tick_50 produces 0/16 Tier 2 dual threshold passes (same as time_5s and volume_100 in R4/R4b).
- [ ] **SC4:** Arm 2 — if actionable-timescale thresholds exist (≥5s median duration), Δ_temporal_book fails dual threshold at those operating points.
- [ ] **SC5:** No regression on R4 baselines — reloaded h={1,5,20,100} Static-Book R² within ±0.002 of R4 values.
- [ ] **SC6:** Results reproducible across 5 CV folds with fold-level variance reported for all primary metrics.

Note: SC4 applies only if Arm 2a calibration identifies viable thresholds. If no threshold produces ≥5s bars, SC4 is vacuously satisfied and documented as such.

The experiment **succeeds** (produces a clear decision) if all applicable SCs pass. A definitive null across all arms is the expected and useful outcome — it permanently closes the temporal encoder investigation.

---

## Minimum Viable Experiment

**Run Arm 3 only.** This is the highest-information, lowest-cost arm:

1. Load existing time_5s CSV (`.kit/results/info-decomposition/features.csv`).
2. Compute fwd_return_{200, 500, 1000} in Python from the `close` column.
3. Validate: compare Python-computed fwd_return_{1,5,20,100} against CSV values to confirm correctness.
4. Run Tier 1 (AR-10 GBT only — single lookback, single model, 3 horizons = 3 configs × 5 folds = 15 GBT fits).
5. Run Tier 2 (Static-Book, Book+Temporal, Temporal-Only — 3 configs × 3 horizons = 9 cells × 5 folds = 45 GBT fits).
6. If all Tier 1 R² < 0 and all Δ_temporal_book fail dual threshold: Arm 3 hypothesis confirmed.

**If Arm 3 MVE passes:** Proceed to Arms 1 and 2 per the full protocol.
**If Arm 3 MVE shows unexpected signal (any Tier 2 Δ > 0 with raw p < 0.10):** Expand Arm 3 to full protocol (AR-10/50/100, all models, all 5 feature configs) before proceeding.

---

## Full Protocol

### Phase 1: Arm 3 — Extended Horizons (~1.5 hr)

1. **Load data.** Read `.kit/results/info-decomposition/features.csv` via polars. Verify 87,970 rows.

2. **Compute extended forward returns in Python.**
   ```python
   # For each day independently (no cross-day computation):
   for h in [200, 500, 1000]:
       # Within each day's bars (sorted by bar_index):
       # fwd_return_h[i] = close[i+h] / close[i] - 1
       # Bars where i+h exceeds the day's bar count get NaN
   ```
   **Validation:** Also compute fwd_return_{1,5,20,100} in Python and compare to CSV values. Max absolute difference must be < 1e-6. If validation fails, stop and investigate — the forward return computation is wrong.

3. **Document data loss.** Report per-horizon how many bars are lost to end-of-session truncation:
   - h=200: last 200 bars/day (~4.3% of ~4,600 RTH bars/day)
   - h=500: last 500 bars/day (~10.9%)
   - h=1000: last 1000 bars/day (~21.7%)
   Total training rows after warmup + lookback + truncation for each horizon.

4. **Run Tier 1** (AR-10 GBT on h={200, 500, 1000}). 3 configs × 5 folds = 15 fits. ~5 min.

5. **Run Tier 2** (Static-Book GBT, Book+Temporal GBT, Temporal-Only GBT on h={200, 500, 1000}). 9 cells × 5 folds = 45 fits. ~15 min.

6. **Compute Δ_temporal_book** for each extended horizon. Paired t-test on 5-fold differences. Holm-Bonferroni across the 3 horizons. Report point estimate, 95% CI, raw p, corrected p.

7. **Evaluate SC1 and SC2.** If both pass, Arm 3 confirms null.

8. **Gate decision:**
   - If SC1 and SC2 pass → proceed to Arms 1 and 2.
   - If either fails (unexpected signal) → expand Arm 3 to full protocol before proceeding.

### Phase 2: Arm 1 — Tick Bars (~3 hr)

9. **Export tick_50 features.**
   ```bash
   ./build/bar_feature_export --bar-type tick --bar-param 50 \
     --output .kit/results/R4c/arm1_tick_bars/features.csv
   ```
   ~2.5 hr. Verify bar count within ±10% of R1's 88,521.

10. **Run R4 protocol on tick_50.** Same as R4b: Tier 1 (36 configs) + Tier 2 (24 configs). ~40 min.
    ```bash
    python research/R4_temporal_predictability.py \
      --input-csv .kit/results/R4c/arm1_tick_bars/features.csv \
      --output-dir .kit/results/R4c/arm1_tick_bars \
      --bar-label tick_50
    ```

11. **Evaluate SC3.** If 0/16 dual threshold passes → tick bars confirm null.

### Phase 3: Arm 2 — Event Bar Calibration + Actionable Timescales (~conditional, 2-10 hr)

12. **Arm 2a: Calibration sweep.**
    Run `bar_feature_export` for each threshold on day 20220103 only. Parse CSV output to extract bar timestamps, compute duration statistics. ~30 min total (10 thresholds × ~3 min each on one day).

    Dollar thresholds: $50k, $100k, $250k, $500k, $1M
    Tick thresholds: 100, 250, 500, 1000, 2500

    Output: `calibration_table.csv` with columns: bar_type, threshold, median_duration_s, mean_duration_s, p10_duration_s, p90_duration_s, bars_per_day.

13. **Arm 2b: Select actionable thresholds.**
    From calibration table, pick up to 2 thresholds per bar type closest to 5s and 30s median duration. If no threshold produces ≥5s median, that bar type is entirely sub-actionable — document and skip.

14. **Arm 2c: Temporal analysis at selected thresholds.** (If any selected)
    For each selected threshold:
    a. Export features for all 19 days: `bar_feature_export --bar-type <type> --bar-param <threshold>` (~1-2.5 hr each)
    b. Run R4 protocol: `python research/R4_temporal_predictability.py` (~40 min each)

15. **Evaluate SC4.** If all selected thresholds fail dual threshold → actionable-timescale event bars confirm null.

### Phase 4: Cross-Arm Analysis

16. **Write `.kit/results/R4c/analysis.md`:**
    - Cross-arm summary table: arm × horizon → best Tier 1 R², Δ_temporal_book, dual threshold pass/fail
    - Decision framework evaluation per the table in this spec
    - Comparison to R4/R4b baselines
    - If any arm passes Rule 2: characterize (linear vs. nonlinear, actionable vs. sub-actionable)
    - Final architectural recommendation

17. **Update state files:** `.kit/RESEARCH_LOG.md`, `CLAUDE.md` Current State section.

---

## Resource Budget

**Tier: Standard** (multiple arms, up to ~4 hr GPU-equivalent compute, though all CPU)

| Phase | Component | Est. Wall-Clock | Cumulative |
|-------|-----------|----------------|------------|
| Arm 3 | Load data + compute fwd returns | 5 min | 5 min |
| Arm 3 | Tier 1 + Tier 2 (60 GBT fits) | 20 min | 25 min |
| Arm 3 | Analysis + gate decision | 5 min | 30 min |
| Arm 1 | Feature export (tick_50, 19 days) | 2.5 hr | 3 hr |
| Arm 1 | Tier 1 + Tier 2 (300 GBT fits) | 40 min | 3.7 hr |
| Arm 2a | Calibration (10 thresholds × 1 day) | 30 min | 4.2 hr |
| Arm 2b | Threshold selection | 5 min | 4.3 hr |
| Arm 2c | Feature export (up to 4 thresholds × 19 days) | 4-10 hr | 8-14 hr |
| Arm 2c | R4 analysis (up to 4 thresholds) | 1-3 hr | 9-17 hr |
| Cross-arm | Analysis + state updates | 30 min | 9.5-17.5 hr |

- **Max GPU-hours:** 0
- **Max wall-clock time:** 18 hours (full protocol). Arm 3 alone: 30 min.
- **Max training runs:** 10 (1 per arm/threshold configuration)
- **Max seeds per configuration:** 1 (deterministic — GBT with fixed seed=42)

**Compute triage:** If wall-clock budget is constrained to 4 hours, run Arm 3 + Arm 1 + Arm 2a (calibration only). This covers 90% of the information value. Arm 2c (feature export + analysis at actionable thresholds) is the most expensive and least likely to yield signal — defer if needed.

---

## Abort Criteria

- **Arm 3 extended horizon Tier 1 R² > +0.01 at any horizon:** Unexpected strong signal. Stop, verify data pipeline, check for lookahead leakage in forward return computation. This has never been observed in any R4/R4b config and would indicate a bug.
- **Feature export fails for tick bars:** Document error, skip Arm 1, proceed with Arm 2.
- **Arm 2a calibration shows no threshold produces ≥5s median bars for either dollar or tick:** Skip Arm 2c entirely. Event bars are sub-actionable by construction. Document the calibration table as a standalone deliverable.
- **Any single GBT fit exceeds 10 minutes:** Subsample training data to 100k rows for that configuration. Document the subsampling.
- **Total wall-clock exceeds 24 hours:** Stop remaining arms. Report partial results with explicit documentation of which arms completed.
- **Arm 3 shows raw p < 0.05 on any Δ_temporal_book:** Do NOT abort. This is unexpected but not impossible. Expand Arm 3 to full protocol (3 lookback × 3 models × 5 feature configs) to verify. Only flag as genuine signal if it survives the expanded protocol AND Holm-Bonferroni correction.

---

## Confounds to Watch For

1. **Forward return computation error (Arm 3).** Computing fwd_return in Python from close prices could diverge from C++ computation if the C++ uses a different formula (log return vs. simple return, different price field). The validation step (protocol step 2) catches this. If validation fails, use the C++ forward returns up to h=100 and investigate the discrepancy.

2. **Data loss at long horizons (Arm 3).** h=1000 loses ~22% of each day's bars. This biases the sample toward early-session bars. If early-session returns have different temporal structure than late-session (e.g., opening auction effects), this could create a spurious positive or negative result. Mitigation: report results with and without the first 30 minutes of each session.

3. **In-sample bias in R1 tick bar priors.** R1's tick_50 AR R²=0.00034 was computed without CV (single-split). R4b showed R1's dollar_25k AR R² was 18× inflated by in-sample bias. The tick_50 prior may also be inflated, but since 0.00034 is already near zero, the practical impact is minimal.

4. **Calibration extrapolation (Arm 2a).** Using 1 day for calibration assumes duration statistics are stationary across 2022. If volatility regimes produce very different bar durations (e.g., FOMC days produce 10× more dollar bars than quiet days), the single-day calibration may be unrepresentative. Mitigation: if time permits, calibrate on 3 days (low-vol, mid-vol, high-vol) and report the range.

5. **Horizon-bar-type confound (cross-arm comparison).** Arm 3 tests extended horizons on time_5s. Arm 2 tests standard horizons on event bars at actionable timescales. A null result on time_5s at h=1000 (~83min) does not logically preclude signal on dollar bars at h=1000 (which would be a different clock time). However, the combined evidence from R4, R4b, and R4c across multiple bar types, timescales, and horizons makes this unlikely. Document the limitation.

6. **Multiple testing across arms.** Three arms × multiple horizons × multiple feature configs = many tests. Holm-Bonferroni is applied within each arm, not across arms. If exactly one test in one arm barely passes, treat with extreme skepticism — the family-wise error rate across the full experiment is higher than within any single arm.

---

## Decision Rules (per R4/R4b framework, applied independently per arm)

**Rule 1 (AR structure):** Best AR model achieves R² > 0 at h > 1 with corrected p < 0.05.

**Rule 2 (Temporal augmentation — dual threshold):**
- R²(Book+Temporal) − R²(Static-Book) > 0 with corrected p < 0.05, AND
- Relative improvement > 20% of Static-Book R²

**Rule 3 (Temporal standalone):** Temporal-Only achieves R² > 0 with corrected p < 0.05.

**Rule 4 (Reconciliation):** Interpret joint pattern across R4, R4b, and R4c.

### Cross-Arm Decision Framework (post-experiment)

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| All arms fail Rule 2 | Temporal encoder question closed permanently across all bar types, timescales (5s-83min), and horizons. | Proceed with CNN+GBT static. Close R4 line. |
| Arm 3 passes Rule 2 at h ≥ 200 | Temporal structure at regime scales (15-83 min) on time bars. | **Architecture change.** Design temporal encoder for long-horizon features. High priority follow-up. |
| Arm 2c passes Rule 2 at ≥5s timescale | Temporal signal at actionable timescale on event bars. | **Architecture change.** Test cross-scale aggregation (dollar temporal features → time_5s grid) before full pipeline switch. |
| Arm 1 passes Rule 2 | Tick bars surface temporal structure others miss. | Investigate carefully — tick_50 prior (AR R²=0.00034) makes this very unlikely. Check for artifacts. |
| Signal found AND linear (Ridge ≈ GBT) | Simple feature augmentation suffices. | Add temporal features to static input vector. No SSM needed. |
| Signal found AND nonlinear (GBT >> Ridge) | Learned temporal encoder warranted. | Design SSM/LSTM experiment on that specific bar type + timescale. |

---

## Output Structure

```
.kit/results/R4c/
├── arm3_extended_horizons/
│   ├── metrics.json              # Tier 1 + Tier 2 R², deltas, p-values
│   ├── analysis.md               # Per-horizon results + SC1/SC2 evaluation
│   ├── fwd_return_validation.json # Python vs C++ fwd_return comparison
│   └── data_loss_report.json     # Bars lost per horizon per day
├── arm1_tick_bars/
│   ├── features.csv              # C++ export
│   ├── metrics.json              # Full R4 protocol output
│   └── analysis.md
├── arm2_calibration/
│   ├── calibration_table.csv     # threshold → duration stats
│   ├── selected_thresholds.json  # Chosen thresholds + rationale
│   ├── <threshold_dir>/          # Per-threshold results (if 2c runs)
│   │   ├── features.csv
│   │   ├── metrics.json
│   │   └── analysis.md
│   └── cross_threshold_comparison.csv
└── analysis.md                   # Cross-arm decision + final recommendation
```

---

## Exit Criteria

- [ ] Arm 3: fwd_return_{200,500,1000} computed in Python and validated against C++ fwd_return_{1,5,20,100}
- [ ] Arm 3: Tier 1 AR R² reported for h={200,500,1000} (at minimum AR-10 GBT)
- [ ] Arm 3: Tier 2 Δ_temporal_book reported for h={200,500,1000} with paired test + Holm-Bonferroni
- [ ] Arm 3: Data loss at extended horizons documented
- [ ] Arm 3: SC1 and SC2 evaluated
- [ ] Arm 1: tick_50 features exported, bar count sanity-checked against R1 (88,521 ±10%)
- [ ] Arm 1: Full R4 protocol (Tier 1 + Tier 2) completed, SC3 evaluated
- [ ] Arm 2a: Calibration table produced for all 10 thresholds
- [ ] Arm 2b: Actionable thresholds selected (or documented as non-existent)
- [ ] Arm 2c: If actionable thresholds exist, R4 protocol completed at those thresholds. SC4 evaluated.
- [ ] Cross-arm analysis.md written with decision framework evaluation
- [ ] All decision rules (Rule 1-4) evaluated per arm
- [ ] Feature importance reported for all GBT models
- [ ] Summary entry ready for `.kit/RESEARCH_LOG.md`
- [ ] `CLAUDE.md` Current State updated
