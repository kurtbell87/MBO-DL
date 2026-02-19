# Experiment: R3b Genuine Tick Bars — CNN Spatial Signal on Trade-Event Tick Bars

## Hypothesis

At least one genuine tick-bar threshold (trade-counting bars, post TB-Fix) produces CNN spatial R² on structured (20,2) book input ≥ 0.107 (≥20% relative improvement above the time_5s baseline of 0.089), indicating that activity-normalized bars give the CNN a more homogeneous prediction task that improves spatial signal quality.

**Direction:** Positive — tick bars improve R² over time bars.
**Magnitude:** ≥20% relative (≥0.107 absolute vs 0.089 baseline).

**H_null:** CNN R² ≤ 0.089 at all tick-bar thresholds tested. Time bars are as good as or better than event bars for spatial book prediction. The within-bar activity heterogeneity of time bars does not impair the CNN.

**Background:**
- **R3b (original):** INCONCLUSIVE — the C++ `bar_feature_export` counted fixed-rate book snapshots (10/s), not trade events. All "tick bar" results were actually time bars at different frequencies. Those results are void.
- **TB-Fix** (PR #19, merged 2026-02-19): Fixed tick bar construction. `book_builder.hpp` now emits `trade_count` per snapshot (counts action='T' MBO events). `tick_bar_builder.hpp` accumulates trade counts and closes bars at threshold with remainder carry-over.
- **9E:** CNN R²=0.089 on time_5s with corrected normalization (3rd independent reproduction). This is the baseline.
- **Economic context:** Hybrid pipeline has exp=-$0.37/trade at base costs. A 20%+ R² improvement could eventually contribute to closing the 2pp win rate gap needed for breakeven.

## Independent Variables

**Bar type and threshold** — the sole independent variable.

- **Baseline:** time_5s bars (R²=0.089, from 9E, not re-run)
- **Treatment:** Genuine tick bars at multiple thresholds, selected adaptively from calibration

Calibration sweep (8 thresholds):

| Threshold | Old (broken) estimate | Notes |
|-----------|----------------------|-------|
| tick_25 | ~2.5s | Finest grain — closest to time_5s bar rate |
| tick_50 | ~5s | |
| tick_100 | ~10s | Was the "peak" in broken R3b (void) |
| tick_250 | ~25s | |
| tick_500 | ~50s | |
| tick_1000 | ~100s | |
| tick_2000 | ~200s | |
| tick_5000 | ~500s | Near upper bound of intraday feasibility |

**Old duration estimates are void.** They assumed 10 snapshots/s counting (broken). Genuine tick bars count actual trade events, which are sparser than 10/s. Expect longer durations and fewer bars/day than these estimates. The calibration phase establishes the true bar characteristics.

CNN training runs on up to 4 viable thresholds selected from calibration. "Viable" = bars_per_day mean ≥ 100 across 19 days.

## Controls

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | Same 19 trading days as R3/9D/9E | Eliminates date selection as variable |
| CNN architecture | Conv1d(2→59→59) + BN×2 + ReLU×2 + AdaptiveAvgPool1d(1) + Linear(59→16) + ReLU + Linear(16→1). **12,128 params exactly.** | R3-exact, confirmed by 9C (0% deviation), 9D (perfect reproduction), 9E (3rd reproduction) |
| Price normalization | Raw CSV values ÷ TICK_SIZE (0.25) → tick offsets. Do NOT z-score channel 0. | 9D/9E-exact. Root cause of 9B/9C failure. |
| Size normalization | log1p(size), z-scored **per day** using train-day stats only | 9D/9E-exact. Per-fold z-scoring is wrong. |
| CV structure | 5-fold expanding window on day boundaries | Same temporal splits as R3/9D/9E |
| Validation | 80/20 split of train days for early stopping. Test data NEVER used for model selection. | Prevents R3's test-as-validation leakage (~36% R² inflation) |
| Optimizer | AdamW(lr=1e-3, weight_decay=1e-4) | R3-exact |
| LR schedule | CosineAnnealingLR(T_max=50, eta_min=1e-5) | R3-exact |
| Batch size | 512 | R3-exact |
| Max epochs / early stopping | 50 max, patience=10 on validation loss | R3-exact with proper validation |
| Seeds | seed = 42 + fold_idx (42,43,44,45,46 for folds 1-5) | Matches 9D/9E for per-fold comparison where applicable |
| Loss | MSE on fwd_return_5 | R3-exact |
| Feature pipeline | Same C++ bar_feature_export binary, same 40 book columns + non-spatial features + forward returns | Only bar construction mode differs |

**The only variable is bar construction method (tick event-counting vs fixed time interval) and tick threshold.** Everything else is held constant.

### CNN Architecture (exact specification — MUST match)

```
Input: (B, 2, 20)                    # channels-first: (price_offset, log1p_size) × 20 levels
Conv1d(in=2, out=59, kernel_size=3, padding=1)  + BatchNorm1d(59) + ReLU
Conv1d(in=59, out=59, kernel_size=3, padding=1) + BatchNorm1d(59) + ReLU
AdaptiveAvgPool1d(1)                 # → (B, 59)
Linear(59, 16) + ReLU               # 16-dim embedding
Linear(16, 1)                        # scalar return prediction
```

**Parameter count breakdown:**
- Conv1d(2→59, k=3): 2×59×3 + 59 = 413
- BN(59): 59×2 = 118
- Conv1d(59→59, k=3): 59×59×3 + 59 = 10,502
- BN(59): 59×2 = 118
- Linear(59→16): 59×16 + 16 = 960
- Linear(16→1): 16×1 + 1 = 17
- **Total: 12,128**

### Expanding-Window Folds

| Fold | Train Days | Val Days (last 20%) | Test Days | CNN Seed |
|------|-----------|---------------------|-----------|----------|
| 1 | Days 1–3 | Day 4 | Days 5–7 | 42 |
| 2 | Days 1–6 | Day 7 | Days 8–10 | 43 |
| 3 | Days 1–8 | Days 9–10 | Days 11–13 | 44 |
| 4 | Days 1–10 | Days 11–13 | Days 14–16 | 45 |
| 5 | Days 1–13 | Days 14–16 | Days 17–19 | 46 |

## Metrics (ALL must be reported)

### Primary

1. **peak_tick_r2**: Highest mean OOS R² across all viable tick-bar thresholds (5-fold mean)
2. **peak_delta**: peak_tick_r2 − 0.089 (improvement over time_5s baseline)

### Secondary

- **r2_by_threshold**: Mean OOS R² for each viable threshold
- **per_fold_r2**: All folds × all thresholds (the full R² matrix)
- **per_fold_train_r2**: Train R² per fold/threshold (must be > 0.05 where data is adequate, i.e., fold train size > 5,000)
- **fold3_r2**: R² at fold 3 for each threshold (Oct 2022 diagnostic — -0.049 on time_5s)
- **r2_std_by_threshold**: Per-fold R² standard deviation per threshold
- **curve_shape**: Monotonic up / Inverted-U / Monotonic down / Flat
- **bars_per_day**: Mean, std, cv per threshold (from calibration)
- **bar_duration**: Mean, median, p10, p90 per threshold (from calibration)
- **total_bar_count**: Per threshold across 19 days

### Sanity Checks

| Check | Expected | Failure means |
|-------|----------|---------------|
| bars_per_day_cv > 0 at all thresholds | cv > 0 (e.g., 2-10%) | Tick-bar-fix didn't work — **FATAL** |
| p10 ≠ p90 duration at all thresholds | p10 < p90 | Bars are time-sampled, not event-sampled — **FATAL** |
| Channel 0 after TICK_SIZE division | Range approx [-25, +25] | TICK_SIZE division not applied |
| Channel 1 per-day z-scored | Per-day mean≈0, std≈1 | Per-day normalization not applied |
| CNN param count | 12,128 ± 5% | Architecture mismatch — **FATAL** |
| No NaN in predictions or losses | 0 NaN | Normalization or forward pass bug |
| LR decays from ~1e-3 toward ~1e-5 | Cosine schedule active | CosineAnnealingLR not applied |

## Baselines

### 1. time_5s CNN R² (9E, proper validation — the baseline to beat)
- **Source:** `.kit/results/hybrid-model-corrected/step1_cnn/fold_results.json`
- **Mean R²:** 0.089
- **Per-fold:** [0.139, 0.086, -0.049, 0.131, 0.140]
- **Std:** ~0.074
- **Bars/day:** 4,630 (constant)
- **Note:** 3rd independent reproduction. Not re-run in this experiment.

### 2. Broken R3b (void — reference only)
- **Peak R²:** 0.057 at "tick_100" (was actually time_10s, not genuine tick bars)
- **Note:** These results are void. Do NOT compare against these values.

## Success Criteria (immutable once RUN begins)

- [ ] **SC-1:** bars_per_day_cv > 0 at ALL 8 calibrated thresholds (genuine event bars confirmed)
- [ ] **SC-2:** At least 3 thresholds viable for CNN training (bars_per_day mean ≥ 100)
- [ ] **SC-3:** MVE gate passes (best viable threshold, fold 5, train R² ≥ 0.05)
- [ ] **SC-4:** Peak tick-bar mean OOS R² ≥ 0.107 (primary hypothesis: ≥20% above 0.089)
- [ ] **SC-5:** No sanity check failures (bars_per_day_cv, normalization, architecture, NaN)
- [ ] **SC-6:** R² vs bar-size curve produced with ≥ 3 data points
- [ ] No regression on sanity checks beyond stated tolerances

## Minimum Viable Experiment

**Phase 0 — Calibration (~10 min):**
1. Export bars for all 8 thresholds (tick_25 through tick_5000) across 19 trading days using the fixed `bar_feature_export --bar-type tick`.
2. For each threshold CSV, compute: bars_per_day (mean, std, cv), total_bars, bar_duration (median, p10, p90).
3. Save to `.kit/results/r3b-genuine-tick-bars/calibration/threshold_sweep.json`.
4. **Gate A (bars_per_day_cv):** If bars_per_day_cv = 0 at ANY threshold → **ABORT**. The tick-bar-fix did not work as expected.
5. **Gate B (viability):** If < 3 thresholds have bars_per_day ≥ 100 → try additional fine-grained thresholds (tick_10, tick_15). If still < 3 → **ABORT** (MES trade frequency too low for tick-bar CNN with 12k-param model).
6. Select up to 4 viable thresholds for CNN training. Selection strategy: pick thresholds that span the viable range evenly in log-space (smallest viable, ~25th percentile, ~50th percentile, largest viable).

**Phase 1 — MVE (~2 min):**
1. Pick the threshold with highest bars_per_day (most data, best chance of learning).
2. Train CNN on fold 5 only (seed=46, largest training set).
3. Verify: param count = 12,128, TICK_SIZE division applied, per-day z-scoring applied.
4. **Gate C:** train R² < 0.05 → normalization or export issue. **ABORT** and investigate.
5. **Gate D:** train R² ≥ 0.05 → proceed to full sweep.

## Full Protocol

### Step 1: Export All Calibration Thresholds

For each threshold in [25, 50, 100, 250, 500, 1000, 2000, 5000]:
```
./build/bar_feature_export --bar-type tick --bar-param <N> \
  --output .kit/results/r3b-genuine-tick-bars/tick_<N>/tick_<N>.csv
```

These CSVs serve both calibration (bar statistics) and CNN training (if threshold is viable). Single export per threshold.

### Step 2: Compute Calibration Statistics

For each exported CSV:
1. Count bars per day across 19 trading days. Compute mean, std, cv.
2. Compute bar duration (time between consecutive bar timestamps): median, p10, p90.
3. Compute total_bar_count.
4. Verify: bars_per_day_cv > 0, p10 ≠ p90.

Save all to `calibration/threshold_sweep.json`.

### Step 3: Threshold Selection

Apply viability filter: bars_per_day mean ≥ 100. From viable thresholds, select up to 4 spanning the range. Log the selection rationale.

Flag data adequacy per fold: for each selected threshold, compute expected fold 1 train bars (bars_per_day × 3 effective train days). If < 2,000, note that fold 1 may be data-starved for this threshold.

### Step 4: MVE — Single Fold

Run CNN training on highest-data threshold, fold 5 only. Apply 9E-exact normalization and training protocol. Verify train R² ≥ 0.05.

### Step 5: Full Sweep

For each selected threshold, for each fold 1-5:
1. Set seed = 42 + fold_idx.
2. Split data by day boundaries per the expanding-window table.
3. Apply validation split: 80/20 of train days (last 20% as val). Early stopping on validation loss.
4. Normalize: channel 0 ÷ 0.25 (TICK_SIZE), channel 1 log1p + per-day z-score.
5. Train CNN with R3-exact protocol.
6. Record: train_r2, val_r2, test_r2, epochs_trained, train_bar_count, test_bar_count.

### Step 6: Analysis

1. Compute mean/std test R² per threshold.
2. Produce R² vs bar-size curve (`r2_vs_barsize_curve.csv`): columns = threshold, mean_duration_sec, mean_test_r2, std_test_r2, fold3_test_r2, mean_train_r2, bars_per_day_mean, bars_per_day_cv.
3. Identify peak threshold, curve shape, peak_delta.
4. Fold 3 diagnostic: if ANY threshold gives positive fold 3 R² (where time_5s gives -0.049), flag prominently.
5. Data volume comparison: report the correlation between bars_per_day and mean R² across thresholds. If strong negative correlation, data volume (not bar type) may be the primary driver.
6. Apply decision framework.
7. Write `analysis.md`.

### Decision Framework

| Outcome | Criterion | Action |
|---------|-----------|--------|
| **BETTER** | Peak R² ≥ 0.107 (≥20% above baseline) | Switch pipeline to optimal tick-bar threshold before full-year export. If peak is broad (2+ adjacent thresholds BETTER), pick the larger one (fewer bars = more meaningful per-bar prediction). |
| **COMPARABLE** | All viable R² in [0.071, 0.107] | Stick with time_5s (simpler, proven, no threshold tuning). Time_5s vindicated. |
| **WORSE** | All viable R² < 0.071 | Time_5s definitively superior. Close this line of inquiry permanently. |
| **Peak found** | Inverted-U shape, peak in BETTER range | Adopt peak threshold. The CNN has a preferred event-bar scale. |
| **Monotonic down** | R² declines with increasing threshold | Fastest tick bars are best but may not beat time_5s. If smallest viable threshold is BETTER, adopt it; otherwise stick with time_5s. |
| **Data-volume-dominated** | R² correlates strongly with bars_per_day (r > 0.8) | Cannot distinguish bar-type effect from data-volume effect. Report as INCONCLUSIVE for the bar-type question. Tick bars with matching bar count to time_5s (~4,630/day) would be needed for a clean comparison — check if any threshold lands near this value. |

## Resource Budget

**Tier:** Standard

- Max GPU-hours: 0 (CPU only)
- Max wall-clock time: 2 hours
- Max training runs: 21 (1 MVE + up to 4 thresholds × 5 folds)
- Max seeds per configuration: 1 (seed = 42 + fold_idx, deterministic per fold)

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 200000
model_type: pytorch
sequential_fits: 21
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 1.0
```

### Wall-Time Estimation

| Component | Per-unit estimate | Count | Subtotal |
|-----------|------------------|-------|----------|
| Bar export (19 days MBO data per threshold) | 60–120s | 8 | ~10-16 min |
| CNN training (12k params, variable rows) | 10–30s | 21 | ~4-10 min |
| Statistics computation | 30s | 8 | ~4 min |
| File I/O + analysis writing | — | — | ~5 min |
| **Total estimated** | | | **~25-40 min** |

Budget of 2 hours provides 3-5× headroom. Original R3b (with broken bars) completed in 14 minutes. Genuine tick bars may produce different bar counts (potentially more or fewer), affecting training time proportionally.

## Abort Criteria

- **bars_per_day_cv = 0 at any threshold** → ABORT. Tick-bar-fix didn't take effect. Investigate C++ code.
- **< 3 viable thresholds** (even after trying tick_10, tick_15) → ABORT. MES trade frequency insufficient for tick-bar CNN sweep.
- **MVE train R² < 0.05** (fold 5 on best threshold) → ABORT. Normalization or export bug. Investigate before any further training.
- **All thresholds produce mean test R² < 0.02** → ABORT. Systematic issue with event-bar feature export schema.
- **NaN in any loss or prediction** → ABORT that threshold, continue others.
- **Wall clock > 2 hours** → ABORT remaining, report partial results for completed thresholds.
- **Per-run time:** Any single CNN fit exceeds 5 minutes (10× expected) → investigate.

Time-based per-run abort at 5 min is 10× expected ~30s — allows for larger-than-expected bar counts without premature kills.

## Confounds to Watch For

1. **Data volume confound (MOST LIKELY).** Tick bars will produce fewer bars/day than time_5s (4,630). If the best tick threshold produces e.g. 1,000 bars/day, the R² comparison is confounded: lower R² may reflect less training data, not worse signal. The original R3b (broken) showed this clearly — at tick_1000 (183 bars/day), the 12,128-param CNN had 16× more params than training samples in fold 1. **Mitigation:** Report correlation between bars_per_day and mean R² across thresholds. If tick bar R² at the threshold closest to 4,630 bars/day matches or exceeds 0.089, the bar-type effect is real despite data confounding at coarser thresholds.

2. **Forward return semantics change.** fwd_return_5 means "5 bars ahead." On tick_100 bars, that might be ~5× the median bar duration. On tick_1000 bars, it spans a much longer clock-time horizon. Lower R² at larger thresholds could reflect harder prediction tasks (longer horizons are inherently less predictable) rather than worse input quality. This is inherent to event-bar-denominated returns and cannot be separated from the bar-type effect in this design.

3. **Fold 3 regime weakness.** Oct 2022 (fold 3 test days) consistently produces weak/negative CNN R² on time_5s (-0.049). May persist on tick bars. Do not over-interpret fold 3 results. The mean across all 5 folds is the primary metric.

4. **Fold 5 dominance.** Fold 5 has the most training data (days 1-13 train, 14-16 val, 17-19 test). At coarser thresholds, it may be the only fold where the CNN has enough data to learn. Original R3b showed this: fold 5 was the only positive fold at tick_500+. **Mitigation:** Report fold-5-only R² alongside 5-fold mean at each threshold. If the two diverge strongly, the 5-fold mean is unreliable at that threshold.

5. **Within-day distribution shift.** Event bars produce more bars during active periods (market open, close) and fewer during quiet periods (midday). The CNN training set will be dominated by active-period book states. If the CNN's spatial signal is strongest during quiet periods (when the book is stable), event bars underrepresent the most predictable regime. Time bars don't have this bias.

6. **Threshold sensitivity / overfitting the sweep.** If only 1 out of 4 thresholds crosses the 0.107 line, it could be noise. The 20% threshold applies to individual thresholds, not to the best-of-4. **Mitigation:** If only 1 threshold is BETTER, note the multiple-testing concern. A broad pattern (2+ adjacent thresholds in the BETTER or high-COMPARABLE range) is much stronger evidence than a single-threshold peak.

## Output

```
.kit/results/r3b-genuine-tick-bars/
├── calibration/
│   └── threshold_sweep.json         # threshold → {bars_per_day_mean, bars_per_day_std, bars_per_day_cv,
│                                    #              total_bars, median_duration, p10_duration, p90_duration}
├── tick_<N>/                        # one per threshold (all 8 exported; CNN results only for viable ones)
│   ├── tick_<N>.csv                 # feature export (bar construction + features + forward returns)
│   ├── fold_results.json            # per-fold: {train_r2, val_r2, test_r2, epochs, train_bars, test_bars}
│   ├── bar_statistics.json          # duration stats, bars_per_day variance
│   └── normalization_verification.txt
├── sweep_summary.json               # all thresholds: mean R², std, fold 3, bar stats, viability
├── r2_vs_barsize_curve.csv          # threshold, mean_duration, mean_r2, std_r2, fold3_r2, bars_per_day
├── comparison_table.md              # side-by-side: all viable thresholds + time_5s baseline
└── analysis.md                      # full analysis with decision per framework
```

## Exit Criteria

- [ ] EC-1: Calibration complete for all 8 thresholds with bars_per_day_cv > 0 (genuine event bars confirmed)
- [ ] EC-2: Viability filtering applied — non-viable thresholds documented and dropped with rationale
- [ ] EC-3: MVE gate passed (fold 5 train R² ≥ 0.05) OR investigation documented
- [ ] EC-4: CNN trained on all viable thresholds × 5 folds with 9E-exact protocol
- [ ] EC-5: R² vs bar-size curve produced (`r2_vs_barsize_curve.csv`) with ≥ 3 data points
- [ ] EC-6: Fold 3 diagnostic reported (R² at each viable threshold vs -0.049 baseline)
- [ ] EC-7: Data volume confound analysis reported (correlation between bars_per_day and mean R²)
- [ ] EC-8: Comparison table produced (`comparison_table.md`) — all viable thresholds + time_5s
- [ ] EC-9: Decision rendered per framework (BETTER / COMPARABLE / WORSE / INCONCLUSIVE)
- [ ] EC-10: `analysis.md` written with all findings, decision, and recommended next steps
- [ ] EC-11: `sweep_summary.json` contains all threshold metrics
