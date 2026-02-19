# Experiment: R3b — CNN Spatial Predictability on Event Bars

**Date:** 2026-02-19
**Depends on:** R3 (CNN book encoder), R4d (event bar calibration table), CNN Reproduction Diagnostic (corrected normalization protocol)
**Priority:** BLOCKING — determines bar type for full-year export and model build

---

## Motivation

The entire model pipeline is built on time_5s bars. This was never empirically justified for spatial predictability. Here's how it happened:

- **R1** tested whether event bars produce more Gaussian/homoskedastic returns than time bars. Answer: no. This is a statistical property of the return series, not a statement about CNN learnability.
- **R3** tested CNN spatial encoding on book snapshots. It used time_5s bars because R1 said time bars were "fine." R3 never tested event bars.
- **R6** locked in time_5s based on R1 + R3. The bar type and the spatial encoder were never tested in combination with event bars.
- **R4b** found dollar_25k static-HC R²=0.080 at h=1 with flat features — but this was interpreted through the temporal lens (is there temporal signal?) rather than the spatial lens (is the book more predictable on event bars?).
- **R4c/R4d** calibrated event bars to actionable timescales and tested temporal features. Spatial features (and especially CNN on structured book) were never tested at those timescales.

The gap: **R3's CNN on event-bar book snapshots at actionable timescales has never been run.** This experiment fills it.

The hypothesis behind event bars for spatial prediction is simple: time bars mix different activity regimes into the same prediction task. A book snapshot at 9:31 AM and a book snapshot at 12:15 PM are both "one 5-second bar," but the former has 50 trades behind it and the latter has 2. The CNN predicts fwd_return_5 for both, but the information content of "5 bars ahead" varies by an order of magnitude. Event bars normalize this — each bar represents the same amount of market activity, so the CNN's prediction task is homogeneous regardless of time of day.

If event-bar CNN R² meaningfully exceeds time_5s CNN R², we switch the entire pipeline before scaling to 250 days. If it's comparable or worse, time_5s is vindicated.

---

## Hypotheses

**H1 (Primary):** There exists a tick-bar threshold at which CNN spatial R² on structured (20,2) book snapshots exceeds the time_5s baseline of 0.084 by at least 20% relative (≥ 0.101). The spatial signal has an optimal bar size that event bars can target.

**H2 (Shape):** CNN spatial R² as a function of bar size is non-monotonic — it rises as bars aggregate more activity (reducing snapshot noise) and then falls as bars become so large that intra-bar book dynamics have decayed by bar close. There is a peak somewhere in the actionable range.

**H3 (Stability):** At the optimal event-bar size, per-fold R² variance is lower than time_5s (std < 0.048), because activity-normalized bars reduce regime-driven heterogeneity in the prediction task.

**H4 (Fold 3):** The fold 3 collapse (R²=-0.047 on time_5s, October 2022) is attenuated at one or more event-bar sizes, because the bar type rather than the market regime was the primary driver of that failure.

**H_null:** CNN spatial R² is ≤ 0.084 at all event-bar sizes tested. Time bars are as good as or better than event bars for spatial book prediction. The heteroskedasticity of time bars does not impair the CNN.

---

## Design

### Bar Type Selection — Timescale Sweep

We have zero empirical evidence about where the CNN's spatial signal peaks as a function of bar size. R4d's diminishing R² at larger timescales measured *temporal* features, not *spatial*. The CNN on structured book input may have a completely different optimal scale. We need a sweep, not a single point.

**Use tick bars for the sweep.** Dollar bars had calibration issues (sub-actionable at thresholds up to $1M per R4c; actionable thresholds from R4d were $5M+ but produced very few bars/day). Tick bars give cleaner threshold-to-timescale control.

**Select 4 tick thresholds spanning the actionable range:**

| Label | Target Duration | Approximate Threshold | Rationale |
|-------|----------------|----------------------|-----------|
| **Small** | ~5-10s | tick_100 – tick_200 | Fast. Near-minimal actionable. High bar count. Book is noisy snapshot. |
| **Medium** | ~30-60s | tick_500 – tick_1000 | Moderate. 1-minute-scale prediction. Book has settled somewhat. |
| **Large** | ~2-5 min | tick_2000 – tick_3000 | What you actually trade on in Sierra Chart. Book represents meaningful equilibrium. |
| **XL** | ~10-15 min | tick_5000 – tick_10000 | Near the upper bound of intraday prediction. Tests whether spatial signal decays at this scale. |

Use the R4d calibration table to map these target durations to exact thresholds. If the calibration table doesn't cover this range, run a quick calibration pass first (bar construction only, no features — just compute bar counts and duration stats for a sweep of thresholds). This is cheap (~30 min).

**Why 4 points:** We're looking for the shape of the curve (CNN R² vs. bar size). A single point tells you nothing about the gradient. Four points gives you monotonic increase, a peak, or monotonic decrease — qualitatively different conclusions that lead to different actions.

**If a dollar threshold from R4d's calibration table lands cleanly at one of these timescales** (e.g., $10M ≈ 14s ≈ the Small operating point), include it as a 5th configuration for a tick-vs-dollar comparison at matched timescale. Otherwise skip dollar bars.

### Feature Export

For each selected tick-bar threshold, export book tensors from the existing C++ pipeline for the same 19 trading days used in R3 and the reproduction diagnostic.

Export format: same as time_5s.csv — flat CSV with 40 book columns (20 price offsets + 20 sizes), non-spatial features, and forward returns. The only thing that changes is the bar construction; the feature computation pipeline is identical.

**Forward return target:** fwd_return_5 denominated in event bars (the return 5 bars ahead), NOT a fixed clock-time horizon. On tick_500 bars (~30s each), fwd_return_5 spans ~150s of order flow. On tick_2000 bars (~2min each), fwd_return_5 spans ~10min. This is by design — the prediction task scales with bar size.

Export all 4 thresholds before training. Bar construction and feature export can be parallelized across thresholds.

### CNN Training Protocol

Use the corrected protocol from the reproduction diagnostic. Zero deviations.

| Parameter | Value |
|-----------|-------|
| Architecture | Conv1d(2→32→32) + BN + ReLU × 2 → AdaptiveAvgPool1d → Linear(32→16→1) |
| Params | ~12,128 |
| Price normalization | Raw values ÷ TICK_SIZE (0.25) to get tick offsets |
| Size normalization | log1p(size), z-scored per day using train-day stats only |
| Optimizer | AdamW(lr=1e-3, wd=1e-4) |
| LR schedule | CosineAnnealingLR(T_max=50, eta_min=1e-5) |
| Batch size | 512 |
| Epochs | 50 max, early stopping patience=10 on validation loss |
| Validation | 80/20 split of training days (NOT test set) |
| Loss | MSE on fwd_return_5 |
| Seed | 42 + fold_idx (matching R3) |
| CV | 5-fold expanding window on day boundaries |

### Expanding Window Folds

Assign the 19 days chronologically. Use the same fold structure as R3/reproduction:

| Fold | Train Days | Test Days |
|------|-----------|-----------|
| 1 | Days 1–4 | Days 5–7 |
| 2 | Days 1–7 | Days 8–10 |
| 3 | Days 1–10 | Days 11–13 |
| 4 | Days 1–13 | Days 14–16 |
| 5 | Days 1–16 | Days 17–19 |

Note: bar counts per day will differ from time_5s. Event bars produce more bars during active periods and fewer during quiet periods. Total bars across 19 days may differ significantly from the 87,970 in time_5s.csv. Log the actual bar count per day and total.

---

## Controls (Fixed)

| Control | Value | Rationale |
|---------|-------|-----------|
| Data | Same 19 trading days as R3 and reproduction diagnostic | Eliminates date selection as a variable |
| CNN architecture | Identical to R3-corrected | Isolates bar type as the only variable |
| Normalization | TICK_SIZE division + per-day log1p z-scoring | Corrected protocol from reproduction diagnostic |
| CV structure | Same 5-fold expanding window | Same temporal splits |
| Training protocol | Same optimizer, LR, patience, batch size | No hyperparameter differences |
| Feature pipeline | Same C++ export code | Only bar construction differs |

**The only independent variable is bar type and size.** Everything else is held constant across all thresholds and against the time_5s baseline. This is a clean comparison.

---

## Metrics (ALL must be reported)

### Primary
- **r2_by_threshold**: Mean OOS R² across 5 folds, for each tick-bar threshold
- **r2_time5s_baseline**: 0.084 (from reproduction diagnostic — reference, not re-computed)
- **peak_threshold**: Which threshold achieves the highest mean R²
- **peak_r2**: Mean R² at the peak
- **peak_delta**: peak_r2 − 0.084
- **curve_shape**: Monotonic up / Inverted-U / Monotonic down / Flat — characterizes the R² vs. bar-size relationship

### Secondary
- Per-fold R² for each threshold (all 5 folds × 4 thresholds = 20 values)
- Per-fold train R² for each threshold (must be > 0.05 to confirm CNN is learning)
- Fold-level R² standard deviation per threshold (compare against time_5s std ≈ 0.048+)
- Fold 3 R² at each threshold (October 2022 diagnostic)
- Bars per day: mean, min, max, std across 19 days, per threshold
- Bar duration statistics: mean, median, p10, p90 across all bars, per threshold (confirms timescale)
- Total bar count per threshold (for power comparison)
- **R² vs. log(bar_size) scatter** — plot and report slope. Linear, sublinear, or peaked?

### Sanity Checks
- Price offsets after TICK_SIZE division are integer-valued tick offsets (print samples)
- Param count = 12,128 ± 5%
- LR decays from ~1e-3 toward ~1e-5 over training
- No NaN in predictions or losses
- Bar count per day is in the 200-5000 range (too few = threshold too high, too many = sub-second bars)

---

## Decision Framework (immutable once RUN begins)

### Per-Threshold Assessment

For each tick-bar threshold, classify against time_5s baseline (R²=0.084):

| Category | Criterion |
|----------|-----------|
| **BETTER** | Mean R² ≥ 0.101 (20%+ relative improvement) |
| **COMPARABLE** | 0.068 ≤ Mean R² < 0.101 (within ±20%) |
| **WORSE** | Mean R² < 0.068 (20%+ relative degradation) |

### Cross-Threshold Decision

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| Peak exists: ≥1 threshold BETTER, with inverted-U shape across sweep | Optimal event-bar size found. Spatial signal has a scale preference. | **Switch pipeline to the peak bar type before full-year export.** If peak is broad (2+ adjacent BETTER), pick the larger bar (fewer bars = more meaningful per-bar prediction). |
| Monotonic improvement: R² rises with bar size across entire sweep | Haven't found the peak yet. Signal keeps improving at larger aggregation. | Run 1-2 additional larger thresholds to find the peak. Do not switch until peak is located. |
| Monotonic decline: R² falls with bar size, Small is best | Spatial signal prefers fast bars. Event bars hurt at larger sizes. | If Small threshold BETTER than time_5s: switch to that threshold. If Small ≈ time_5s: stick with time_5s (simpler, no threshold tuning). |
| All COMPARABLE: no threshold clearly beats time_5s | Bar type doesn't matter much for CNN spatial prediction | Stick with time_5s. The heteroskedasticity doesn't impair the CNN. Proceed with confidence. |
| All WORSE: time_5s beats every event-bar threshold | Event bars hurt CNN spatial prediction | Time_5s definitively vindicated. Close this line of inquiry. |
| Mixed: fold variance on event bars is much lower despite similar mean R² | Event bars don't improve mean but improve consistency | Worth considering the switch — a model with R²=0.080±0.02 is more deployable than R²=0.084±0.10. Evaluate on fold-min rather than fold-mean. |

### Fold 3 Diagnostic (Informational, not decisive)

If fold 3 R² is positive at ANY event-bar threshold where it was -0.047 on time_5s, this is strong evidence that the fold 3 failure was bar-type-driven, not regime-driven. Note this finding but don't change the decision framework — it's one fold.

**The 20% threshold is deliberate.** Switching bar types has downstream costs — re-exporting 250 days, revalidating oracle expectancy on event bars, recomputing all features. The improvement must be meaningful enough to justify that work.

---

## Minimum Viable Experiment

Run the sweep incrementally, not all at once. This lets you abort early if the direction is clear.

**Step 0: Calibration (if R4d table doesn't cover the full range)**
- Run bar construction only (no features, no CNN) for tick thresholds: 100, 200, 500, 1000, 2000, 3000, 5000, 10000
- Compute bar count/day and median duration for each
- Select the 4 thresholds closest to target durations (5-10s, 30-60s, 2-5min, 10-15min)
- This takes ~30-60 minutes and is prerequisite infrastructure

**Step 1: Medium threshold first (fold 5 only)**
- Export features for the Medium threshold (~30-60s bars)
- Train CNN fold 5. Check train R² > 0.05, test R² > 0.
- This is the most informative single data point — far enough from time_5s to see a difference, not so large that bar count becomes a problem.
- If train R² < 0.05: investigate before expanding to more thresholds.

**Step 2: Medium threshold, all 5 folds**
- If Step 1 passes, run all 5 folds on Medium.
- Compare mean R² against 0.084. This gives you a preliminary answer.

**Step 3: Full sweep (Small, Large, XL)**
- Export and run the remaining 3 thresholds, all 5 folds each.
- Produce the R² vs. bar-size curve.
- Apply decision framework.

---

## Resource Budget

| Step | Compute | Time (est.) |
|------|---------|-------------|
| Calibration sweep (if needed) | Local | ~30-60 min |
| Event bar construction (4 thresholds × 19 days) | Local | ~4-6 hr |
| Feature export to CSV (4 thresholds) | Local | ~4-6 hr |
| CNN training (4 thresholds × 5 folds, CPU) | Local | ~1-2 hr |
| **Total** | | **~10-15 hr** |

This is more compute than the single-point version but still cheap relative to the decision it informs. The incremental MVE approach (Medium first, then expand) means you can abort after ~4 hours if the direction is clear.

Bar construction and feature export can be parallelized across thresholds if multiple cores are available.

### Compute Profile
```yaml
compute_type: cpu
estimated_rows: 400000
model_type: pytorch
sequential_fits: 20
parallelizable: true
memory_gb: 4
gpu_type: none
estimated_wall_hours: 15
```

---

## Abort Criteria

- Calibration produces < 100 bars/day at any threshold → that threshold is too coarse. Drop it, pick a smaller threshold.
- Calibration produces > 50,000 bars/day at any threshold → that threshold is too fine (sub-second bars). Drop it, pick a larger threshold.
- Train R² < 0.05 on Medium fold 5 (MVE gate) → investigate preprocessing before expanding sweep. This is the same failure mode as Phase B/9C.
- All 4 thresholds produce mean R² < 0.03 → something is systematically wrong with event-bar feature export. Stop and investigate rather than concluding event bars are worse.
- Export schema doesn't match time_5s.csv → C++ export needs bar-type-specific handling. Fix before proceeding.
- Wall clock exceeds 20 hours → abort, report partial results for completed thresholds
- If after Medium + one other threshold, both are WORSE by > 40%: consider early termination of remaining thresholds. But run at least 3 of 4 before concluding.

---

## Context for Research Agent

Read RESEARCH_AUDIT.md for full program context. Key points relevant to this experiment:

- R3 demonstrated CNN R²=0.084 on time_5s book snapshots (corrected from 0.132 after fixing test-as-validation leakage). The CNN's spatial signal is real — it exploits local book structure (queue imbalance, depth gradients).
- R4 chain (168+ tests) confirmed MES returns are martingale — no temporal signal at any bar type or timescale. **This experiment is NOT about temporal predictability.** It's about whether the CNN's SPATIAL signal is stronger when bars are activity-normalized.
- **Critical:** R4d showed diminishing temporal R² at larger bar sizes. That finding is IRRELEVANT here. Temporal R² measures autoregressive return structure. Spatial R² measures book-state predictiveness. These are different quantities. Do not assume larger bars are worse because R4d said so — R4d tested a different thing.
- The corrected CNN normalization protocol (raw prices ÷ TICK_SIZE (0.25), per-day z-scoring of log1p sizes, proper 80/20 train/val split) is mandatory. Three prior attempts failed because of normalization errors.
- Time bars are heteroskedastic by construction — they mix different activity regimes. Event bars normalize this. The question is whether that normalization helps the CNN, and at what scale.
- **This is a sweep, not a single point.** We don't know whether the CNN prefers small (noisy snapshot, many bars), medium, large (settled snapshot, fewer bars), or very large event bars. The R² vs. bar-size curve is the primary deliverable.

**Do not test temporal features. Do not add lookback windows. Do not modify the CNN architecture. The only variable is bar type and size.**

---

## Output

```
.kit/results/R3b-event-bar-cnn/
├── calibration/
│   └── threshold_sweep.json         # threshold → {bar_count/day, median_duration, p10, p90}
├── tick_{threshold_small}/
│   ├── export_verification.txt
│   ├── fold_results.json            # Per-fold: {train_r2, test_r2, epochs, bar_count}
│   ├── bar_statistics.json
│   └── normalization_verification.txt
├── tick_{threshold_medium}/
│   └── [same structure]
├── tick_{threshold_large}/
│   └── [same structure]
├── tick_{threshold_xl}/
│   └── [same structure]
├── sweep_summary.json               # All thresholds: mean R², std, fold 3, bar stats
├── r2_vs_barsize_curve.csv          # threshold, mean_duration, mean_r2, std_r2, fold3_r2
├── comparison_table.md              # Side-by-side: all thresholds + time_5s baseline
└── analysis.md                      # Full analysis with decision per framework
```
