# Analysis: R3b — CNN Spatial Predictability on Event Bars

## Verdict: INCONCLUSIVE

The experiment's bar construction produced **time bars at different frequencies, not activity-normalized event bars.** Every threshold yielded identical bar counts per day (std=0.0) and zero duration variance (p10=p90) across all 19 trading days — impossible for genuine tick/event bars. The primary hypothesis (activity normalization helps the CNN) was never testable with this data. All thresholds scored WORSE than the time_5s baseline, but this finding pertains to time-bar frequency selection, not to the event-bar question.

---

## Results vs. Success Criteria

- [ ] **H1 (Primary): CNN R² ≥ 0.101 at any threshold** — **FAIL.** Peak R² = 0.057 at tick_100 (Δ = -0.027 from 0.084 baseline). No threshold reaches even the COMPARABLE band (≥0.068).
- [ ] **H2 (Shape): Inverted-U R² vs. bar size** — **FAIL.** Curve is monotonic down. Slope = -0.029 per log-unit. R² decreases strictly as bar size increases.
- [ ] **H3 (Stability): Per-fold R² std < 0.048 at optimal point** — **FAIL.** tick_100 std = 0.104 (2.2× the baseline). tick_500 std = 0.096 (2.0×). tick_1000/tick_1500 have lower std (0.038/0.030) but with negative mean R² — lower variance of a worse estimator.
- [ ] **H4 (Fold 3): Positive at any threshold** — **FAIL.** Fold 3 R² is negative at all thresholds: tick_100 = -0.108, tick_500 = -0.009, tick_1000 = -0.036, tick_1500 = -0.072. Fold 3 failure is regime-driven, not bar-type-driven.
- [x] **H_null: CNN R² ≤ 0.084 at all sizes** — **PASS (confirmed).** All thresholds below baseline. All classified WORSE.
- [x] **Sanity: Price tick-quantized** — **PASS.** Fraction = 1.0 at all thresholds. Normalization correct.
- [x] **Sanity: Param count = 12,128 ± 5%** — **PASS.** Exactly 12,128 (0.0% deviation).
- [x] **Sanity: LR decays properly** — **PASS.** Decays from ~1e-3 toward ~1e-5 across epochs.
- [x] **Sanity: No NaN** — **PASS.** Zero NaN in predictions or losses at any threshold.
- [ ] **Sanity: Bars per day in 200–5000 range** — **PARTIAL FAIL.** tick_100 (2,289) and tick_500 (417) pass. tick_1000 (183) and tick_1500 (105) are below 200 but above the 100 abort criterion.
- [x] **Reproducibility** — **PASS (on its own terms).** tick_100 mean R² = 0.057 ± 0.104 across 5 folds. But the high std (1.8× the mean) means this average is unreliable for any individual comparison.

**Primary criteria: 0/4 pass. Null hypothesis confirmed. But experiment validity is compromised (see Confounds).**

---

## Metric-by-Metric Breakdown

### Primary Metrics

| Metric | Value | Spec Criterion | Verdict |
|--------|-------|----------------|---------|
| r2_by_threshold (tick_100) | 0.0566 | ≥0.101 for BETTER | WORSE (< 0.068) |
| r2_by_threshold (tick_500) | 0.0465 | ≥0.101 for BETTER | WORSE |
| r2_by_threshold (tick_1000) | -0.0027 | ≥0.101 for BETTER | WORSE |
| r2_by_threshold (tick_1500) | -0.0219 | ≥0.101 for BETTER | WORSE |
| r2_time5s_baseline | 0.084 | Reference value | N/A |
| peak_threshold | tick_100 | — | Smallest bar |
| peak_r2 | 0.0566 | ≥0.101 for H1 PASS | FAIL by 44% |
| peak_delta | -0.0274 | ≥+0.017 for 20% improvement | Negative (33% degradation) |
| curve_shape | Monotonic down | Inverted-U for H2 PASS | FAIL |

### Secondary Metrics

**Per-Fold Test R²** (all 5 folds × 4 thresholds = 20 values):

| Threshold | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|-----------|--------|--------|--------|--------|--------|------|-----|
| tick_100 | +0.181 | +0.060 | -0.108 | +0.001 | +0.149 | 0.057 | 0.104 |
| tick_500 | +0.018 | +0.000 | -0.009 | -0.014 | +0.238 | 0.047 | 0.096 |
| tick_1000 | -0.020 | +0.000 | -0.036 | -0.026 | +0.069 | -0.003 | 0.038 |
| tick_1500 | -0.026 | -0.006 | -0.072 | -0.025 | +0.019 | -0.022 | 0.030 |

**Per-Fold Train R²** (confirms learning / data starvation):

| Threshold | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|-----------|--------|--------|--------|--------|--------|------|
| tick_100 | 0.165 | 0.198 | 0.178 | 0.030 | 0.195 | 0.153 |
| tick_500 | 0.040 | -0.005 | 0.017 | 0.000 | 0.143 | 0.039 |
| tick_1000 | -0.009 | -0.012 | -0.002 | -0.001 | 0.031 | 0.001 |
| tick_1500 | -0.019 | -0.016 | -0.011 | -0.005 | 0.006 | -0.009 |

**Train R² interpretation:** The spec requires train R² > 0.05 to confirm the CNN is learning. Only tick_100 meets this threshold (mean 0.153). tick_500 is marginal (0.039). tick_1000 and tick_1500 show **negative train R²** in folds 1–4 — the CNN cannot fit its own training data. This is diagnostic of **data starvation**, not of absent signal.

**Fold 3 R² (October 2022 diagnostic):** Negative at all thresholds. tick_500 shows the mildest failure (-0.009 vs -0.047 on time_5s), but still negative. No evidence that bar type attenuates the fold 3 collapse.

**Bars per day and duration statistics:**

| Threshold | Bars/Day | Duration (s) | Total Bars | Train Size (fold 1) | Params/Train Ratio |
|-----------|----------|-------------|------------|---------------------|--------------------|
| tick_100 | 2,289 | 10.0 | 43,491 | 9,156 | 1.3:1 |
| tick_500 | 417 | 50.0 | 7,923 | 1,668 | 7.3:1 |
| tick_1000 | 183 | 100.0 | 3,477 | 732 | 16.6:1 |
| tick_1500 | 105 | 150.0 | 1,995 | 420 | 28.9:1 |
| **time_5s** | **4,630** | **5.0** | **87,970** | **~18,520** | **0.65:1** |

At tick_1000/tick_1500, the parameter-to-training-sample ratio exceeds 16:1 — the 12,128-parameter CNN has more parameters than samples in early folds. This makes negative train R² expected, independent of any signal in the data.

**R² vs. log(bar_size) scatter:** Slope = -0.0285 per log-unit. Linear monotonic decline. No peak, no plateau.

### Sanity Checks

| Check | Result | Notes |
|-------|--------|-------|
| Price tick-quantized | PASS (1.0 at all thresholds) | Tick offsets: [-12.5, +11.0] at tick_100. Normalization correct. |
| Param count | PASS (12,128 exact) | 0.0% deviation |
| LR decay | PASS | lr_start ≈ 1e-3, lr_end approaches 1e-5 at max epochs |
| No NaN | PASS | All 20 runs clean |
| Bars/day 200–5000 | **PARTIAL FAIL** | tick_1000 (183) and tick_1500 (105) below 200. Above 100 abort criterion. |
| **Zero bar-count variance** | **CRITICAL ANOMALY** | All thresholds: bars_per_day_std = 0.0. Duration p10 = p90. See Confounds. |

---

## Resource Usage

| Item | Budgeted | Actual | Notes |
|------|----------|--------|-------|
| Total wall clock | 10–15 hours | ~14 minutes (838s) | 60× faster than budget — far fewer bars than anticipated |
| Export time | 4–6 hours | ~3 minutes (3 × ~60s + 1 reused) | tick_500 reused from R4d |
| Training time | 1–2 hours | ~14 minutes | tick_100 dominates (639s). Larger thresholds train in <1 min. |
| GPU hours | 0 | 0 | CPU only, as spec'd |
| Total runs | 20 | 20 | 4 thresholds × 5 folds |

The resource budget was wildly overestimated. The spec assumed event bars would produce variable and potentially large bar counts. The actual bar counts (105–2,289/day) are much smaller than anticipated, making the experiment trivially fast.

---

## Confounds and Alternative Explanations

### CRITICAL: Bar Construction Defect — These Are NOT Event Bars

**This is the single most important finding of this experiment.** The data proves the "tick" bars from `bar_feature_export` are time bars, not event bars:

1. **Identical bar counts every day.** tick_100 produces exactly 2,289 bars on all 19 trading days — January 3 (first trading day, typically volatile), July 1 (quiet summer Friday), September 30 (quarter-end), December 5. MES daily trade volume varies by 3–5× across these dates. Genuine tick bars would show proportional variation. Zero variance is only possible with clock-based bar construction.

2. **Zero duration variance.** At every threshold, p10 = median = mean = p90. Every single bar has the same duration to the reported decimal. Real tick bars on MES would show heavy right-tail duration distribution (fast during opens, slow during lunch).

3. **Consistent pattern across all thresholds.** This is not a single-threshold anomaly. Every threshold shows the identical pattern: constant bars/day, constant duration.

4. **The math confirms snapshot counting.** tick_100 at 2,289 bars/day corresponds to a fixed session length of ~22,890 seconds divided into 10-second windows (100 snapshots at 10/s = 10s). The C++ export tool counts book-state snapshots arriving at a fixed 10/second rate from the data feed, not variable-rate trade events.

**Consequence:** The experiment compared time_10s, time_50s, time_100s, and time_150s bars against time_5s — a time-frequency sweep, not an event-bar sweep. The hypothesis that activity normalization helps CNN spatial prediction **was never tested.** The experimental variable (bar type: event vs. time) was not actually manipulated.

### Confound 2: Data Starvation at Larger Bar Sizes

| Threshold | Fold 1 Train Size | Params | Ratio | Train R² (fold 1) |
|-----------|-------------------|--------|-------|-------------------|
| tick_100 | 9,156 | 12,128 | 0.75:1 | +0.165 |
| tick_500 | 1,668 | 12,128 | 0.14:1 | +0.040 |
| tick_1000 | 732 | 12,128 | 0.06:1 | -0.009 |
| tick_1500 | 420 | 12,128 | 0.03:1 | -0.019 |

At tick_1000/tick_1500, the CNN has 16–29× more parameters than training samples in early folds. Negative train R² is expected regardless of whether the data contains signal. The comparison of tick_1000/tick_1500 against time_5s is invalid on statistical grounds alone — you cannot conclude "larger bars are worse for CNN" when the CNN physically cannot learn from the available sample size.

Even tick_100 (the "best" threshold) has half the data of time_5s (43,491 vs 87,970 bars). Some of the R² gap (0.057 vs 0.084) may be driven by data volume rather than bar-type quality.

### Confound 3: Fold 5 Dominance

Fold 5 is the only fold with positive test R² at tick_500, tick_1000, and tick_1500. Its test R² values are dramatic outliers:

| Threshold | Fold 5 Test R² | Mean of Folds 1–4 | Fold 5 as % of positive signal |
|-----------|---------------|--------------------|---------------------------------|
| tick_500 | +0.238 | -0.002 | 100% |
| tick_1000 | +0.069 | -0.017 | 100% |
| tick_1500 | +0.019 | -0.021 | 100% |
| tick_100 | +0.149 | +0.034 | dominant |

Fold 5 has the most training data (days 1–16, the largest set) and tests on days 17–19. The +0.238 at tick_500 is 5× the threshold mean — suggesting either (a) days 17–19 are uniquely predictable, (b) the full training set enables overfitting that happens to work on these specific test days, or (c) a seed-specific lucky initialization. Without fold 5, the means would be substantially negative at all thresholds except tick_100.

This inflates the reported means and makes the "monotonic down" shape partially an artifact of fold 5's decreasing leverage at larger bar sizes (where its outlier has less pull).

### Confound 4: Forward Return Scale Changes with Bar Size

fwd_return_5 means "5 bars ahead." On tick_100 (10s bars), this is a 50-second horizon. On tick_1500 (150s bars), this is a 750-second (~12.5 min) horizon. The prediction task changes qualitatively across thresholds. Lower R² at larger bar sizes could reflect that 12-minute returns are inherently harder to predict than 50-second returns, not that larger bars are worse inputs.

This is by design per the spec ("the prediction task scales with bar size"), but it means the R² vs. bar-size curve confounds input quality with target difficulty.

---

## What This Changes About Our Understanding

### What we learned (high confidence):

1. **CNN spatial R² degrades at slower time frequencies.** time_5s > time_10s > time_50s > time_100s > time_150s. The CNN's spatial signal from book snapshots is strongest at the fastest available bar rate. This makes physical sense — at 150-second bars, the book snapshot at bar close has decayed from the events that generated the bar's return.

2. **Data volume is a binding constraint for the CNN.** Below ~5,000 training samples, the 12,128-parameter CNN cannot learn. This sets a floor on bar size for any future CNN experiments — the bar type must produce enough data to train the model.

3. **The C++ `bar_feature_export` tool's "tick" bar construction is broken.** It counts fixed-rate snapshots, not trade events. Any experiment claiming to use tick/event bars from this tool actually used time bars at a different frequency. This invalidates the R4d tick_500 and tick_3000 "tick bar" data as well — those were also time bars.

### What we did NOT learn (hypothesis survives):

4. **Whether genuine activity-normalized event bars help the CNN** remains untested. The theoretical argument (event bars normalize activity regimes, making the CNN's prediction task more homogeneous) was never empirically evaluated. It may be right or wrong, but this experiment provides no evidence either way.

5. **Whether the fold 3 collapse is bar-type-driven** — we only tested different time frequencies, not different bar types. The fold 3 failure at tick_500 (-0.009) being milder than at tick_100 (-0.108) is more parsimoniously explained by data volume (fewer samples = more regularized = smaller negative R²) than by bar type.

### Impact on project direction:

The **practical decision is unchanged.** The main pipeline direction is time_5s CNN+GBT with corrected normalization. This experiment reinforces that time_5s is the right frequency — it's the fastest available bar rate, and faster is better for CNN spatial prediction. The event-bar question, while scientifically interesting, is low priority relative to fixing the CNN normalization pipeline and proceeding with the model build.

---

## Proposed Next Experiments

1. **HANDOFF (non-blocking): Fix `bar_feature_export` tick bar construction.** The tool should count actual trade events (MBO messages with `action=Trade`), not fixed-rate book snapshots. This is an infrastructure fix, not a research task. If fixed, a re-run of R3b with genuine tick bars would be the cleanest test of the event-bar hypothesis.

2. **Proceed with CNN+GBT on time_5s (high priority).** The corrected normalization pipeline (TICK_SIZE ÷ 0.25 + per-day log1p z-scoring) with proper validation (80/20 train/val split, no test leakage) should yield CNN R² ≈ 0.084. This is the main pipeline direction regardless of R3b's outcome.

3. **Multi-seed robustness study on time_5s (medium priority).** Run 5 seeds × 5 folds on time_5s with corrected normalization to confirm R² ≈ 0.084 ± [std] and characterize fold variance. This would establish the time_5s baseline with uncertainty bounds before CNN+GBT integration.

4. **If bar construction is fixed (low priority): R3b rerun.** Test 4 genuine tick-bar thresholds (calibrated to 5s, 30s, 2min, 10min actual durations, which will NOW vary by day). Use the same CNN protocol. This would properly test the event-bar hypothesis. If genuine event bars show bars_per_day std > 0 and R² > 0.101, switch to event bars. If R² is comparable or worse, time_5s is definitively vindicated.

---

## Program Status

- Questions answered this cycle: 0 (hypothesis untested due to bar construction defect)
- New questions added this cycle: 1 (genuine event-bar CNN test, blocked on bar construction fix)
- Questions remaining (open, not blocked): 1 (P2: transaction cost sensitivity)
- Questions remaining (blocked): 1 (event-bar CNN, blocked on bar_feature_export fix)
- Handoff required: **YES** — bar_feature_export tick bar construction counts snapshots, not trades
