# Analysis: R3b Genuine Tick Bars — CNN Spatial Signal on Trade-Event Tick Bars

## Verdict: CONFIRMED (low confidence)

The primary criterion (SC-4: peak mean OOS R² ≥ 0.107) passes — tick_100 achieves mean R² = 0.124. However, the evidence is statistically weak: the improvement over the time_5s baseline is not significant in a paired t-test (p ≈ 0.21, df=4), the result depends critically on fold 5's anomalously high R² = 0.259, and only 1 of 3 tested thresholds crosses the criterion. This is a positive signal that warrants replication, not a robust finding to act on.

## Results vs. Success Criteria

- [x] SC-1: bars_per_day_cv > 0 at ALL 8 thresholds — **PASS**. Range 0.188–0.467. Genuine trade-event bars confirmed. The tick-bar fix worked.
- [x] SC-2: At least 3 thresholds viable (bars_per_day mean ≥ 100) — **PASS**. 7 of 8 viable (tick_5000 excluded at 34 bars/day). 4 selected for training.
- [x] SC-3: MVE gate passes (fold 5 train R² ≥ 0.05) — **PASS**. tick_25 fold 5 train R² = 0.224.
- [x] SC-4: Peak tick-bar mean OOS R² ≥ 0.107 — **PASS** (nominally). tick_100 mean R² = 0.124. Single-threshold peak; paired t-test p ≈ 0.21; fold-5-dependent. See Confounds.
- [x] SC-5: No sanity check failures — **PASS**. All checks pass at all trained thresholds.
- [x] SC-6: R² vs bar-size curve with ≥ 3 data points — **PASS**. 3 data points produced.
- [x] No regression on sanity checks — **PASS**.

## Metric-by-Metric Breakdown

### Primary Metrics

| Metric | Value | Threshold | Assessment |
|--------|-------|-----------|------------|
| peak_tick_r2 | 0.124 (tick_100) | ≥ 0.107 | PASS |
| peak_delta | +0.035 | ≥ 0.018 | PASS |

The peak represents a 39% relative improvement over the time_5s baseline (0.089). However, the absolute delta (+0.035) is small compared to within-threshold fold variance (std = 0.107).

### Secondary Metrics

**R² by threshold (r2_by_threshold):**

| Threshold | Bars/Day | CV | Duration (med) | Folds | Mean R² | Std R² | Delta vs 0.089 | Category |
|-----------|----------|------|----------------|-------|---------|--------|-----------------|----------|
| time_5s (baseline) | 4,630 | 0.000 | 5.0s | 5/5 | 0.089 | 0.074 | — | BASELINE |
| tick_25 | 16,836 | 0.189 | 1.4s | 5/5 | 0.064 | 0.054 | -0.025 | WORSE |
| tick_100 | 4,171 | 0.190 | 5.7s | 5/5 | 0.124 | 0.107 | +0.035 | BETTER |
| tick_500 | 794 | 0.200 | 28.5s | 3/5 | 0.050 | 0.055 | -0.039 | WORSE (incomplete) |
| tick_2000 | 161 | 0.247 | 117.6s | 0/5 | — | — | — | NOT TESTED |

**Per-fold R² matrix (per_fold_r2) — tick_100 vs time_5s:**

| Fold | tick_100 | time_5s | Delta | Train R² (tick_100) | Train Size |
|------|----------|---------|-------|---------------------|------------|
| 1 | 0.198 | 0.139 | **+0.059** | 0.287 | 20,894 |
| 2 | 0.095 | 0.086 | +0.009 | 0.290 | 32,054 |
| 3 | -0.058 | -0.049 | -0.009 | 0.254 | 44,187 |
| 4 | 0.128 | 0.131 | -0.003 | 0.223 | 54,739 |
| 5 | **0.259** | 0.140 | **+0.119** | 0.248 | 67,078 |
| **Mean** | **0.124** | **0.089** | **+0.035** | **0.260** | — |

**Critical finding: Excluding fold 5, tick_100 mean R² = (0.198 + 0.095 - 0.058 + 0.128) / 4 = 0.091 — firmly in the COMPARABLE range [0.071, 0.107]. The BETTER verdict depends entirely on fold 5.**

**Paired t-test (tick_100 vs time_5s):**
- Fold deltas: [+0.059, +0.009, -0.009, -0.003, +0.119]
- Mean delta = 0.035, Std delta = 0.053
- t = 0.035 / (0.053 / sqrt(5)) = 1.48
- p = 0.21 (two-tailed, df=4)
- **Not statistically significant.**

**Per-fold train R² (per_fold_train_r2) — data adequacy check:**

| Threshold | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | All > 0.05? |
|-----------|--------|--------|--------|--------|--------|-------------|
| tick_25 | 0.314 | 0.302 | 0.233 | 0.231 | 0.224 | YES |
| tick_100 | 0.287 | 0.290 | 0.254 | 0.223 | 0.248 | YES |
| tick_500 | 0.180 | 0.215 | **0.0001** | — | — | **NO** (fold 3) |

tick_500 fold 3: train R² = 0.0001 with 8,433 training samples for 12,128 parameters (0.7 samples/param). The CNN could not learn. This fold is data-starved and invalid for comparison purposes.

**Fold 3 diagnostic (fold3_r2):**

| Threshold | Fold 3 R² | Delta vs time_5s fold 3 (-0.049) |
|-----------|-----------|----------------------------------|
| tick_25 | +0.004 | **+0.053 (eliminates deficit)** |
| tick_100 | -0.058 | -0.009 (no improvement) |
| tick_500 | -0.004 | +0.045 (but train R² = 0) |

tick_25 eliminates the Oct 2022 fold 3 weakness entirely. tick_100 does not — fold 3 remains negative, slightly worse than time_5s. This is a notable divergence: tick_25's finer granularity apparently helps in the difficult Oct 2022 regime, while tick_100 does not.

**R² std by threshold (r2_std_by_threshold):**

| Threshold | R² Std | R² CV (std/mean) |
|-----------|--------|-------------------|
| tick_25 | 0.054 | 0.85 |
| tick_100 | 0.107 | 0.86 |
| tick_500 | 0.055 | 1.09 |
| time_5s | 0.074 | 0.83 |

All thresholds have extremely high coefficient of variation (>80%). tick_100 has the highest absolute std (0.107) — its fold-to-fold results are the most variable. The improvement (+0.035) is only 0.33 standard deviations of the fold variance.

**Curve shape:** Inverted-U. R² = 0.064 -> 0.124 -> 0.050 as threshold increases (tick_25 -> tick_100 -> tick_500). Physically interpretable: too-fine bars (tick_25, ~1.4s) produce noisier returns and a harder prediction task; too-coarse bars (tick_500, ~28.5s) cause data starvation for the 12k-param CNN. Peak at tick_100 (~5.7s, ~4,171 bars/day) — near-matched to time_5s in both bar rate and forward horizon.

**Bars per day (calibration):**

| Threshold | Mean | Std | CV | Total Bars |
|-----------|------|-----|----|------------|
| tick_25 | 16,836 | 3,173 | 0.189 | 319,876 |
| tick_50 | 8,393 | 1,587 | 0.189 | 159,464 |
| tick_100 | 4,171 | 793 | 0.190 | 79,252 |
| tick_250 | 1,638 | 317 | 0.194 | 31,125 |
| tick_500 | 794 | 159 | 0.200 | 15,082 |
| tick_1000 | 372 | 79 | 0.214 | 7,062 |
| tick_2000 | 161 | 40 | 0.247 | 3,051 |
| tick_5000 | 34 | 16 | 0.467 | 646 |

All CVs in 0.189-0.467. CV increases with threshold — expected for genuine event bars (coarser thresholds amplify day-to-day activity differences). Contrast with broken R3b where ALL CVs were 0.0.

**Bar duration (bar_duration):**

| Threshold | Median | p10 | p90 | p90/p10 |
|-----------|--------|-----|-----|---------|
| tick_25 | 1.4s | 0.4s | 2.3s | 5.8x |
| tick_50 | 2.8s | 1.0s | 4.5s | 4.5x |
| tick_100 | 5.7s | 2.3s | 8.7s | 3.8x |
| tick_250 | 14.2s | 6.5s | 21.1s | 3.2x |
| tick_500 | 28.5s | 14.0s | 41.7s | 3.0x |
| tick_1000 | 57.8s | 30.8s | 82.8s | 2.7x |
| tick_2000 | 117.6s | 66.9s | 168.5s | 2.5x |
| tick_5000 | 279.6s | 173.2s | 474.5s | 2.7x |

p10 != p90 at all thresholds (ratio 2.5-5.8x). Confirms genuine within-day duration variability. Bars are shorter during active periods and longer during quiet periods.

**Total bar count (total_bar_count):** See calibration table above.

**Data volume correlation:** r = -0.149 (weak negative). Bar-type quality, not data volume, drives R² differences. tick_25 has 4x more data than time_5s but performs worse; tick_100 has 10% fewer bars than time_5s but outperforms it.

**Fold 3 improvements (fold3_improvements):** tick_25 (+0.053 vs baseline) and tick_500 (+0.045, but fold 3 train R² = 0) show improvement. tick_100 shows no fold 3 improvement (-0.009).

**Selected thresholds:** [tick_25, tick_100, tick_500, tick_2000]. Of these, tick_2000 was never reached due to wall clock budget.

### Sanity Checks

| Check | tick_25 | tick_100 | tick_500 | Status |
|-------|---------|----------|----------|--------|
| bars_per_day_cv > 0 | 0.189 | 0.190 | 0.200 | **PASS** — genuine event bars |
| p10 != p90 duration | 0.4 != 2.3 | 2.3 != 8.7 | 14.0 != 41.7 | **PASS** — not time-sampled |
| Ch0 TICK_SIZE division | [-22.5, 22.5] | [-19.5, 19.5] | [-19.5, 19.5] | **PASS** |
| Ch1 per-day z-scored | mean~0, std~1.0 | mean~0, std~1.0 | mean~0, std~1.0 | **PASS** |
| CNN param count | 12,128 | 12,128 | 12,128 | **PASS** |
| No NaN | yes | yes | yes | **PASS** |
| LR cosine schedule | 1e-3 -> 1e-5 | 1e-3 -> 1e-5 | 1e-3 -> 1e-5 | **PASS** |

All 7 sanity checks pass at all trained thresholds.

## Resource Usage

| Resource | Budget | Actual | Status |
|----------|--------|--------|--------|
| GPU-hours | 0 | 0 | OK |
| Wall clock | 7,200s (2h) | 7,331s (2h 2m) | **EXCEEDED** (+131s) |
| Training runs | 21 (1 MVE + 20 sweep) | 13 completed | 62% complete |
| Thresholds fully trained | 4 | 2 (tick_25 5/5, tick_100 5/5) | 50% complete |

Wall clock budget was calibrated for the broken R3b (14 min). Genuine tick bars at lower thresholds produce far more data than anticipated. Individual fold times ranged from 31s (tick_500 fold 2) to 1,887s (tick_100 fold 5). The budget was insufficient; 4 hours would have completed all thresholds.

**Impact of incomplete data:** tick_500 missing folds 4-5 (historically the highest-R² folds). tick_2000 not tested. The R² vs bar-size curve has only 3 points — the inverted-U shape is underdetermined.

## Confounds and Alternative Explanations

### 1. Fold 5 Dependence (CRITICAL)

The BETTER verdict rests on fold 5. At tick_100, fold 5 test R² = 0.259 — higher than any single-fold result in R3's leaked analysis (R3 mean = 0.132), and higher than its own train R² (0.248). Without fold 5, tick_100 mean R² = 0.091 (COMPARABLE).

Fold 5 anomalies:
- Test R² (0.259) > train R² (0.248) — unusual; suggests the test period (Dec 2022, days 17-19) is peculiarly favorable, not that generalization improved
- Training time 1,887s (31 min) vs ~150s for folds 1-4 — 10x slower
- 45 epochs trained (near 50-epoch max) vs 26-49 for other folds
- Largest training set (67,078 bars) — but time_5s fold 5 with comparable data produces only R² = 0.140

**Assessment:** Cannot determine if fold 5's extreme result is signal or regime coincidence with n=1.

### 2. Multiple Testing (SIGNIFICANT)

Three thresholds tested, best selected. The 0.107 threshold was calibrated for a single comparison, not best-of-3. Under the global null, P(at least 1 of 3 exceeds 0.107) > P(1 exceeds 0.107). No adjacent thresholds support the result — tick_25 and tick_500 are both WORSE. A broad peak (2+ adjacent thresholds in BETTER range) would be much stronger evidence.

### 3. Forward Return Horizon Confound (MODERATE)

fwd_return_5 = "5 bars ahead" corresponds to different clock-time horizons:

| Threshold | fwd_return_5 clock span |
|-----------|------------------------|
| tick_25 | ~7s |
| tick_100 | ~28.5s |
| time_5s | 25s |
| tick_500 | ~142s |

tick_100 (~28.5s) is a near-match to time_5s (25s) — the comparison at this threshold is relatively clean. tick_25 predicts at ~7s (noisier) and tick_500 at ~142s (longer horizon, harder per R4). The inverted-U could partly reflect horizon difficulty rather than bar-type quality.

### 4. Incomplete Sweep (MODERATE)

tick_500 has 3/5 folds (missing the highest-data folds). tick_2000 was never tested. With only 3 data points, the inverted-U shape is underdetermined. tick_50 and tick_200 (not tested with CNN) might reveal whether the peak is broad or narrow.

### 5. In-Run Code Fix (NOTABLE — protocol deviation)

The run agent modified `bar_feature_export.cpp` during the experiment to fix `StreamingBookBuilder.emit_snapshot()` not populating the `trade_count` field. This:
- Was NOT validated by a TDD cycle (protocol deviation)
- Touches shared infrastructure code that the TB-Fix TDD cycle (PR #19) was supposed to fix
- Suggests the TDD fix did not fully propagate to `bar_feature_export.cpp`'s internal `StreamingBookBuilder`
- Was validated by post-fix calibration (all 8 thresholds cv > 0, p10 != p90)
- All 8 CSVs were re-exported after the fix

The fix should be formalized with a proper TDD cycle to prevent regression.

### 6. tick_25 Underperformance (INFORMATIVE, not a confound)

tick_25 has 4x more data than time_5s but achieves only R² = 0.064 (WORSE). This:
- Rules out data-volume as the driver of R² differences
- Shows the CNN spatial signal degrades at sub-second bar rates (~1.4s duration)
- High train R² (0.224-0.314) with poor generalization = overfitting to noise at very fine granularity
- Consistent with the hypothesis that there is an optimal event-bar scale, but the WORSE performance also means tick bars are not universally better

## What This Changes About Our Understanding

**Before this experiment:**
- The "genuine tick bar" question was blocked pending the infrastructure fix
- time_5s was the only validated bar type for CNN spatial prediction (R² = 0.089)
- All prior "tick bar" results (R1, R3b, R4c, R4d) were void — they tested broken snapshot-counting bars

**After this experiment:**
- The tick-bar fix is **validated**: genuine trade-event bars produce variable daily counts (cv = 19%) and within-day duration variance (p90/p10 = 2.5-5.8x)
- tick_100 (100 trades/bar, ~5.7s median, ~4,171 bars/day) shows R² = 0.124 — a 39% relative improvement over time_5s
- **BUT** this improvement is not statistically robust (paired t p = 0.21) and depends on fold 5 (R² = 0.259, an outlier)
- The data volume confound is ruled out (r = -0.149)
- tick_25 (sub-second bars) degrades CNN spatial signal — there is an optimal bar granularity, not a monotonic benefit
- The result is promising enough to warrant replication but not reliable enough to switch the pipeline from time_5s to tick_100

**Updated mental model:** Genuine tick bars at ~100 trades/bar may offer modest improvement for CNN spatial prediction. The evidence is a positive signal, not a robust finding. The main pipeline direction (time_5s) should continue unless a properly powered replication confirms the tick_100 advantage.

**HANDOFF.md status:** The existing HANDOFF.md (requesting tick-bar fix in bar_feature_export) is now resolved. The TB-Fix TDD cycle (PR #19) fixed the library code; the run agent completed propagation to bar_feature_export.cpp. Recommend archiving to `handoffs/completed/`.

## Proposed Next Experiments

1. **Tick_100 replication with multi-seed and broader curve (if pursuing event bars).** Run tick_50, tick_100, tick_200, tick_500, tick_2000 each with 3 seeds per fold (75 total CNN fits). If tick_100's 3-seed mean still exceeds 0.107 AND at least one adjacent threshold (tick_50 or tick_200) is also BETTER, the result is robust. Budget: ~6 hours.

2. **End-to-end CNN classification on tb_label (HIGHEST PRIORITY — unchanged from 9E).** The regression-to-classification bottleneck remains the primary obstacle to economic viability regardless of bar type. Run on time_5s first.

3. **XGBoost hyperparameter tuning (P2 — unchanged from 9E).** The 2pp win rate gap to breakeven may be closable through tuning.

## Exit Criteria Audit

- [x] EC-1: Calibration complete for all 8 thresholds with bars_per_day_cv > 0
- [x] EC-2: Viability filtering applied — tick_5000 dropped (34 bars/day < 100)
- [x] EC-3: MVE gate passed — tick_25 fold 5 train R² = 0.224 >= 0.05
- [~] EC-4: CNN trained on viable thresholds x 5 folds — **PARTIAL**: tick_25 5/5, tick_100 5/5, tick_500 3/5 (wall clock), tick_2000 0/5 (not reached)
- [x] EC-5: R² vs bar-size curve produced with 3 data points
- [x] EC-6: Fold 3 diagnostic reported
- [x] EC-7: Data volume confound analysis reported (r = -0.149)
- [x] EC-8: Comparison table produced
- [x] EC-9: Decision rendered per framework — BETTER (tick_100, with low-confidence caveats)
- [x] EC-10: analysis.md written
- [x] EC-11: sweep_summary.json contains all threshold metrics

## Program Status

- Questions answered this cycle: 1 (P3 tick-bar question — low-confidence CONFIRMED)
- New questions added this cycle: 0
- Questions remaining (open, not blocked): 3 (P1 end-to-end classification, P2 cost sensitivity, P2 XGBoost tuning)
- Handoff required: NO (existing HANDOFF.md resolved; recommend archival)
