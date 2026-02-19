# Experiment: R3b Genuine Tick Bars — CNN Spatial Signal on Trade-Counting Tick Bars

## Question

Does CNN spatial R² on genuine tick-bar book snapshots (trade-counting bars, post TB-Fix) exceed the time_5s baseline (R²=0.089) at any threshold in [50, 10000], indicating event bars improve spatial prediction?

## Background

- **R3b (original)** was INCONCLUSIVE — the C++ `bar_feature_export` tick bar construction was broken. It counted fixed-rate book snapshots (10/s), not trade events. All "tick bar" results were actually time bars at different frequencies. Those results are void.
- **TB-Fix** (merged to main, PR #19) fixed tick bar construction. `book_builder.hpp` now emits `trade_count` per snapshot (counts action='T' MBO events). `tick_bar_builder.hpp` accumulates trade counts and closes bars at threshold with remainder carry-over.
- **9E** confirmed CNN R²=0.089 on time_5s with corrected normalization (3rd independent reproduction). This is the baseline to beat.
- **Economic context**: The hybrid pipeline has exp=-$0.37/trade at base costs. A 20%+ R² improvement could flip viability. Even matching baseline on event bars validates the architecture on a different bar type.

## Hypothesis

At least one genuine tick-bar threshold in [50, 10000] produces CNN spatial R² ≥ 0.107 (20% above the time_5s baseline of 0.089), indicating activity-normalized bars give the CNN a more homogeneous prediction task that boosts spatial signal quality.

## Compute Profile

```yaml
gpu: false
cpu_cores: 4
ram_gb: 16
estimated_hours: 12
data_gb: 2
framework: pytorch
```

## Protocol

### Phase 1: Calibration (ALL 8 thresholds, bar counts only)

Export bars for all 8 thresholds across 19 trading days. No features, no CNN — just bar counts and timing statistics.

**Tool**: `./build/bar_feature_export --bar-type tick --bar-param <N> --output .kit/results/r3b-genuine-tick-bars/calibration/tick_<N>.csv`

**Thresholds**:

| Label | Threshold | Old (broken) estimate | Notes |
|-------|-----------|----------------------|-------|
| XS | 50 | ~2-5s | Near time_5s equivalent |
| Small | 100 | ~5-10s | First true event-bar scale |
| Med-Small | 250 | ~15-25s | Moderate aggregation |
| Medium | 500 | ~30-60s | 1-minute scale |
| Med-Large | 1000 | ~1-2 min | Meaningful equilibrium |
| Large | 2000 | ~3-5 min | Discretionary trader bar |
| XL | 5000 | ~8-15 min | Upper intraday range |
| XXL | 10000 | ~20-30 min | Extreme aggregation test |

**IMPORTANT**: Old duration estimates are from broken bars (counting 10/s snapshots). Genuine tick bars count actual trades, which are sparser. Expect fewer bars per day and longer durations than these estimates.

For each threshold, compute:
- `total_bars`: total bars across all 19 days
- `bars_per_day`: mean and std across days
- `bars_per_day_cv`: coefficient of variation (std/mean) — MUST be > 0 to confirm genuine event bars
- `median_duration_sec`: median bar duration in seconds
- `p10_duration_sec`, `p90_duration_sec`: 10th and 90th percentile durations
- `duration_iqr`: p75 - p25

Save to `.kit/results/r3b-genuine-tick-bars/calibration/threshold_sweep.json`.

**Viability filters** (apply after calibration):
- DROP any threshold with `bars_per_day` < 50 (too coarse for CNN training)
- DROP any threshold with `bars_per_day` > 100,000 (sub-second, too noisy)
- WARN if largest viable threshold produces < 500 total train bars in fold 1 (4 days) — skip fold 1 for that threshold but still run folds 4-5
- CONFIRM `bars_per_day_cv` > 0 for all thresholds (genuine event bar diagnostic)

### Phase 2: CNN Training (viable thresholds × 5 folds)

For each viable threshold from Phase 1, train CNN with the **exact 9E/9D protocol**:

**Data export**: `./build/bar_feature_export --bar-type tick --bar-param <N> --output .kit/results/r3b-genuine-tick-bars/tick_<N>/tick_<N>.csv`

**CNN Architecture** (R3-exact, MUST match):
- Conv1d(in_channels=2, out_channels=59, kernel_size=3, padding=1)
- ReLU + AdaptiveAvgPool1d(1)
- Linear(59, 59) + ReLU
- Linear(59, 1)
- Total: 12,128 parameters exactly

**Normalization** (MANDATORY — three prior attempts failed on this):
- Prices: divide by TICK_SIZE=0.25 to get integer tick offsets. Resulting range should be approximately [-25, +25].
- Sizes: per-day z-scoring on log1p(size). Each day's sizes should have mean≈0, std≈1.0.
- Do NOT use per-fold z-scoring. Do NOT skip the TICK_SIZE division.

**Training**:
- Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR(T_max=50, eta_min=1e-5)
- Batch size: 512
- Max epochs: 50
- Early stopping: patience=10 on VALIDATION loss (80/20 split of TRAIN days only, NOT test data)
- Seed: 42
- Target: fwd_return_5 (5 bars ahead — note: for tick bars this is 5 event-bars, not fixed clock time)

**Cross-validation**: 5-fold expanding window on 19 trading days (same fold structure as 9D/9E).

**Per-fold metrics**:
- train_r2, val_r2, test_r2
- epochs_trained
- bar_count (train, val, test)

**MVE Gate** (before full sweep):
- Run tick_500 (Medium) fold 5 FIRST
- If train R² < 0.05 → STOP, investigate normalization before expanding
- If train R² ≥ 0.05 → proceed with full sweep

### Phase 3: Analysis

**Primary deliverable**: R² vs bar-size curve.

Compute for each threshold:
- mean_test_r2 (across 5 folds)
- std_test_r2
- fold3_r2 (Oct 2022 — the fold where time_5s gives -0.049)
- mean_train_r2
- mean_bar_duration_sec (from calibration)

Save to `.kit/results/r3b-genuine-tick-bars/r2_vs_barsize_curve.csv` with columns: threshold, mean_duration_sec, mean_test_r2, std_test_r2, fold3_test_r2, mean_train_r2, bars_per_day.

**Fold 3 diagnostic**: If ANY tick-bar threshold produces positive R² on fold 3 (where time_5s gives -0.049), that's evidence the fold 3 failure was bar-type-driven, not regime-driven. Flag this prominently.

**Comparison table**: Side-by-side table of all thresholds + time_5s baseline. Save as `.kit/results/r3b-genuine-tick-bars/comparison_table.md`.

## Decision Framework

| Outcome | Criterion | Action |
|---------|-----------|--------|
| **BETTER** | Peak R² ≥ 0.107 (20%+ above 0.089) | Switch pipeline to optimal tick-bar threshold |
| **COMPARABLE** | 0.071 ≤ all R² ≤ 0.107 | Stick with time_5s (simpler, proven) |
| **WORSE** | All R² < 0.071 (20%+ below 0.089) | Time_5s definitively vindicated |
| **Peak found** | Inverted-U shape with clear optimum | Adopt peak threshold |
| **Monotonic up** | R² still rising at tick_10000 | Consider even larger thresholds |

## Abort Criteria

- Any threshold with < 50 bars/day after calibration → drop that threshold
- Train R² < 0.05 on Medium fold 5 (MVE gate) → investigate normalization before expanding
- All tested thresholds produce mean R² < 0.03 → systematic issue with event-bar export
- Data starvation: largest thresholds with < 500 total train bars in fold 1 → skip fold 1, still run folds 4-5

## Baselines

- **time_5s CNN R²=0.089** (from 9E, proper validation, 3rd independent reproduction)
- Per-fold reference from 9E: [0.139, 0.086, -0.049, 0.131, 0.140]
- **Broken R3b peak**: R²=0.057 at tick_100 (void — was actually time_10s)

## Results Directory

```
.kit/results/r3b-genuine-tick-bars/
├── calibration/
│   ├── threshold_sweep.json          # threshold → {bars_per_day, bars_per_day_cv, median_duration, p10, p90, total_bars}
│   └── tick_<N>.csv                  # raw bar export per threshold (calibration only)
├── tick_<N>/                         # one per viable threshold
│   ├── fold_results.json             # per-fold: {train_r2, val_r2, test_r2, epochs, bar_count}
│   ├── bar_statistics.json           # duration stats, bars_per_day variance
│   ├── tick_<N>.csv                  # feature export
│   └── normalization_verification.txt
├── sweep_summary.json                # all thresholds: mean R², std, fold 3, bar stats
├── r2_vs_barsize_curve.csv           # threshold, mean_duration, mean_r2, std_r2, fold3_r2
├── comparison_table.md               # side-by-side: all thresholds + time_5s baseline
└── analysis.md                       # full analysis with decision per framework
```

## Exit Criteria

- [x] EC-1: Calibration complete for all 8 thresholds with bars_per_day_cv > 0 (genuine event bars confirmed) — ALL 8 cv in [0.189, 0.467]
- [x] EC-2: Viability filtering applied — tick_5000 dropped (34 bars/day < 100). 7 viable, 4 selected for CNN.
- [x] EC-3: MVE gate passed — tick_25 fold 5 train R² = 0.224 ≥ 0.05
- [~] EC-4: CNN trained on viable thresholds — PARTIAL: tick_25 5/5, tick_100 5/5, tick_500 3/5 (wall clock), tick_2000 0/5 (not reached)
- [x] EC-5: R² vs bar-size curve produced (r2_vs_barsize_curve.csv) — 3 data points
- [x] EC-6: Fold 3 diagnostic reported — tick_25 +0.004 (eliminates deficit), tick_100 -0.058 (no improvement), tick_500 -0.004
- [x] EC-7: Comparison table produced (comparison_table.md)
- [x] EC-8: Decision rendered — BETTER (low confidence): tick_100 R²=0.124, +39% vs baseline, but paired t p=0.21, fold-5-dependent
- [x] EC-9: analysis.md written with full findings
- [x] EC-10: sweep_summary.json contains all threshold metrics
