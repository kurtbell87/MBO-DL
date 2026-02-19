# R3b Event Bar CNN — Comparison Table

## Mean OOS R² Across 5-Fold Expanding Window CV

| Threshold | Duration | Total Bars | Bars/Day | Mean R² | Std R² | Fold 3 R² | Assessment |
|-----------|----------|------------|----------|---------|--------|-----------|------------|
| time_5s (baseline) | 5.0s | 87,970 | 4,630 | **0.084** | ~0.048 | -0.047 | — |
| tick_100 | 10.0s | 43,491 | 2,289 | 0.057 | 0.104 | -0.108 | WORSE |
| tick_500 | 50.0s | 7,923 | 417 | 0.047 | 0.096 | -0.009 | WORSE |
| tick_1000 | 100.0s | 3,477 | 183 | -0.003 | 0.038 | -0.036 | WORSE |
| tick_1500 | 150.0s | 1,995 | 105 | -0.022 | 0.030 | -0.072 | WORSE |

## Per-Fold Test R²

| Threshold | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-----------|--------|--------|--------|--------|--------|
| tick_100 | 0.181 | 0.060 | -0.108 | 0.001 | 0.149 |
| tick_500 | 0.018 | 0.000 | -0.009 | -0.014 | 0.238 |
| tick_1000 | -0.020 | 0.000 | -0.036 | -0.026 | 0.069 |
| tick_1500 | -0.026 | -0.006 | -0.072 | -0.025 | 0.019 |

## Per-Fold Train R²

| Threshold | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|-----------|--------|--------|--------|--------|--------|
| tick_100 | 0.165 | 0.198 | 0.178 | 0.030 | 0.195 |
| tick_500 | 0.040 | -0.005 | 0.017 | 0.000 | 0.143 |
| tick_1000 | -0.009 | -0.012 | -0.002 | -0.001 | 0.031 |
| tick_1500 | -0.019 | -0.016 | -0.011 | -0.005 | 0.006 |

## Decision Per Spec Framework

- **BETTER** threshold: Mean R² >= 0.101 (20% above baseline)
- **COMPARABLE** threshold: 0.068 <= Mean R² < 0.101
- **WORSE** threshold: Mean R² < 0.068

All 4 thresholds: **WORSE**.

Cross-threshold outcome: **All WORSE: time_5s beats every tick-bar threshold.**

## R² vs Bar Size Curve

Shape: **Monotonic down**. R² decreases as bar size increases.
Slope (R² vs log threshold): -0.029 per log unit.

## Observation

Bar counts per day are identical across all 19 trading days for each threshold (std = 0.0):
- tick_100: 2,289 bars every day
- tick_500: 417 bars every day
- tick_1000: 183 bars every day
- tick_1500: 105 bars every day

Bar durations show zero variance (p10 = median = p90). This is consistent with the C++ export tool counting fixed-frequency book snapshots (10/second) rather than variable-rate trade events.
