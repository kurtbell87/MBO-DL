# R4: Temporal Predictability — Analysis [tick_50]

**Date:** 2026-02-17  
**Bar type:** tick_50

## Table 1: Tier 1 — Pure Return AR

| Lookback | Model | return_1 | return_5 | return_20 | return_100 |
|----------|-------|----------|----------|-----------|------------|
| AR-10 | linear | -0.000972 ± 0.000434 | -0.002322 ± 0.001496 | -0.002568 ± 0.002626 | -0.012200 ± 0.016927 |
| AR-10 | ridge | -0.000966 ± 0.000432 | -0.002311 ± 0.001487 | -0.002565 ± 0.002627 | -0.012196 ± 0.016928 |
| AR-10 | gbt | -0.000186 ± 0.000225 | -0.001679 ± 0.001866 | -0.002326 ± 0.002728 | -0.014597 ± 0.015308 |
| AR-50 | linear | -0.002842 ± 0.001531 | -0.004673 ± 0.003656 | -0.004197 ± 0.002954 | -0.018221 ± 0.016336 |
| AR-50 | ridge | -0.002823 ± 0.001510 | -0.004646 ± 0.003622 | -0.004188 ± 0.002952 | -0.018171 ± 0.016326 |
| AR-50 | gbt | -0.000517 ± 0.000571 | -0.001774 ± 0.001947 | -0.002690 ± 0.003656 | -0.011218 ± 0.015161 |
| AR-100 | linear | -0.005218 ± 0.003189 | -0.007274 ± 0.005955 | -0.007180 ± 0.003591 | -0.033016 ± 0.023228 |
| AR-100 | ridge | -0.005180 ± 0.003147 | -0.007229 ± 0.005897 | -0.007147 ± 0.003554 | -0.032847 ± 0.023030 |
| AR-100 | gbt | -0.000349 ± 0.000426 | -0.000641 ± 0.000983 | -0.002480 ± 0.003005 | -0.010859 ± 0.012840 |

## Table 2: Tier 2 — Temporal Feature Augmentation (GBT)

| Config | return_1 | return_5 | return_20 | return_100 |
|--------|----------|----------|-----------|------------|
| Static-Book | -0.002970 ± 0.021455 | -0.000707 ± 0.001135 | -0.003582 ± 0.003580 | -0.014171 ± 0.014977 |
| Static-HC | 0.000967 ± 0.009846 | -0.000418 ± 0.001700 | -0.006052 ± 0.002743 | -0.020460 ± 0.015234 |
| Book+Temporal | -0.001441 ± 0.016785 | -0.000637 ± 0.000968 | -0.002317 ± 0.003667 | -0.013195 ± 0.018442 |
| HC+Temporal | 0.000607 ± 0.012036 | -0.000861 ± 0.000995 | -0.004638 ± 0.002494 | -0.015075 ± 0.015530 |
| Temporal-Only | -0.000003 ± 0.000325 | -0.000453 ± 0.000958 | -0.004315 ± 0.006493 | -0.012977 ± 0.019336 |

## Table 3: Information Gaps (GBT, Tier 2)

| Gap | Horizon | Δ_R² | 95% CI | Raw p | Corrected p | Passes? |
|-----|---------|------|--------|-------|-------------|---------|
| delta_temporal_book | 1 | 0.001529 | [-0.005050, 0.008108] | 1.0000 | 1.0000 | no |
| delta_temporal_book | 5 | 0.000070 | [-0.000783, 0.000923] | 0.6250 | 1.0000 | no |
| delta_temporal_book | 20 | 0.001265 | [-0.002730, 0.005260] | 1.0000 | 1.0000 | no |
| delta_temporal_book | 100 | 0.000975 | [-0.010162, 0.012112] | 0.8199 | 1.0000 | no |
| delta_temporal_hc | 1 | -0.000360 | [-0.007421, 0.006701] | 0.8943 | 1.0000 | no |
| delta_temporal_hc | 5 | -0.000443 | [-0.002534, 0.001649] | 0.5883 | 1.0000 | no |
| delta_temporal_hc | 20 | 0.001415 | [-0.001791, 0.004620] | 0.2876 | 0.8629 | no |
| delta_temporal_hc | 100 | 0.005385 | [-0.002188, 0.012958] | 0.1196 | 0.4784 | no |
| delta_temporal_only | 1 | -0.000003 | [-0.000455, 0.000449] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 5 | -0.000453 | [-0.001783, 0.000878] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 20 | -0.004315 | [-0.013329, 0.004699] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 100 | -0.012977 | [-0.039819, 0.013865] | 1.0000 | 1.0000 | no |
| delta_static_comparison | 1 | -0.003937 | [-0.023570, 0.015697] | 1.0000 | 1.0000 | no |
| delta_static_comparison | 5 | -0.000289 | [-0.001278, 0.000701] | 0.4634 | 0.9267 | no |
| delta_static_comparison | 20 | 0.002470 | [-0.000600, 0.005541] | 0.0893 | 0.2793 | no |
| delta_static_comparison | 100 | 0.006289 | [-0.000814, 0.013393] | 0.0698 | 0.2793 | no |

## Table 4: Feature Importance (GBT, Fold 5)

### Book+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_19 | 0.036678 | static |
| 2 | book_snap_31 | 0.036277 | static |
| 3 | book_snap_21 | 0.031865 | static |
| 4 | book_snap_33 | 0.031188 | static |
| 5 | lag_return_2 | 0.030065 | temporal |
| 6 | vol_ratio | 0.029791 | temporal |
| 7 | book_snap_29 | 0.029110 | static |
| 8 | lag_return_3 | 0.029023 | temporal |
| 9 | book_snap_23 | 0.028440 | static |
| 10 | book_snap_39 | 0.027607 | static |

Temporal feature share: 49.8% of total importance

### Book+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_33 | 0.039905 | static |
| 2 | rolling_vol_20 | 0.039321 | temporal |
| 3 | rolling_vol_5 | 0.036527 | temporal |
| 4 | rolling_vol_100 | 0.033361 | temporal |
| 5 | momentum_20 | 0.032077 | temporal |
| 6 | vol_ratio | 0.030663 | temporal |
| 7 | book_snap_37 | 0.029950 | static |
| 8 | book_snap_23 | 0.029253 | static |
| 9 | book_snap_29 | 0.029160 | static |
| 10 | momentum_100 | 0.029073 | temporal |

Temporal feature share: 52.5% of total importance

### Book+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | momentum_100 | 0.055437 | temporal |
| 2 | book_snap_25 | 0.048345 | static |
| 3 | book_snap_27 | 0.045779 | static |
| 4 | vol_ratio | 0.045688 | temporal |
| 5 | rolling_vol_100 | 0.045472 | temporal |
| 6 | momentum_20 | 0.045259 | temporal |
| 7 | rolling_vol_5 | 0.044903 | temporal |
| 8 | book_snap_31 | 0.042558 | static |
| 9 | abs_return_lag1 | 0.039933 | temporal |
| 10 | book_snap_33 | 0.038923 | static |

Temporal feature share: 45.6% of total importance

### Book+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | momentum_100 | 0.098425 | temporal |
| 2 | rolling_vol_100 | 0.070020 | temporal |
| 3 | book_snap_1 | 0.052048 | static |
| 4 | book_snap_11 | 0.050900 | static |
| 5 | book_snap_35 | 0.046402 | static |
| 6 | book_snap_37 | 0.045053 | static |
| 7 | book_snap_23 | 0.044795 | static |
| 8 | book_snap_33 | 0.038696 | static |
| 9 | book_snap_13 | 0.037867 | static |
| 10 | rolling_vol_20 | 0.037523 | temporal |

Temporal feature share: 33.5% of total importance

### HC+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | ask_depth_profile_3 | 0.035681 | static |
| 2 | close_position | 0.022994 | static |
| 3 | book_imbalance_1 | 0.021112 | static |
| 4 | bid_depth_profile_0 | 0.018667 | static |
| 5 | ask_depth_profile_8 | 0.018453 | static |
| 6 | weighted_imbalance | 0.018065 | static |
| 7 | vol_ratio | 0.018010 | temporal |
| 8 | net_volume | 0.017943 | static |
| 9 | time_sin | 0.016230 | static |
| 10 | rolling_vol_5 | 0.016190 | temporal |

Temporal feature share: 26.6% of total importance

### HC+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | minutes_to_close | 0.027642 | static |
| 2 | ask_depth_profile_6 | 0.026424 | static |
| 3 | time_cos | 0.025395 | static |
| 4 | time_sin | 0.024897 | static |
| 5 | volatility_20 | 0.024653 | static |
| 6 | rolling_vol_20 | 0.021948 | temporal |
| 7 | bid_depth_profile_7 | 0.021268 | static |
| 8 | ask_depth_profile_1 | 0.021182 | static |
| 9 | bid_depth_profile_4 | 0.020758 | static |
| 10 | high_low_range_50 | 0.020156 | static |

Temporal feature share: 26.9% of total importance

### HC+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_slope_ask | 0.037665 | static |
| 2 | momentum | 0.036322 | static |
| 3 | time_cos | 0.034509 | static |
| 4 | high_low_range_50 | 0.032689 | static |
| 5 | return_20 | 0.032093 | static |
| 6 | minutes_since_open | 0.031761 | static |
| 7 | close_position | 0.030747 | static |
| 8 | momentum_100 | 0.030590 | temporal |
| 9 | ask_depth_profile_1 | 0.029388 | static |
| 10 | ask_depth_profile_6 | 0.029271 | static |

Temporal feature share: 13.1% of total importance

### HC+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | minutes_to_close | 0.067485 | static |
| 2 | bid_depth_profile_8 | 0.055078 | static |
| 3 | momentum_100 | 0.052895 | temporal |
| 4 | ask_depth_profile_6 | 0.049330 | static |
| 5 | rolling_vol_100 | 0.035192 | temporal |
| 6 | time_cos | 0.033933 | static |
| 7 | time_sin | 0.033888 | static |
| 8 | bid_depth_profile_3 | 0.033096 | static |
| 9 | minutes_since_open | 0.031442 | static |
| 10 | ask_depth_profile_4 | 0.031310 | static |

Temporal feature share: 10.4% of total importance


## Tier 1 Pairwise: GBT vs. Linear

| Lookback | Horizon | GBT R² | Linear R² | Δ | Corrected p | Cohen's d |
|----------|---------|--------|-----------|---|-------------|-----------|
| AR-10 | 1 | -0.000186 | -0.000972 | 0.000785 | 0.1292 | 2.016 |
| AR-10 | 5 | -0.001679 | -0.002322 | 0.000642 | 0.7756 | 0.585 |
| AR-10 | 20 | -0.002326 | -0.002568 | 0.000242 | 0.7756 | 0.435 |
| AR-10 | 100 | -0.014597 | -0.012200 | -0.002397 | 0.7756 | -0.589 |
| AR-50 | 1 | -0.000517 | -0.002842 | 0.002325 | 0.5499 | 1.174 |
| AR-50 | 5 | -0.001774 | -0.004673 | 0.002899 | 0.5026 | 1.281 |
| AR-50 | 20 | -0.002690 | -0.004197 | 0.001508 | 0.5499 | 1.200 |
| AR-50 | 100 | -0.011218 | -0.018221 | 0.007003 | 0.5499 | 1.192 |
| AR-100 | 1 | -0.000349 | -0.005218 | 0.004869 | 0.5499 | 1.357 |
| AR-100 | 5 | -0.000641 | -0.007274 | 0.006633 | 0.5499 | 1.021 |
| AR-100 | 20 | -0.002480 | -0.007180 | 0.004700 | 0.6188 | 0.870 |
| AR-100 | 100 | -0.010859 | -0.033016 | 0.022157 | 0.6188 | 0.794 |

## Decision Rules

### rule1_ar_structure
- **Passes**: False
- **Interpretation**: Returns are martingale at horizons > 5s
- **Passing configs**: []

### rule2_temporal_augmentation
- **Passes**: False
- **Interpretation**: Current-bar features sufficient. Drop temporal encoder/SSM.
- **Passing horizons**: []

### rule3_temporal_only
- **Passes**: False
- **Interpretation**: No temporal signal. Martingale confirmed.
- **Passing horizons**: []

### rule4_reconciliation
- **Passes**: N/A
- **Interpretation**: R4 confirms R2: temporal encoder adds no value for MES at 5s bars. Strong recommendation to drop SSM.

## Summary Finding

**NO TEMPORAL SIGNAL**: Returns are a martingale difference sequence at the 5s scale. Drop SSM. Converges with R2 Δ_temporal finding.
