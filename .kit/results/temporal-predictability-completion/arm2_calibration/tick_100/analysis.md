# R4: Temporal Predictability — Analysis [tick_100]

**Date:** 2026-02-17  
**Bar type:** tick_100

## Table 1: Tier 1 — Pure Return AR

| Lookback | Model | return_1 | return_5 | return_20 | return_100 |
|----------|-------|----------|----------|-----------|------------|
| AR-10 | linear | -0.001593 ± 0.001866 | -0.003232 ± 0.002913 | -0.005493 ± 0.005929 | -0.023932 ± 0.032622 |
| AR-10 | ridge | -0.001575 ± 0.001844 | -0.003208 ± 0.002890 | -0.005488 ± 0.005933 | -0.023896 ± 0.032634 |
| AR-10 | gbt | -0.000634 ± 0.000626 | -0.001617 ± 0.001821 | -0.005199 ± 0.006630 | -0.023890 ± 0.031976 |
| AR-50 | linear | -0.005145 ± 0.003114 | -0.006871 ± 0.005889 | -0.010302 ± 0.005038 | -0.028684 ± 0.026171 |
| AR-50 | ridge | -0.005078 ± 0.003039 | -0.006789 ± 0.005779 | -0.010218 ± 0.004989 | -0.028542 ± 0.026143 |
| AR-50 | gbt | -0.000304 ± 0.000341 | -0.001106 ± 0.001260 | -0.004051 ± 0.004676 | -0.020968 ± 0.028290 |
| AR-100 | linear | -0.012508 ± 0.008898 | -0.014914 ± 0.014348 | -0.023141 ± 0.017882 | -0.029991 ± 0.024415 |
| AR-100 | ridge | -0.012327 ± 0.008666 | -0.014707 ± 0.014047 | -0.022882 ± 0.017525 | -0.029812 ± 0.024338 |
| AR-100 | gbt | -0.000297 ± 0.000460 | -0.000924 ± 0.001010 | -0.004194 ± 0.003992 | -0.014783 ± 0.024078 |

## Table 2: Tier 2 — Temporal Feature Augmentation (GBT)

| Config | return_1 | return_5 | return_20 | return_100 |
|--------|----------|----------|-----------|------------|
| Static-Book | 0.002221 ± 0.002434 | -0.000909 ± 0.001556 | -0.004630 ± 0.005372 | -0.019599 ± 0.030759 |
| Static-HC | 0.002126 ± 0.004014 | -0.001578 ± 0.001848 | -0.011867 ± 0.013123 | -0.028016 ± 0.030242 |
| Book+Temporal | -0.002295 ± 0.006077 | -0.001274 ± 0.002334 | -0.004510 ± 0.004040 | -0.030871 ± 0.034901 |
| HC+Temporal | 0.001793 ± 0.003805 | -0.001811 ± 0.003108 | -0.007789 ± 0.008720 | -0.025373 ± 0.030091 |
| Temporal-Only | -0.000926 ± 0.000892 | -0.000453 ± 0.001995 | -0.003893 ± 0.004270 | -0.018823 ± 0.031858 |

## Table 3: Information Gaps (GBT, Tier 2)

| Gap | Horizon | Δ_R² | 95% CI | Raw p | Corrected p | Passes? |
|-----|---------|------|--------|-------|-------------|---------|
| delta_temporal_book | 1 | -0.004516 | [-0.010944, 0.001913] | 0.1229 | 0.4917 | no |
| delta_temporal_book | 5 | -0.000365 | [-0.002225, 0.001496] | 0.6152 | 1.0000 | no |
| delta_temporal_book | 20 | 0.000120 | [-0.002219, 0.002458] | 0.8939 | 1.0000 | no |
| delta_temporal_book | 100 | -0.011272 | [-0.040718, 0.018174] | 0.3125 | 0.9375 | no |
| delta_temporal_hc | 1 | -0.000334 | [-0.002043, 0.001376] | 0.6167 | 1.0000 | no |
| delta_temporal_hc | 5 | -0.000233 | [-0.002306, 0.001839] | 0.7701 | 1.0000 | no |
| delta_temporal_hc | 20 | 0.004078 | [-0.002813, 0.010969] | 0.1757 | 0.7029 | no |
| delta_temporal_hc | 100 | 0.002643 | [-0.005551, 0.010837] | 0.4211 | 1.0000 | no |
| delta_temporal_only | 1 | -0.000926 | [-0.002164, 0.000313] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 5 | -0.000453 | [-0.003222, 0.002317] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 20 | -0.003893 | [-0.009820, 0.002035] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 100 | -0.018823 | [-0.063049, 0.025404] | 1.0000 | 1.0000 | no |
| delta_static_comparison | 1 | 0.000095 | [-0.004589, 0.004778] | 0.9580 | 0.9580 | no |
| delta_static_comparison | 5 | 0.000668 | [-0.000492, 0.001829] | 0.1849 | 0.5547 | no |
| delta_static_comparison | 20 | 0.007237 | [-0.004017, 0.018491] | 0.0625 | 0.2500 | no |
| delta_static_comparison | 100 | 0.008417 | [-0.011066, 0.027900] | 0.2965 | 0.5931 | no |

## Table 4: Feature Importance (GBT, Fold 5)

### Book+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | vol_ratio | 0.033800 | temporal |
| 2 | book_snap_33 | 0.032789 | static |
| 3 | book_snap_35 | 0.032238 | static |
| 4 | book_snap_29 | 0.031907 | static |
| 5 | book_snap_39 | 0.031360 | static |
| 6 | abs_return_lag1 | 0.029280 | temporal |
| 7 | mean_reversion_20 | 0.028569 | temporal |
| 8 | book_snap_25 | 0.028532 | static |
| 9 | book_snap_17 | 0.028078 | static |
| 10 | book_snap_11 | 0.027650 | static |

Temporal feature share: 48.3% of total importance

### Book+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_37 | 0.064164 | static |
| 2 | book_snap_9 | 0.048820 | static |
| 3 | book_snap_21 | 0.038965 | static |
| 4 | book_snap_25 | 0.038388 | static |
| 5 | book_snap_31 | 0.037718 | static |
| 6 | book_snap_33 | 0.033040 | static |
| 7 | book_snap_23 | 0.032662 | static |
| 8 | momentum_100 | 0.030126 | temporal |
| 9 | book_snap_13 | 0.029303 | static |
| 10 | rolling_vol_100 | 0.028614 | temporal |

Temporal feature share: 44.3% of total importance

### Book+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_27 | 0.048035 | static |
| 2 | book_snap_33 | 0.046744 | static |
| 3 | momentum_100 | 0.044559 | temporal |
| 4 | rolling_vol_100 | 0.042567 | temporal |
| 5 | rolling_vol_20 | 0.036126 | temporal |
| 6 | book_snap_29 | 0.036116 | static |
| 7 | book_snap_11 | 0.035491 | static |
| 8 | book_snap_25 | 0.034754 | static |
| 9 | book_snap_31 | 0.032589 | static |
| 10 | book_snap_39 | 0.032017 | static |

Temporal feature share: 43.6% of total importance

### Book+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_13 | 0.057276 | static |
| 2 | rolling_vol_100 | 0.054245 | temporal |
| 3 | book_snap_11 | 0.053400 | static |
| 4 | rolling_vol_20 | 0.052260 | temporal |
| 5 | book_snap_15 | 0.048632 | static |
| 6 | momentum_20 | 0.048525 | temporal |
| 7 | book_snap_7 | 0.047945 | static |
| 8 | momentum_100 | 0.047942 | temporal |
| 9 | book_snap_23 | 0.045318 | static |
| 10 | book_snap_1 | 0.043507 | static |

Temporal feature share: 29.3% of total importance

### HC+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | ask_depth_profile_0 | 0.021985 | static |
| 2 | momentum | 0.021475 | static |
| 3 | ask_depth_profile_6 | 0.019258 | static |
| 4 | ask_depth_profile_4 | 0.018791 | static |
| 5 | cancel_concentration | 0.018557 | static |
| 6 | order_flow_toxicity | 0.018542 | static |
| 7 | rolling_vol_20 | 0.018429 | temporal |
| 8 | ask_depth_profile_1 | 0.018155 | static |
| 9 | close_position | 0.017704 | static |
| 10 | volume_imbalance | 0.017564 | static |

Temporal feature share: 26.7% of total importance

### HC+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | bid_depth_profile_8 | 0.036589 | static |
| 2 | ask_depth_profile_5 | 0.031782 | static |
| 3 | ask_depth_profile_4 | 0.025722 | static |
| 4 | momentum_5 | 0.024563 | temporal |
| 5 | ask_depth_profile_6 | 0.024068 | static |
| 6 | rolling_vol_100 | 0.023697 | temporal |
| 7 | ask_depth_profile_2 | 0.023151 | static |
| 8 | bid_depth_profile_4 | 0.022604 | static |
| 9 | close_position | 0.022108 | static |
| 10 | minutes_since_open | 0.020628 | static |

Temporal feature share: 24.5% of total importance

### HC+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_slope_ask | 0.037496 | static |
| 2 | minutes_to_close | 0.036513 | static |
| 3 | bid_depth_profile_4 | 0.035509 | static |
| 4 | avg_trade_size | 0.035030 | static |
| 5 | time_sin | 0.034968 | static |
| 6 | ask_depth_profile_4 | 0.032895 | static |
| 7 | minutes_since_open | 0.032873 | static |
| 8 | ask_depth_profile_5 | 0.030319 | static |
| 9 | time_cos | 0.029957 | static |
| 10 | rolling_vol_20 | 0.029578 | temporal |

Temporal feature share: 19.3% of total importance

### HC+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | time_cos | 0.051974 | static |
| 2 | minutes_to_close | 0.049488 | static |
| 3 | message_rate | 0.047634 | static |
| 4 | volatility_50 | 0.047265 | static |
| 5 | modify_fraction | 0.044519 | static |
| 6 | bid_depth_profile_9 | 0.042764 | static |
| 7 | rolling_vol_20 | 0.042503 | temporal |
| 8 | cancel_concentration | 0.039048 | static |
| 9 | rolling_vol_100 | 0.036535 | temporal |
| 10 | minutes_since_open | 0.036409 | static |

Temporal feature share: 13.4% of total importance


## Tier 1 Pairwise: GBT vs. Linear

| Lookback | Horizon | GBT R² | Linear R² | Δ | Corrected p | Cohen's d |
|----------|---------|--------|-----------|---|-------------|-----------|
| AR-10 | 1 | -0.000634 | -0.001593 | 0.000960 | 1.0000 | 0.481 |
| AR-10 | 5 | -0.001617 | -0.003232 | 0.001615 | 0.8559 | 0.778 |
| AR-10 | 20 | -0.005199 | -0.005493 | 0.000294 | 1.0000 | 0.175 |
| AR-10 | 100 | -0.023890 | -0.023932 | 0.000042 | 1.0000 | 0.012 |
| AR-50 | 1 | -0.000304 | -0.005145 | 0.004841 | 0.6875 | 1.369 |
| AR-50 | 5 | -0.001106 | -0.006871 | 0.005765 | 0.6875 | 0.826 |
| AR-50 | 20 | -0.004051 | -0.010302 | 0.006251 | 0.5576 | 1.274 |
| AR-50 | 100 | -0.020968 | -0.028684 | 0.007717 | 1.0000 | 0.393 |
| AR-100 | 1 | -0.000297 | -0.012508 | 0.012211 | 0.6875 | 1.213 |
| AR-100 | 5 | -0.000924 | -0.014914 | 0.013990 | 0.6875 | 0.856 |
| AR-100 | 20 | -0.004194 | -0.023141 | 0.018947 | 0.7787 | 0.911 |
| AR-100 | 100 | -0.014783 | -0.029991 | 0.015208 | 0.8559 | 0.815 |

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
