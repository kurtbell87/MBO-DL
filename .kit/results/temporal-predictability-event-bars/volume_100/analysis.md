# R4: Temporal Predictability — Analysis [volume_100]

**Date:** 2026-02-17  
**Bar type:** volume_100

## Table 1: Tier 1 — Pure Return AR

| Lookback | Model | return_1 | return_5 | return_20 | return_100 |
|----------|-------|----------|----------|-----------|------------|
| AR-10 | linear | -0.000878 ± 0.000527 | -0.001794 ± 0.001611 | -0.002329 ± 0.002377 | -0.008928 ± 0.013022 |
| AR-10 | ridge | -0.000874 ± 0.000526 | -0.001787 ± 0.001611 | -0.002326 ± 0.002376 | -0.008927 ± 0.013022 |
| AR-10 | gbt | -0.000306 ± 0.000262 | -0.000540 ± 0.000493 | -0.003055 ± 0.002947 | -0.010658 ± 0.011648 |
| AR-50 | linear | -0.002234 ± 0.000923 | -0.002830 ± 0.001642 | -0.003983 ± 0.003037 | -0.011526 ± 0.012981 |
| AR-50 | ridge | -0.002223 ± 0.000920 | -0.002815 ± 0.001634 | -0.003974 ± 0.003035 | -0.011511 ± 0.012981 |
| AR-50 | gbt | -0.000135 ± 0.000151 | -0.000861 ± 0.000669 | -0.003579 ± 0.006241 | -0.010558 ± 0.012544 |
| AR-100 | linear | -0.004077 ± 0.002314 | -0.005236 ± 0.004090 | -0.005710 ± 0.003389 | -0.023357 ± 0.016757 |
| AR-100 | ridge | -0.004054 ± 0.002291 | -0.005207 ± 0.004056 | -0.005691 ± 0.003380 | -0.023274 ± 0.016713 |
| AR-100 | gbt | -0.000222 ± 0.000237 | -0.000776 ± 0.000917 | -0.001775 ± 0.002430 | -0.012247 ± 0.017456 |

## Table 2: Tier 2 — Temporal Feature Augmentation (GBT)

| Config | return_1 | return_5 | return_20 | return_100 |
|--------|----------|----------|-----------|------------|
| Static-Book | 0.005581 ± 0.005714 | -0.000006 ± 0.001115 | -0.002507 ± 0.002593 | -0.010912 ± 0.011426 |
| Static-HC | 0.007732 ± 0.004876 | -0.000136 ± 0.001449 | -0.004602 ± 0.005117 | -0.016211 ± 0.013950 |
| Book+Temporal | 0.005624 ± 0.003378 | -0.000596 ± 0.001058 | -0.006892 ± 0.007329 | -0.010301 ± 0.012828 |
| HC+Temporal | 0.007261 ± 0.003215 | -0.000286 ± 0.002128 | -0.004920 ± 0.003999 | -0.012243 ± 0.014960 |
| Temporal-Only | 0.000088 ± 0.000434 | -0.000810 ± 0.000604 | -0.002174 ± 0.003359 | -0.012995 ± 0.017164 |

## Table 3: Information Gaps (GBT, Tier 2)

| Gap | Horizon | Δ_R² | 95% CI | Raw p | Corrected p | Passes? |
|-----|---------|------|--------|-------|-------------|---------|
| delta_temporal_book | 1 | 0.000043 | [-0.003293, 0.003380] | 0.6250 | 1.0000 | no |
| delta_temporal_book | 5 | -0.000589 | [-0.001865, 0.000686] | 0.2686 | 1.0000 | no |
| delta_temporal_book | 20 | -0.004384 | [-0.013994, 0.005225] | 0.4375 | 1.0000 | no |
| delta_temporal_book | 100 | 0.000611 | [-0.003831, 0.005053] | 0.7219 | 1.0000 | no |
| delta_temporal_hc | 1 | -0.000471 | [-0.003335, 0.002393] | 0.6716 | 1.0000 | no |
| delta_temporal_hc | 5 | -0.000149 | [-0.002208, 0.001910] | 0.8504 | 1.0000 | no |
| delta_temporal_hc | 20 | -0.000318 | [-0.004195, 0.003558] | 0.8309 | 1.0000 | no |
| delta_temporal_hc | 100 | 0.003969 | [-0.013765, 0.021702] | 0.5680 | 1.0000 | no |
| delta_temporal_only | 1 | 0.000088 | [-0.000514, 0.000691] | 0.3522 | 1.0000 | no |
| delta_temporal_only | 5 | -0.000810 | [-0.001648, 0.000028] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 20 | -0.002174 | [-0.006837, 0.002489] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 100 | -0.012995 | [-0.036823, 0.010833] | 1.0000 | 1.0000 | no |
| delta_static_comparison | 1 | -0.002151 | [-0.004298, -0.000005] | 0.0497 | 0.1988 | no |
| delta_static_comparison | 5 | 0.000130 | [-0.001235, 0.001495] | 0.8045 | 1.0000 | no |
| delta_static_comparison | 20 | 0.002095 | [-0.003783, 0.007973] | 0.6250 | 1.0000 | no |
| delta_static_comparison | 100 | 0.005300 | [-0.006746, 0.017345] | 0.2889 | 0.8668 | no |

## Table 4: Feature Importance (GBT, Fold 5)

### Book+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_19 | 0.039853 | static |
| 2 | book_snap_21 | 0.038212 | static |
| 3 | book_snap_27 | 0.030134 | static |
| 4 | rolling_vol_100 | 0.029850 | temporal |
| 5 | vol_ratio | 0.029602 | temporal |
| 6 | rolling_vol_20 | 0.029414 | temporal |
| 7 | book_snap_35 | 0.029381 | static |
| 8 | lag_return_7 | 0.028854 | temporal |
| 9 | book_snap_7 | 0.028241 | static |
| 10 | lag_return_6 | 0.027734 | temporal |

Temporal feature share: 49.9% of total importance

### Book+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_33 | 0.050231 | static |
| 2 | lag_return_2 | 0.041686 | temporal |
| 3 | momentum_20 | 0.039161 | temporal |
| 4 | lag_return_1 | 0.037096 | temporal |
| 5 | rolling_vol_100 | 0.034562 | temporal |
| 6 | book_snap_23 | 0.032845 | static |
| 7 | momentum_100 | 0.032422 | temporal |
| 8 | book_snap_15 | 0.032326 | static |
| 9 | lag_return_4 | 0.032308 | temporal |
| 10 | momentum_5 | 0.032063 | temporal |

Temporal feature share: 53.8% of total importance

### Book+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_33 | 0.059034 | static |
| 2 | rolling_vol_20 | 0.052402 | temporal |
| 3 | rolling_vol_100 | 0.051206 | temporal |
| 4 | book_snap_29 | 0.046987 | static |
| 5 | momentum_20 | 0.043619 | temporal |
| 6 | momentum_100 | 0.041582 | temporal |
| 7 | book_snap_27 | 0.037258 | static |
| 8 | book_snap_17 | 0.034142 | static |
| 9 | book_snap_11 | 0.033840 | static |
| 10 | book_snap_23 | 0.031600 | static |

Temporal feature share: 44.6% of total importance

### Book+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | rolling_vol_100 | 0.089432 | temporal |
| 2 | book_snap_11 | 0.081096 | static |
| 3 | book_snap_27 | 0.070553 | static |
| 4 | momentum_100 | 0.067245 | temporal |
| 5 | rolling_vol_20 | 0.055352 | temporal |
| 6 | book_snap_29 | 0.053055 | static |
| 7 | book_snap_13 | 0.049540 | static |
| 8 | book_snap_15 | 0.046708 | static |
| 9 | book_snap_9 | 0.041679 | static |
| 10 | rolling_vol_5 | 0.039622 | temporal |

Temporal feature share: 32.6% of total importance

### HC+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | volume_imbalance | 0.026619 | static |
| 2 | book_imbalance_1 | 0.026564 | static |
| 3 | weighted_imbalance | 0.021458 | static |
| 4 | message_rate | 0.020581 | static |
| 5 | modify_fraction | 0.020573 | static |
| 6 | volatility_50 | 0.020150 | static |
| 7 | vol_price_corr | 0.019979 | static |
| 8 | close_position | 0.019795 | static |
| 9 | depth_concentration_ask | 0.019645 | static |
| 10 | volatility_20 | 0.019201 | static |

Temporal feature share: 26.0% of total importance

### HC+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | minutes_since_open | 0.053762 | static |
| 2 | high_low_range_50 | 0.022480 | static |
| 3 | trade_count | 0.022145 | static |
| 4 | time_sin | 0.021477 | static |
| 5 | rolling_vol_20 | 0.020719 | temporal |
| 6 | rolling_vol_100 | 0.019427 | temporal |
| 7 | ask_depth_profile_5 | 0.018807 | static |
| 8 | avg_trade_size | 0.018683 | static |
| 9 | time_cos | 0.018537 | static |
| 10 | volatility_50 | 0.018348 | static |

Temporal feature share: 27.6% of total importance

### HC+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | ask_depth_profile_3 | 0.038847 | static |
| 2 | time_sin | 0.037605 | static |
| 3 | high_low_range_50 | 0.035006 | static |
| 4 | vol_price_corr | 0.034447 | static |
| 5 | rolling_vol_100 | 0.032063 | temporal |
| 6 | time_cos | 0.030030 | static |
| 7 | cancel_concentration | 0.029829 | static |
| 8 | minutes_since_open | 0.029405 | static |
| 9 | close_position | 0.028056 | static |
| 10 | ask_depth_profile_6 | 0.027949 | static |

Temporal feature share: 21.4% of total importance

### HC+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | volatility_50 | 0.074920 | static |
| 2 | rolling_vol_100 | 0.064898 | temporal |
| 3 | time_cos | 0.060402 | static |
| 4 | time_sin | 0.055479 | static |
| 5 | momentum_100 | 0.051220 | temporal |
| 6 | volatility_20 | 0.051011 | static |
| 7 | minutes_since_open | 0.050437 | static |
| 8 | cancel_concentration | 0.049652 | static |
| 9 | modify_fraction | 0.044333 | static |
| 10 | bid_depth_profile_3 | 0.044029 | static |

Temporal feature share: 15.2% of total importance


## Tier 1 Pairwise: GBT vs. Linear

| Lookback | Horizon | GBT R² | Linear R² | Δ | Corrected p | Cohen's d |
|----------|---------|--------|-----------|---|-------------|-----------|
| AR-10 | 1 | -0.000306 | -0.000878 | 0.000572 | 0.5192 | 1.130 |
| AR-10 | 5 | -0.000540 | -0.001794 | 0.001254 | 0.9390 | 0.710 |
| AR-10 | 20 | -0.003055 | -0.002329 | -0.000726 | 1.0000 | -0.411 |
| AR-10 | 100 | -0.010658 | -0.008928 | -0.001730 | 1.0000 | -0.534 |
| AR-50 | 1 | -0.000135 | -0.002234 | 0.002099 | 0.0850 | 2.272 |
| AR-50 | 5 | -0.000861 | -0.002830 | 0.001969 | 0.4985 | 1.197 |
| AR-50 | 20 | -0.003579 | -0.003983 | 0.000404 | 1.0000 | 0.112 |
| AR-50 | 100 | -0.010558 | -0.011526 | 0.000969 | 1.0000 | 0.226 |
| AR-100 | 1 | -0.000222 | -0.004077 | 0.003855 | 0.2862 | 1.497 |
| AR-100 | 5 | -0.000776 | -0.005236 | 0.004461 | 0.6658 | 0.967 |
| AR-100 | 20 | -0.001775 | -0.005710 | 0.003935 | 0.1690 | 1.815 |
| AR-100 | 100 | -0.012247 | -0.023357 | 0.011109 | 0.6658 | 0.973 |

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
