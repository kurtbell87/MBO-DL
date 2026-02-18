# R4: Temporal Predictability — Analysis [tick_250]

**Date:** 2026-02-17  
**Bar type:** tick_250

## Table 1: Tier 1 — Pure Return AR

| Lookback | Model | return_1 | return_5 | return_20 | return_100 |
|----------|-------|----------|----------|-----------|------------|
| AR-10 | linear | -0.003519 ± 0.003572 | -0.004173 ± 0.002000 | -0.015327 ± 0.013929 | -0.062269 ± 0.104031 |
| AR-10 | ridge | -0.003403 ± 0.003470 | -0.004103 ± 0.001973 | -0.015065 ± 0.013833 | -0.062203 ± 0.104028 |
| AR-10 | gbt | -0.000762 ± 0.000705 | -0.001856 ± 0.003546 | -0.012724 ± 0.013568 | -0.063941 ± 0.099502 |
| AR-50 | linear | -0.016385 ± 0.015841 | -0.024572 ± 0.021755 | -0.030794 ± 0.026118 | -0.078757 ± 0.097098 |
| AR-50 | ridge | -0.015690 ± 0.014890 | -0.023701 ± 0.020493 | -0.029781 ± 0.025225 | -0.078306 ± 0.097219 |
| AR-50 | gbt | -0.000935 ± 0.000760 | -0.002813 ± 0.002586 | -0.009179 ± 0.011495 | -0.089304 ± 0.099035 |
| AR-100 | linear | -0.032384 ± 0.024336 | -0.038629 ± 0.029471 | -0.045485 ± 0.030032 | -0.074853 ± 0.081315 |
| AR-100 | ridge | -0.030731 ± 0.022249 | -0.036909 ± 0.027464 | -0.043894 ± 0.028876 | -0.073392 ± 0.080823 |
| AR-100 | gbt | -0.001189 ± 0.000719 | -0.002046 ± 0.002448 | -0.008528 ± 0.009773 | -0.063700 ± 0.077583 |

## Table 2: Tier 2 — Temporal Feature Augmentation (GBT)

| Config | return_1 | return_5 | return_20 | return_100 |
|--------|----------|----------|-----------|------------|
| Static-Book | -0.001282 ± 0.002243 | -0.004860 ± 0.005427 | -0.010812 ± 0.011018 | -0.064559 ± 0.091071 |
| Static-HC | -0.001784 ± 0.001431 | -0.009561 ± 0.016127 | -0.020225 ± 0.028048 | -0.077055 ± 0.097018 |
| Book+Temporal | -0.005375 ± 0.009103 | -0.002270 ± 0.002454 | -0.018053 ± 0.031808 | -0.207490 ± 0.374604 |
| HC+Temporal | -0.004674 ± 0.007631 | -0.007434 ± 0.008565 | -0.021034 ± 0.030078 | -0.073098 ± 0.089574 |
| Temporal-Only | -0.001415 ± 0.002032 | -0.002324 ± 0.002201 | -0.017792 ± 0.031235 | -0.092540 ± 0.134417 |

## Table 3: Information Gaps (GBT, Tier 2)

| Gap | Horizon | Δ_R² | 95% CI | Raw p | Corrected p | Passes? |
|-----|---------|------|--------|-------|-------------|---------|
| delta_temporal_book | 1 | -0.004093 | [-0.013873, 0.005687] | 0.3125 | 0.9375 | no |
| delta_temporal_book | 5 | 0.002591 | [-0.003783, 0.008964] | 0.1250 | 0.5000 | no |
| delta_temporal_book | 20 | -0.007241 | [-0.039174, 0.024691] | 1.0000 | 1.0000 | no |
| delta_temporal_book | 100 | -0.142932 | [-0.537226, 0.251363] | 0.4375 | 0.9375 | no |
| delta_temporal_hc | 1 | -0.002890 | [-0.012940, 0.007159] | 0.4693 | 1.0000 | no |
| delta_temporal_hc | 5 | 0.002127 | [-0.008725, 0.012979] | 0.6250 | 1.0000 | no |
| delta_temporal_hc | 20 | -0.000809 | [-0.005163, 0.003545] | 0.6333 | 1.0000 | no |
| delta_temporal_hc | 100 | 0.003957 | [-0.006631, 0.014545] | 0.6250 | 1.0000 | no |
| delta_temporal_only | 1 | -0.001415 | [-0.004237, 0.001407] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 5 | -0.002324 | [-0.005379, 0.000731] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 20 | -0.017792 | [-0.061152, 0.025569] | 1.0000 | 1.0000 | no |
| delta_temporal_only | 100 | -0.092540 | [-0.279142, 0.094061] | 1.0000 | 1.0000 | no |
| delta_static_comparison | 1 | 0.000502 | [-0.002658, 0.003661] | 0.6820 | 1.0000 | no |
| delta_static_comparison | 5 | 0.004701 | [-0.015927, 0.025328] | 0.6250 | 1.0000 | no |
| delta_static_comparison | 20 | 0.009413 | [-0.014759, 0.033585] | 0.3404 | 1.0000 | no |
| delta_static_comparison | 100 | 0.012496 | [-0.001541, 0.026534] | 0.0688 | 0.2753 | no |

## Table 4: Feature Importance (GBT, Fold 5)

### Book+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_39 | 0.039983 | static |
| 2 | book_snap_9 | 0.036554 | static |
| 3 | book_snap_33 | 0.035875 | static |
| 4 | book_snap_35 | 0.035615 | static |
| 5 | book_snap_25 | 0.033438 | static |
| 6 | lag_return_1 | 0.031867 | temporal |
| 7 | signed_vol | 0.031526 | temporal |
| 8 | rolling_vol_20 | 0.031074 | temporal |
| 9 | book_snap_31 | 0.030655 | static |
| 10 | book_snap_23 | 0.029271 | static |

Temporal feature share: 50.5% of total importance

### Book+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | rolling_vol_20 | 0.043253 | temporal |
| 2 | book_snap_29 | 0.038457 | static |
| 3 | rolling_vol_5 | 0.037618 | temporal |
| 4 | book_snap_21 | 0.035483 | static |
| 5 | rolling_vol_100 | 0.035363 | temporal |
| 6 | book_snap_23 | 0.035028 | static |
| 7 | signed_vol | 0.034303 | temporal |
| 8 | vol_ratio | 0.034130 | temporal |
| 9 | book_snap_15 | 0.032950 | static |
| 10 | momentum_100 | 0.032494 | temporal |

Temporal feature share: 53.5% of total importance

### Book+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_11 | 0.072077 | static |
| 2 | book_snap_27 | 0.062692 | static |
| 3 | book_snap_13 | 0.062036 | static |
| 4 | book_snap_23 | 0.059276 | static |
| 5 | book_snap_35 | 0.054668 | static |
| 6 | momentum_20 | 0.052826 | temporal |
| 7 | momentum_100 | 0.048074 | temporal |
| 8 | rolling_vol_100 | 0.047196 | temporal |
| 9 | signed_vol | 0.041209 | temporal |
| 10 | book_snap_25 | 0.040198 | static |

Temporal feature share: 47.2% of total importance

### Book+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_29 | 0.089554 | static |
| 2 | book_snap_27 | 0.083480 | static |
| 3 | book_snap_19 | 0.075986 | static |
| 4 | book_snap_25 | 0.061974 | static |
| 5 | book_snap_13 | 0.056746 | static |
| 6 | rolling_vol_100 | 0.051330 | temporal |
| 7 | book_snap_23 | 0.051029 | static |
| 8 | book_snap_17 | 0.048847 | static |
| 9 | book_snap_5 | 0.047444 | static |
| 10 | momentum_100 | 0.044611 | temporal |

Temporal feature share: 21.0% of total importance

### HC+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | signed_vol | 0.024537 | temporal |
| 2 | bid_depth_profile_6 | 0.024432 | static |
| 3 | trade_count | 0.023607 | static |
| 4 | ask_depth_profile_9 | 0.021832 | static |
| 5 | close_position | 0.019564 | static |
| 6 | bid_depth_profile_5 | 0.019465 | static |
| 7 | depth_concentration_bid | 0.019027 | static |
| 8 | ask_depth_profile_3 | 0.018775 | static |
| 9 | time_cos | 0.018042 | static |
| 10 | volatility_20 | 0.017685 | static |

Temporal feature share: 27.4% of total importance

### HC+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | minutes_to_close | 0.032698 | static |
| 2 | time_cos | 0.029236 | static |
| 3 | rolling_vol_20 | 0.024786 | temporal |
| 4 | minutes_since_open | 0.024524 | static |
| 5 | return_5 | 0.022578 | static |
| 6 | momentum_5 | 0.022108 | temporal |
| 7 | ask_depth_profile_5 | 0.020825 | static |
| 8 | volatility_50 | 0.020606 | static |
| 9 | close_position | 0.020546 | static |
| 10 | momentum | 0.020514 | static |

Temporal feature share: 25.3% of total importance

### HC+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | minutes_to_close | 0.062222 | static |
| 2 | message_rate | 0.040929 | static |
| 3 | close_position | 0.040776 | static |
| 4 | time_sin | 0.037947 | static |
| 5 | minutes_since_open | 0.037820 | static |
| 6 | vol_price_corr | 0.035497 | static |
| 7 | bid_depth_profile_4 | 0.034864 | static |
| 8 | return_20 | 0.033437 | static |
| 9 | volatility_50 | 0.032423 | static |
| 10 | high_low_range_50 | 0.032363 | static |

Temporal feature share: 18.3% of total importance

### HC+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | ask_depth_profile_5 | 0.091893 | static |
| 2 | high_low_range_50 | 0.056919 | static |
| 3 | kyle_lambda | 0.048519 | static |
| 4 | message_rate | 0.048457 | static |
| 5 | ask_depth_profile_3 | 0.042572 | static |
| 6 | momentum_100 | 0.040154 | temporal |
| 7 | depth_concentration_ask | 0.039240 | static |
| 8 | minutes_since_open | 0.036663 | static |
| 9 | cancel_concentration | 0.035999 | static |
| 10 | time_sin | 0.035855 | static |

Temporal feature share: 10.7% of total importance


## Tier 1 Pairwise: GBT vs. Linear

| Lookback | Horizon | GBT R² | Linear R² | Δ | Corrected p | Cohen's d |
|----------|---------|--------|-----------|---|-------------|-----------|
| AR-10 | 1 | -0.000762 | -0.003519 | 0.002757 | 0.9062 | 0.723 |
| AR-10 | 5 | -0.001856 | -0.004173 | 0.002316 | 0.6875 | 1.097 |
| AR-10 | 20 | -0.012724 | -0.015327 | 0.002603 | 1.0000 | 0.296 |
| AR-10 | 100 | -0.063941 | -0.062269 | -0.001672 | 1.0000 | -0.083 |
| AR-50 | 1 | -0.000935 | -0.016385 | 0.015450 | 0.6875 | 0.868 |
| AR-50 | 5 | -0.002813 | -0.024572 | 0.021759 | 0.7118 | 0.904 |
| AR-50 | 20 | -0.009179 | -0.030794 | 0.021615 | 0.7118 | 0.947 |
| AR-50 | 100 | -0.089304 | -0.078757 | -0.010546 | 1.0000 | -0.209 |
| AR-100 | 1 | -0.001189 | -0.032384 | 0.031195 | 0.6875 | 1.173 |
| AR-100 | 5 | -0.002046 | -0.038629 | 0.036583 | 0.6875 | 1.085 |
| AR-100 | 20 | -0.008528 | -0.045485 | 0.036958 | 0.5933 | 1.247 |
| AR-100 | 100 | -0.063700 | -0.074853 | 0.011153 | 1.0000 | 0.113 |

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
