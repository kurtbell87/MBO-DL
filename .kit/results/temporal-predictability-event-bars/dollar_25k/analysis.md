# R4: Temporal Predictability — Analysis [dollar_25k]

**Date:** 2026-02-17  
**Bar type:** dollar_25k

## Table 1: Tier 1 — Pure Return AR

| Lookback | Model | return_1 | return_5 | return_20 | return_100 |
|----------|-------|----------|----------|-----------|------------|
| AR-10 | linear | **0.000633 ± 0.000184** | 0.000364 ± 0.000247 | 0.000080 ± 0.000242 | -0.000294 ± 0.000408 |
| AR-10 | ridge | **0.000633 ± 0.000184** | 0.000364 ± 0.000247 | 0.000080 ± 0.000241 | -0.000294 ± 0.000408 |
| AR-10 | gbt | **0.000471 ± 0.000176** | 0.000203 ± 0.000223 | -0.000034 ± 0.000211 | -0.000359 ± 0.000388 |
| AR-50 | linear | **0.000584 ± 0.000220** | 0.000223 ± 0.000442 | -0.000166 ± 0.000517 | -0.000326 ± 0.000522 |
| AR-50 | ridge | **0.000584 ± 0.000220** | 0.000223 ± 0.000442 | -0.000166 ± 0.000516 | -0.000326 ± 0.000522 |
| AR-50 | gbt | 0.000468 ± 0.000224 | 0.000214 ± 0.000139 | 0.000047 ± 0.000059 | -0.000321 ± 0.000438 |
| AR-100 | linear | 0.000528 ± 0.000244 | 0.000111 ± 0.000468 | -0.000323 ± 0.000479 | -0.000464 ± 0.000704 |
| AR-100 | ridge | 0.000528 ± 0.000244 | 0.000111 ± 0.000467 | -0.000323 ± 0.000479 | -0.000464 ± 0.000704 |
| AR-100 | gbt | 0.000385 ± 0.000174 | 0.000193 ± 0.000132 | 0.000035 ± 0.000069 | -0.000286 ± 0.000391 |

## Table 2: Tier 2 — Temporal Feature Augmentation (GBT)

| Config | return_1 | return_5 | return_20 | return_100 |
|--------|----------|----------|-----------|------------|
| Static-Book | 0.067321 ± 0.020525 | 0.040100 ± 0.010395 | 0.006859 ± 0.012966 | 0.000798 ± 0.000895 |
| Static-HC | 0.080347 ± 0.010713 | 0.043513 ± 0.006834 | 0.012311 ± 0.003574 | 0.001282 ± 0.000929 |
| Book+Temporal | 0.080529 ± 0.006966 | 0.037165 ± 0.013697 | 0.008060 ± 0.009259 | -0.000807 ± 0.001431 |
| HC+Temporal | 0.080088 ± 0.012273 | 0.043300 ± 0.006897 | 0.012759 ± 0.003189 | 0.001459 ± 0.001043 |
| Temporal-Only | 0.011811 ± 0.001882 | 0.004255 ± 0.001178 | 0.000353 ± 0.000629 | -0.000853 ± 0.000583 |

## Table 3: Information Gaps (GBT, Tier 2)

| Gap | Horizon | Δ_R² | 95% CI | Raw p | Corrected p | Passes? |
|-----|---------|------|--------|-------|-------------|---------|
| delta_temporal_book | 1 | 0.013208 | [-0.008290, 0.034706] | 0.0625 | 0.2500 | no |
| delta_temporal_book | 5 | -0.002935 | [-0.009012, 0.003141] | 0.3125 | 0.6469 | no |
| delta_temporal_book | 20 | 0.001201 | [-0.004124, 0.006527] | 0.5650 | 0.6469 | no |
| delta_temporal_book | 100 | -0.001605 | [-0.004637, 0.001427] | 0.2156 | 0.6469 | no |
| delta_temporal_hc | 1 | -0.000259 | [-0.002713, 0.002196] | 0.6250 | 1.0000 | no |
| delta_temporal_hc | 5 | -0.000213 | [-0.001371, 0.000944] | 0.6358 | 1.0000 | no |
| delta_temporal_hc | 20 | 0.000448 | [-0.000160, 0.001057] | 0.1104 | 0.4416 | no |
| delta_temporal_hc | 100 | 0.000177 | [-0.000619, 0.000973] | 0.5702 | 1.0000 | no |
| delta_temporal_only | 1 | 0.011811 | [0.009198, 0.014424] | 0.0001 | 0.0005 | no |
| delta_temporal_only | 5 | 0.004255 | [0.002620, 0.005890] | 0.0010 | 0.0029 | no |
| delta_temporal_only | 20 | 0.000353 | [-0.000520, 0.001226] | 0.1625 | 0.3249 | no |
| delta_temporal_only | 100 | -0.000853 | [-0.001662, -0.000044] | 1.0000 | 1.0000 | no |
| delta_static_comparison | 1 | -0.013026 | [-0.029072, 0.003020] | 0.0873 | 0.3491 | no |
| delta_static_comparison | 5 | -0.003414 | [-0.011436, 0.004609] | 0.8125 | 1.0000 | no |
| delta_static_comparison | 20 | -0.005453 | [-0.020028, 0.009123] | 0.6250 | 1.0000 | no |
| delta_static_comparison | 100 | -0.000484 | [-0.002143, 0.001176] | 0.4636 | 1.0000 | no |

## Table 4: Feature Importance (GBT, Fold 5)

### Book+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_19 | 0.149059 | static |
| 2 | book_snap_21 | 0.139394 | static |
| 3 | book_snap_15 | 0.104055 | static |
| 4 | book_snap_25 | 0.053950 | static |
| 5 | book_snap_27 | 0.040271 | static |
| 6 | book_snap_2 | 0.037056 | static |
| 7 | book_snap_0 | 0.034023 | static |
| 8 | book_snap_11 | 0.031686 | static |
| 9 | rolling_vol_100 | 0.030925 | temporal |
| 10 | book_snap_23 | 0.027978 | static |

Temporal feature share: 22.5% of total importance

### Book+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_19 | 0.144317 | static |
| 2 | book_snap_21 | 0.138312 | static |
| 3 | book_snap_15 | 0.080065 | static |
| 4 | book_snap_25 | 0.049518 | static |
| 5 | book_snap_27 | 0.045036 | static |
| 6 | book_snap_2 | 0.039443 | static |
| 7 | book_snap_0 | 0.030193 | static |
| 8 | book_snap_18 | 0.028117 | static |
| 9 | book_snap_23 | 0.024688 | static |
| 10 | book_snap_22 | 0.020640 | static |

Temporal feature share: 21.6% of total importance

### Book+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_19 | 0.114951 | static |
| 2 | book_snap_21 | 0.104629 | static |
| 3 | book_snap_25 | 0.038909 | static |
| 4 | book_snap_15 | 0.037028 | static |
| 5 | book_snap_7 | 0.034644 | static |
| 6 | book_snap_0 | 0.029098 | static |
| 7 | book_snap_31 | 0.028495 | static |
| 8 | momentum_20 | 0.028226 | temporal |
| 9 | book_snap_29 | 0.028088 | static |
| 10 | book_snap_23 | 0.023982 | static |

Temporal feature share: 27.0% of total importance

### Book+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_snap_19 | 0.081385 | static |
| 2 | book_snap_21 | 0.075973 | static |
| 3 | book_snap_0 | 0.051793 | static |
| 4 | momentum_100 | 0.046153 | temporal |
| 5 | book_snap_23 | 0.044328 | static |
| 6 | book_snap_17 | 0.044240 | static |
| 7 | rolling_vol_100 | 0.038283 | temporal |
| 8 | book_snap_29 | 0.036751 | static |
| 9 | book_snap_31 | 0.036489 | static |
| 10 | book_snap_27 | 0.035907 | static |

Temporal feature share: 24.8% of total importance

### HC+Temporal_gbt_h1

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | weighted_imbalance | 0.289475 | static |
| 2 | book_imbalance_1 | 0.115238 | static |
| 3 | bid_depth_profile_0 | 0.048525 | static |
| 4 | ask_depth_profile_0 | 0.047321 | static |
| 5 | book_slope_bid | 0.042496 | static |
| 6 | book_slope_ask | 0.027422 | static |
| 7 | volatility_50 | 0.025916 | static |
| 8 | bid_depth_profile_2 | 0.023085 | static |
| 9 | avg_trade_size | 0.021001 | static |
| 10 | rolling_vol_100 | 0.020976 | temporal |

Temporal feature share: 9.7% of total importance

### HC+Temporal_gbt_h5

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | weighted_imbalance | 0.330274 | static |
| 2 | book_imbalance_1 | 0.125518 | static |
| 3 | book_slope_bid | 0.056261 | static |
| 4 | book_slope_ask | 0.049029 | static |
| 5 | depth_concentration_bid | 0.035721 | static |
| 6 | bid_depth_profile_0 | 0.022372 | static |
| 7 | ask_depth_profile_0 | 0.019391 | static |
| 8 | ask_depth_profile_3 | 0.011355 | static |
| 9 | modify_fraction | 0.011178 | static |
| 10 | momentum_100 | 0.010425 | temporal |

Temporal feature share: 9.8% of total importance

### HC+Temporal_gbt_h20

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | weighted_imbalance | 0.209337 | static |
| 2 | book_imbalance_1 | 0.092335 | static |
| 3 | book_slope_bid | 0.056895 | static |
| 4 | book_imbalance_3 | 0.042816 | static |
| 5 | book_slope_ask | 0.036199 | static |
| 6 | depth_concentration_bid | 0.033891 | static |
| 7 | bid_depth_profile_0 | 0.021117 | static |
| 8 | ask_depth_profile_0 | 0.018205 | static |
| 9 | time_sin | 0.016527 | static |
| 10 | minutes_since_open | 0.016247 | static |

Temporal feature share: 8.3% of total importance

### HC+Temporal_gbt_h100

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | book_slope_bid | 0.109422 | static |
| 2 | book_imbalance_1 | 0.060859 | static |
| 3 | rolling_vol_100 | 0.037993 | temporal |
| 4 | bid_depth_profile_2 | 0.035484 | static |
| 5 | depth_concentration_bid | 0.034212 | static |
| 6 | minutes_to_close | 0.031596 | static |
| 7 | ask_depth_profile_5 | 0.031585 | static |
| 8 | minutes_since_open | 0.029756 | static |
| 9 | time_sin | 0.029134 | static |
| 10 | time_cos | 0.027998 | static |

Temporal feature share: 8.1% of total importance


## Tier 1 Pairwise: GBT vs. Linear

| Lookback | Horizon | GBT R² | Linear R² | Δ | Corrected p | Cohen's d |
|----------|---------|--------|-----------|---|-------------|-----------|
| AR-10 | 1 | 0.000471 | 0.000633 | -0.000162 | 0.3476 | -1.492 |
| AR-10 | 5 | 0.000203 | 0.000364 | -0.000161 | 1.0000 | -0.591 |
| AR-10 | 20 | -0.000034 | 0.000080 | -0.000114 | 1.0000 | -0.613 |
| AR-10 | 100 | -0.000359 | -0.000294 | -0.000065 | 1.0000 | -0.351 |
| AR-50 | 1 | 0.000468 | 0.000584 | -0.000116 | 0.9807 | -0.961 |
| AR-50 | 5 | 0.000214 | 0.000223 | -0.000009 | 1.0000 | -0.024 |
| AR-50 | 20 | 0.000047 | -0.000166 | 0.000213 | 1.0000 | 0.394 |
| AR-50 | 100 | -0.000321 | -0.000326 | 0.000005 | 1.0000 | 0.008 |
| AR-100 | 1 | 0.000385 | 0.000528 | -0.000143 | 0.8667 | -1.050 |
| AR-100 | 5 | 0.000193 | 0.000111 | 0.000082 | 1.0000 | 0.195 |
| AR-100 | 20 | 0.000035 | -0.000323 | 0.000358 | 1.0000 | 0.688 |
| AR-100 | 100 | -0.000286 | -0.000464 | 0.000178 | 1.0000 | 0.212 |

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
- **Passes**: True
- **Interpretation**: Temporal features have standalone predictive power
- **Passing horizons**: ['delta_temporal_only_h1', 'delta_temporal_only_h5']

### rule4_reconciliation
- **Passes**: N/A
- **Interpretation**: R4 confirms R2: temporal encoder adds no value for MES at 5s bars. Strong recommendation to drop SSM.

## Summary Finding

**MARGINAL SIGNAL**: Positive R² in temporal-only but augmentation fails threshold. Temporal structure exists but is redundant with static features.
