"""Shared configuration for the hybrid model pipeline."""

# 20 non-spatial features from the 62 Track A features.
# Selected based on R4 importance analysis and R6 synthesis.
NON_SPATIAL_FEATURES = [
    "weighted_imbalance", "spread",
    "net_volume", "volume_imbalance", "trade_count",
    "avg_trade_size", "vwap_distance",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50",
    "high_low_range_50", "close_position",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "time_sin", "time_cos", "minutes_since_open",
]

# 19 selected days from bar_feature_export (matching C++ SELECTED_DAYS)
SELECTED_DAYS = [
    20220103, 20220121, 20220211, 20220304, 20220331, 20220401, 20220422,
    20220513, 20220603, 20220630, 20220701, 20220722, 20220812, 20220902,
    20220930, 20221003, 20221024, 20221114, 20221205,
]

# Transaction cost scenarios
COST_SCENARIOS = [
    {
        "name": "optimistic",
        "commission_per_side": 0.62,
        "spread_ticks": 1,
        "slippage_ticks": 0,
        "total_rt_cost": 2.49,
    },
    {
        "name": "base",
        "commission_per_side": 0.62,
        "spread_ticks": 1,
        "slippage_ticks": 0.5,
        "total_rt_cost": 3.74,
    },
    {
        "name": "pessimistic",
        "commission_per_side": 1.00,
        "spread_ticks": 1,
        "slippage_ticks": 1,
        "total_rt_cost": 6.25,
    },
]
