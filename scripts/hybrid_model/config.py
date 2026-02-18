"""Shared configuration for CNN+GBT Hybrid model pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CSV = PROJECT_ROOT / ".kit" / "results" / "hybrid-model" / "time_5s.csv"
RESULTS_DIR = PROJECT_ROOT / ".kit" / "results" / "hybrid-model"

# ---------------------------------------------------------------------------
# Selected days (same order as C++ SELECTED_DAYS)
# ---------------------------------------------------------------------------
SELECTED_DAYS = [
    20220103, 20220121, 20220211, 20220304, 20220331, 20220401, 20220422,
    20220513, 20220603, 20220630, 20220701, 20220722, 20220812, 20220902,
    20220930, 20221003, 20221024, 20221114, 20221205,
]

# ---------------------------------------------------------------------------
# 5-fold expanding window CV (spec Cross-Validation Protocol)
# ---------------------------------------------------------------------------
FOLDS = [
    {"train": SELECTED_DAYS[0:4],   "test": SELECTED_DAYS[4:7]},    # Fold 1
    {"train": SELECTED_DAYS[0:7],   "test": SELECTED_DAYS[7:10]},   # Fold 2
    {"train": SELECTED_DAYS[0:10],  "test": SELECTED_DAYS[10:13]},  # Fold 3
    {"train": SELECTED_DAYS[0:13],  "test": SELECTED_DAYS[13:16]},  # Fold 4
    {"train": SELECTED_DAYS[0:16],  "test": SELECTED_DAYS[16:19]},  # Fold 5
]

# ---------------------------------------------------------------------------
# 62 Track A feature names (order matches C++ BarFeatureRow::feature_names())
# ---------------------------------------------------------------------------
TRACK_A_FEATURES = [
    # Cat 1: Book Shape (32)
    "book_imbalance_1", "book_imbalance_3", "book_imbalance_5", "book_imbalance_10",
    "weighted_imbalance", "spread",
    *[f"bid_depth_profile_{i}" for i in range(10)],
    *[f"ask_depth_profile_{i}" for i in range(10)],
    "depth_concentration_bid", "depth_concentration_ask",
    "book_slope_bid", "book_slope_ask",
    "level_count_bid", "level_count_ask",
    # Cat 2: Order Flow (7)
    "net_volume", "volume_imbalance", "trade_count", "avg_trade_size",
    "large_trade_count", "vwap_distance", "kyle_lambda",
    # Cat 3: Price Dynamics (10)
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50", "momentum",
    "high_low_range_20", "high_low_range_50", "close_position",
    # Cat 4: Cross-Scale Dynamics (4)
    "volume_surprise", "duration_surprise", "acceleration", "vol_price_corr",
    # Cat 5: Time Context (5)
    "time_sin", "time_cos", "minutes_since_open", "minutes_to_close", "session_volume_frac",
    # Cat 6: Message Microstructure (5)
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "order_flow_toxicity", "cancel_concentration",
]
assert len(TRACK_A_FEATURES) == 62

# ---------------------------------------------------------------------------
# 20 non-spatial features for XGBoost (spec Non-Spatial Feature Set)
# ---------------------------------------------------------------------------
NON_SPATIAL_FEATURES = [
    "weighted_imbalance", "spread",
    "net_volume", "volume_imbalance", "trade_count", "avg_trade_size", "vwap_distance",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50",
    "high_low_range_50", "close_position",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "time_sin", "time_cos", "minutes_since_open",
]
assert len(NON_SPATIAL_FEATURES) == 20

# Book snapshot columns in CSV: book_snap_0 .. book_snap_39
# Layout: 20 levels x (price_offset, size) interleaved
# Rows 0-9: bids (deepest first), Rows 10-19: asks (best first)
BOOK_SNAP_COLS = [f"book_snap_{i}" for i in range(40)]

# Forward return columns (used as CNN regression targets)
# Note: CSV has duplicate names with Track A return_1/5/20.
# We rename forward returns on load to fwd_return_*.
FWD_RETURN_H1 = "fwd_return_1"
FWD_RETURN_H5 = "fwd_return_5"

# ---------------------------------------------------------------------------
# Triple barrier columns
# ---------------------------------------------------------------------------
TB_LABEL_COL = "tb_label"
TB_EXIT_TYPE_COL = "tb_exit_type"
TB_BARS_HELD_COL = "tb_bars_held"

# ---------------------------------------------------------------------------
# CNN hyperparameters (spec Two-Stage Training, Stage 1)
# ---------------------------------------------------------------------------
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-5
CNN_BATCH_SIZE = 256
CNN_MAX_EPOCHS = 50
CNN_PATIENCE = 5
CNN_EMBEDDING_DIM = 16
CNN_SEED = 42

# ---------------------------------------------------------------------------
# XGBoost hyperparameters (spec Two-Stage Training, Stage 2)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "multi:softmax",
    "num_class": 3,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
    "verbosity": 0,
}

# ---------------------------------------------------------------------------
# Triple barrier parameters (for PnL computation)
# ---------------------------------------------------------------------------
TB_TARGET_TICKS = 10
TB_STOP_TICKS = 5
TB_TICK_SIZE = 0.25
TB_POINT_VALUE = 5.00  # $/point for MES

# ---------------------------------------------------------------------------
# Transaction cost scenarios (spec Transaction cost sensitivity)
# ---------------------------------------------------------------------------
COST_SCENARIOS = {
    "optimistic":  {"commission_per_side": 0.62, "spread_ticks": 1, "slippage_ticks": 0.0},
    "base":        {"commission_per_side": 0.62, "spread_ticks": 1, "slippage_ticks": 0.5},
    "pessimistic": {"commission_per_side": 1.00, "spread_ticks": 1, "slippage_ticks": 1.0},
}


def round_trip_cost(scenario_name: str) -> float:
    """Compute total round-trip cost in dollars for a scenario."""
    s = COST_SCENARIOS[scenario_name]
    commission_rt = s["commission_per_side"] * 2
    spread_cost = s["spread_ticks"] * TB_TICK_SIZE * TB_POINT_VALUE
    slippage_cost = s["slippage_ticks"] * TB_TICK_SIZE * TB_POINT_VALUE
    return commission_rt + spread_cost + slippage_cost
