"""Evaluation metrics: R-squared, accuracy, PnL, Sharpe."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from . import config


def r_squared(y_true, y_pred):
    """Compute R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def classification_metrics(y_true, y_pred):
    """Compute classification accuracy and macro F1."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "f1_macro": float(f1)}


def compute_pnl(predictions, tb_labels, tb_exit_types, tb_bars_held,
                entry_mids, bars_df, cost_per_trade=0.0):
    """Compute realized PnL for model predictions.

    For each prediction:
    - pred = +1 or -1: trade in that direction
    - pred = 0: no trade (PnL = 0)

    PnL per trade = direction * (exit_price - entry_price) * $5/point - cost

    The actual exit is determined by the triple barrier applied to the
    forward price path (already computed in the export: tb_label tells us
    the outcome, and we use the TB parameters to compute exit PnL).

    Args:
        predictions: (N,) model predictions in {-1, 0, +1}
        tb_labels: (N,) actual TB labels (oracle)
        tb_exit_types: (N,) "target", "stop", "expiry", "timeout"
        tb_bars_held: (N,) bars held until exit
        entry_mids: (N,) mid price at entry bar
        bars_df: dataframe with bar data (unused if we use TB outcome directly)
        cost_per_trade: round-trip cost in dollars

    Returns:
        dict with expectancy, profit_factor, trade_count, total_pnl, sharpe, pnl_array
    """
    N = len(predictions)
    pnl_per_bar = np.zeros(N)
    trade_mask = predictions != 0

    for i in range(N):
        if predictions[i] == 0:
            continue

        direction = predictions[i]
        exit_type = tb_exit_types[i] if isinstance(tb_exit_types, np.ndarray) else tb_exit_types.iloc[i]
        oracle_label = tb_labels[i] if isinstance(tb_labels, np.ndarray) else tb_labels.iloc[i]

        # Compute PnL based on actual barrier outcome
        if exit_type == "target":
            # Oracle hit target: PnL = oracle_direction * target_ticks * tick_size * point_value
            barrier_pnl = oracle_label * config.TB_TARGET_TICKS * config.TB_TICK_SIZE * config.TB_POINT_VALUE
        elif exit_type == "stop":
            # Oracle hit stop: PnL = oracle_direction * stop_ticks * tick_size * point_value
            # oracle_label is -1 for stop (short side), but the stop loss amount is always negative
            barrier_pnl = oracle_label * config.TB_STOP_TICKS * config.TB_TICK_SIZE * config.TB_POINT_VALUE
        else:
            # Expiry/timeout: use oracle_label sign and min_return as proxy
            # In practice, the actual return is somewhere between 0 and the barriers
            # Use a conservative estimate: min_return_ticks for non-zero labels, 0 for HOLD
            if oracle_label != 0:
                barrier_pnl = oracle_label * config.TB_STOP_TICKS * config.TB_TICK_SIZE * config.TB_POINT_VALUE * 0.5
            else:
                barrier_pnl = 0.0

        # Model PnL: if we predicted the same direction as oracle outcome, we capture the PnL
        # If we predicted opposite, we get negative of the PnL
        if direction == oracle_label:
            pnl_per_bar[i] = abs(barrier_pnl) - cost_per_trade
        elif oracle_label == 0:
            # Oracle was HOLD, we traded: small loss from costs
            pnl_per_bar[i] = -cost_per_trade
        else:
            # We predicted wrong direction
            pnl_per_bar[i] = -abs(barrier_pnl) - cost_per_trade

    trade_pnls = pnl_per_bar[trade_mask]
    trade_count = int(trade_mask.sum())

    if trade_count == 0:
        return {
            "expectancy": 0.0, "profit_factor": 0.0, "trade_count": 0,
            "total_pnl": 0.0, "sharpe": 0.0, "pnl_array": pnl_per_bar,
        }

    gross_profit = trade_pnls[trade_pnls > 0].sum()
    gross_loss = abs(trade_pnls[trade_pnls < 0].sum())

    expectancy = float(trade_pnls.mean())
    profit_factor = float(gross_profit / (gross_loss + 1e-8))

    # Annualized Sharpe: assume ~4600 bars/day for time_5s, 252 trading days
    if trade_pnls.std() > 0:
        daily_sharpe = trade_pnls.mean() / trade_pnls.std()
        sharpe = float(daily_sharpe * np.sqrt(252))
    else:
        sharpe = 0.0

    return {
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "trade_count": trade_count,
        "total_pnl": float(trade_pnls.sum()),
        "sharpe": sharpe,
        "pnl_array": pnl_per_bar,
    }
