"""Evaluation metrics for the hybrid model pipeline."""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def compute_pnl(direction, entry_price, exit_price, point_value, cost_rt):
    """Compute PnL for a single trade.

    PnL = direction * (exit_price - entry_price) * point_value - cost_rt
    Direction 0 (hold) = no trade = PnL 0.
    """
    if direction == 0:
        return 0.0
    return direction * (exit_price - entry_price) * point_value - cost_rt


def compute_expectancy(pnls):
    """Compute expectancy (mean PnL per trade)."""
    if len(pnls) == 0:
        return 0.0
    return float(np.mean(pnls))


def compute_profit_factor(pnls):
    """Compute profit factor = gross_profit / gross_loss."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = sum(abs(p) for p in pnls if p < 0)
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def compute_sharpe(daily_pnls):
    """Compute Sharpe ratio from daily PnL series."""
    if len(daily_pnls) < 2:
        return 0.0
    arr = np.array(daily_pnls, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if std < 1e-10:
        return 0.0
    return float(mean / std * np.sqrt(252))


def compute_accuracy(predictions, actuals):
    """Compute classification accuracy."""
    return float(accuracy_score(actuals, predictions))


def compute_f1_macro(predictions, actuals):
    """Compute macro F1 score across all classes."""
    return float(f1_score(actuals, predictions, average="macro", zero_division=0))


def compute_metrics_suite(predictions, actuals, entry_prices, exit_prices,
                          point_value, cost_rt):
    """Compute the full evaluation metrics suite.

    Returns a dict with all per-fold metrics.
    """
    # Compute PnL for each trade
    pnls = []
    trade_count = 0
    for pred, actual, entry, exit_p in zip(predictions, actuals, entry_prices, exit_prices):
        direction = int(pred)
        pnl = compute_pnl(direction, entry, exit_p, point_value, cost_rt)
        if direction != 0:
            pnls.append(pnl)
            trade_count += 1

    return {
        "xgb_accuracy": compute_accuracy(predictions, actuals),
        "xgb_f1_macro": compute_f1_macro(predictions, actuals),
        "expectancy": compute_expectancy(pnls),
        "profit_factor": compute_profit_factor(pnls),
        "trade_count": int(trade_count),
        "sharpe": compute_sharpe(pnls) if len(pnls) > 1 else 0.0,
    }
