#!/usr/bin/env python3
"""
Experiment: Daily Stop Loss — Sequential Execution
Spec: .kit/experiments/daily-stop-loss-sequential.md

Forked from timeout-filtered-sequential (PR #40). Training, prediction, and
time cutoff (270) are identical. Adds daily stop loss (DSL) sweep: when
cumulative intra-day P&L drops to -$X, stop trading for the rest of that day.

9 DSL levels × 45 CPCV splits = 405 simulations.
"""

import json
import os
import sys
import time
import random
import hashlib
import csv
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from scipy import stats as scipy_stats

# ==========================================================================
# Config
# ==========================================================================
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / ".kit" / "results" / "daily-stop-loss-sequential"
DATA_DIR = PROJECT_ROOT / ".kit" / "results" / "label-geometry-1h" / "geom_19_7"

# Fixed cutoff from PR #40
FIXED_CUTOFF = 270

# DSL levels (None = baseline = identical to PR #40 cutoff=270)
DSL_LEVELS = [None, 5000, 4000, 3000, 2500, 2000, 1500, 1000, 500]

TUNED_XGB_PARAMS_BINARY = {
    "max_depth": 6,
    "learning_rate": 0.0134,
    "min_child_weight": 20,
    "subsample": 0.561,
    "colsample_bytree": 0.748,
    "reg_alpha": 0.0014,
    "reg_lambda": 6.586,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": SEED,
    "verbosity": 0,
    "n_jobs": -1,
}

NON_SPATIAL_FEATURES = [
    "weighted_imbalance", "spread", "net_volume", "volume_imbalance",
    "trade_count", "avg_trade_size", "vwap_distance",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50", "high_low_range_50", "close_position",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "time_sin", "time_cos", "minutes_since_open",
]

RT_COST_BASE = 2.49
TARGET_TICKS = 19
STOP_TICKS = 7
TICK_VALUE = 1.25
HORIZON_BARS = 720  # 3600s / 5s

N_GROUPS = 10
K_TEST = 2
PURGE_BARS = 500
EMBARGO_BARS = 4600
DEV_DAYS_COUNT = 201
STAGE1_THRESHOLD = 0.50
WALL_CLOCK_LIMIT_S = 15 * 60

WIN_PNL = TARGET_TICKS * TICK_VALUE   # $23.75
LOSS_PNL = STOP_TICKS * TICK_VALUE    # $8.75

# Baseline references for sanity checks
PR38_SPLIT0_EXP = 1.065186
PR40_CUTOFF270_EXP = 3.016
PR40_CUTOFF270_TPD = 116.8
PR40_CUTOFF270_MIN_ACCT = 34000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def dsl_label(dsl):
    """Human-readable DSL label."""
    return "None" if dsl is None else f"${dsl}"


# ==========================================================================
# Data Loading (identical to PR #39/#40)
# ==========================================================================
def load_data():
    """Load 19:7 geometry Parquet data with sequential simulation columns."""
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files in {DATA_DIR}")

    print(f"  Loading {len(parquet_files)} Parquet files...")

    all_features = []
    all_labels = []
    all_day_raw = []
    all_fwd_return_1 = []
    all_bar_pos_in_day = []
    all_day_lengths = []
    all_tb_bars_held = []
    all_tb_exit_type = []
    all_minutes_since_open = []

    global_offset = 0
    day_offsets = {}

    for pf in parquet_files:
        df = pl.read_parquet(pf)
        if "is_warmup" in df.columns:
            df = df.filter(pl.col("is_warmup") == 0)

        n = len(df)
        features = df.select(NON_SPATIAL_FEATURES).to_numpy().astype(np.float64)
        labels = df["tb_label"].to_numpy().astype(np.float64).astype(np.int64)
        day_raw = df["day"].to_numpy()
        fwd1 = df["fwd_return_1"].to_numpy().astype(np.float64)
        tb_bh = df["tb_bars_held"].to_numpy().astype(np.float64)
        tb_et = df["tb_exit_type"].to_list()
        mso = df["minutes_since_open"].to_numpy().astype(np.float64)

        bar_pos = np.arange(n)

        all_features.append(features)
        all_labels.append(labels)
        all_day_raw.append(day_raw)
        all_fwd_return_1.append(fwd1)
        all_bar_pos_in_day.append(bar_pos)
        all_day_lengths.append(np.full(n, n))
        all_tb_bars_held.append(tb_bh)
        all_tb_exit_type.extend(tb_et)
        all_minutes_since_open.append(mso)

        unique_day = day_raw[0]
        day_offsets[unique_day] = (global_offset, n)
        global_offset += n

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    day_raw = np.concatenate(all_day_raw)
    fwd_return_1 = np.concatenate(all_fwd_return_1)
    bar_pos_in_day = np.concatenate(all_bar_pos_in_day)
    day_lengths = np.concatenate(all_day_lengths)
    tb_bars_held = np.concatenate(all_tb_bars_held)
    minutes_since_open = np.concatenate(all_minutes_since_open)

    unique_days_raw = sorted(set(day_raw.tolist()))
    day_map = {d: i + 1 for i, d in enumerate(unique_days_raw)}
    day_indices = np.array([day_map[d] for d in day_raw])

    tb_exit_type_arr = np.array(all_tb_exit_type)

    assert features.shape[1] == 20

    n_total = len(labels)
    print(f"  Total rows: {n_total}")
    print(f"  Days: {len(unique_days_raw)}, Range: day {min(day_indices)}-{max(day_indices)}")
    n_short = (labels == -1).sum()
    n_hold = (labels == 0).sum()
    n_long = (labels == 1).sum()
    print(f"  Labels: short={n_short} ({100*n_short/n_total:.1f}%), "
          f"hold={n_hold} ({100*n_hold/n_total:.1f}%), "
          f"long={n_long} ({100*n_long/n_total:.1f}%)")

    extra = {
        "fwd_return_1": fwd_return_1,
        "bar_pos_in_day": bar_pos_in_day,
        "day_lengths": day_lengths,
        "day_offsets": day_offsets,
        "day_raw": day_raw,
        "unique_days_raw": unique_days_raw,
        "day_map": day_map,
        "tb_bars_held": tb_bars_held,
        "tb_exit_type": tb_exit_type_arr,
        "minutes_since_open": minutes_since_open,
    }
    return features, labels, day_indices, unique_days_raw, extra


def compute_fwd_return_720(indices, extra):
    """Compute 720-bar forward return for given bar indices."""
    fwd1 = extra["fwd_return_1"]
    bar_pos = extra["bar_pos_in_day"]
    day_raw = extra["day_raw"]
    day_offsets = extra["day_offsets"]

    fwd_returns = np.zeros(len(indices))
    for k, idx in enumerate(indices):
        dr = day_raw[idx]
        day_start, day_len = day_offsets[dr]
        pos = bar_pos[idx]
        remaining = day_len - pos
        n_forward = min(HORIZON_BARS, remaining)
        fwd_returns[k] = np.sum(fwd1[idx:idx + n_forward])
    return fwd_returns


# ==========================================================================
# Purge and Embargo (identical to CPCV)
# ==========================================================================
def apply_purge_embargo_cpcv(train_indices, test_indices, purge_bars, embargo_bars, n_total):
    train_set = set(train_indices.tolist())
    test_sorted = np.sort(test_indices)
    if len(test_sorted) == 0:
        return np.array(sorted(train_set)), 0, 0

    boundaries = []
    block_start = test_sorted[0]
    prev = test_sorted[0]
    for idx in test_sorted[1:]:
        if idx != prev + 1:
            boundaries.append((block_start, prev))
            block_start = idx
        prev = idx
    boundaries.append((block_start, prev))

    purged = set()
    embargoed = set()
    for block_start, block_end in boundaries:
        for i in range(max(0, block_start - purge_bars), block_start):
            if i in train_set:
                purged.add(i)
        for i in range(block_end + 1, min(n_total, block_end + 1 + purge_bars)):
            if i in train_set:
                purged.add(i)
        for i in range(max(0, block_start - purge_bars - embargo_bars),
                       max(0, block_start - purge_bars)):
            if i in train_set:
                embargoed.add(i)
        for i in range(min(n_total, block_end + 1 + purge_bars),
                       min(n_total, block_end + 1 + purge_bars + embargo_bars)):
            if i in train_set:
                embargoed.add(i)

    excluded = purged | embargoed
    clean_train = np.array(sorted(train_set - excluded))
    return clean_train, len(purged), len(embargoed)


# ==========================================================================
# Two-Stage Training (identical to PR #38/#39/#40)
# ==========================================================================
def train_two_stage(features, labels, day_indices, extra,
                    train_indices, test_indices, val_indices, seed):
    set_seed(seed)

    ft_train = features[train_indices].copy()
    ft_val = features[val_indices].copy()
    ft_test = features[test_indices].copy()
    f_mean = np.nanmean(ft_train, axis=0)
    f_std = np.nanstd(ft_train, axis=0)
    f_std[f_std < 1e-10] = 1.0
    ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
    ft_val_z = np.nan_to_num((ft_val - f_mean) / f_std, nan=0.0)
    ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)

    labels_train = labels[train_indices]
    labels_val = labels[val_indices]
    labels_test = labels[test_indices]

    # STAGE 1: Binary is_directional
    s1_train_labels = (labels_train != 0).astype(np.int64)
    s1_val_labels = (labels_val != 0).astype(np.int64)

    params_s1 = {**TUNED_XGB_PARAMS_BINARY, "random_state": seed}
    clf_s1 = xgb.XGBClassifier(**params_s1)
    if len(ft_val_z) > 0:
        clf_s1.fit(ft_train_z, s1_train_labels,
                   eval_set=[(ft_val_z, s1_val_labels)], verbose=False)
    else:
        clf_s1.fit(ft_train_z, s1_train_labels, verbose=False)

    s1_pred_proba = clf_s1.predict_proba(ft_test_z)[:, 1]

    # STAGE 2: Binary is_long on directional-only bars
    dir_mask_train = (labels_train != 0)
    dir_mask_val = (labels_val != 0)

    s2_train_z = ft_train_z[dir_mask_train]
    s2_val_z = ft_val_z[dir_mask_val]
    s2_train_labels = (labels_train[dir_mask_train] == 1).astype(np.int64)
    s2_val_labels = (labels_val[dir_mask_val] == 1).astype(np.int64)

    params_s2 = {**TUNED_XGB_PARAMS_BINARY, "random_state": seed}
    clf_s2 = xgb.XGBClassifier(**params_s2)
    if len(s2_val_z) > 0:
        clf_s2.fit(s2_train_z, s2_train_labels,
                   eval_set=[(s2_val_z, s2_val_labels)], verbose=False)
    else:
        clf_s2.fit(s2_train_z, s2_train_labels, verbose=False)

    s2_pred_proba = clf_s2.predict_proba(ft_test_z)[:, 1]

    # COMBINE: Stage 1 P(dir) > T=0.50 -> use Stage 2 direction
    combined_pred = np.zeros(len(labels_test), dtype=np.int64)
    s1_dir_mask = (s1_pred_proba > STAGE1_THRESHOLD)
    s2_direction = np.where(s2_pred_proba > 0.5, 1, -1)
    combined_pred[s1_dir_mask] = s2_direction[s1_dir_mask]

    del clf_s1, clf_s2

    return {
        "combined_pred": combined_pred,
        "labels_test": labels_test,
        "test_indices": test_indices,
    }


# ==========================================================================
# Bar-Level PnL (for verification against CPCV PR #38)
# ==========================================================================
def compute_bar_level_pnl(true_labels, pred_labels, test_indices, extra, rt_cost):
    n = len(true_labels)
    trades = (pred_labels != 0)
    dir_bars = (true_labels != 0)
    hold_bars = (true_labels == 0)
    dir_trades = trades & dir_bars
    hold_trades = trades & hold_bars

    n_trades = int(trades.sum())
    if n_trades == 0:
        return {"expectancy": 0.0, "n_trades": 0, "trade_rate": 0.0,
                "hold_fraction": 0.0, "dir_accuracy": 0.0}

    correct = (pred_labels == true_labels) & dir_trades
    wrong = (pred_labels != true_labels) & dir_trades

    pnl = np.zeros(n)
    pnl[correct] = WIN_PNL - rt_cost
    pnl[wrong] = -LOSS_PNL - rt_cost

    n_hold_trades = int(hold_trades.sum())
    if n_hold_trades > 0:
        hold_indices_global = test_indices[hold_trades]
        fwd_720 = compute_fwd_return_720(hold_indices_global, extra)
        hold_pred_signs = np.sign(pred_labels[hold_trades])
        pnl[hold_trades] = fwd_720 * TICK_VALUE * hold_pred_signs - rt_cost

    traded_pnl = pnl[trades]
    expectancy = float(traded_pnl.sum()) / n_trades

    both_nonzero = (pred_labels != 0) & (true_labels != 0)
    dir_correct = (pred_labels == true_labels) & both_nonzero
    dir_acc = float(dir_correct.sum() / both_nonzero.sum()) if both_nonzero.sum() > 0 else 0.0

    return {
        "expectancy": float(expectancy),
        "n_trades": n_trades,
        "trade_rate": float(n_trades / n),
        "hold_fraction": float(n_hold_trades / n_trades),
        "dir_accuracy": dir_acc,
    }


# ==========================================================================
# Sequential Execution Simulator (with time cutoff + daily stop loss)
# ==========================================================================
def simulate_sequential(test_indices, predictions, labels, extra, rt_cost,
                        cutoff=270, daily_stop_loss=None):
    """Simulate sequential 1-contract execution with time-of-day cutoff
    and optional daily stop loss.

    daily_stop_loss: if not None, when cumulative intra-day P&L drops to
    -$daily_stop_loss, stop trading for the rest of that day. The triggering
    trade completes; no new trades are entered after.

    Returns (trade_log, daily_pnl_records, stats).
    """
    day_raw = extra["day_raw"]
    tb_bars_held = extra["tb_bars_held"]
    minutes_since_open = extra["minutes_since_open"]

    sorted_order = np.argsort(test_indices)
    test_sorted = test_indices[sorted_order]
    pred_sorted = predictions[sorted_order]
    label_sorted = labels[sorted_order]

    test_days_raw = day_raw[test_sorted]
    unique_test_days = sorted(set(test_days_raw.tolist()))

    trade_log = []
    daily_pnl_records = []
    total_hold_skips = 0
    total_time_skips = 0
    total_dsl_skips = 0
    total_bars_seen = 0

    for day_val in unique_test_days:
        day_mask = (test_days_raw == day_val)
        day_positions = np.where(day_mask)[0]
        n_day_bars = len(day_positions)

        day_pnl_total = 0.0
        day_n_trades = 0
        day_hold_skips = 0
        day_time_skips = 0
        day_dsl_skips = 0
        dsl_triggered = False

        i = 0
        while i < n_day_bars:
            test_pos = day_positions[i]
            global_idx = test_sorted[test_pos]
            pred = pred_sorted[test_pos]

            if pred == 0:
                day_hold_skips += 1
                i += 1
                continue

            # pred != 0: this is an entry opportunity.

            # Check daily stop loss FIRST (before time cutoff)
            if daily_stop_loss is not None and dsl_triggered:
                day_dsl_skips += 1
                i += 1
                continue

            # Check time cutoff
            mso = minutes_since_open[global_idx]
            if mso > cutoff:
                day_time_skips += 1
                i += 1
                continue

            # ENTER position
            label = label_sorted[test_pos]
            bars_held = max(1, int(tb_bars_held[global_idx]))
            exit_type = extra["tb_exit_type"][global_idx]

            # PnL computation
            if label != 0:
                if np.sign(pred) == np.sign(label):
                    pnl = WIN_PNL - rt_cost
                else:
                    pnl = -LOSS_PNL - rt_cost
            else:
                fwd_720 = compute_fwd_return_720(np.array([global_idx]), extra)
                pnl = float(fwd_720[0]) * TICK_VALUE * float(np.sign(pred)) - rt_cost

            trade_log.append({
                'day': int(day_val),
                'entry_bar_global': int(global_idx),
                'direction': int(pred),
                'true_label': int(label),
                'pnl': float(pnl),
                'bars_held': bars_held,
                'minutes_since_open': float(mso),
                'exit_type': str(exit_type),
            })

            day_pnl_total += pnl
            day_n_trades += 1

            # Check DSL trigger AFTER trade completes
            if daily_stop_loss is not None and day_pnl_total <= -daily_stop_loss:
                dsl_triggered = True

            # Advance past holding period
            i += bars_held

        daily_pnl_records.append({
            'day': int(day_val),
            'pnl': float(day_pnl_total),
            'n_trades': day_n_trades,
            'hold_skips': day_hold_skips,
            'time_skips': day_time_skips,
            'dsl_skips': day_dsl_skips,
            'dsl_triggered': dsl_triggered,
            'total_bars': n_day_bars,
        })

        total_hold_skips += day_hold_skips
        total_time_skips += day_time_skips
        total_dsl_skips += day_dsl_skips
        total_bars_seen += n_day_bars

    n_trades = len(trade_log)
    n_dsl_triggered_days = sum(1 for d in daily_pnl_records if d['dsl_triggered'])
    stats = {
        'total_hold_skips': total_hold_skips,
        'total_time_skips': total_time_skips,
        'total_dsl_skips': total_dsl_skips,
        'total_bars_seen': total_bars_seen,
        'n_trades': n_trades,
        'n_dsl_triggered_days': n_dsl_triggered_days,
        'n_test_days': len(unique_test_days),
    }

    return trade_log, daily_pnl_records, stats


# ==========================================================================
# Risk Metrics
# ==========================================================================
def compute_equity_curve(trade_log):
    pnls = [t['pnl'] for t in trade_log]
    return np.cumsum(pnls) if pnls else np.array([])


def compute_max_drawdown(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    peak = np.maximum(peak, 0.0)
    drawdowns = peak - equity_curve
    return float(np.max(drawdowns))


def compute_max_consecutive_losses(trade_log):
    if not trade_log:
        return 0
    max_streak = 0
    current_streak = 0
    for t in trade_log:
        if t['pnl'] < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def compute_drawdown_duration_days(daily_pnl_records):
    if not daily_pnl_records:
        return 0
    daily_pnls = [r['pnl'] for r in daily_pnl_records]
    equity = np.cumsum(daily_pnls)

    peak = 0.0
    peak_day = 0
    max_duration = 0
    for i, eq in enumerate(equity):
        if eq >= peak:
            peak = eq
            peak_day = i
        else:
            duration = i - peak_day
            max_duration = max(max_duration, duration)
    if equity[-1] < peak:
        max_duration = max(max_duration, len(equity) - 1 - peak_day)
    return max_duration


def compute_account_sizing(split_equity_curves, max_account=50000, step=500):
    """For each account level ($500-$50K, $500 steps), count surviving paths."""
    results = []
    n_paths = len(split_equity_curves)

    path_max_dd = []
    for ec in split_equity_curves:
        if len(ec) > 0:
            peak = np.maximum.accumulate(np.concatenate([[0.0], ec]))
            curve_with_zero = np.concatenate([[0.0], ec])
            dd = peak - curve_with_zero
            path_max_dd.append(float(np.max(dd)))
        else:
            path_max_dd.append(0.0)

    for account_size in range(500, max_account + 1, step):
        survived = sum(1 for dd in path_max_dd if account_size > dd)
        results.append({
            'account_size': account_size,
            'survival_count': survived,
            'survival_rate': survived / n_paths if n_paths > 0 else 0.0,
        })

    return results, path_max_dd


# ==========================================================================
# CPCV Group Assignment (identical)
# ==========================================================================
def assign_groups(day_indices_dev, n_groups):
    unique_days = sorted(set(day_indices_dev.tolist()))
    n_days = len(unique_days)
    days_per_group = n_days // n_groups
    remainder = n_days % n_groups

    groups = {}
    day_to_group = {}
    start = 0
    for g in range(n_groups):
        size = days_per_group + (1 if g < remainder else 0)
        group_days = unique_days[start:start + size]
        groups[g] = group_days
        for d in group_days:
            day_to_group[d] = g
        start += size
    return groups, day_to_group


# ==========================================================================
# Compute Per-Split Simulation Metrics
# ==========================================================================
def compute_split_sim_metrics(trade_log, daily_pnl_records, stats):
    """Compute per-split metrics from a single simulation run."""
    n_trades = len(trade_log)
    n_test_days = len(daily_pnl_records)

    trades_per_day = [r['n_trades'] for r in daily_pnl_records]

    seq_pnls = [t['pnl'] for t in trade_log]
    seq_expectancy = float(np.mean(seq_pnls)) if seq_pnls else 0.0
    seq_win_rate = float(np.mean([p > 0 for p in seq_pnls])) if seq_pnls else 0.0

    # Win rate on directional bars only
    dir_trades = [t for t in trade_log if t['true_label'] != 0]
    seq_win_rate_dir = (float(np.mean([t['pnl'] > 0 for t in dir_trades]))
                       if dir_trades else 0.0)

    # Avg bars held
    avg_bars_held = (float(np.mean([t['bars_held'] for t in trade_log]))
                    if trade_log else 0.0)

    # Timeout fraction
    n_timeout = sum(1 for t in trade_log if t['true_label'] == 0)
    timeout_fraction = n_timeout / n_trades if n_trades > 0 else 0.0

    # DSL metrics
    dsl_trigger_rate = (stats['n_dsl_triggered_days'] / stats['n_test_days']
                       if stats['n_test_days'] > 0 else 0.0)

    daily_pnls = [r['pnl'] for r in daily_pnl_records]

    equity_curve = compute_equity_curve(trade_log)
    max_dd = compute_max_drawdown(equity_curve)
    max_consec_losses = compute_max_consecutive_losses(trade_log)
    dd_duration = compute_drawdown_duration_days(daily_pnl_records)

    # Daily PnL statistics
    daily_arr = np.array(daily_pnls) if daily_pnls else np.array([0.0])
    daily_pnl_mean = float(np.mean(daily_arr))
    daily_pnl_std = float(np.std(daily_arr, ddof=1)) if len(daily_arr) > 1 else 0.0
    daily_pnl_skew = float(scipy_stats.skew(daily_arr)) if len(daily_arr) > 2 else 0.0
    daily_pnl_kurtosis = float(scipy_stats.kurtosis(daily_arr)) if len(daily_arr) > 3 else 0.0

    return {
        "n_trades": n_trades,
        "n_test_days": n_test_days,
        "trades_per_day_mean": float(np.mean(trades_per_day)) if trades_per_day else 0.0,
        "trades_per_day_std": float(np.std(trades_per_day, ddof=1)) if len(trades_per_day) > 1 else 0.0,
        "expectancy": seq_expectancy,
        "win_rate": seq_win_rate,
        "win_rate_dir_bars": seq_win_rate_dir,
        "avg_bars_held": avg_bars_held,
        "timeout_fraction": timeout_fraction,
        "daily_pnl_mean": daily_pnl_mean,
        "daily_pnl_std": daily_pnl_std,
        "daily_pnl_skew": daily_pnl_skew,
        "daily_pnl_kurtosis": daily_pnl_kurtosis,
        "max_drawdown": max_dd,
        "max_consecutive_losses": max_consec_losses,
        "drawdown_duration_days": dd_duration,
        "dsl_trigger_rate": dsl_trigger_rate,
        "equity_curve": equity_curve.tolist() if len(equity_curve) > 0 else [],
        "trade_log": trade_log,
        "daily_pnl_records": daily_pnl_records,
    }


# ==========================================================================
# Recovery Sacrifice Analysis
# ==========================================================================
def compute_recovery_sacrifice(baseline_daily_records):
    """Using DSL=None baseline trade logs, compute per-DSL threshold:
    For each trading day, track cumulative intra-day P&L path.
    For each DSL level $X:
      (a) Days where cumulative P&L dipped below -$X at some point
      (b) Of those, days that ended positive (recovered)
      Recovery sacrifice rate = (b) / (a)
    """
    # baseline_daily_records is list of per-split lists of daily records
    # We also need the trade-level data per day to compute cumulative path.
    # But we already have the trade logs from baseline.
    # We'll compute from the trade logs directly.
    pass  # Implemented inline below using trade logs


def compute_recovery_sacrifice_from_trade_logs(all_baseline_trade_logs, all_baseline_daily_records):
    """Compute recovery sacrifice using baseline (DSL=None) trade logs.

    For each DSL level, for each day:
    - Track cumulative intra-day P&L after each trade
    - Check if it ever dips below -$X (DSL trigger)
    - Check if final day P&L is positive (recovery)

    Returns dict keyed by DSL level -> {dipped_days, recovered_days, sacrifice_rate}
    """
    dsl_thresholds = [d for d in DSL_LEVELS if d is not None]

    # Collect per-day cumulative P&L paths across all splits
    # Each day may appear in multiple splits (different test sets)
    # We treat each (split, day) as an independent observation
    day_paths = []  # list of (day_id, cumulative_pnl_list, final_pnl)

    for split_idx, trade_log in enumerate(all_baseline_trade_logs):
        # Group trades by day
        day_trades = {}
        for t in trade_log:
            day = t['day']
            if day not in day_trades:
                day_trades[day] = []
            day_trades[day].append(t['pnl'])

        for day, pnls in day_trades.items():
            cum_path = np.cumsum(pnls)
            day_paths.append({
                'split_idx': split_idx,
                'day': day,
                'cum_min': float(np.min(cum_path)),
                'final_pnl': float(cum_path[-1]),
                'n_trades': len(pnls),
            })

    # For each DSL threshold, compute sacrifice
    sacrifice_results = {}
    for dsl in dsl_thresholds:
        dipped_count = 0
        recovered_count = 0
        for dp in day_paths:
            if dp['cum_min'] <= -dsl:
                dipped_count += 1
                if dp['final_pnl'] > 0:
                    recovered_count += 1

        sacrifice_rate = recovered_count / dipped_count if dipped_count > 0 else 0.0
        sacrifice_results[dsl] = {
            'dsl_threshold': dsl,
            'total_day_observations': len(day_paths),
            'dipped_days': dipped_count,
            'dipped_rate': dipped_count / len(day_paths) if day_paths else 0.0,
            'recovered_days': recovered_count,
            'sacrifice_rate': sacrifice_rate,
        }

    return sacrifice_results, day_paths


# ==========================================================================
# Intra-day P&L Path Statistics
# ==========================================================================
def compute_intraday_pnl_stats(all_baseline_trade_logs):
    """For each day in the DSL=None baseline, compute:
    - Intra-day min cumulative P&L
    - Intra-day max cumulative P&L
    - Final P&L
    - Trade count
    - Max consecutive losses within the day
    """
    # Gather per (split, day) stats
    intraday_stats = []

    for split_idx, trade_log in enumerate(all_baseline_trade_logs):
        day_trades = {}
        for t in trade_log:
            day = t['day']
            if day not in day_trades:
                day_trades[day] = []
            day_trades[day].append(t['pnl'])

        for day, pnls in day_trades.items():
            cum_path = np.cumsum(pnls)
            # Max consecutive losses
            max_consec = 0
            current_streak = 0
            for p in pnls:
                if p < 0:
                    current_streak += 1
                    max_consec = max(max_consec, current_streak)
                else:
                    current_streak = 0

            intraday_stats.append({
                'split_idx': split_idx,
                'day': day,
                'intraday_min_pnl': float(np.min(cum_path)),
                'intraday_max_pnl': float(np.max(cum_path)),
                'final_pnl': float(cum_path[-1]),
                'n_trades': len(pnls),
                'max_consecutive_losses': max_consec,
            })

    return intraday_stats


# ==========================================================================
# Main CPCV Loop with DSL Sweep
# ==========================================================================
def run_cpcv_with_dsl_sweep(features, labels, day_indices, extra,
                            mve_only=False, cpcv_baseline=None):
    """Run 45-split CPCV, train once per split, simulate 9 DSL levels each.

    Returns (all_split_results, abort_reason).
    """
    n_total = len(labels)
    t0_global = time.time()

    dev_mask = (day_indices >= 1) & (day_indices <= DEV_DAYS_COUNT)
    dev_indices = np.where(dev_mask)[0]
    dev_day_indices = day_indices[dev_mask]

    groups, day_to_group = assign_groups(dev_day_indices, N_GROUPS)
    dev_group_arr = np.array([day_to_group.get(d, -1) for d in dev_day_indices])

    print(f"\n  CPCV groups:")
    for g in range(N_GROUPS):
        g_bars = int((dev_group_arr == g).sum())
        print(f"    Group {g}: {len(groups[g])} days, {g_bars} bars")

    splits = list(combinations(range(N_GROUPS), K_TEST))
    n_splits_to_run = 1 if mve_only else len(splits)
    print(f"  Running {n_splits_to_run} of {len(splits)} splits")

    all_split_results = []

    for s_idx in range(n_splits_to_run):
        g1, g2 = splits[s_idx]
        t0_split = time.time()
        split_seed = SEED + s_idx

        test_groups = {g1, g2}
        train_groups = set(range(N_GROUPS)) - test_groups

        test_mask_local = np.isin(dev_group_arr, list(test_groups))
        train_mask_local = np.isin(dev_group_arr, list(train_groups))

        test_indices_local = np.where(test_mask_local)[0]
        train_indices_local = np.where(train_mask_local)[0]

        test_indices_global = dev_indices[test_indices_local]
        train_indices_global_pre = dev_indices[train_indices_local]

        clean_train_global, n_purged, n_embargoed = apply_purge_embargo_cpcv(
            train_indices_global_pre, test_indices_global,
            PURGE_BARS, EMBARGO_BARS, n_total)

        # Internal train/val split
        train_days = sorted(set(day_indices[clean_train_global].tolist()))
        n_val_days = max(1, len(train_days) // 5)
        val_day_set = set(train_days[-n_val_days:])
        inner_train_day_set = set(train_days[:-n_val_days])

        inner_train_global = np.array([i for i in clean_train_global
                                       if day_indices[i] in inner_train_day_set])
        inner_val_global = np.array([i for i in clean_train_global
                                     if day_indices[i] in val_day_set])

        if len(inner_val_global) > 0 and len(inner_train_global) > 0:
            val_sorted = np.sort(inner_val_global)
            val_start = val_sorted[0]
            val_end = val_sorted[-1]
            inner_train_global = inner_train_global[
                (inner_train_global < val_start - PURGE_BARS) |
                (inner_train_global > val_end + PURGE_BARS)
            ]

        # Train ONCE per split (same as PR #38/#39/#40)
        fold_result = train_two_stage(
            features, labels, day_indices, extra,
            inner_train_global, test_indices_global, inner_val_global,
            seed=split_seed)

        combined_pred = fold_result["combined_pred"]
        labels_test = fold_result["labels_test"]

        # Bar-level PnL (for SC-S1 verification)
        bar_pnl = compute_bar_level_pnl(
            labels_test, combined_pred, test_indices_global, extra, RT_COST_BASE)

        # Run simulation for each DSL level (fixed cutoff=270)
        dsl_results = {}
        for dsl in DSL_LEVELS:
            trade_log, daily_records, stats = simulate_sequential(
                test_indices_global, combined_pred, labels_test,
                extra, RT_COST_BASE, cutoff=FIXED_CUTOFF, daily_stop_loss=dsl)
            sim_metrics = compute_split_sim_metrics(trade_log, daily_records, stats)
            dsl_results[dsl_label(dsl)] = sim_metrics

        elapsed = time.time() - t0_split

        split_record = {
            "split_idx": s_idx,
            "test_groups": (g1, g2),
            "n_test": len(labels_test),
            "n_purged": n_purged,
            "n_embargoed": n_embargoed,
            "bar_level_expectancy": bar_pnl["expectancy"],
            "bar_level_n_trades": bar_pnl["n_trades"],
            "bar_level_dir_accuracy": bar_pnl["dir_accuracy"],
            "dsl_results": dsl_results,
            "wall_seconds": elapsed,
        }
        all_split_results.append(split_record)

        # Print progress
        dsl_none = dsl_results["None"]
        dsl_1500 = dsl_results["$1500"]
        if (s_idx + 1) % 5 == 0 or s_idx == 0:
            print(f"    Split {s_idx:2d} ({g1},{g2}): bar_exp=${bar_pnl['expectancy']:+.4f}, "
                  f"dsl=None exp=${dsl_none['expectancy']:+.4f} ({dsl_none['trades_per_day_mean']:.0f}t/d), "
                  f"dsl=$1500 exp=${dsl_1500['expectancy']:+.4f} ({dsl_1500['trades_per_day_mean']:.0f}t/d), "
                  f"{elapsed:.1f}s")

        # MVE checks (split 0 only)
        if s_idx == 0:
            # SC-S1: bar-level exp match PR #38
            if cpcv_baseline is not None:
                ref_exp = cpcv_baseline.get("split_0_base_exp", None)
                if ref_exp is not None:
                    diff = abs(bar_pnl["expectancy"] - ref_exp)
                    if diff > 0.05:
                        print(f"  *** MVE ABORT: bar-level exp differs from PR #38 by ${diff:.4f} (>$0.05) ***")
                        return all_split_results, f"MVE: bar-level exp mismatch (delta=${diff:.4f})"
                    print(f"    MVE PASS SC-S1: bar_exp=${bar_pnl['expectancy']:.4f} vs ref={ref_exp:.6f} (delta=${diff:.4f})")

            # DSL=None should match PR #40 cutoff=270
            if dsl_none['n_trades'] == 0:
                print(f"  *** MVE ABORT: DSL=None produced 0 trades ***")
                return all_split_results, "MVE: DSL=None zero trades"

            # DSL=$1500 should produce fewer or equal trades than DSL=None
            if dsl_1500['trades_per_day_mean'] > dsl_none['trades_per_day_mean'] + 0.001:
                print(f"  *** MVE ABORT: DSL=$1500 ({dsl_1500['trades_per_day_mean']:.1f}) > DSL=None ({dsl_none['trades_per_day_mean']:.1f}) ***")
                return all_split_results, "MVE: monotonicity violation (DSL=$1500 > DSL=None)"

            # NaN check
            if np.isnan(dsl_none['expectancy']) or np.isnan(dsl_1500['expectancy']):
                print(f"  *** MVE ABORT: NaN in expectancy ***")
                return all_split_results, "MVE: NaN in expectancy"

            print(f"    MVE PASS: DSL=None tpd={dsl_none['trades_per_day_mean']:.1f}, "
                  f"DSL=$1500 tpd={dsl_1500['trades_per_day_mean']:.1f} (monotonic)")

        # Wall-clock check
        elapsed_total = time.time() - t0_global
        if elapsed_total > WALL_CLOCK_LIMIT_S:
            print(f"  *** ABORT: wall-clock {elapsed_total:.0f}s > {WALL_CLOCK_LIMIT_S}s ***")
            return all_split_results, f"Wall-clock limit exceeded ({elapsed_total:.0f}s)"

    return all_split_results, None


# ==========================================================================
# Aggregation Per DSL Level
# ==========================================================================
def aggregate_per_dsl(all_split_results):
    """For each DSL level, aggregate metrics across all 45 splits."""
    dsl_summaries = {}

    for dsl in DSL_LEVELS:
        key = dsl_label(dsl)
        per_split_exp = []
        per_split_tpd = []
        per_split_daily_mean = []
        per_split_daily_std = []
        per_split_daily_skew = []
        per_split_daily_kurtosis = []
        per_split_max_dd = []
        per_split_max_consec = []
        per_split_dd_dur = []
        per_split_win_rate = []
        per_split_dsl_trigger_rate = []

        all_trade_pnls = []
        all_daily_pnls = []
        all_equity_curves = []
        all_trade_logs = []
        all_daily_pnl_records = []

        for sr in all_split_results:
            cr = sr["dsl_results"][key]
            per_split_exp.append(cr["expectancy"])
            per_split_tpd.append(cr["trades_per_day_mean"])
            per_split_daily_mean.append(cr["daily_pnl_mean"])
            per_split_daily_std.append(cr["daily_pnl_std"])
            per_split_daily_skew.append(cr["daily_pnl_skew"])
            per_split_daily_kurtosis.append(cr["daily_pnl_kurtosis"])
            per_split_max_dd.append(cr["max_drawdown"])
            per_split_max_consec.append(cr["max_consecutive_losses"])
            per_split_dd_dur.append(cr["drawdown_duration_days"])
            per_split_win_rate.append(cr["win_rate"])
            per_split_dsl_trigger_rate.append(cr["dsl_trigger_rate"])

            all_trade_pnls.extend([t['pnl'] for t in cr['trade_log']])
            all_daily_pnls.extend([d['pnl'] for d in cr['daily_pnl_records']])
            all_equity_curves.append(np.array(cr['equity_curve']))
            all_trade_logs.append(cr['trade_log'])
            all_daily_pnl_records.append(cr['daily_pnl_records'])

        per_split_exp = np.array(per_split_exp)
        per_split_tpd = np.array(per_split_tpd)
        per_split_max_dd = np.array(per_split_max_dd)

        # Account sizing
        account_sizing, path_max_dds = compute_account_sizing(all_equity_curves)

        # Find min_account thresholds
        min_account_all = None
        min_account_95 = None
        for entry in account_sizing:
            if entry['survival_rate'] >= 0.95 and min_account_95 is None:
                min_account_95 = entry['account_size']
            if entry['survival_rate'] >= 1.0 and min_account_all is None:
                min_account_all = entry['account_size']

        if min_account_all is None:
            worst_dd = max(path_max_dds) if path_max_dds else 0
            min_account_all = int(np.ceil(worst_dd / 500) * 500) + 500

        if min_account_95 is None:
            sorted_dds = sorted(path_max_dds)
            idx_95 = int(np.ceil(0.95 * len(sorted_dds))) - 1
            dd_95 = sorted_dds[idx_95] if idx_95 < len(sorted_dds) else max(sorted_dds)
            min_account_95 = int(np.ceil(dd_95 / 500) * 500) + 500

        # Calmar ratio
        mean_daily = float(np.mean(per_split_daily_mean))
        annual_pnl = mean_daily * 251
        worst_dd = float(np.max(per_split_max_dd)) if len(per_split_max_dd) > 0 else 1.0
        calmar = annual_pnl / worst_dd if worst_dd > 0 else 0.0

        # Sharpe
        all_daily_arr = np.array(all_daily_pnls)
        if len(all_daily_arr) > 1 and np.std(all_daily_arr) > 0:
            sharpe = (np.mean(all_daily_arr) / np.std(all_daily_arr)) * np.sqrt(251)
        else:
            sharpe = 0.0

        # Daily PnL percentiles
        daily_pnl_pcts = {}
        if len(all_daily_arr) > 0:
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                daily_pnl_pcts[f"p{p}"] = float(np.percentile(all_daily_arr, p))

        # Aggregated daily statistics
        agg_daily_skew = float(scipy_stats.skew(all_daily_arr)) if len(all_daily_arr) > 2 else 0.0
        agg_daily_kurtosis = float(scipy_stats.kurtosis(all_daily_arr)) if len(all_daily_arr) > 3 else 0.0

        dsl_summaries[key] = {
            "dsl_level": dsl,
            "dsl_label": key,
            "trades_per_day": float(np.mean(per_split_tpd)),
            "exp_per_trade": float(np.mean(per_split_exp)),
            "daily_pnl_mean": mean_daily,
            "daily_pnl_std": float(np.mean(per_split_daily_std)),
            "daily_pnl_skew": agg_daily_skew,
            "daily_pnl_kurtosis": agg_daily_kurtosis,
            "dd_worst": float(np.max(per_split_max_dd)),
            "dd_median": float(np.median(per_split_max_dd)),
            "min_acct_all": min_account_all,
            "min_acct_95": min_account_95,
            "win_rate": float(np.mean(per_split_win_rate)),
            "dsl_trigger_rate": float(np.mean(per_split_dsl_trigger_rate)),
            "calmar": calmar,
            "sharpe": float(sharpe),
            "annual_pnl": annual_pnl,
            "daily_pnl_percentiles": daily_pnl_pcts,
            # Per-split arrays
            "per_split_exp": per_split_exp.tolist(),
            "per_split_max_dd": per_split_max_dd.tolist(),
            "per_split_tpd": per_split_tpd.tolist(),
            "path_max_dds": path_max_dds,
            "account_sizing": account_sizing,
            "all_equity_curves": all_equity_curves,
            "all_trade_logs": all_trade_logs,
            "all_daily_pnl_records": all_daily_pnl_records,
        }

    return dsl_summaries


# ==========================================================================
# Optimal DSL Selection
# ==========================================================================
def select_optimal_dsl(dsl_summaries, sacrifice_results):
    """Select per spec rules:
    Loosest (highest dollar) threshold achieving ALL of:
      SC-1: min_acct <= $20K
      SC-2: annual PnL >= $50K
      SC-3: Calmar >= 2.0
      SC-4: recovery sacrifice <= 20%

    Fallback: max Calmar subject to annual PnL >= $40K.
    Final fallback: min min_account_all.
    """
    # Check each non-None DSL level from loosest to tightest
    dsl_dollar_levels = [d for d in DSL_LEVELS if d is not None]
    dsl_dollar_levels_sorted = sorted(dsl_dollar_levels, reverse=True)  # loosest first

    # Rule 1: All four criteria
    for dsl in dsl_dollar_levels_sorted:
        key = dsl_label(dsl)
        cs = dsl_summaries[key]
        sac = sacrifice_results.get(dsl, {})
        sac_rate = sac.get('sacrifice_rate', 0.0)

        if (cs["min_acct_all"] <= 20000 and
            cs["annual_pnl"] >= 50000 and
            cs["calmar"] >= 2.0 and
            sac_rate <= 0.20):
            return key, (f"Rule 1: loosest DSL achieving min_acct<=$20K, annual_pnl>=$50K, "
                        f"Calmar>=2.0, sacrifice<=20%")

    # Rule 2 (fallback): max Calmar subject to annual PnL >= $40K
    best_calmar = -float('inf')
    best_key = None
    for dsl in dsl_dollar_levels_sorted:
        key = dsl_label(dsl)
        cs = dsl_summaries[key]
        if cs["annual_pnl"] >= 40000 and cs["calmar"] > best_calmar:
            best_calmar = cs["calmar"]
            best_key = key

    if best_key is not None:
        return best_key, f"Rule 2: max Calmar (>={best_calmar:.2f}) subject to annual PnL >= $40K (relaxed SC-2)"

    # Rule 3 (final fallback): min min_account_all
    best_min_acct = float('inf')
    best_key = "None"
    for dsl in dsl_dollar_levels_sorted:
        key = dsl_label(dsl)
        cs = dsl_summaries[key]
        if cs["min_acct_all"] < best_min_acct:
            best_min_acct = cs["min_acct_all"]
            best_key = key

    return best_key, f"Rule 3: min min_account_all (${best_min_acct:,})"


# ==========================================================================
# Output Writers
# ==========================================================================
def write_dsl_sweep_csv(dsl_summaries, sacrifice_results):
    """Write 9-row DSL sweep table."""
    fieldnames = ["dsl_level", "trades_per_day", "exp_per_trade",
                  "daily_pnl_mean", "daily_pnl_std", "daily_pnl_skew", "daily_pnl_kurtosis",
                  "dd_worst", "dd_median", "min_acct_all", "min_acct_95",
                  "calmar", "sharpe", "annual_pnl", "win_rate",
                  "dsl_trigger_rate", "recovery_sacrifice_rate"]
    with open(RESULTS_DIR / "dsl_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dsl in DSL_LEVELS:
            key = dsl_label(dsl)
            cs = dsl_summaries[key]
            sac = sacrifice_results.get(dsl, {})
            sac_rate = sac.get('sacrifice_rate', 0.0) if dsl is not None else 0.0
            writer.writerow({
                "dsl_level": key,
                "trades_per_day": f"{cs['trades_per_day']:.2f}",
                "exp_per_trade": f"{cs['exp_per_trade']:.4f}",
                "daily_pnl_mean": f"{cs['daily_pnl_mean']:.2f}",
                "daily_pnl_std": f"{cs['daily_pnl_std']:.2f}",
                "daily_pnl_skew": f"{cs['daily_pnl_skew']:.4f}",
                "daily_pnl_kurtosis": f"{cs['daily_pnl_kurtosis']:.4f}",
                "dd_worst": f"{cs['dd_worst']:.2f}",
                "dd_median": f"{cs['dd_median']:.2f}",
                "min_acct_all": cs['min_acct_all'],
                "min_acct_95": cs['min_acct_95'],
                "calmar": f"{cs['calmar']:.4f}",
                "sharpe": f"{cs['sharpe']:.4f}",
                "annual_pnl": f"{cs['annual_pnl']:.2f}",
                "win_rate": f"{cs['win_rate']:.4f}",
                "dsl_trigger_rate": f"{cs['dsl_trigger_rate']:.4f}",
                "recovery_sacrifice_rate": f"{sac_rate:.4f}",
            })
    print(f"  Wrote dsl_sweep.csv")


def write_recovery_sacrifice_csv(sacrifice_results):
    """Write 8-row recovery sacrifice table (one per non-None DSL)."""
    fieldnames = ["dsl_threshold", "total_day_observations", "dipped_days",
                  "dipped_rate", "recovered_days", "sacrifice_rate"]
    with open(RESULTS_DIR / "recovery_sacrifice.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dsl in [d for d in DSL_LEVELS if d is not None]:
            sac = sacrifice_results[dsl]
            writer.writerow({
                "dsl_threshold": f"${dsl}",
                "total_day_observations": sac['total_day_observations'],
                "dipped_days": sac['dipped_days'],
                "dipped_rate": f"{sac['dipped_rate']:.4f}",
                "recovered_days": sac['recovered_days'],
                "sacrifice_rate": f"{sac['sacrifice_rate']:.4f}",
            })
    print(f"  Wrote recovery_sacrifice.csv")


def write_intraday_pnl_stats_csv(intraday_stats):
    """Write per-day intra-day P&L path statistics from baseline."""
    fieldnames = ["split_idx", "day", "intraday_min_pnl", "intraday_max_pnl",
                  "final_pnl", "n_trades", "max_consecutive_losses"]
    with open(RESULTS_DIR / "intraday_pnl_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in intraday_stats:
            writer.writerow({
                "split_idx": s['split_idx'],
                "day": s['day'],
                "intraday_min_pnl": f"{s['intraday_min_pnl']:.4f}",
                "intraday_max_pnl": f"{s['intraday_max_pnl']:.4f}",
                "final_pnl": f"{s['final_pnl']:.4f}",
                "n_trades": s['n_trades'],
                "max_consecutive_losses": s['max_consecutive_losses'],
            })
    print(f"  Wrote intraday_pnl_stats.csv")


def write_optimal_trade_log_csv(dsl_summaries, optimal_key, all_split_results):
    """Write trade log at recommended DSL."""
    fieldnames = ["split_idx", "day", "entry_bar_global", "direction",
                  "true_label", "exit_type", "pnl", "bars_held", "minutes_since_open"]
    with open(RESULTS_DIR / "optimal_trade_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sr in all_split_results:
            for t in sr["dsl_results"][optimal_key]["trade_log"]:
                writer.writerow({
                    "split_idx": sr["split_idx"],
                    "day": t["day"],
                    "entry_bar_global": t["entry_bar_global"],
                    "direction": t["direction"],
                    "true_label": t["true_label"],
                    "exit_type": t["exit_type"],
                    "pnl": f"{t['pnl']:.4f}",
                    "bars_held": t["bars_held"],
                    "minutes_since_open": f"{t['minutes_since_open']:.2f}",
                })
    print(f"  Wrote optimal_trade_log.csv")


def write_optimal_equity_curves_csv(dsl_summaries, optimal_key, all_split_results):
    """Write equity curves at recommended DSL."""
    curves = []
    for sr in all_split_results:
        ec = sr["dsl_results"][optimal_key]["equity_curve"]
        curves.append(ec)

    max_trades = max(len(c) for c in curves) if curves else 0
    with open(RESULTS_DIR / "optimal_equity_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["trade_idx"] + [f"split_{sr['split_idx']:02d}" for sr in all_split_results]
        writer.writerow(header)
        for i in range(max_trades):
            row = [i]
            for c in curves:
                row.append(f"{c[i]:.4f}" if i < len(c) else "")
            writer.writerow(row)
    print(f"  Wrote optimal_equity_curves.csv")


def write_optimal_drawdown_summary_csv(dsl_summaries, optimal_key, all_split_results):
    """Write drawdown summary at recommended DSL."""
    with open(RESULTS_DIR / "optimal_drawdown_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split_idx", "test_groups", "max_drawdown", "max_consecutive_losses",
                        "drawdown_duration_days", "n_trades", "expectancy",
                        "trades_per_day_mean", "win_rate", "dsl_trigger_rate"])
        for sr in all_split_results:
            cr = sr["dsl_results"][optimal_key]
            writer.writerow([
                sr["split_idx"],
                f"{sr['test_groups']}",
                f"{cr['max_drawdown']:.2f}",
                cr["max_consecutive_losses"],
                cr["drawdown_duration_days"],
                cr["n_trades"],
                f"{cr['expectancy']:.4f}",
                f"{cr['trades_per_day_mean']:.2f}",
                f"{cr['win_rate']:.4f}",
                f"{cr['dsl_trigger_rate']:.4f}",
            ])
    print(f"  Wrote optimal_drawdown_summary.csv")


def write_optimal_daily_pnl_csv(optimal_key, all_split_results):
    """Write daily PnL at recommended DSL."""
    with open(RESULTS_DIR / "optimal_daily_pnl.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split_idx", "day", "pnl", "n_trades", "hold_skips",
                        "time_skips", "dsl_skips", "dsl_triggered", "total_bars"])
        for sr in all_split_results:
            cr = sr["dsl_results"][optimal_key]
            for d in cr["daily_pnl_records"]:
                writer.writerow([
                    sr["split_idx"], d["day"], f"{d['pnl']:.4f}",
                    d["n_trades"], d["hold_skips"],
                    d.get("time_skips", 0), d.get("dsl_skips", 0),
                    d.get("dsl_triggered", False), d["total_bars"],
                ])
    print(f"  Wrote optimal_daily_pnl.csv")


def write_optimal_account_sizing_csv(dsl_summaries, optimal_key):
    """Write account sizing curve at recommended DSL ($500-$50K, $500 steps)."""
    cs = dsl_summaries[optimal_key]
    with open(RESULTS_DIR / "optimal_account_sizing.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["account_size", "survival_count", "survival_rate"])
        for entry in cs["account_sizing"]:
            writer.writerow([
                entry["account_size"],
                entry["survival_count"],
                f"{entry['survival_rate']:.4f}",
            ])
    print(f"  Wrote optimal_account_sizing.csv")


def write_comparison_table_csv(dsl_summaries, optimal_key):
    """Write DSL=None vs recommended DSL comparison."""
    baseline = dsl_summaries["None"]
    optimal = dsl_summaries[optimal_key]

    rows = [
        ("trades_per_day", baseline["trades_per_day"], optimal["trades_per_day"]),
        ("exp_per_trade", baseline["exp_per_trade"], optimal["exp_per_trade"]),
        ("daily_pnl_mean", baseline["daily_pnl_mean"], optimal["daily_pnl_mean"]),
        ("daily_pnl_std", baseline["daily_pnl_std"], optimal["daily_pnl_std"]),
        ("dd_worst", baseline["dd_worst"], optimal["dd_worst"]),
        ("dd_median", baseline["dd_median"], optimal["dd_median"]),
        ("min_acct_all", baseline["min_acct_all"], optimal["min_acct_all"]),
        ("min_acct_95", baseline["min_acct_95"], optimal["min_acct_95"]),
        ("win_rate", baseline["win_rate"], optimal["win_rate"]),
        ("calmar", baseline["calmar"], optimal["calmar"]),
        ("sharpe", baseline["sharpe"], optimal["sharpe"]),
        ("annual_pnl", baseline["annual_pnl"], optimal["annual_pnl"]),
        ("dsl_trigger_rate", baseline["dsl_trigger_rate"], optimal["dsl_trigger_rate"]),
    ]

    with open(RESULTS_DIR / "comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "dsl_none", f"dsl_{optimal_key}", "delta", "pct_change"])
        for name, v_none, v_opt in rows:
            delta = v_opt - v_none
            pct = (delta / abs(v_none) * 100) if v_none != 0 else 0.0
            writer.writerow([name, f"{v_none:.4f}", f"{v_opt:.4f}", f"{delta:.4f}", f"{pct:.2f}%"])
    print(f"  Wrote comparison_table.csv")


# ==========================================================================
# Metrics JSON
# ==========================================================================
def write_metrics_json(all_split_results, dsl_summaries, optimal_key,
                       optimal_reason, sacrifice_results, intraday_stats,
                       t0_global):
    """Write metrics.json with ALL required metrics."""
    elapsed = time.time() - t0_global
    baseline = dsl_summaries["None"]
    optimal = dsl_summaries[optimal_key]

    # Build DSL sweep table (compact — no raw trade logs)
    sweep_table = []
    for dsl in DSL_LEVELS:
        key = dsl_label(dsl)
        cs = dsl_summaries[key]
        sac = sacrifice_results.get(dsl, {})
        sac_rate = sac.get('sacrifice_rate', 0.0) if dsl is not None else 0.0
        sweep_table.append({
            "dsl_level": key,
            "trades_per_day": cs["trades_per_day"],
            "exp_per_trade": cs["exp_per_trade"],
            "daily_pnl_mean": cs["daily_pnl_mean"],
            "daily_pnl_std": cs["daily_pnl_std"],
            "daily_pnl_skew": cs["daily_pnl_skew"],
            "daily_pnl_kurtosis": cs["daily_pnl_kurtosis"],
            "dd_worst": cs["dd_worst"],
            "dd_median": cs["dd_median"],
            "min_acct_all": cs["min_acct_all"],
            "min_acct_95": cs["min_acct_95"],
            "calmar": cs["calmar"],
            "sharpe": cs["sharpe"],
            "annual_pnl": cs["annual_pnl"],
            "win_rate": cs["win_rate"],
            "dsl_trigger_rate": cs["dsl_trigger_rate"],
            "recovery_sacrifice_rate": sac_rate,
        })

    # Recovery sacrifice table
    recovery_sacrifice_table = []
    for dsl in [d for d in DSL_LEVELS if d is not None]:
        sac = sacrifice_results[dsl]
        recovery_sacrifice_table.append({
            "dsl_threshold": dsl,
            "total_day_observations": sac['total_day_observations'],
            "dipped_days": sac['dipped_days'],
            "dipped_rate": sac['dipped_rate'],
            "recovered_days": sac['recovered_days'],
            "sacrifice_rate": sac['sacrifice_rate'],
        })

    # DSL trigger rate by level
    dsl_trigger_by_level = {}
    for dsl in DSL_LEVELS:
        key = dsl_label(dsl)
        dsl_trigger_by_level[key] = dsl_summaries[key]["dsl_trigger_rate"]

    # Intra-day PnL stats summary
    if intraday_stats:
        min_pnls = [s['intraday_min_pnl'] for s in intraday_stats]
        max_pnls = [s['intraday_max_pnl'] for s in intraday_stats]
        final_pnls = [s['final_pnl'] for s in intraday_stats]
        trade_counts = [s['n_trades'] for s in intraday_stats]
        max_consec = [s['max_consecutive_losses'] for s in intraday_stats]
        intraday_summary = {
            "n_day_observations": len(intraday_stats),
            "intraday_min_pnl": {
                "mean": float(np.mean(min_pnls)),
                "p5": float(np.percentile(min_pnls, 5)),
                "p25": float(np.percentile(min_pnls, 25)),
                "p50": float(np.percentile(min_pnls, 50)),
                "p75": float(np.percentile(min_pnls, 75)),
                "p95": float(np.percentile(min_pnls, 95)),
                "worst": float(np.min(min_pnls)),
            },
            "intraday_max_pnl": {
                "mean": float(np.mean(max_pnls)),
                "p50": float(np.percentile(max_pnls, 50)),
                "best": float(np.max(max_pnls)),
            },
            "final_pnl": {
                "mean": float(np.mean(final_pnls)),
                "p50": float(np.percentile(final_pnls, 50)),
            },
            "trade_count_per_day": {
                "mean": float(np.mean(trade_counts)),
                "p50": float(np.median(trade_counts)),
            },
            "max_consecutive_losses_per_day": {
                "mean": float(np.mean(max_consec)),
                "p50": float(np.median(max_consec)),
                "max": int(np.max(max_consec)),
            },
        }
    else:
        intraday_summary = {}

    # Sanity checks
    s1_exp = all_split_results[0]["bar_level_expectancy"] if all_split_results else 0.0
    sc_s1 = abs(s1_exp - PR38_SPLIT0_EXP) <= 0.01

    sc_s2_exp = baseline["exp_per_trade"]
    sc_s2 = abs(sc_s2_exp - PR40_CUTOFF270_EXP) <= 0.10

    sc_s3_tpd = baseline["trades_per_day"]
    sc_s3 = abs(sc_s3_tpd - PR40_CUTOFF270_TPD) <= 5

    sc_s4_min_acct = baseline["min_acct_all"]
    sc_s4 = abs(sc_s4_min_acct - PR40_CUTOFF270_MIN_ACCT) <= 1000

    # SC-S5: monotonicity of trades/day (non-increasing as DSL tightens)
    tpds = [dsl_summaries[dsl_label(d)]["trades_per_day"] for d in DSL_LEVELS]
    sc_s5 = all(tpds[i] >= tpds[i+1] - 0.001 for i in range(len(tpds) - 1))

    # Success criteria
    n_splits = len(all_split_results)
    sc1 = baseline["min_acct_all"] is not None  # placeholder — checked via optimal below
    sc2 = optimal["annual_pnl"] >= 50000
    sc3 = optimal["calmar"] >= 2.0

    optimal_dsl_dollar = optimal["dsl_level"]
    if optimal_dsl_dollar is not None:
        opt_sac = sacrifice_results.get(optimal_dsl_dollar, {})
        sc4_rate = opt_sac.get('sacrifice_rate', 1.0)
    else:
        sc4_rate = 0.0
    sc4 = sc4_rate <= 0.20

    sc5 = (n_splits == 45) and all(
        len(sr["dsl_results"]) == 9 for sr in all_split_results)
    sc6 = len(sweep_table) == 9 and all(
        all(k in row for k in ["trades_per_day", "exp_per_trade", "daily_pnl_mean",
            "dd_worst", "dd_median", "min_acct_all", "min_acct_95", "win_rate",
            "dsl_trigger_rate", "recovery_sacrifice_rate"])
        for row in sweep_table)
    sc7 = len(recovery_sacrifice_table) == 8
    sc8 = True  # files written (checked at end)

    sc1_val = optimal["min_acct_all"] <= 20000
    sc9 = abs(s1_exp - PR38_SPLIT0_EXP) <= 0.01
    sc_all_sanity = sc_s1 and sc_s2 and sc_s3 and sc_s4 and sc_s5

    # Outcome determination
    all_primary_pass = sc1_val and sc2 and sc3 and sc4
    if all_primary_pass and sc5 and sc6 and sc7 and sc9 and sc_all_sanity:
        outcome = "A"
        outcome_desc = (f"DSL effective. {optimal_key} achieves min_acct=${optimal['min_acct_all']:,}, "
                       f"annual PnL=${optimal['annual_pnl']:,.0f}, Calmar={optimal['calmar']:.2f}.")
    elif sc5 and sc6 and sc7 and sc_all_sanity:
        # Partial
        if sc1_val and not sc2:
            outcome = "B"
            outcome_desc = (f"min_acct ${optimal['min_acct_all']:,} meets SC-1 but annual PnL "
                           f"${optimal['annual_pnl']:,.0f} below $50K (SC-2 fail).")
        elif sc2 and not sc1_val:
            outcome = "B"
            outcome_desc = (f"Annual PnL ${optimal['annual_pnl']:,.0f} meets SC-2 but min_acct "
                           f"${optimal['min_acct_all']:,} exceeds $20K (SC-1 fail). "
                           f"Drawdown is inter-day, not intra-day.")
        elif not sc3:
            outcome = "B"
            outcome_desc = (f"Calmar {optimal['calmar']:.2f} below 2.0 (SC-3 fail).")
        elif not sc4:
            outcome = "B"
            outcome_desc = (f"Recovery sacrifice {sc4_rate:.1%} exceeds 20% (SC-4 fail).")
        else:
            # Check if DSL had any effect
            dd_delta = abs(baseline["dd_worst"] - optimal["dd_worst"])
            if dd_delta < 2000:
                outcome = "C"
                outcome_desc = (f"DSL has no meaningful effect. DD delta=${dd_delta:.0f} (<$2K). "
                               f"Drawdown driven by multi-day streaks, not intra-day clustering.")
            else:
                outcome = "B"
                outcome_desc = (f"Partial improvement but not all SC met. "
                               f"Best DSL: {optimal_key}, min_acct=${optimal['min_acct_all']:,}.")
    elif not sc_all_sanity:
        outcome = "D"
        outcome_desc = "Sanity check failure — baseline does not reproduce."
    else:
        outcome = "D"
        outcome_desc = "Simulation or sanity check failure."

    metrics = {
        "experiment": "daily-stop-loss-sequential",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),

        # Primary metrics
        "dsl_sweep_table": sweep_table,
        "optimal_dsl_min_account_all": optimal["min_acct_all"],
        "optimal_dsl_annual_pnl": optimal["annual_pnl"],

        # Secondary metrics
        "recovery_sacrifice_table": recovery_sacrifice_table,
        "intraday_pnl_stats": intraday_summary,
        "optimal_dsl_calmar": optimal["calmar"],
        "optimal_dsl_sharpe": optimal["sharpe"],
        "optimal_dsl_trades_per_day": optimal["trades_per_day"],
        "daily_pnl_percentiles_optimal": optimal["daily_pnl_percentiles"],
        "dsl_trigger_rate_by_level": dsl_trigger_by_level,

        # Optimal DSL identification
        "optimal_dsl": optimal_key,
        "optimal_dsl_reason": optimal_reason,
        "optimal_dsl_level_dollar": optimal["dsl_level"],

        # Baseline (DSL=None) for comparison
        "baseline_dsl_none": {
            "exp_per_trade": baseline["exp_per_trade"],
            "trades_per_day": baseline["trades_per_day"],
            "daily_pnl_mean": baseline["daily_pnl_mean"],
            "dd_worst": baseline["dd_worst"],
            "min_acct_all": baseline["min_acct_all"],
            "calmar": baseline["calmar"],
            "sharpe": baseline["sharpe"],
            "annual_pnl": baseline["annual_pnl"],
        },

        # Success criteria
        "success_criteria": {
            "SC-1": {"description": "Min account (all) <= $20,000 at recommended DSL",
                     "pass": sc1_val, "value": optimal["min_acct_all"]},
            "SC-2": {"description": "Annual PnL >= $50,000 at recommended DSL",
                     "pass": sc2, "value": optimal["annual_pnl"]},
            "SC-3": {"description": "Calmar >= 2.0 at recommended DSL",
                     "pass": sc3, "value": optimal["calmar"]},
            "SC-4": {"description": "Recovery sacrifice rate <= 20% at recommended DSL",
                     "pass": sc4, "value": sc4_rate},
            "SC-5": {"description": "All 9 DSL levels × 45 splits = 405 simulations completed",
                     "pass": sc5, "value": f"{n_splits} splits × {len(DSL_LEVELS)} DSL levels"},
            "SC-6": {"description": "Per-DSL sweep table fully populated (9 × 16+ columns)",
                     "pass": sc6},
            "SC-7": {"description": "Recovery sacrifice analysis for all 8 non-baseline DSL levels",
                     "pass": sc7},
            "SC-8": {"description": "All output files written to results directory",
                     "pass": sc8},
            "SC-9": {"description": "Bar-level split 0 matches PR #38 within $0.01",
                     "pass": sc9, "value": s1_exp, "reference": PR38_SPLIT0_EXP},
        },

        "sanity_checks": {
            "SC-S1": {"description": "bar-level split 0 within $0.01 of PR #38",
                      "pass": sc_s1, "value": s1_exp, "reference": PR38_SPLIT0_EXP},
            "SC-S2": {"description": f"DSL=None exp within $0.10 of PR #40's ${PR40_CUTOFF270_EXP}",
                      "pass": sc_s2, "value": sc_s2_exp, "reference": PR40_CUTOFF270_EXP},
            "SC-S3": {"description": f"DSL=None trades/day within 5 of PR #40's {PR40_CUTOFF270_TPD}",
                      "pass": sc_s3, "value": sc_s3_tpd, "reference": PR40_CUTOFF270_TPD},
            "SC-S4": {"description": f"DSL=None min_acct within $1K of PR #40's ${PR40_CUTOFF270_MIN_ACCT:,}",
                      "pass": sc_s4, "value": sc_s4_min_acct, "reference": PR40_CUTOFF270_MIN_ACCT},
            "SC-S5": {"description": "Trades/day non-increasing as DSL tightens (None -> $500)",
                      "pass": sc_s5,
                      "values": {dsl_label(d): dsl_summaries[dsl_label(d)]["trades_per_day"]
                                 for d in DSL_LEVELS}},
        },

        "outcome": outcome,
        "outcome_description": outcome_desc,

        "resource_usage": {
            "wall_clock_seconds": elapsed,
            "wall_clock_minutes": elapsed / 60,
            "total_training_runs": n_splits * 2,
            "total_simulations": n_splits * len(DSL_LEVELS),
            "gpu_hours": 0,
            "total_runs": n_splits,
        },

        "abort_triggered": False,
        "abort_reason": None,
        "notes": ("Local execution on Apple Silicon. "
                  "Sequential 1-contract simulation with fixed cutoff=270 and DSL sweep. "
                  "Models trained ONCE per split, simulation repeated 9x per DSL level. "
                  f"Corrected-base cost ${RT_COST_BASE} RT."),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote metrics.json")
    return metrics


# ==========================================================================
# Analysis MD
# ==========================================================================
def write_analysis_md(metrics, dsl_summaries, optimal_key, sacrifice_results,
                      intraday_stats, all_split_results):
    """Write analysis.md with all required sections per spec."""
    lines = []
    optimal = dsl_summaries[optimal_key]
    baseline = dsl_summaries["None"]

    lines.append("# Daily Stop Loss — Sequential Execution — Analysis\n")
    lines.append(f"**Date:** {metrics['timestamp']}")
    lines.append(f"**Outcome:** {metrics['outcome']} — {metrics['outcome_description']}\n")

    # 1. Executive summary
    lines.append("## 1. Executive Summary\n")
    lines.append(f"Daily stop loss sweep on sequential 1-contract execution across 9 DSL levels "
                 f"(None, $5000→$500) and 45 CPCV splits (405 simulations). Fixed cutoff=270 (from PR #40). "
                 f"Recommended DSL: **{optimal_key}** ({metrics['optimal_dsl_reason']}). "
                 f"At DSL={optimal_key}: **${optimal['exp_per_trade']:.2f}/trade**, "
                 f"**{optimal['trades_per_day']:.1f} trades/day**, "
                 f"min account **${optimal['min_acct_all']:,}**, "
                 f"annual PnL **${optimal['annual_pnl']:,.0f}**, "
                 f"Calmar **{optimal['calmar']:.2f}**.\n")

    # 2. DSL sweep results
    lines.append("## 2. DSL Sweep Results\n")
    lines.append("| DSL Level | Trades/Day | Exp/Trade | Daily PnL Mean | Daily PnL Std | "
                 "Skew | Kurtosis | DD Worst | DD Median | Min Acct All | Min Acct 95% | "
                 "Calmar | Sharpe | Annual PnL | Win Rate | DSL Trigger Rate | Recovery Sacrifice |")
    lines.append("|" + "|".join(["---"] * 17) + "|")
    for dsl in DSL_LEVELS:
        key = dsl_label(dsl)
        cs = dsl_summaries[key]
        sac = sacrifice_results.get(dsl, {})
        sac_rate = sac.get('sacrifice_rate', 0.0) if dsl is not None else 0.0
        marker = " **" if key == optimal_key else ""
        lines.append(f"| {key}{marker} | {cs['trades_per_day']:.1f} | ${cs['exp_per_trade']:.2f} | "
                     f"${cs['daily_pnl_mean']:.0f} | ${cs['daily_pnl_std']:.0f} | "
                     f"{cs['daily_pnl_skew']:.3f} | {cs['daily_pnl_kurtosis']:.3f} | "
                     f"${cs['dd_worst']:,.0f} | ${cs['dd_median']:,.0f} | "
                     f"${cs['min_acct_all']:,} | ${cs['min_acct_95']:,} | "
                     f"{cs['calmar']:.2f} | {cs['sharpe']:.2f} | ${cs['annual_pnl']:,.0f} | "
                     f"{cs['win_rate']:.3f} | {cs['dsl_trigger_rate']:.3f} | {sac_rate:.3f} |")

    # 3. Recovery sacrifice analysis
    lines.append("\n## 3. Recovery Sacrifice Analysis\n")
    lines.append("For each DSL threshold, of the days where intra-day P&L dipped below -$X, "
                 "what fraction would have ended positive (recovered) without DSL?\n")
    lines.append("| DSL Threshold | Total Day Obs | Dipped Days | Dipped Rate | Recovered Days | Sacrifice Rate |")
    lines.append("|" + "|".join(["---"] * 6) + "|")
    for dsl in [d for d in DSL_LEVELS if d is not None]:
        sac = sacrifice_results[dsl]
        lines.append(f"| ${dsl} | {sac['total_day_observations']} | {sac['dipped_days']} | "
                     f"{sac['dipped_rate']:.4f} | {sac['recovered_days']} | "
                     f"{sac['sacrifice_rate']:.4f} |")

    # 4. Intra-day P&L path analysis
    lines.append("\n## 4. Intra-Day P&L Path Analysis (Baseline DSL=None)\n")
    if metrics.get("intraday_pnl_stats"):
        ids = metrics["intraday_pnl_stats"]
        lines.append(f"Observations: {ids['n_day_observations']} (split, day) pairs\n")
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Intra-day min P&L (mean) | ${ids['intraday_min_pnl']['mean']:,.0f} |")
        lines.append(f"| Intra-day min P&L (worst) | ${ids['intraday_min_pnl']['worst']:,.0f} |")
        lines.append(f"| Intra-day min P&L (p5) | ${ids['intraday_min_pnl']['p5']:,.0f} |")
        lines.append(f"| Intra-day min P&L (p25) | ${ids['intraday_min_pnl']['p25']:,.0f} |")
        lines.append(f"| Intra-day min P&L (p50) | ${ids['intraday_min_pnl']['p50']:,.0f} |")
        lines.append(f"| Intra-day max P&L (mean) | ${ids['intraday_max_pnl']['mean']:,.0f} |")
        lines.append(f"| Intra-day max P&L (best) | ${ids['intraday_max_pnl']['best']:,.0f} |")
        lines.append(f"| Final P&L (mean) | ${ids['final_pnl']['mean']:,.0f} |")
        lines.append(f"| Trade count/day (mean) | {ids['trade_count_per_day']['mean']:.1f} |")
        lines.append(f"| Max consecutive losses/day (mean) | {ids['max_consecutive_losses_per_day']['mean']:.1f} |")
        lines.append(f"| Max consecutive losses/day (worst) | {ids['max_consecutive_losses_per_day']['max']} |")

    # 5. Optimal DSL selection
    lines.append(f"\n## 5. Optimal DSL Selection\n")
    lines.append(f"**Recommended DSL: {optimal_key}**\n")
    lines.append(f"**Reason:** {metrics['optimal_dsl_reason']}\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Per-trade expectancy | **${optimal['exp_per_trade']:.4f}** |")
    lines.append(f"| Trades/day | {optimal['trades_per_day']:.2f} |")
    lines.append(f"| Daily PnL | **${optimal['daily_pnl_mean']:.2f}** |")
    lines.append(f"| Annual PnL (1 MES) | **${optimal['annual_pnl']:,.0f}** |")
    lines.append(f"| Min account (all survive) | **${optimal['min_acct_all']:,}** |")
    lines.append(f"| Min account (95%) | ${optimal['min_acct_95']:,} |")
    lines.append(f"| Calmar ratio | {optimal['calmar']:.4f} |")
    lines.append(f"| Annualized Sharpe | {optimal['sharpe']:.4f} |")
    lines.append(f"| Win rate | {optimal['win_rate']:.4f} |")
    lines.append(f"| DSL trigger rate | {optimal['dsl_trigger_rate']:.4f} |")

    # 6. Detailed results at recommended DSL
    lines.append(f"\n## 6. Detailed Risk Metrics at DSL={optimal_key}\n")
    lines.append("| Metric | Worst | Median |")
    lines.append("|--------|-------|--------|")
    lines.append(f"| Max drawdown ($) | ${optimal['dd_worst']:,.0f} | ${optimal['dd_median']:,.0f} |")

    # 7. DSL=None vs recommended DSL comparison
    lines.append(f"\n## 7. DSL=None vs DSL={optimal_key} Comparison\n")
    lines.append(f"| Metric | DSL=None | DSL={optimal_key} | Delta | % Change |")
    lines.append(f"|--------|---------|" + "-" * (len(optimal_key) + 6) + "|-------|----------|")
    comparisons = [
        ("Trades/day", baseline["trades_per_day"], optimal["trades_per_day"]),
        ("Exp/trade ($)", baseline["exp_per_trade"], optimal["exp_per_trade"]),
        ("Daily PnL ($)", baseline["daily_pnl_mean"], optimal["daily_pnl_mean"]),
        ("Daily PnL Std ($)", baseline["daily_pnl_std"], optimal["daily_pnl_std"]),
        ("DD worst ($)", baseline["dd_worst"], optimal["dd_worst"]),
        ("DD median ($)", baseline["dd_median"], optimal["dd_median"]),
        ("Min acct all ($)", baseline["min_acct_all"], optimal["min_acct_all"]),
        ("Min acct 95% ($)", baseline["min_acct_95"], optimal["min_acct_95"]),
        ("Win rate", baseline["win_rate"], optimal["win_rate"]),
        ("Calmar", baseline["calmar"], optimal["calmar"]),
        ("Sharpe", baseline["sharpe"], optimal["sharpe"]),
        ("Annual PnL ($)", baseline["annual_pnl"], optimal["annual_pnl"]),
        ("DSL trigger rate", baseline["dsl_trigger_rate"], optimal["dsl_trigger_rate"]),
    ]
    for name, v_none, v_opt in comparisons:
        delta = v_opt - v_none
        pct = (delta / abs(v_none) * 100) if v_none != 0 else 0.0
        lines.append(f"| {name} | {v_none:.2f} | {v_opt:.2f} | {delta:+.2f} | {pct:+.1f}% |")

    # 8. Account sizing at recommended DSL
    lines.append(f"\n## 8. Account Sizing at DSL={optimal_key}\n")
    lines.append(f"| Threshold | Account Size |")
    lines.append(f"|-----------|-------------|")
    lines.append(f"| All 45 paths survive | **${optimal['min_acct_all']:,}** |")
    lines.append(f"| 95% paths survive | **${optimal['min_acct_95']:,}** |")

    # 9. Daily PnL distribution at recommended DSL
    lines.append(f"\n## 9. Daily PnL Distribution at DSL={optimal_key}\n")
    lines.append("| Percentile | Daily PnL ($) |")
    lines.append("|------------|--------------|")
    for p, v in optimal["daily_pnl_percentiles"].items():
        lines.append(f"| {p} | ${v:.2f} |")

    # 10. SC and sanity check pass/fail
    lines.append("\n## 10. Success Criteria and Sanity Checks\n")
    lines.append("### Success Criteria\n")
    lines.append("| Criterion | Description | Pass | Value |")
    lines.append("|-----------|-------------|------|-------|")
    for sc_key, sc_val in metrics["success_criteria"].items():
        val_str = str(sc_val.get('value', 'N/A'))
        lines.append(f"| {sc_key} | {sc_val['description']} | "
                     f"{'PASS' if sc_val['pass'] else 'FAIL'} | {val_str} |")

    lines.append("\n### Sanity Checks\n")
    lines.append("| Check | Description | Pass | Value | Reference |")
    lines.append("|-------|-------------|------|-------|-----------|")
    for sk, sv in metrics["sanity_checks"].items():
        val_str = str(sv.get('value', 'N/A'))
        ref_str = str(sv.get('reference', 'N/A'))
        lines.append(f"| {sk} | {sv['description']} | "
                     f"{'PASS' if sv['pass'] else 'FAIL'} | {val_str} | {ref_str} |")

    # 11. Outcome verdict
    lines.append(f"\n## 11. Outcome Verdict\n")
    lines.append(f"**{metrics['outcome']}** — {metrics['outcome_description']}\n")

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote analysis.md")


# ==========================================================================
# Main
# ==========================================================================
def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Daily Stop Loss — Sequential Execution")
    print(f"DSL Levels: {[dsl_label(d) for d in DSL_LEVELS]}")
    print(f"Fixed cutoff: {FIXED_CUTOFF}")
    print(f"Splits: 45, Simulations: {45 * len(DSL_LEVELS)}")
    print("=" * 70)

    # [1/7] Load data
    print("\n[1/7] Loading data...")
    features, labels, day_indices, unique_days_raw, extra = load_data()

    # Baseline for verification
    cpcv_baseline = {"split_0_base_exp": PR38_SPLIT0_EXP}

    # [2/7] MVE: split 0 at DSL=None and DSL=$1500
    print("\n[2/7] Running MVE (split 0 only, 9 DSL levels)...")
    mve_results, mve_abort = run_cpcv_with_dsl_sweep(
        features, labels, day_indices, extra,
        mve_only=True, cpcv_baseline=cpcv_baseline)

    if mve_abort:
        print(f"\n*** ABORT: {mve_abort} ***")
        metrics = {
            "experiment": "daily-stop-loss-sequential",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": mve_abort,
            "resource_usage": {"wall_clock_seconds": time.time() - t0},
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # MVE additional checks
    mve_split = mve_results[0]
    dsl_none_mve = mve_split["dsl_results"]["None"]
    dsl_1500_mve = mve_split["dsl_results"]["$1500"]
    print(f"    MVE DSL=None: exp=${dsl_none_mve['expectancy']:.4f}, tpd={dsl_none_mve['trades_per_day_mean']:.1f}")
    print(f"    MVE DSL=$1500: exp=${dsl_1500_mve['expectancy']:.4f}, tpd={dsl_1500_mve['trades_per_day_mean']:.1f}")
    print(f"    MVE bar-level: ${mve_split['bar_level_expectancy']:.4f}")

    # [3/7] Full sweep: all 45 splits × 9 DSL levels
    print(f"\n[3/7] Running full CPCV with DSL sweep (45 splits × 9 DSL = 405 sims)...")
    all_split_results, abort_reason = run_cpcv_with_dsl_sweep(
        features, labels, day_indices, extra,
        mve_only=False, cpcv_baseline=cpcv_baseline)

    if abort_reason:
        print(f"\n*** ABORT: {abort_reason} ***")
        metrics = {
            "experiment": "daily-stop-loss-sequential",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": abort_reason,
            "resource_usage": {"wall_clock_seconds": time.time() - t0},
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # [4/7] Aggregate per DSL level
    print("\n[4/7] Aggregating results per DSL level...")
    dsl_summaries = aggregate_per_dsl(all_split_results)

    # Quick summary
    for dsl in DSL_LEVELS:
        key = dsl_label(dsl)
        cs = dsl_summaries[key]
        print(f"    DSL={key:>5s}: exp=${cs['exp_per_trade']:.2f}, tpd={cs['trades_per_day']:.1f}, "
              f"daily=${cs['daily_pnl_mean']:.0f}, dd_worst=${cs['dd_worst']:,.0f}, "
              f"min_acct=${cs['min_acct_all']:,}, trigger_rate={cs['dsl_trigger_rate']:.3f}")

    # [5/7] Recovery sacrifice analysis + intra-day stats
    print("\n[5/7] Computing recovery sacrifice and intra-day P&L path stats...")
    baseline_trade_logs = dsl_summaries["None"]["all_trade_logs"]
    baseline_daily_records = dsl_summaries["None"]["all_daily_pnl_records"]
    sacrifice_results, day_paths = compute_recovery_sacrifice_from_trade_logs(
        baseline_trade_logs, baseline_daily_records)
    intraday_stats = compute_intraday_pnl_stats(baseline_trade_logs)

    for dsl in [d for d in DSL_LEVELS if d is not None]:
        sac = sacrifice_results[dsl]
        print(f"    DSL=${dsl}: dipped={sac['dipped_days']}/{sac['total_day_observations']}, "
              f"recovered={sac['recovered_days']}, sacrifice_rate={sac['sacrifice_rate']:.4f}")

    # [6/7] Select optimal DSL
    print("\n[6/7] Selecting optimal DSL...")
    optimal_key, optimal_reason = select_optimal_dsl(dsl_summaries, sacrifice_results)
    opt_cs = dsl_summaries[optimal_key]
    print(f"    Optimal: DSL={optimal_key}")
    print(f"    Reason: {optimal_reason}")
    print(f"    Exp=${opt_cs['exp_per_trade']:.2f}, min_acct=${opt_cs['min_acct_all']:,}, "
          f"annual PnL=${opt_cs['annual_pnl']:,.0f}, Calmar={opt_cs['calmar']:.2f}")

    # [7/7] Write outputs
    print("\n[7/7] Writing outputs...")
    write_dsl_sweep_csv(dsl_summaries, sacrifice_results)
    write_recovery_sacrifice_csv(sacrifice_results)
    write_intraday_pnl_stats_csv(intraday_stats)
    write_optimal_trade_log_csv(dsl_summaries, optimal_key, all_split_results)
    write_optimal_equity_curves_csv(dsl_summaries, optimal_key, all_split_results)
    write_optimal_drawdown_summary_csv(dsl_summaries, optimal_key, all_split_results)
    write_optimal_daily_pnl_csv(optimal_key, all_split_results)
    write_optimal_account_sizing_csv(dsl_summaries, optimal_key)
    write_comparison_table_csv(dsl_summaries, optimal_key)
    metrics = write_metrics_json(all_split_results, dsl_summaries, optimal_key,
                                 optimal_reason, sacrifice_results, intraday_stats, t0)
    write_analysis_md(metrics, dsl_summaries, optimal_key, sacrifice_results,
                      intraday_stats, all_split_results)

    # Copy spec
    spec_src = PROJECT_ROOT / ".kit" / "experiments" / "daily-stop-loss-sequential.md"
    try:
        with open(spec_src, "r") as f_in:
            spec_content = f_in.read()
        with open(RESULTS_DIR / "spec.md", "w") as f_out:
            f_out.write(spec_content)
        print(f"  Copied spec.md")
    except Exception as e:
        print(f"  Warning: could not copy spec.md: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Outcome: {metrics['outcome']} — {metrics['outcome_description']}")
    print(f"Optimal DSL: {optimal_key}")
    print(f"  Expectancy: ${opt_cs['exp_per_trade']:.4f}/trade")
    print(f"  Trades/day: {opt_cs['trades_per_day']:.1f}")
    print(f"  Daily PnL: ${opt_cs['daily_pnl_mean']:.2f}")
    print(f"  Annual PnL: ${opt_cs['annual_pnl']:,.0f}")
    print(f"  Min account (all): ${opt_cs['min_acct_all']:,}")
    print(f"  Calmar: {opt_cs['calmar']:.4f}")
    print(f"  Sharpe: {opt_cs['sharpe']:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
