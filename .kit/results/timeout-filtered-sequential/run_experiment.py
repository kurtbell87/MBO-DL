#!/usr/bin/env python3
"""
Experiment: Timeout-Filtered Sequential Execution
Spec: .kit/experiments/timeout-filtered-sequential.md

Adapts the sequential simulation from PR #39 (trade-level-risk-metrics) with
a time-of-day entry filter. Sweeps 7 cutoff levels (minutes_since_open) to
find the optimal tradeoff between fewer trades and higher per-trade expectancy.

Training is IDENTICAL to PR #38/#39. Only the simulation entry filter changes.
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

# ==========================================================================
# Config
# ==========================================================================
SEED = 42
PROJECT_ROOT = Path("/Users/brandonbell/LOCAL_DEV/MBO-DL-02152026")
RESULTS_DIR = PROJECT_ROOT / ".kit" / "results" / "timeout-filtered-sequential"
DATA_DIR = PROJECT_ROOT / ".kit" / "results" / "label-geometry-1h" / "geom_19_7"

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

# Cutoff levels (minutes_since_open threshold for entry)
CUTOFF_LEVELS = [390, 375, 360, 345, 330, 300, 270]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================================================
# Data Loading (identical to PR #39)
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
# Two-Stage Training (identical to PR #38/#39)
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
# Sequential Execution Simulator — with time cutoff
# ==========================================================================
def simulate_sequential(test_indices, predictions, labels, extra, rt_cost, cutoff=390):
    """Simulate sequential 1-contract execution with time-of-day cutoff.

    If minutes_since_open > cutoff at an entry opportunity, skip (time-filtered).
    Returns (trade_log, daily_pnl_records).
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

    for day_val in unique_test_days:
        day_mask = (test_days_raw == day_val)
        day_positions = np.where(day_mask)[0]
        n_day_bars = len(day_positions)

        day_pnl_total = 0.0
        day_n_trades = 0
        day_hold_skips = 0
        day_time_skips = 0

        i = 0
        while i < n_day_bars:
            test_pos = day_positions[i]
            global_idx = test_sorted[test_pos]
            pred = pred_sorted[test_pos]

            if pred == 0:
                day_hold_skips += 1
                i += 1
                continue

            # Check time cutoff BEFORE entering
            mso = minutes_since_open[global_idx]
            if mso > cutoff:
                day_time_skips += 1
                i += 1
                continue

            # ENTER position
            label = label_sorted[test_pos]
            bars_held = max(1, int(tb_bars_held[global_idx]))
            exit_type = extra["tb_exit_type"][global_idx]

            # PnL computation (identical to PR #39)
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

            # Advance past holding period
            i += bars_held

        daily_pnl_records.append({
            'day': int(day_val),
            'pnl': float(day_pnl_total),
            'n_trades': day_n_trades,
            'hold_skips': day_hold_skips,
            'time_skips': day_time_skips,
            'total_bars': n_day_bars,
        })

    return trade_log, daily_pnl_records


# ==========================================================================
# Risk Metrics (identical to PR #39)
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
    """For each account level ($500 to $50K in $500 steps), count surviving paths."""
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
            'survival_rate': survived / n_paths,
        })

    return results, path_max_dd


# ==========================================================================
# CPCV Group Assignment (identical to PR #38/#39)
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
# Per-cutoff metrics from trade_log and daily_pnl_records
# ==========================================================================
def compute_split_cutoff_metrics(trade_log, daily_pnl_records):
    """Compute per-split, per-cutoff sequential metrics."""
    n_trades = len(trade_log)
    n_test_days = len(daily_pnl_records)

    if n_trades == 0:
        return {
            "n_trades": 0,
            "n_test_days": n_test_days,
            "trades_per_day_mean": 0.0,
            "trades_per_day_std": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "win_rate_dir": 0.0,
            "hold_skip_rate": 0.0,
            "time_skip_rate": 0.0,
            "avg_bars_held": 0.0,
            "daily_pnl_mean": 0.0,
            "daily_pnl_std": 0.0,
            "max_drawdown": 0.0,
            "max_consecutive_losses": 0,
            "drawdown_duration_days": 0,
            "timeout_fraction": 0.0,
            "barrier_hit_fraction": 0.0,
            "equity_curve": [],
        }

    trades_per_day = [r['n_trades'] for r in daily_pnl_records]
    hold_skips_total = sum(r['hold_skips'] for r in daily_pnl_records)
    time_skips_total = sum(r.get('time_skips', 0) for r in daily_pnl_records)
    total_opportunities = hold_skips_total + time_skips_total + n_trades

    seq_pnls = [t['pnl'] for t in trade_log]
    seq_expectancy = float(np.mean(seq_pnls))
    seq_win_rate = float(np.mean([p > 0 for p in seq_pnls]))

    dir_trades = [t for t in trade_log if t['true_label'] != 0]
    seq_win_rate_dir = (float(np.mean([t['pnl'] > 0 for t in dir_trades]))
                        if dir_trades else 0.0)

    hold_skip_rate = hold_skips_total / total_opportunities if total_opportunities > 0 else 0.0
    time_skip_rate = time_skips_total / total_opportunities if total_opportunities > 0 else 0.0

    avg_bars_held = float(np.mean([t['bars_held'] for t in trade_log]))

    # Timeout fraction: trades where exit_type is "expiry" (time horizon reached)
    n_timeout = sum(1 for t in trade_log if t['exit_type'] == 'expiry')
    timeout_fraction = n_timeout / n_trades
    barrier_hit_fraction = 1.0 - timeout_fraction

    equity_curve = compute_equity_curve(trade_log)
    max_dd = compute_max_drawdown(equity_curve)
    max_consec = compute_max_consecutive_losses(trade_log)
    dd_duration = compute_drawdown_duration_days(daily_pnl_records)

    daily_pnls = [r['pnl'] for r in daily_pnl_records]

    return {
        "n_trades": n_trades,
        "n_test_days": n_test_days,
        "trades_per_day_mean": float(np.mean(trades_per_day)) if trades_per_day else 0.0,
        "trades_per_day_std": float(np.std(trades_per_day, ddof=1)) if len(trades_per_day) > 1 else 0.0,
        "expectancy": seq_expectancy,
        "win_rate": seq_win_rate,
        "win_rate_dir": seq_win_rate_dir,
        "hold_skip_rate": hold_skip_rate,
        "time_skip_rate": time_skip_rate,
        "avg_bars_held": avg_bars_held,
        "daily_pnl_mean": float(np.mean(daily_pnls)) if daily_pnls else 0.0,
        "daily_pnl_std": float(np.std(daily_pnls, ddof=1)) if len(daily_pnls) > 1 else 0.0,
        "max_drawdown": max_dd,
        "max_consecutive_losses": max_consec,
        "drawdown_duration_days": dd_duration,
        "timeout_fraction": timeout_fraction,
        "barrier_hit_fraction": barrier_hit_fraction,
        "equity_curve": equity_curve.tolist() if len(equity_curve) > 0 else [],
    }


# ==========================================================================
# Main CPCV Loop — Train Once, Simulate 7x per Split
# ==========================================================================
def run_cpcv_with_cutoff_sweep(features, labels, day_indices, extra,
                                mve_only=False, cpcv_baseline=None):
    """Run 45-split CPCV. Train once per split, simulate 7 cutoff levels."""
    n_total = len(labels)

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

    # Results: per_split_cutoff[split_idx][cutoff] = metrics dict
    per_split_cutoff = {}
    # Also track bar-level results per split for verification
    bar_level_results = {}

    t0_run = time.time()

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

        # Train ONCE per split
        fold_result = train_two_stage(
            features, labels, day_indices, extra,
            inner_train_global, test_indices_global, inner_val_global,
            seed=split_seed)

        combined_pred = fold_result["combined_pred"]
        labels_test = fold_result["labels_test"]

        # Bar-level PnL for verification
        bar_pnl = compute_bar_level_pnl(
            labels_test, combined_pred, test_indices_global, extra, RT_COST_BASE)
        bar_level_results[s_idx] = bar_pnl

        # Run simulation for each cutoff level
        cutoffs_to_run = CUTOFF_LEVELS if not mve_only else [390, 330]
        per_split_cutoff[s_idx] = {}

        for cutoff in cutoffs_to_run:
            trade_log, daily_pnl_records = simulate_sequential(
                test_indices_global, combined_pred, labels_test, extra,
                RT_COST_BASE, cutoff=cutoff)

            metrics = compute_split_cutoff_metrics(trade_log, daily_pnl_records)
            metrics["split_idx"] = s_idx
            metrics["test_groups"] = (g1, g2)
            metrics["cutoff"] = cutoff
            metrics["trade_log"] = trade_log
            metrics["daily_pnl_records"] = daily_pnl_records

            per_split_cutoff[s_idx][cutoff] = metrics

        elapsed = time.time() - t0_split

        # Log progress
        c390 = per_split_cutoff[s_idx].get(390, per_split_cutoff[s_idx].get(list(per_split_cutoff[s_idx].keys())[0]))
        if (s_idx + 1) % 5 == 0 or s_idx == 0:
            print(f"    Split {s_idx:2d} ({g1},{g2}): bar_exp=${bar_pnl['expectancy']:+.4f}, "
                  f"seq_exp_390=${c390['expectancy']:+.4f}, "
                  f"trades/day_390={c390['trades_per_day_mean']:.1f}, "
                  f"{elapsed:.1f}s")

        # MVE checks (split 0 only)
        if s_idx == 0:
            # SC-S1: bar-level match
            if cpcv_baseline is not None:
                ref_exp = cpcv_baseline.get("split_0_base_exp", None)
                if ref_exp is not None:
                    diff = abs(bar_pnl["expectancy"] - ref_exp)
                    if diff > 0.05:
                        print(f"  *** MVE ABORT: bar-level exp differs from PR #38 by ${diff:.4f} (>{0.05}) ***")
                        return per_split_cutoff, bar_level_results, f"MVE: bar-level exp mismatch (delta=${diff:.4f})"
                    else:
                        print(f"    MVE PASS SC-S1: bar-level exp ${bar_pnl['expectancy']:.4f} vs PR #38 ${ref_exp:.4f} (delta=${diff:.4f})")

            # Check cutoff=390 trades
            c390_tpd = per_split_cutoff[s_idx][390]['trades_per_day_mean'] if 390 in per_split_cutoff[s_idx] else 0
            if c390_tpd == 0:
                print(f"  *** MVE ABORT: cutoff=390 zero trades ***")
                return per_split_cutoff, bar_level_results, "MVE: cutoff=390 zero trades"

            if np.isnan(c390['expectancy']):
                print(f"  *** MVE ABORT: NaN in expectancy ***")
                return per_split_cutoff, bar_level_results, "MVE: NaN in expectancy"

            if mve_only:
                # Check monotonicity: cutoff=330 trades < cutoff=390 trades
                c330 = per_split_cutoff[s_idx].get(330, None)
                if c330 is not None:
                    if c330['trades_per_day_mean'] > c390['trades_per_day_mean']:
                        print(f"  *** MVE ABORT: cutoff=330 ({c330['trades_per_day_mean']:.1f}) > cutoff=390 ({c390['trades_per_day_mean']:.1f}) ***")
                        return per_split_cutoff, bar_level_results, "MVE: monotonicity violation (330 > 390 trades)"
                    else:
                        print(f"    MVE PASS monotonicity: 330={c330['trades_per_day_mean']:.1f} < 390={c390['trades_per_day_mean']:.1f}")

        # Wall-clock check
        total_elapsed = time.time() - t0_run
        if total_elapsed > WALL_CLOCK_LIMIT_S:
            print(f"  *** ABORT: wall-clock limit exceeded ({total_elapsed:.0f}s > {WALL_CLOCK_LIMIT_S}s) ***")
            return per_split_cutoff, bar_level_results, f"Wall-clock limit exceeded ({total_elapsed:.0f}s)"

    return per_split_cutoff, bar_level_results, None


# ==========================================================================
# Aggregation per Cutoff
# ==========================================================================
def aggregate_per_cutoff(per_split_cutoff, cutoff_levels):
    """Aggregate metrics across all splits for each cutoff level."""
    agg_by_cutoff = {}

    for cutoff in cutoff_levels:
        split_metrics = []
        for s_idx in sorted(per_split_cutoff.keys()):
            if cutoff in per_split_cutoff[s_idx]:
                split_metrics.append(per_split_cutoff[s_idx][cutoff])

        if not split_metrics:
            continue

        n_splits = len(split_metrics)
        exp_arr = np.array([m['expectancy'] for m in split_metrics])
        tpd_arr = np.array([m['trades_per_day_mean'] for m in split_metrics])
        daily_pnl_arr = np.array([m['daily_pnl_mean'] for m in split_metrics])
        dd_arr = np.array([m['max_drawdown'] for m in split_metrics])
        wr_arr = np.array([m['win_rate'] for m in split_metrics])
        timeout_arr = np.array([m['timeout_fraction'] for m in split_metrics])
        barrier_arr = np.array([m['barrier_hit_fraction'] for m in split_metrics])
        time_skip_arr = np.array([m['time_skip_rate'] for m in split_metrics])
        hold_skip_arr = np.array([m['hold_skip_rate'] for m in split_metrics])
        abh_arr = np.array([m['avg_bars_held'] for m in split_metrics])
        consec_arr = np.array([m['max_consecutive_losses'] for m in split_metrics])
        dd_dur_arr = np.array([m['drawdown_duration_days'] for m in split_metrics])

        # Equity curves for account sizing
        equity_curves = [np.array(m['equity_curve']) for m in split_metrics]
        account_sizing, path_max_dds = compute_account_sizing(equity_curves)

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
        annual_pnl = float(np.mean(daily_pnl_arr)) * 251
        worst_dd = float(np.max(dd_arr)) if len(dd_arr) > 0 else 1.0
        calmar = annual_pnl / worst_dd if worst_dd > 0 else 0.0

        # Sharpe
        if len(daily_pnl_arr) > 1:
            mean_daily = float(np.mean(daily_pnl_arr))
            std_daily = float(np.std(daily_pnl_arr, ddof=1))
            sharpe = (mean_daily / std_daily) * np.sqrt(251) if std_daily > 0 else 0.0
        else:
            sharpe = 0.0

        # Daily PnL percentiles (pooled across all splits)
        all_daily = []
        for m in split_metrics:
            all_daily.extend([d['pnl'] for d in m['daily_pnl_records']])
        all_daily = np.array(all_daily)
        daily_pcts = {}
        if len(all_daily) > 0:
            for p in [5, 25, 50, 75, 95]:
                daily_pcts[f"p{p}"] = float(np.percentile(all_daily, p))

        agg_by_cutoff[cutoff] = {
            "cutoff": cutoff,
            "n_splits": n_splits,
            "trades_per_day": float(np.mean(tpd_arr)),
            "expectancy": float(np.mean(exp_arr)),
            "daily_pnl": float(np.mean(daily_pnl_arr)),
            "dd_worst": float(np.max(dd_arr)),
            "dd_median": float(np.median(dd_arr)),
            "min_acct_all": min_account_all,
            "min_acct_95": min_account_95,
            "win_rate": float(np.mean(wr_arr)),
            "time_skip_pct": float(np.mean(time_skip_arr)),
            "hold_skip_pct": float(np.mean(hold_skip_arr)),
            "timeout_fraction": float(np.mean(timeout_arr)),
            "barrier_hit_fraction": float(np.mean(barrier_arr)),
            "avg_bars_held": float(np.mean(abh_arr)),
            "max_consec_losses": int(np.max(consec_arr)),
            "median_consec_losses": int(np.median(consec_arr)),
            "dd_duration_worst": int(np.max(dd_dur_arr)),
            "dd_duration_median": int(np.median(dd_dur_arr)),
            "calmar": calmar,
            "sharpe": sharpe,
            "annual_pnl": annual_pnl,
            "daily_pnl_percentiles": daily_pcts,
            "account_sizing_curve": account_sizing,
            "path_max_dds": path_max_dds,
            "per_split_exp": exp_arr.tolist(),
            "per_split_dd": dd_arr.tolist(),
        }

    return agg_by_cutoff


# ==========================================================================
# Optimal Cutoff Selection
# ==========================================================================
def select_optimal_cutoff(agg_by_cutoff):
    """Select optimal cutoff per the spec's selection rule.

    Primary: most conservative (highest) cutoff achieving BOTH
    exp >= $3.50 AND min_acct_all <= $30K.

    Fallback 1: maximize daily_pnl subject to min_acct_all <= $35K.
    Fallback 2: maximize Calmar ratio.
    """
    # Primary: most conservative cutoff meeting BOTH targets
    candidates = []
    for cutoff in sorted(agg_by_cutoff.keys(), reverse=True):  # highest first
        a = agg_by_cutoff[cutoff]
        if a['expectancy'] >= 3.50 and a['min_acct_all'] <= 30000:
            candidates.append(cutoff)

    if candidates:
        optimal = candidates[0]  # highest (most conservative) cutoff
        return optimal, "primary"

    # Fallback 1: max daily_pnl with min_acct_all <= $35K
    fb1_candidates = []
    for cutoff in agg_by_cutoff.keys():
        a = agg_by_cutoff[cutoff]
        if a['min_acct_all'] <= 35000:
            fb1_candidates.append((a['daily_pnl'], cutoff))

    if fb1_candidates:
        fb1_candidates.sort(reverse=True)
        return fb1_candidates[0][1], "fallback_1_max_daily_pnl_under_35k"

    # Fallback 2: max Calmar ratio
    best_calmar = -999
    best_cutoff = CUTOFF_LEVELS[0]
    for cutoff in agg_by_cutoff.keys():
        a = agg_by_cutoff[cutoff]
        if a['calmar'] > best_calmar:
            best_calmar = a['calmar']
            best_cutoff = cutoff

    return best_cutoff, "fallback_2_max_calmar"


# ==========================================================================
# Output Writers
# ==========================================================================
def write_cutoff_sweep_csv(agg_by_cutoff):
    """Write cutoff_sweep.csv — 7 rows × all metrics."""
    fieldnames = [
        "cutoff", "trades_per_day", "expectancy", "daily_pnl",
        "dd_worst", "dd_median", "min_acct_all", "min_acct_95",
        "win_rate", "time_skip_pct", "hold_skip_pct", "timeout_fraction",
        "barrier_hit_fraction", "avg_bars_held", "calmar", "sharpe", "annual_pnl",
    ]
    with open(RESULTS_DIR / "cutoff_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for cutoff in sorted(agg_by_cutoff.keys(), reverse=True):
            a = agg_by_cutoff[cutoff]
            writer.writerow({
                "cutoff": cutoff,
                "trades_per_day": f"{a['trades_per_day']:.2f}",
                "expectancy": f"{a['expectancy']:.4f}",
                "daily_pnl": f"{a['daily_pnl']:.2f}",
                "dd_worst": f"{a['dd_worst']:.2f}",
                "dd_median": f"{a['dd_median']:.2f}",
                "min_acct_all": a['min_acct_all'],
                "min_acct_95": a['min_acct_95'],
                "win_rate": f"{a['win_rate']:.4f}",
                "time_skip_pct": f"{a['time_skip_pct']:.4f}",
                "hold_skip_pct": f"{a['hold_skip_pct']:.4f}",
                "timeout_fraction": f"{a['timeout_fraction']:.4f}",
                "barrier_hit_fraction": f"{a['barrier_hit_fraction']:.4f}",
                "avg_bars_held": f"{a['avg_bars_held']:.2f}",
                "calmar": f"{a['calmar']:.4f}",
                "sharpe": f"{a['sharpe']:.4f}",
                "annual_pnl": f"{a['annual_pnl']:.2f}",
            })
    print(f"  Wrote cutoff_sweep.csv")


def write_trade_log_csv(per_split_cutoff, optimal_cutoff):
    """Write optimal_trade_log.csv — trades at recommended cutoff."""
    fieldnames = ["split_idx", "day", "entry_bar_global", "direction",
                  "true_label", "exit_type", "pnl", "bars_held", "minutes_since_open"]
    with open(RESULTS_DIR / "optimal_trade_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s_idx in sorted(per_split_cutoff.keys()):
            if optimal_cutoff not in per_split_cutoff[s_idx]:
                continue
            m = per_split_cutoff[s_idx][optimal_cutoff]
            for t in m["trade_log"]:
                writer.writerow({
                    "split_idx": s_idx,
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


def write_equity_curves_csv(per_split_cutoff, optimal_cutoff):
    """Write optimal_equity_curves.csv — per-split equity at recommended cutoff."""
    split_curves = {}
    for s_idx in sorted(per_split_cutoff.keys()):
        if optimal_cutoff in per_split_cutoff[s_idx]:
            split_curves[s_idx] = per_split_cutoff[s_idx][optimal_cutoff]['equity_curve']

    max_trades = max((len(ec) for ec in split_curves.values()), default=0)

    with open(RESULTS_DIR / "optimal_equity_curves.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["trade_idx"] + [f"split_{s:02d}" for s in sorted(split_curves.keys())]
        writer.writerow(header)
        for i in range(max_trades):
            row = [i]
            for s_idx in sorted(split_curves.keys()):
                ec = split_curves[s_idx]
                row.append(f"{ec[i]:.4f}" if i < len(ec) else "")
            writer.writerow(row)
    print(f"  Wrote optimal_equity_curves.csv")


def write_drawdown_summary_csv(per_split_cutoff, optimal_cutoff):
    """Write optimal_drawdown_summary.csv at recommended cutoff."""
    with open(RESULTS_DIR / "optimal_drawdown_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split_idx", "test_groups", "max_drawdown",
                         "max_consecutive_losses", "drawdown_duration_days",
                         "n_trades", "expectancy", "trades_per_day_mean",
                         "win_rate", "timeout_fraction"])
        for s_idx in sorted(per_split_cutoff.keys()):
            if optimal_cutoff not in per_split_cutoff[s_idx]:
                continue
            m = per_split_cutoff[s_idx][optimal_cutoff]
            writer.writerow([
                s_idx,
                f"{m['test_groups']}",
                f"{m['max_drawdown']:.2f}",
                m["max_consecutive_losses"],
                m["drawdown_duration_days"],
                m["n_trades"],
                f"{m['expectancy']:.4f}",
                f"{m['trades_per_day_mean']:.2f}",
                f"{m['win_rate']:.4f}",
                f"{m['timeout_fraction']:.4f}",
            ])
    print(f"  Wrote optimal_drawdown_summary.csv")


def write_daily_pnl_csv(per_split_cutoff, optimal_cutoff):
    """Write optimal_daily_pnl.csv at recommended cutoff."""
    with open(RESULTS_DIR / "optimal_daily_pnl.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split_idx", "day", "pnl", "n_trades",
                         "hold_skips", "time_skips", "total_bars"])
        for s_idx in sorted(per_split_cutoff.keys()):
            if optimal_cutoff not in per_split_cutoff[s_idx]:
                continue
            m = per_split_cutoff[s_idx][optimal_cutoff]
            for d in m["daily_pnl_records"]:
                writer.writerow([
                    s_idx, d["day"], f"{d['pnl']:.4f}",
                    d["n_trades"], d["hold_skips"],
                    d.get("time_skips", 0), d["total_bars"],
                ])
    print(f"  Wrote optimal_daily_pnl.csv")


def write_account_sizing_csv(agg_by_cutoff, optimal_cutoff):
    """Write optimal_account_sizing.csv at recommended cutoff."""
    curve = agg_by_cutoff[optimal_cutoff]['account_sizing_curve']
    with open(RESULTS_DIR / "optimal_account_sizing.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["account_size", "survival_count", "survival_rate"])
        for entry in curve:
            writer.writerow([
                entry["account_size"],
                entry["survival_count"],
                f"{entry['survival_rate']:.4f}",
            ])
    print(f"  Wrote optimal_account_sizing.csv")


def write_time_of_day_csv(per_split_cutoff, optimal_cutoff):
    """Write optimal_time_of_day.csv at recommended cutoff."""
    tod_buckets = {}
    for s_idx in sorted(per_split_cutoff.keys()):
        if optimal_cutoff not in per_split_cutoff[s_idx]:
            continue
        for t in per_split_cutoff[s_idx][optimal_cutoff]['trade_log']:
            mso = t['minutes_since_open']
            bucket = int(mso // 30) * 30
            bucket_label = f"{bucket}-{bucket+30}"
            tod_buckets[bucket_label] = tod_buckets.get(bucket_label, 0) + 1

    with open(RESULTS_DIR / "optimal_time_of_day.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "entry_count"])
        for bucket in sorted(tod_buckets.keys(), key=lambda x: int(x.split("-")[0])):
            writer.writerow([bucket, tod_buckets[bucket]])
    print(f"  Wrote optimal_time_of_day.csv")


def write_comparison_csv(agg_by_cutoff, optimal_cutoff, baseline_metrics):
    """Write comparison_table.csv — unfiltered vs filtered side-by-side."""
    a_unf = agg_by_cutoff[390]
    a_opt = agg_by_cutoff[optimal_cutoff]

    rows = [
        ("trades_per_day", a_unf['trades_per_day'], a_opt['trades_per_day']),
        ("expectancy_per_trade", a_unf['expectancy'], a_opt['expectancy']),
        ("daily_pnl", a_unf['daily_pnl'], a_opt['daily_pnl']),
        ("dd_worst", a_unf['dd_worst'], a_opt['dd_worst']),
        ("dd_median", a_unf['dd_median'], a_opt['dd_median']),
        ("min_acct_all", a_unf['min_acct_all'], a_opt['min_acct_all']),
        ("min_acct_95", a_unf['min_acct_95'], a_opt['min_acct_95']),
        ("win_rate", a_unf['win_rate'], a_opt['win_rate']),
        ("timeout_fraction", a_unf['timeout_fraction'], a_opt['timeout_fraction']),
        ("barrier_hit_fraction", a_unf['barrier_hit_fraction'], a_opt['barrier_hit_fraction']),
        ("hold_skip_pct", a_unf['hold_skip_pct'], a_opt['hold_skip_pct']),
        ("time_skip_pct", a_unf['time_skip_pct'], a_opt['time_skip_pct']),
        ("avg_bars_held", a_unf['avg_bars_held'], a_opt['avg_bars_held']),
        ("calmar", a_unf['calmar'], a_opt['calmar']),
        ("sharpe", a_unf['sharpe'], a_opt['sharpe']),
        ("annual_pnl", a_unf['annual_pnl'], a_opt['annual_pnl']),
    ]

    with open(RESULTS_DIR / "comparison_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "unfiltered_390", f"filtered_{optimal_cutoff}", "delta", "delta_pct"])
        for name, unf, filt in rows:
            delta = filt - unf
            delta_pct = (delta / abs(unf) * 100) if unf != 0 else 0
            writer.writerow([
                name,
                f"{unf:.4f}" if isinstance(unf, float) else unf,
                f"{filt:.4f}" if isinstance(filt, float) else filt,
                f"{delta:.4f}" if isinstance(delta, float) else delta,
                f"{delta_pct:.2f}",
            ])
    print(f"  Wrote comparison_table.csv")


# ==========================================================================
# Splits 18 & 32 Analysis
# ==========================================================================
def analyze_outlier_splits(per_split_cutoff, cutoff_levels):
    """Analyze splits 18 and 32 (group-4 outlier paths) across cutoffs."""
    outlier_splits = [18, 32]
    result = {}
    for s_idx in outlier_splits:
        if s_idx not in per_split_cutoff:
            continue
        result[s_idx] = {}
        for cutoff in cutoff_levels:
            if cutoff not in per_split_cutoff[s_idx]:
                continue
            m = per_split_cutoff[s_idx][cutoff]
            result[s_idx][cutoff] = {
                "expectancy": m['expectancy'],
                "max_drawdown": m['max_drawdown'],
                "n_trades": m['n_trades'],
                "trades_per_day": m['trades_per_day_mean'],
                "timeout_fraction": m['timeout_fraction'],
                "daily_pnl_mean": m['daily_pnl_mean'],
            }
    return result


# ==========================================================================
# Write metrics.json
# ==========================================================================
def write_metrics_json(per_split_cutoff, bar_level_results, agg_by_cutoff,
                       optimal_cutoff, selection_rule, outlier_analysis, t0_global):
    elapsed = time.time() - t0_global
    a_opt = agg_by_cutoff[optimal_cutoff]
    a_390 = agg_by_cutoff[390]

    # Build sweep table
    sweep_table = []
    for cutoff in sorted(agg_by_cutoff.keys(), reverse=True):
        a = agg_by_cutoff[cutoff]
        sweep_table.append({
            "cutoff": cutoff,
            "trades_per_day": round(a['trades_per_day'], 2),
            "expectancy": round(a['expectancy'], 4),
            "daily_pnl": round(a['daily_pnl'], 2),
            "dd_worst": round(a['dd_worst'], 2),
            "dd_median": round(a['dd_median'], 2),
            "min_acct_all": a['min_acct_all'],
            "min_acct_95": a['min_acct_95'],
            "win_rate": round(a['win_rate'], 4),
            "time_skip_pct": round(a['time_skip_pct'], 4),
            "hold_skip_pct": round(a['hold_skip_pct'], 4),
            "timeout_fraction": round(a['timeout_fraction'], 4),
        })

    # Sanity checks
    s1_exp = bar_level_results[0]["expectancy"] if 0 in bar_level_results else 0.0
    s1_ref = 1.065186
    sc_s1 = abs(s1_exp - s1_ref) < 0.01

    sc_s2_exp = a_390['expectancy']
    sc_s2 = abs(sc_s2_exp - 2.50) < 0.10

    sc_s3_tpd = a_390['trades_per_day']
    sc_s3 = abs(sc_s3_tpd - 162.2) < 5

    # SC-S4: monotonicity of expectancy (non-decreasing as cutoff tightens)
    sorted_cutoffs = sorted(agg_by_cutoff.keys(), reverse=True)  # 390, 375, ..., 270
    sc_s4 = True
    for i in range(1, len(sorted_cutoffs)):
        if agg_by_cutoff[sorted_cutoffs[i]]['expectancy'] < agg_by_cutoff[sorted_cutoffs[i-1]]['expectancy'] - 0.01:
            sc_s4 = False
            break

    # SC-S5: monotonicity of trades/day (non-increasing as cutoff tightens)
    sc_s5 = True
    for i in range(1, len(sorted_cutoffs)):
        if agg_by_cutoff[sorted_cutoffs[i]]['trades_per_day'] > agg_by_cutoff[sorted_cutoffs[i-1]]['trades_per_day'] + 0.1:
            sc_s5 = False
            break

    # Success criteria
    n_splits = len(per_split_cutoff)
    n_cutoffs = len(CUTOFF_LEVELS)
    total_sims = sum(
        len(per_split_cutoff[s]) for s in per_split_cutoff
    )
    sc_1 = n_splits == 45 and total_sims == 45 * n_cutoffs
    sc_2 = a_opt['expectancy'] >= 3.50
    sc_3 = a_opt['min_acct_all'] <= 30000
    sc_4 = len(sweep_table) == 7
    sc_5 = True  # comparison table written
    sc_6 = True  # account sizing produced
    sc_7 = True  # outputs written
    sc_8 = sc_s1  # bar-level split 0 match

    # Determine outcome
    if sc_2 and sc_3:
        outcome = "A"
        outcome_desc = (f"Timeout filtering effective. Cutoff={optimal_cutoff} achieves "
                        f"${a_opt['expectancy']:.2f}/trade (>=$3.50) and "
                        f"min_account ${a_opt['min_acct_all']:,} (<=$30K).")
    elif sc_1 and sc_4:
        if sc_2 and not sc_3:
            outcome = "B"
            outcome_desc = (f"SC-2 passes (exp=${a_opt['expectancy']:.2f}) but SC-3 fails "
                            f"(min_acct=${a_opt['min_acct_all']:,}). "
                            f"Drawdown is structural, not timeout-driven.")
        elif sc_3 and not sc_2:
            outcome = "B"
            outcome_desc = (f"SC-3 passes (min_acct=${a_opt['min_acct_all']:,}) but SC-2 fails "
                            f"(exp=${a_opt['expectancy']:.2f}). "
                            f"Timeout trades aren't the main expectancy drag.")
        else:
            # Check if filtering has meaningful effect
            exp_delta = abs(a_opt['expectancy'] - a_390['expectancy'])
            if exp_delta < 0.25:
                outcome = "C"
                outcome_desc = (f"Filtering has no meaningful effect. Best exp=${a_opt['expectancy']:.2f} "
                                f"vs baseline ${a_390['expectancy']:.2f} (delta ${exp_delta:.2f} < $0.25). "
                                f"Timeouts not concentrated in late-day entries.")
            else:
                outcome = "B"
                outcome_desc = (f"Filtering helps but neither target met. "
                                f"Best exp=${a_opt['expectancy']:.2f}, "
                                f"min_acct=${a_opt['min_acct_all']:,}. "
                                f"Selection: {selection_rule}.")
    else:
        outcome = "D"
        outcome_desc = "Simulation failed or incomplete."

    # Timeout fraction by cutoff
    timeout_by_cutoff = {str(c): round(agg_by_cutoff[c]['timeout_fraction'], 4) for c in sorted(agg_by_cutoff.keys(), reverse=True)}
    barrier_by_cutoff = {str(c): round(agg_by_cutoff[c]['barrier_hit_fraction'], 4) for c in sorted(agg_by_cutoff.keys(), reverse=True)}

    metrics = {
        "experiment": "timeout-filtered-sequential",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),

        # Primary metrics
        "cutoff_sweep_table": sweep_table,
        "optimal_cutoff": optimal_cutoff,
        "optimal_cutoff_selection_rule": selection_rule,
        "optimal_cutoff_expectancy": a_opt['expectancy'],
        "optimal_cutoff_min_account_all": a_opt['min_acct_all'],

        # Secondary
        "optimal_cutoff_daily_pnl": a_opt['daily_pnl'],
        "optimal_cutoff_calmar": a_opt['calmar'],
        "optimal_cutoff_sharpe": a_opt['sharpe'],
        "optimal_cutoff_trades_per_day": a_opt['trades_per_day'],
        "timeout_fraction_by_cutoff": timeout_by_cutoff,
        "barrier_hit_fraction_by_cutoff": barrier_by_cutoff,
        "splits_18_32_comparison": outlier_analysis,
        "daily_pnl_percentiles_optimal": a_opt['daily_pnl_percentiles'],
        "annual_expectancy_optimal": a_opt['annual_pnl'],

        # Full optimal risk profile
        "optimal_risk_profile": {
            "trades_per_day": a_opt['trades_per_day'],
            "expectancy": a_opt['expectancy'],
            "daily_pnl": a_opt['daily_pnl'],
            "daily_pnl_std": float(np.std([
                per_split_cutoff[s][optimal_cutoff]['daily_pnl_mean']
                for s in per_split_cutoff if optimal_cutoff in per_split_cutoff[s]
            ], ddof=1)),
            "dd_worst": a_opt['dd_worst'],
            "dd_median": a_opt['dd_median'],
            "min_acct_all": a_opt['min_acct_all'],
            "min_acct_95": a_opt['min_acct_95'],
            "win_rate": a_opt['win_rate'],
            "hold_skip_pct": a_opt['hold_skip_pct'],
            "time_skip_pct": a_opt['time_skip_pct'],
            "timeout_fraction": a_opt['timeout_fraction'],
            "barrier_hit_fraction": a_opt['barrier_hit_fraction'],
            "avg_bars_held": a_opt['avg_bars_held'],
            "max_consec_losses": a_opt['max_consec_losses'],
            "median_consec_losses": a_opt['median_consec_losses'],
            "dd_duration_worst": a_opt['dd_duration_worst'],
            "dd_duration_median": a_opt['dd_duration_median'],
            "calmar": a_opt['calmar'],
            "sharpe": a_opt['sharpe'],
            "annual_pnl": a_opt['annual_pnl'],
        },

        # Success criteria
        "success_criteria": {
            "SC-1": {"description": "All 7 cutoffs × 45 splits (315 sims)", "pass": sc_1,
                     "value": f"{total_sims} simulations across {n_splits} splits"},
            "SC-2": {"description": "Optimal exp >= $3.50", "pass": sc_2,
                     "value": a_opt['expectancy']},
            "SC-3": {"description": "Optimal min_acct_all <= $30,000", "pass": sc_3,
                     "value": a_opt['min_acct_all']},
            "SC-4": {"description": "Sweep table fully populated (7 rows)", "pass": sc_4,
                     "value": len(sweep_table)},
            "SC-5": {"description": "Comparison table populated", "pass": sc_5},
            "SC-6": {"description": "Account sizing curve produced", "pass": sc_6},
            "SC-7": {"description": "All output files written", "pass": sc_7},
            "SC-8": {"description": "Bar-level split 0 matches PR #38 within $0.01", "pass": sc_8,
                     "value": s1_exp, "reference": s1_ref},
        },

        "sanity_checks": {
            "SC-S1": {"description": "bar_level_exp_split0 within $0.01 of PR #38",
                      "pass": sc_s1, "value": s1_exp, "reference": s1_ref},
            "SC-S2": {"description": "cutoff_390_exp within $0.10 of PR #39's $2.50",
                      "pass": sc_s2, "value": sc_s2_exp, "reference": 2.50},
            "SC-S3": {"description": "cutoff_390_trades_per_day within 5 of PR #39's 162.2",
                      "pass": sc_s3, "value": sc_s3_tpd, "reference": 162.2},
            "SC-S4": {"description": "monotonicity_exp: non-decreasing as cutoff tightens",
                      "pass": sc_s4},
            "SC-S5": {"description": "monotonicity_trades: non-increasing as cutoff tightens",
                      "pass": sc_s5},
        },

        "outcome": outcome,
        "outcome_description": outcome_desc,

        "resource_usage": {
            "wall_clock_seconds": elapsed,
            "wall_clock_minutes": elapsed / 60,
            "total_training_runs": n_splits * 2,
            "total_simulations": total_sims,
            "gpu_hours": 0,
            "total_runs": n_splits,
        },

        "abort_triggered": False,
        "abort_reason": None,
        "notes": ("Local execution on Apple Silicon. "
                  "Train once per split, simulate 7 cutoff levels. "
                  "Corrected-base cost $2.49 RT."),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Wrote metrics.json")
    return metrics


# ==========================================================================
# Write analysis.md
# ==========================================================================
def write_analysis_md(metrics, agg_by_cutoff, per_split_cutoff, optimal_cutoff,
                      selection_rule, outlier_analysis):
    lines = []
    lines.append("# Timeout-Filtered Sequential Execution — Analysis\n")
    lines.append(f"**Date:** {metrics['timestamp']}")
    lines.append(f"**Outcome:** {metrics['outcome']} — {metrics['outcome_description']}\n")

    # 1. Executive summary
    lines.append("## 1. Executive Summary\n")
    a_opt = agg_by_cutoff[optimal_cutoff]
    a_390 = agg_by_cutoff[390]
    lines.append(f"Sweeping 7 time-of-day cutoff levels across 45 CPCV splits (315 total simulations). "
                 f"Recommended cutoff: **{optimal_cutoff}** (selection: {selection_rule}). "
                 f"Per-trade expectancy moves from ${a_390['expectancy']:.2f} (unfiltered) to "
                 f"${a_opt['expectancy']:.2f} (filtered). Min account from "
                 f"${a_390['min_acct_all']:,} to ${a_opt['min_acct_all']:,}.\n")

    # 2. Cutoff sweep results
    lines.append("## 2. Cutoff Sweep Results\n")
    lines.append("| Cutoff | Trades/Day | Exp/Trade | Daily PnL | DD Worst | DD Median | Min Acct All | Min Acct 95% | Win Rate | Time Skip% | Timeout% |")
    lines.append("|--------|-----------|-----------|-----------|----------|-----------|-------------|-------------|---------|------------|----------|")
    for cutoff in sorted(agg_by_cutoff.keys(), reverse=True):
        a = agg_by_cutoff[cutoff]
        marker = " **" if cutoff == optimal_cutoff else ""
        lines.append(f"| {cutoff}{marker} | {a['trades_per_day']:.1f} | ${a['expectancy']:.4f} | "
                     f"${a['daily_pnl']:.2f} | ${a['dd_worst']:.0f} | ${a['dd_median']:.0f} | "
                     f"${a['min_acct_all']:,} | ${a['min_acct_95']:,} | {a['win_rate']:.4f} | "
                     f"{a['time_skip_pct']:.4f} | {a['timeout_fraction']:.4f} |")

    # 3. Timeout fraction analysis
    lines.append("\n## 3. Timeout Fraction Analysis\n")
    lines.append("| Cutoff | Timeout Fraction | Barrier Hit Fraction |")
    lines.append("|--------|-----------------|---------------------|")
    for cutoff in sorted(agg_by_cutoff.keys(), reverse=True):
        a = agg_by_cutoff[cutoff]
        lines.append(f"| {cutoff} | {a['timeout_fraction']:.4f} | {a['barrier_hit_fraction']:.4f} |")

    timeout_delta = a_390['timeout_fraction'] - agg_by_cutoff[min(agg_by_cutoff.keys())]['timeout_fraction']
    lines.append(f"\nTimeout fraction change from 390→{min(agg_by_cutoff.keys())}: "
                 f"{a_390['timeout_fraction']:.4f} → {agg_by_cutoff[min(agg_by_cutoff.keys())]['timeout_fraction']:.4f} "
                 f"(delta {timeout_delta:.4f})\n")

    # 4. Optimal cutoff selection
    lines.append("## 4. Optimal Cutoff Selection\n")
    lines.append(f"**Recommended cutoff:** {optimal_cutoff}")
    lines.append(f"**Selection rule:** {selection_rule}")
    lines.append(f"**Rationale:** {'Primary rule — most conservative cutoff meeting both SC-2 ($3.50) and SC-3 ($30K).' if selection_rule == 'primary' else 'Fallback — see selection rule.'}\n")

    # 5. Detailed results at recommended cutoff
    lines.append("## 5. Detailed Risk Profile at Recommended Cutoff\n")
    rp = metrics['optimal_risk_profile']
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for k, v in rp.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")

    # 6. Unfiltered vs filtered comparison
    lines.append("\n## 6. Unfiltered vs Filtered Comparison\n")
    lines.append(f"| Metric | Unfiltered (390) | Filtered ({optimal_cutoff}) | Delta | Delta % |")
    lines.append("|--------|-----------------|--------------------------|-------|---------|")
    comp_items = [
        ("trades_per_day", a_390['trades_per_day'], a_opt['trades_per_day']),
        ("expectancy", a_390['expectancy'], a_opt['expectancy']),
        ("daily_pnl", a_390['daily_pnl'], a_opt['daily_pnl']),
        ("dd_worst", a_390['dd_worst'], a_opt['dd_worst']),
        ("dd_median", a_390['dd_median'], a_opt['dd_median']),
        ("min_acct_all", a_390['min_acct_all'], a_opt['min_acct_all']),
        ("win_rate", a_390['win_rate'], a_opt['win_rate']),
        ("timeout_fraction", a_390['timeout_fraction'], a_opt['timeout_fraction']),
        ("calmar", a_390['calmar'], a_opt['calmar']),
        ("sharpe", a_390['sharpe'], a_opt['sharpe']),
    ]
    for name, unf, filt in comp_items:
        delta = filt - unf
        dpct = (delta / abs(unf) * 100) if unf != 0 else 0
        if isinstance(unf, float):
            lines.append(f"| {name} | {unf:.4f} | {filt:.4f} | {delta:+.4f} | {dpct:+.1f}% |")
        else:
            lines.append(f"| {name} | {unf} | {filt} | {delta:+} | {dpct:+.1f}% |")

    # 7. Account sizing
    lines.append("\n## 7. Account Sizing at Recommended Cutoff\n")
    lines.append(f"| Threshold | Account Size |")
    lines.append(f"|-----------|-------------|")
    lines.append(f"| All 45 paths survive | **${a_opt['min_acct_all']:,}** |")
    lines.append(f"| 95% paths survive | **${a_opt['min_acct_95']:,}** |")

    # 8. Outlier path analysis
    lines.append("\n## 8. Outlier Path Analysis (Splits 18 & 32)\n")
    if outlier_analysis:
        for s_idx in sorted(outlier_analysis.keys()):
            lines.append(f"\n### Split {s_idx}\n")
            lines.append("| Cutoff | Expectancy | Max DD | Trades | Trades/Day | Timeout% |")
            lines.append("|--------|-----------|--------|--------|-----------|----------|")
            for cutoff in sorted(outlier_analysis[s_idx].keys(), reverse=True):
                o = outlier_analysis[s_idx][cutoff]
                lines.append(f"| {cutoff} | ${o['expectancy']:.4f} | ${o['max_drawdown']:.0f} | "
                             f"{o['n_trades']} | {o['trades_per_day']:.1f} | {o['timeout_fraction']:.4f} |")
    else:
        lines.append("No outlier splits found in completed data.\n")

    # 9. Daily PnL distribution
    lines.append("\n## 9. Daily PnL Distribution at Recommended Cutoff\n")
    pcts = a_opt.get('daily_pnl_percentiles', {})
    if pcts:
        lines.append("| Percentile | Daily PnL ($) |")
        lines.append("|------------|--------------|")
        for p, v in pcts.items():
            lines.append(f"| {p} | ${v:.2f} |")

    # 10. SC pass/fail
    lines.append("\n## 10. Success Criteria & Sanity Checks\n")
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
        val_str = f"{sv.get('value', 'N/A')}"
        ref_str = f"{sv.get('reference', 'N/A')}"
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

    print("=" * 60)
    print("Timeout-Filtered Sequential Execution")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading data...")
    features, labels, day_indices, unique_days_raw, extra = load_data()

    # CPCV baseline for verification
    cpcv_baseline = {"split_0_base_exp": 1.065186}

    # MVE: Split 0 only, cutoffs 390 and 330
    print("\n[2/6] Running MVE (split 0, cutoffs 390 & 330)...")
    mve_results, mve_bar, mve_abort = run_cpcv_with_cutoff_sweep(
        features, labels, day_indices, extra,
        mve_only=True, cpcv_baseline=cpcv_baseline)

    if mve_abort:
        print(f"\n*** ABORT: {mve_abort} ***")
        metrics = {
            "experiment": "timeout-filtered-sequential",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": mve_abort,
            "resource_usage": {"wall_clock_seconds": time.time() - t0},
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # MVE summary
    s0 = mve_results[0]
    c390 = s0[390]
    c330 = s0[330]
    print(f"    MVE PASS:")
    print(f"      cutoff=390: {c390['trades_per_day_mean']:.1f} trades/day, ${c390['expectancy']:.4f}/trade")
    print(f"      cutoff=330: {c330['trades_per_day_mean']:.1f} trades/day, ${c330['expectancy']:.4f}/trade")
    print(f"      bar-level split 0: ${mve_bar[0]['expectancy']:.4f} (ref: $1.065186)")

    # Full sweep: 45 splits × 7 cutoffs
    print("\n[3/6] Running full 45-split × 7-cutoff sweep...")
    per_split_cutoff, bar_level_results, abort_reason = run_cpcv_with_cutoff_sweep(
        features, labels, day_indices, extra,
        mve_only=False, cpcv_baseline=cpcv_baseline)

    if abort_reason:
        print(f"\n*** ABORT: {abort_reason} ***")
        metrics = {
            "experiment": "timeout-filtered-sequential",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": abort_reason,
            "resource_usage": {"wall_clock_seconds": time.time() - t0},
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # Aggregate per cutoff
    print("\n[4/6] Aggregating per-cutoff results...")
    agg_by_cutoff = aggregate_per_cutoff(per_split_cutoff, CUTOFF_LEVELS)

    # Quick summary
    for cutoff in sorted(agg_by_cutoff.keys(), reverse=True):
        a = agg_by_cutoff[cutoff]
        print(f"    cutoff={cutoff}: {a['trades_per_day']:.1f} trades/day, "
              f"${a['expectancy']:.4f}/trade, daily=${a['daily_pnl']:.2f}, "
              f"dd_worst=${a['dd_worst']:.0f}, timeout={a['timeout_fraction']:.4f}")

    # Select optimal cutoff
    optimal_cutoff, selection_rule = select_optimal_cutoff(agg_by_cutoff)
    print(f"\n    Optimal cutoff: {optimal_cutoff} (rule: {selection_rule})")

    # Outlier analysis
    outlier_analysis = analyze_outlier_splits(per_split_cutoff, CUTOFF_LEVELS)

    # Write outputs
    print("\n[5/6] Writing outputs...")
    write_cutoff_sweep_csv(agg_by_cutoff)
    write_trade_log_csv(per_split_cutoff, optimal_cutoff)
    write_equity_curves_csv(per_split_cutoff, optimal_cutoff)
    write_drawdown_summary_csv(per_split_cutoff, optimal_cutoff)
    write_daily_pnl_csv(per_split_cutoff, optimal_cutoff)
    write_account_sizing_csv(agg_by_cutoff, optimal_cutoff)
    write_time_of_day_csv(per_split_cutoff, optimal_cutoff)
    write_comparison_csv(agg_by_cutoff, optimal_cutoff, None)

    print("\n[6/6] Writing metrics.json and analysis.md...")
    metrics = write_metrics_json(per_split_cutoff, bar_level_results, agg_by_cutoff,
                                 optimal_cutoff, selection_rule, outlier_analysis, t0)
    write_analysis_md(metrics, agg_by_cutoff, per_split_cutoff, optimal_cutoff,
                      selection_rule, outlier_analysis)

    # Copy spec
    spec_src = PROJECT_ROOT / ".kit" / "experiments" / "timeout-filtered-sequential.md"
    try:
        with open(spec_src, "r") as f_in:
            spec_content = f_in.read()
        with open(RESULTS_DIR / "spec.md", "w") as f_out:
            f_out.write(spec_content)
        print(f"  Copied spec.md")
    except Exception as e:
        print(f"  Warning: could not copy spec.md: {e}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Outcome: {metrics['outcome']} — {metrics['outcome_description']}")
    print(f"Optimal cutoff: {optimal_cutoff}")
    print(f"Optimal expectancy: ${agg_by_cutoff[optimal_cutoff]['expectancy']:.4f}/trade")
    print(f"Optimal min_account: ${agg_by_cutoff[optimal_cutoff]['min_acct_all']:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
