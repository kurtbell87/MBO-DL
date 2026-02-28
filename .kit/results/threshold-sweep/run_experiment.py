#!/usr/bin/env python3
"""
Experiment: Stage 1 Threshold Sweep — Optimizing Hold-Bar Exposure
Spec: .kit/experiments/threshold-sweep.md

Adapts the PnL Realized Return pipeline (PR #35) to sweep Stage 1
decision thresholds from 0.50 to 0.90. Models are trained ONCE per fold;
only the thresholding on P(directional) changes.

No re-training needed. This is a post-hoc re-scoring exercise.
"""

import json
import os
import sys
import time
import random
import hashlib
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================================================
# Config
# ==========================================================================
SEED = 42
WORKTREE_ROOT = Path("/Users/brandonbell/LOCAL_DEV/MBO-DL-label-geom-p1")
RESULTS_DIR = WORKTREE_ROOT / ".kit" / "results" / "threshold-sweep"
DATA_BASE = WORKTREE_ROOT / ".kit" / "results" / "label-geometry-1h"

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

COST_SCENARIOS = {"optimistic": 2.49, "base": 3.74, "pessimistic": 6.25}
TICK_VALUE = 1.25
TICK_SIZE = 0.25
HORIZON_BARS = 720  # 3600s / 5s

GEOMETRIES = {
    "19_7": {"target": 19, "stop": 7, "label": "19:7", "bev_wr": 0.384,
             "data_dir": DATA_BASE / "geom_19_7"},
    "10_5": {"target": 10, "stop": 5, "label": "10:5", "bev_wr": 0.533,
             "data_dir": DATA_BASE / "geom_10_5"},
}

WF_FOLDS = [
    {"train_range": (1, 100), "test_range": (101, 150), "name": "Fold 1"},
    {"train_range": (1, 150), "test_range": (151, 201), "name": "Fold 2"},
    {"train_range": (1, 201), "test_range": (202, 251), "name": "Fold 3 (holdout)"},
]

PURGE_BARS = 500
WALL_CLOCK_LIMIT_S = 15 * 60  # 15 min per spec

# Threshold sweep levels
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Baseline from pnl-realized-return (PR #35) for sanity check
BASELINE_19_7 = {
    "realized_wf_expectancy": 0.901517613505994,
    "trade_rate": 0.8518110743722463,
    "hold_fraction": 0.44389057472507454,
    "dir_bar_pnl": 3.77471732462107,
    "per_fold_realized": [0.010826135364183474, 2.5384366083743046, 0.1552900967794935],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================================================
# Data Loading (identical to pnl-realized-return)
# ==========================================================================
def load_geometry_data(geom_key):
    geom = GEOMETRIES[geom_key]
    data_dir = geom["data_dir"]

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files in {data_dir}")

    print(f"  Loading {len(parquet_files)} Parquet files from {data_dir.name}...")

    all_features = []
    all_labels = []
    all_day_raw = []
    all_fwd_return_1 = []
    all_tb_long_triggered = []
    all_tb_short_triggered = []
    all_tb_both_triggered = []
    all_bar_pos_in_day = []
    all_day_lengths = []

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
        tb_long = df["tb_long_triggered"].to_numpy().astype(np.float64)
        tb_short = df["tb_short_triggered"].to_numpy().astype(np.float64)
        tb_both = df["tb_both_triggered"].to_numpy().astype(np.float64)

        bar_pos = np.arange(n)

        all_features.append(features)
        all_labels.append(labels)
        all_day_raw.append(day_raw)
        all_fwd_return_1.append(fwd1)
        all_tb_long_triggered.append(tb_long)
        all_tb_short_triggered.append(tb_short)
        all_tb_both_triggered.append(tb_both)
        all_bar_pos_in_day.append(bar_pos)
        all_day_lengths.append(np.full(n, n))

        unique_day = day_raw[0]
        day_offsets[unique_day] = (global_offset, n)
        global_offset += n

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)
    day_raw = np.concatenate(all_day_raw)
    fwd_return_1 = np.concatenate(all_fwd_return_1)
    tb_long_triggered = np.concatenate(all_tb_long_triggered)
    tb_short_triggered = np.concatenate(all_tb_short_triggered)
    tb_both_triggered = np.concatenate(all_tb_both_triggered)
    bar_pos_in_day = np.concatenate(all_bar_pos_in_day)
    day_lengths = np.concatenate(all_day_lengths)

    print(f"  Total rows after warmup filter: {len(labels)}")

    unique_days_raw = sorted(set(day_raw.tolist()))
    day_map = {d: i + 1 for i, d in enumerate(unique_days_raw)}
    day_indices = np.array([day_map[d] for d in day_raw])

    assert features.shape[1] == 20
    assert set(np.unique(labels)).issubset({-1, 0, 1})

    n_total = len(labels)
    n_short = (labels == -1).sum()
    n_hold = (labels == 0).sum()
    n_long = (labels == 1).sum()
    print(f"  Classes: short={n_short} ({100*n_short/n_total:.1f}%), "
          f"hold={n_hold} ({100*n_hold/n_total:.1f}%), "
          f"long={n_long} ({100*n_long/n_total:.1f}%)")
    print(f"  Day range: {min(day_indices)}-{max(day_indices)}, "
          f"Unique days: {len(unique_days_raw)}")

    extra = {
        "fwd_return_1": fwd_return_1,
        "tb_long_triggered": tb_long_triggered,
        "tb_short_triggered": tb_short_triggered,
        "tb_both_triggered": tb_both_triggered,
        "bar_pos_in_day": bar_pos_in_day,
        "day_lengths": day_lengths,
        "day_offsets": day_offsets,
        "day_raw": day_raw,
    }
    return features, labels, day_indices, unique_days_raw, extra


def compute_fwd_return_720(indices, extra):
    fwd1 = extra["fwd_return_1"]
    bar_pos = extra["bar_pos_in_day"]
    day_raw = extra["day_raw"]
    day_offsets = extra["day_offsets"]

    fwd_returns = np.zeros(len(indices))
    actual_bars = np.zeros(len(indices), dtype=np.int64)

    for k, idx in enumerate(indices):
        dr = day_raw[idx]
        day_start, day_len = day_offsets[dr]
        pos = bar_pos[idx]
        remaining = day_len - pos
        n_forward = min(HORIZON_BARS, remaining)
        fwd_returns[k] = np.sum(fwd1[idx:idx + n_forward])
        actual_bars[k] = n_forward

    return fwd_returns, actual_bars


# ==========================================================================
# Train/Test Split
# ==========================================================================
def apply_purge(train_indices, test_indices, day_indices, purge_bars):
    train_set = set(train_indices.tolist())
    test_sorted = np.sort(test_indices)
    if len(test_sorted) == 0:
        return np.array(sorted(train_set)), 0

    excluded = set()
    test_start = test_sorted[0]
    test_end = test_sorted[-1]

    for i in range(max(0, test_start - purge_bars), test_start):
        if i in train_set:
            excluded.add(i)
    for i in range(test_end + 1, min(len(day_indices), test_end + 1 + purge_bars)):
        if i in train_set:
            excluded.add(i)

    clean_train = np.array(sorted(train_set - excluded))
    return clean_train, len(excluded)


# ==========================================================================
# PnL Computation (realized return model — identical to baseline)
# ==========================================================================
def compute_pnl_realized(true_labels, pred_labels, target, stop, rt_cost,
                         test_indices, extra):
    """Compute realized-return PnL for given predictions.

    Directional bars: barrier payoff based on prediction correctness.
    Hold bars: 720-bar forward return * sign(pred) - rt_cost.
    """
    n = len(true_labels)
    win_pnl = target * TICK_VALUE
    loss_pnl = stop * TICK_VALUE

    trades = (pred_labels != 0)
    dir_bars = (true_labels != 0)
    hold_bars = (true_labels == 0)
    dir_trades = trades & dir_bars
    hold_trades = trades & hold_bars

    n_trades = int(trades.sum())
    n_dir_trades = int(dir_trades.sum())
    n_hold_trades = int(hold_trades.sum())

    # Directional-bar PnL
    correct = (pred_labels == true_labels) & dir_trades
    wrong = (pred_labels != true_labels) & dir_trades

    dir_pnl = np.zeros(n)
    dir_pnl[correct] = win_pnl - rt_cost
    dir_pnl[wrong] = -loss_pnl - rt_cost

    # Realized PnL = dir_pnl + hold bar realized returns
    realized_pnl = dir_pnl.copy()

    if n_hold_trades > 0:
        hold_trade_indices_global = test_indices[hold_trades]
        fwd_returns_720, actual_bars_arr = compute_fwd_return_720(
            hold_trade_indices_global, extra)

        hold_pred_signs = np.sign(pred_labels[hold_trades])
        hold_gross_pnl = fwd_returns_720 * TICK_VALUE * hold_pred_signs
        hold_net_pnl = hold_gross_pnl - rt_cost
        realized_pnl[hold_trades] = hold_net_pnl
    else:
        fwd_returns_720 = np.array([])
        hold_net_pnl = np.array([])

    # Compute per-trade metrics
    realized_per_trade = float(realized_pnl[trades].sum()) / n_trades if n_trades > 0 else 0.0
    dir_mean_pnl = float(dir_pnl[dir_trades].mean()) if n_dir_trades > 0 else 0.0
    hold_mean_pnl = float(realized_pnl[hold_trades].mean()) if n_hold_trades > 0 else 0.0

    # Inflated = dir_pnl / n_dir_trades (per directional trade)
    inflated_per_dir_trade = float(dir_pnl[dir_trades].sum()) / n_dir_trades if n_dir_trades > 0 else 0.0

    frac_dir = n_dir_trades / n_trades if n_trades > 0 else 0.0
    frac_hold = n_hold_trades / n_trades if n_trades > 0 else 0.0

    return {
        "n_trades": n_trades,
        "n_dir_trades": n_dir_trades,
        "n_hold_trades": n_hold_trades,
        "realized_per_trade": realized_per_trade,
        "inflated_per_dir_trade": inflated_per_dir_trade,
        "dir_mean_pnl": dir_mean_pnl,
        "hold_mean_pnl": hold_mean_pnl,
        "frac_directional": float(frac_dir),
        "frac_hold": float(frac_hold),
        "realized_total_pnl": float(realized_pnl[trades].sum()),
        "dir_total_pnl": float(dir_pnl[dir_trades].sum()),
    }


# ==========================================================================
# Train one fold and return raw probabilities for threshold sweep
# ==========================================================================
def train_fold_return_probas(features, labels, day_indices, extra,
                             train_range, test_range, geom_key, fold_name):
    """Train Stage 1 + Stage 2 models ONCE. Return raw probabilities for post-hoc thresholding."""
    geom = GEOMETRIES[geom_key]
    target = geom["target"]
    stop = geom["stop"]

    train_mask = (day_indices >= train_range[0]) & (day_indices <= train_range[1])
    test_mask = (day_indices >= test_range[0]) & (day_indices <= test_range[1])

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(test_indices) == 0 or len(train_indices) == 0:
        return None

    clean_train, n_purged = apply_purge(train_indices, test_indices, day_indices, PURGE_BARS)
    if len(clean_train) == 0:
        return None

    # z-score normalize using training stats
    ft_train = features[clean_train].copy()
    ft_test = features[test_indices].copy()
    f_mean = np.nanmean(ft_train, axis=0)
    f_std = np.nanstd(ft_train, axis=0)
    f_std[f_std < 1e-10] = 1.0
    ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
    ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)

    # Internal train/val split
    train_days_in_fold = sorted(set(day_indices[clean_train].tolist()))
    n_val_days = max(1, len(train_days_in_fold) // 5)
    val_day_set = set(train_days_in_fold[-n_val_days:])

    val_mask_inner = np.array([day_indices[i] in val_day_set for i in clean_train])
    train_mask_inner = ~val_mask_inner

    inner_train_z = ft_train_z[train_mask_inner]
    inner_val_z = ft_train_z[val_mask_inner]

    labels_train_all = labels[clean_train]
    labels_test = labels[test_indices]

    inner_train_labels = labels_train_all[train_mask_inner]
    inner_val_labels = labels_train_all[val_mask_inner]

    # STAGE 1: directional vs hold
    s1_train_labels = (inner_train_labels != 0).astype(np.int64)
    s1_val_labels = (inner_val_labels != 0).astype(np.int64)

    clf_s1 = xgb.XGBClassifier(**TUNED_XGB_PARAMS_BINARY)
    if len(inner_val_z) > 0:
        clf_s1.fit(inner_train_z, s1_train_labels,
                   eval_set=[(inner_val_z, s1_val_labels)],
                   verbose=False)
    else:
        clf_s1.fit(inner_train_z, s1_train_labels, verbose=False)

    # Raw P(directional) — the key for threshold sweep
    s1_pred_proba = clf_s1.predict_proba(ft_test_z)[:, 1]

    # STAGE 2: long vs short (trained on directional bars only)
    dir_mask_train = (inner_train_labels != 0)
    dir_mask_val = (inner_val_labels != 0)

    s2_train_z = inner_train_z[dir_mask_train]
    s2_val_z = inner_val_z[dir_mask_val]
    s2_train_labels = (inner_train_labels[dir_mask_train] == 1).astype(np.int64)
    s2_val_labels = (inner_val_labels[dir_mask_val] == 1).astype(np.int64)

    clf_s2 = xgb.XGBClassifier(**TUNED_XGB_PARAMS_BINARY)
    if len(s2_val_z) > 0:
        clf_s2.fit(s2_train_z, s2_train_labels,
                   eval_set=[(s2_val_z, s2_val_labels)],
                   verbose=False)
    else:
        clf_s2.fit(s2_train_z, s2_train_labels, verbose=False)

    # Stage 2: direction predictions (applied to all test bars; thresholding decides which are used)
    s2_pred_proba = clf_s2.predict_proba(ft_test_z)[:, 1]
    s2_pred = np.where(s2_pred_proba > 0.5, 1, -1)

    del clf_s1, clf_s2

    return {
        "fold_name": fold_name,
        "geom_key": geom_key,
        "n_test": len(test_indices),
        "test_indices": test_indices,
        "labels_test": labels_test,
        "s1_pred_proba": s1_pred_proba,
        "s2_pred": s2_pred,
        "day_indices": day_indices,
    }


def apply_threshold(fold_data, threshold, extra, geom_key):
    """Apply a Stage 1 threshold to pre-computed probabilities and compute PnL."""
    geom = GEOMETRIES[geom_key]
    target = geom["target"]
    stop = geom["stop"]

    labels_test = fold_data["labels_test"]
    s1_proba = fold_data["s1_pred_proba"]
    s2_pred = fold_data["s2_pred"]
    test_indices = fold_data["test_indices"]
    n_test = fold_data["n_test"]

    # Apply threshold: bars with P(directional) > threshold get Stage 2 direction
    s1_dir_mask = (s1_proba > threshold)
    combined_pred = np.zeros(len(labels_test), dtype=np.int64)
    combined_pred[s1_dir_mask] = s2_pred[s1_dir_mask]

    n_trades = int((combined_pred != 0).sum())
    trade_rate = float(n_trades / n_test) if n_test > 0 else 0.0

    # Compute PnL at all cost scenarios
    pnl_results = {}
    for scenario, rt_cost in COST_SCENARIOS.items():
        pnl = compute_pnl_realized(
            labels_test, combined_pred, target, stop, rt_cost,
            test_indices, extra)
        pnl_results[scenario] = pnl

    # Compute break-even RT (at zero cost)
    pnl_nocost = compute_pnl_realized(
        labels_test, combined_pred, target, stop, 0.0,
        test_indices, extra)
    breakeven_rt = pnl_nocost["realized_total_pnl"] / n_trades if n_trades > 0 else 0.0

    # Compute directional accuracy on traded bars
    trades = (combined_pred != 0)
    dir_bars = (labels_test != 0)
    dir_trades = trades & dir_bars
    both_nonzero = (combined_pred != 0) & (labels_test != 0)
    n_dir_pairs = both_nonzero.sum()
    dir_acc = float((combined_pred == labels_test)[both_nonzero].sum() / n_dir_pairs) if n_dir_pairs > 0 else 0.0

    pnl_base = pnl_results["base"]

    # Daily PnL at base cost
    day_indices_test = fold_data["day_indices"][test_indices]
    n_test_days = len(set(day_indices_test.tolist()))
    daily_pnl = pnl_base["realized_total_pnl"] / n_test_days if n_test_days > 0 else 0.0

    return {
        "threshold": threshold,
        "fold_name": fold_data["fold_name"],
        "n_test": n_test,
        "n_trades": n_trades,
        "trade_rate": trade_rate,
        "hold_fraction": pnl_base["frac_hold"],
        "dir_fraction": pnl_base["frac_directional"],
        "realized_exp_base": pnl_base["realized_per_trade"],
        "realized_exp_optimistic": pnl_results["optimistic"]["realized_per_trade"],
        "realized_exp_pessimistic": pnl_results["pessimistic"]["realized_per_trade"],
        "dir_bar_exp": pnl_base["inflated_per_dir_trade"],
        "hold_bar_exp": pnl_base["hold_mean_pnl"],
        "n_dir_trades": pnl_base["n_dir_trades"],
        "n_hold_trades": pnl_base["n_hold_trades"],
        "directional_accuracy": dir_acc,
        "breakeven_rt": breakeven_rt,
        "daily_pnl": daily_pnl,
        "n_test_days": n_test_days,
    }


# ==========================================================================
# P(directional) Distribution Analysis
# ==========================================================================
def analyze_p_directional_distribution(all_fold_data):
    """Analyze the distribution of Stage 1 P(directional) across all folds."""
    all_proba = np.concatenate([fd["s1_pred_proba"] for fd in all_fold_data])

    percentiles = {
        "p5": float(np.percentile(all_proba, 5)),
        "p10": float(np.percentile(all_proba, 10)),
        "p25": float(np.percentile(all_proba, 25)),
        "p50": float(np.percentile(all_proba, 50)),
        "p75": float(np.percentile(all_proba, 75)),
        "p90": float(np.percentile(all_proba, 90)),
        "p95": float(np.percentile(all_proba, 95)),
    }

    # Histogram bins for threshold analysis
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.50, 0.55, 0.60, 0.65,
            0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    hist_counts, hist_edges = np.histogram(all_proba, bins=bins)
    histogram = []
    for i in range(len(hist_counts)):
        histogram.append({
            "bin_low": float(hist_edges[i]),
            "bin_high": float(hist_edges[i + 1]),
            "count": int(hist_counts[i]),
            "fraction": float(hist_counts[i] / len(all_proba)),
        })

    # Key diagnostic: fraction within [0.45, 0.55]
    frac_near_05 = float(np.mean((all_proba >= 0.45) & (all_proba <= 0.55)))
    # Fraction above each threshold
    frac_above = {}
    for t in THRESHOLDS:
        frac_above[f"{t:.2f}"] = float(np.mean(all_proba > t))

    return {
        "n_total": len(all_proba),
        "mean": float(all_proba.mean()),
        "std": float(all_proba.std()),
        "min": float(all_proba.min()),
        "max": float(all_proba.max()),
        "percentiles": percentiles,
        "histogram": histogram,
        "fraction_within_045_055": frac_near_05,
        "fraction_above_threshold": frac_above,
    }


# ==========================================================================
# MVE
# ==========================================================================
def run_mve(features, labels, day_indices, extra):
    """Run 3 thresholds (0.50, 0.70, 0.90) on Fold 1 only at 19:7."""
    print("\n" + "=" * 70)
    print("MVE — 3 thresholds, Fold 1 only, 19:7")
    print("=" * 70)

    fold_data = train_fold_return_probas(
        features, labels, day_indices, extra,
        train_range=(1, 100), test_range=(101, 150),
        geom_key="19_7", fold_name="MVE Fold 1")

    if fold_data is None:
        return {"abort": True, "abort_reason": "MVE fold returned None"}

    mve_thresholds = [0.50, 0.70, 0.90]
    mve_results = []
    for t in mve_thresholds:
        result = apply_threshold(fold_data, t, extra, "19_7")
        mve_results.append(result)
        print(f"\n  Threshold {t:.2f}:")
        print(f"    Trade rate: {result['trade_rate']:.4f}")
        print(f"    Hold fraction: {result['hold_fraction']:.4f}")
        print(f"    Realized exp (base): ${result['realized_exp_base']:.4f}")
        print(f"    Dir-bar exp: ${result['dir_bar_exp']:.4f}")

    # MVE checks
    gates = {}

    # Check 1: T=0.50 reproduces baseline
    r050 = mve_results[0]
    baseline_fold1_realized = BASELINE_19_7["per_fold_realized"][0]  # $0.0108
    realized_diff = abs(r050["realized_exp_base"] - baseline_fold1_realized)
    gates["baseline_reproduces"] = realized_diff < 0.02
    print(f"\n  Baseline reproduction: realized diff=${realized_diff:.6f} "
          f"({'PASS' if gates['baseline_reproduces'] else 'FAIL'})")

    # Check 2: Trade rate monotonically decreases
    tr_mono = (mve_results[0]["trade_rate"] >= mve_results[1]["trade_rate"] >=
               mve_results[2]["trade_rate"])
    gates["trade_rate_monotonic"] = tr_mono
    print(f"  Trade rate monotonicity: {tr_mono} "
          f"({mve_results[0]['trade_rate']:.3f} >= {mve_results[1]['trade_rate']:.3f} >= {mve_results[2]['trade_rate']:.3f})")

    # Check 3: Hold fraction monotonically decreases
    hf_mono = (mve_results[0]["hold_fraction"] >= mve_results[1]["hold_fraction"] >=
               mve_results[2]["hold_fraction"])
    gates["hold_fraction_monotonic"] = hf_mono
    print(f"  Hold fraction monotonicity: {hf_mono} "
          f"({mve_results[0]['hold_fraction']:.3f} >= {mve_results[1]['hold_fraction']:.3f} >= {mve_results[2]['hold_fraction']:.3f})")

    # Check 4: Expectancy increases from 0.50 to 0.70
    exp_improves = mve_results[1]["realized_exp_base"] > mve_results[0]["realized_exp_base"]
    gates["exp_improves_050_to_070"] = exp_improves
    print(f"  Exp improves 0.50->0.70: {exp_improves} "
          f"(${mve_results[0]['realized_exp_base']:.4f} -> ${mve_results[1]['realized_exp_base']:.4f})")

    gates["all_passed"] = all(gates.values())
    print(f"\n  MVE gates: {'ALL PASSED' if gates['all_passed'] else 'FAILED'}")

    if not gates["all_passed"]:
        return {"abort": True, "abort_reason": f"MVE gates failed: {gates}", "gates": gates}

    return {"abort": False, "gates": gates, "mve_results": mve_results}


# ==========================================================================
# Full Protocol
# ==========================================================================
def run_full_protocol(data, t0_global):
    """Run threshold sweep across all folds and geometries."""
    all_results = {}  # geom_key -> list of {threshold, fold, metrics}
    all_fold_data = {}  # geom_key -> list of fold_data dicts

    for geom_key in ["19_7", "10_5"]:
        print(f"\n{'=' * 70}")
        print(f"THRESHOLD SWEEP: Geometry {GEOMETRIES[geom_key]['label']}")
        print(f"{'=' * 70}")

        geom_data = data[geom_key]
        fold_data_list = []

        # Step 1: Train models for each fold (ONCE)
        for wf_idx, wf in enumerate(WF_FOLDS):
            elapsed = time.time() - t0_global
            if elapsed > WALL_CLOCK_LIMIT_S:
                print(f"\n** WALL-CLOCK ABORT: {elapsed:.0f}s > {WALL_CLOCK_LIMIT_S}s **")
                return all_results, all_fold_data, True, f"Wall-clock {elapsed:.0f}s > {WALL_CLOCK_LIMIT_S}s"

            print(f"\n  Training {wf['name']} (train {wf['train_range']}, test {wf['test_range']})...")
            t_fold = time.time()

            fold_data = train_fold_return_probas(
                geom_data["features"],
                geom_data["labels"],
                geom_data["day_indices"],
                geom_data["extra"],
                wf["train_range"],
                wf["test_range"],
                geom_key,
                wf["name"],
            )

            if fold_data is None:
                print(f"  !! Fold skipped (empty split)")
                continue

            fold_data_list.append(fold_data)
            print(f"  Trained in {time.time() - t_fold:.1f}s. "
                  f"Test set: {fold_data['n_test']} bars.")

        all_fold_data[geom_key] = fold_data_list

        # Step 2: Sweep thresholds (post-hoc, no retraining)
        geom_results = []
        for threshold in THRESHOLDS:
            for fold_data in fold_data_list:
                result = apply_threshold(
                    fold_data, threshold,
                    geom_data["extra"], geom_key)
                geom_results.append(result)

        all_results[geom_key] = geom_results

        # Print summary for this geometry
        print(f"\n  Summary for {GEOMETRIES[geom_key]['label']}:")
        print(f"  {'Threshold':>9} {'Trade Rate':>10} {'Hold Frac':>9} "
              f"{'Realized':>10} {'Dir Bar':>10} {'Trades':>8}")
        for threshold in THRESHOLDS:
            t_results = [r for r in geom_results if r["threshold"] == threshold]
            mean_tr = np.mean([r["trade_rate"] for r in t_results])
            mean_hf = np.mean([r["hold_fraction"] for r in t_results])
            mean_re = np.mean([r["realized_exp_base"] for r in t_results])
            mean_db = np.mean([r["dir_bar_exp"] for r in t_results])
            mean_nt = np.mean([r["n_trades"] for r in t_results])
            print(f"  {threshold:>9.2f} {mean_tr:>10.4f} {mean_hf:>9.4f} "
                  f"${mean_re:>9.4f} ${mean_db:>9.4f} {mean_nt:>8.0f}")

    return all_results, all_fold_data, False, None


# ==========================================================================
# Metrics Builder
# ==========================================================================
def build_threshold_curve(geom_results):
    """Build threshold -> metrics curve from per-fold results."""
    curve = []
    for threshold in THRESHOLDS:
        t_results = [r for r in geom_results if r["threshold"] == threshold]
        if not t_results:
            continue

        trade_rates = [r["trade_rate"] for r in t_results]
        hold_fracs = [r["hold_fraction"] for r in t_results]
        realized_exps = [r["realized_exp_base"] for r in t_results]
        dir_bar_exps = [r["dir_bar_exp"] for r in t_results]
        hold_bar_exps = [r["hold_bar_exp"] for r in t_results]
        breakeven_rts = [r["breakeven_rt"] for r in t_results]
        n_trades_list = [r["n_trades"] for r in t_results]
        daily_pnls = [r["daily_pnl"] for r in t_results]

        exp_std = float(np.std(realized_exps))
        exp_mean = float(np.mean(realized_exps))
        cv = abs(exp_std / exp_mean) * 100 if abs(exp_mean) > 1e-6 else 9999.0

        curve.append({
            "threshold": threshold,
            "trade_rate_mean": float(np.mean(trade_rates)),
            "hold_fraction_mean": float(np.mean(hold_fracs)),
            "realized_exp_mean": exp_mean,
            "realized_exp_std": exp_std,
            "realized_exp_cv_pct": cv,
            "realized_exp_per_fold": realized_exps,
            "dir_bar_exp_mean": float(np.mean(dir_bar_exps)),
            "hold_bar_exp_mean": float(np.mean(hold_bar_exps)),
            "breakeven_rt_mean": float(np.mean(breakeven_rts)),
            "n_trades_mean": float(np.mean(n_trades_list)),
            "n_trades_per_fold": n_trades_list,
            "daily_pnl_mean": float(np.mean(daily_pnls)),
        })
    return curve


def find_optimal_threshold(curve):
    """Find T* that maximizes realized expectancy with trade rate > 15%."""
    eligible = [pt for pt in curve if pt["trade_rate_mean"] > 0.15]
    if not eligible:
        return None, "No threshold has trade rate > 15%"
    best = max(eligible, key=lambda x: x["realized_exp_mean"])
    return best, None


def compute_pareto_frontier(curve):
    """Find thresholds on the Pareto frontier of expectancy x trade_rate."""
    frontier = []
    for pt in curve:
        dominated = False
        for other in curve:
            if (other["realized_exp_mean"] > pt["realized_exp_mean"] and
                    other["trade_rate_mean"] > pt["trade_rate_mean"]):
                dominated = True
                break
        if not dominated:
            frontier.append(pt["threshold"])
    return frontier


def build_metrics(all_results, all_fold_data, mve_results, t0, elapsed_total,
                  abort_triggered=False, abort_reason=None):
    """Build the complete metrics.json structure."""
    metrics = {
        "experiment": "threshold-sweep",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # P(directional) distribution (19:7 only — primary geometry)
    if "19_7" in all_fold_data and all_fold_data["19_7"]:
        p_dist = analyze_p_directional_distribution(all_fold_data["19_7"])
        metrics["p_directional_distribution"] = p_dist

    # Per-geometry threshold curves
    for geom_key in ["19_7", "10_5"]:
        geom_results = all_results.get(geom_key, [])
        if not geom_results:
            continue

        curve = build_threshold_curve(geom_results)
        optimal, opt_reason = find_optimal_threshold(curve)
        pareto = compute_pareto_frontier(curve)

        geom_metrics = {
            "geometry": GEOMETRIES[geom_key]["label"],
            "threshold_curve": curve,
            "pareto_frontier": pareto,
        }

        if optimal:
            geom_metrics["optimal_threshold"] = optimal["threshold"]
            geom_metrics["optimal_realized_expectancy"] = optimal["realized_exp_mean"]
            geom_metrics["optimal_trade_rate"] = optimal["trade_rate_mean"]
            geom_metrics["optimal_hold_fraction"] = optimal["hold_fraction_mean"]
            geom_metrics["optimal_dir_bar_exp"] = optimal["dir_bar_exp_mean"]
            geom_metrics["optimal_breakeven_rt"] = optimal["breakeven_rt_mean"]
            geom_metrics["optimal_daily_pnl"] = optimal["daily_pnl_mean"]
            geom_metrics["optimal_realized_exp_cv_pct"] = optimal["realized_exp_cv_pct"]
            geom_metrics["optimal_per_fold"] = optimal["realized_exp_per_fold"]
            geom_metrics["optimal_n_trades_per_fold"] = optimal["n_trades_per_fold"]

            # Cost sensitivity at optimal threshold
            t_star = optimal["threshold"]
            t_star_results = [r for r in geom_results if r["threshold"] == t_star]
            cost_sens = {}
            for scenario in ["optimistic", "base", "pessimistic"]:
                key = f"realized_exp_{scenario}"
                vals = [r[key] for r in t_star_results]
                cost_sens[scenario] = {
                    "rt_cost": COST_SCENARIOS[scenario],
                    "mean": float(np.mean(vals)),
                    "per_fold": vals,
                }
            geom_metrics["cost_sensitivity_at_optimal"] = cost_sens
        else:
            geom_metrics["optimal_threshold"] = None
            geom_metrics["optimal_reason"] = opt_reason

        metrics[geom_key] = geom_metrics

    # Primary metrics (from 19:7)
    m19 = metrics.get("19_7", {})

    # Sanity checks
    sanity = {}

    # SC-S1: T=0.50 reproduces baseline
    curve_19 = m19.get("threshold_curve", [])
    baseline_pt = next((pt for pt in curve_19 if pt["threshold"] == 0.50), None)
    if baseline_pt:
        bl_exp = baseline_pt["realized_exp_mean"]
        bl_diff = abs(bl_exp - BASELINE_19_7["realized_wf_expectancy"])
        sanity["SC-S1"] = {
            "description": "T=0.50 reproduces baseline realized exp $0.90 +/- $0.01",
            "pass": bl_diff < 0.01,
            "value": bl_exp,
            "reference": BASELINE_19_7["realized_wf_expectancy"],
            "diff": bl_diff,
        }

        # Also check trade rate
        bl_tr = baseline_pt["trade_rate_mean"]
        bl_tr_diff = abs(bl_tr - BASELINE_19_7["trade_rate"])
        sanity["SC-S1b"] = {
            "description": "T=0.50 reproduces baseline trade rate 85.18%",
            "pass": bl_tr_diff < 0.001,
            "value": bl_tr,
            "reference": BASELINE_19_7["trade_rate"],
            "diff": bl_tr_diff,
        }

    # SC-S2: Trade rate monotonically decreases
    trade_rates = [pt["trade_rate_mean"] for pt in curve_19]
    tr_mono = all(trade_rates[i] >= trade_rates[i + 1]
                  for i in range(len(trade_rates) - 1)) if len(trade_rates) > 1 else False
    sanity["SC-S2"] = {
        "description": "Trade rate monotonically decreases with threshold",
        "pass": tr_mono,
        "values": trade_rates,
    }

    # SC-S3: Hold fraction monotonically decreases
    hold_fracs = [pt["hold_fraction_mean"] for pt in curve_19]
    hf_mono = all(hold_fracs[i] >= hold_fracs[i + 1]
                  for i in range(len(hold_fracs) - 1)) if len(hold_fracs) > 1 else False
    sanity["SC-S3"] = {
        "description": "Hold fraction monotonically decreases with threshold",
        "pass": hf_mono,
        "values": hold_fracs,
    }

    # SC-S4: >= 1000 trades per fold at optimal
    optimal_trades = m19.get("optimal_n_trades_per_fold", [])
    min_trades = min(optimal_trades) if optimal_trades else 0
    sanity["SC-S4"] = {
        "description": "At least 1000 trades per fold at optimal threshold",
        "pass": min_trades >= 1000,
        "min_trades": min_trades,
        "per_fold": optimal_trades,
    }

    metrics["sanity_checks"] = sanity

    # Success Criteria
    sc = {}
    opt_exp = m19.get("optimal_realized_expectancy", -999)
    opt_tr = m19.get("optimal_trade_rate", 0)
    opt_cv = m19.get("optimal_realized_exp_cv_pct", 999)
    opt_hf = m19.get("optimal_hold_fraction", 1.0)
    opt_dir = m19.get("optimal_dir_bar_exp", 0)

    sc["SC-1"] = {
        "description": "Realized WF exp > $1.50 at 19:7 (base) with trade rate > 15%",
        "pass": opt_exp > 1.50 and opt_tr > 0.15,
        "value": opt_exp,
        "trade_rate": opt_tr,
    }
    sc["SC-2"] = {
        "description": "Per-fold CV < 80% at optimal threshold",
        "pass": opt_cv < 80,
        "value": opt_cv,
    }
    sc["SC-3"] = {
        "description": "Hold fraction at optimal < 25%",
        "pass": opt_hf < 0.25,
        "value": opt_hf,
    }
    sc["SC-4"] = {
        "description": "Dir-bar PnL at optimal > $3.00/trade",
        "pass": opt_dir > 3.00,
        "value": opt_dir,
    }
    metrics["success_criteria"] = sc

    # Determine outcome
    if not sanity.get("SC-S1", {}).get("pass", False) or not tr_mono:
        outcome = "D"
        outcome_desc = "INVALID — baseline doesn't reproduce or monotonicity fails."
    elif sc["SC-1"]["pass"] and sc["SC-2"]["pass"]:
        outcome = "A"
        outcome_desc = "CONFIRMED — threshold optimization produces a robust, economically viable strategy."
    elif sc["SC-1"]["pass"] and not sc["SC-2"]["pass"]:
        outcome = "B"
        outcome_desc = "PARTIAL — exp > $1.50 but per-fold CV > 80%. Better economics but still fold-unstable."
    else:
        outcome = "C"
        outcome_desc = "REFUTED — no threshold achieves exp > $1.50 at >15% trade rate."

    metrics["outcome"] = outcome
    metrics["outcome_description"] = outcome_desc

    # Comparison table: baseline vs optimal
    if baseline_pt and m19.get("optimal_threshold") is not None:
        metrics["baseline_vs_optimal"] = {
            "baseline": {
                "threshold": 0.50,
                "realized_exp": baseline_pt["realized_exp_mean"],
                "trade_rate": baseline_pt["trade_rate_mean"],
                "hold_fraction": baseline_pt["hold_fraction_mean"],
                "per_fold_cv": baseline_pt["realized_exp_cv_pct"],
                "breakeven_rt": baseline_pt["breakeven_rt_mean"],
                "daily_pnl": baseline_pt["daily_pnl_mean"],
                "dir_bar_exp": baseline_pt["dir_bar_exp_mean"],
            },
            "optimal": {
                "threshold": m19["optimal_threshold"],
                "realized_exp": m19["optimal_realized_expectancy"],
                "trade_rate": m19["optimal_trade_rate"],
                "hold_fraction": m19["optimal_hold_fraction"],
                "per_fold_cv": m19["optimal_realized_exp_cv_pct"],
                "breakeven_rt": m19["optimal_breakeven_rt"],
                "daily_pnl": m19["optimal_daily_pnl"],
                "dir_bar_exp": m19["optimal_dir_bar_exp"],
            },
        }

    # Dir-bar quality vs threshold
    dir_bar_vs_threshold = []
    for pt in curve_19:
        dir_bar_vs_threshold.append({
            "threshold": pt["threshold"],
            "dir_bar_exp": pt["dir_bar_exp_mean"],
        })
    metrics["dir_bar_quality_vs_threshold"] = dir_bar_vs_threshold

    # MVE
    metrics["mve_gates"] = mve_results.get("gates", {})

    # Resource usage
    metrics["resource_usage"] = {
        "wall_clock_seconds": elapsed_total,
        "wall_clock_minutes": elapsed_total / 60,
        "total_training_runs": 12,  # 3 folds x 2 geometries x 2 models (S1+S2) = 12 fits
        "gpu_hours": 0,
        "total_runs": len(THRESHOLDS) * len(WF_FOLDS) * 2,  # threshold x folds x geometries
    }

    metrics["abort_triggered"] = abort_triggered
    metrics["abort_reason"] = abort_reason
    metrics["notes"] = (
        "Executed locally on Apple Silicon. Models trained ONCE per fold; "
        "threshold sweep is pure numpy (no retraining). "
        "Realized return PnL model identical to pnl-realized-return (PR #35)."
    )

    return metrics


# ==========================================================================
# Write Outputs
# ==========================================================================
def write_metrics(metrics):
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {RESULTS_DIR / 'metrics.json'}")


def write_config():
    config = {
        "seed": SEED,
        "horizon_bars": HORIZON_BARS,
        "tick_value": TICK_VALUE,
        "tick_size": TICK_SIZE,
        "geometries": {k: {"target": v["target"], "stop": v["stop"]}
                       for k, v in GEOMETRIES.items()},
        "xgb_params": TUNED_XGB_PARAMS_BINARY,
        "features": NON_SPATIAL_FEATURES,
        "cost_scenarios": COST_SCENARIOS,
        "walk_forward_folds": WF_FOLDS,
        "purge_bars": PURGE_BARS,
        "thresholds": THRESHOLDS,
        "data_source": str(DATA_BASE),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def write_threshold_curve_csv(all_results):
    """Write threshold curve CSV: 9 thresholds x 2 geometries."""
    rows = []
    for geom_key in ["19_7", "10_5"]:
        geom_results = all_results.get(geom_key, [])
        curve = build_threshold_curve(geom_results)
        for pt in curve:
            rows.append({
                "geometry": GEOMETRIES[geom_key]["label"],
                "threshold": pt["threshold"],
                "trade_rate_mean": pt["trade_rate_mean"],
                "hold_fraction_mean": pt["hold_fraction_mean"],
                "realized_exp_mean": pt["realized_exp_mean"],
                "realized_exp_std": pt["realized_exp_std"],
                "realized_exp_cv_pct": pt["realized_exp_cv_pct"],
                "dir_bar_exp_mean": pt["dir_bar_exp_mean"],
                "hold_bar_exp_mean": pt["hold_bar_exp_mean"],
                "breakeven_rt_mean": pt["breakeven_rt_mean"],
                "n_trades_mean": pt["n_trades_mean"],
                "daily_pnl_mean": pt["daily_pnl_mean"],
                "realized_fold1": pt["realized_exp_per_fold"][0] if len(pt["realized_exp_per_fold"]) > 0 else None,
                "realized_fold2": pt["realized_exp_per_fold"][1] if len(pt["realized_exp_per_fold"]) > 1 else None,
                "realized_fold3": pt["realized_exp_per_fold"][2] if len(pt["realized_exp_per_fold"]) > 2 else None,
            })
    if rows:
        with open(RESULTS_DIR / "threshold_curve.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Threshold curve written to {RESULTS_DIR / 'threshold_curve.csv'}")


def write_per_fold_at_optimal_csv(all_results, metrics):
    """Write per-fold details at T*."""
    rows = []
    for geom_key in ["19_7", "10_5"]:
        m = metrics.get(geom_key, {})
        t_star = m.get("optimal_threshold")
        if t_star is None:
            continue

        geom_results = all_results.get(geom_key, [])
        t_star_results = [r for r in geom_results if r["threshold"] == t_star]
        for r in t_star_results:
            rows.append({
                "geometry": GEOMETRIES[geom_key]["label"],
                "threshold": t_star,
                "fold": r["fold_name"],
                "n_test": r["n_test"],
                "n_trades": r["n_trades"],
                "trade_rate": r["trade_rate"],
                "hold_fraction": r["hold_fraction"],
                "n_dir_trades": r["n_dir_trades"],
                "n_hold_trades": r["n_hold_trades"],
                "realized_exp_base": r["realized_exp_base"],
                "realized_exp_optimistic": r["realized_exp_optimistic"],
                "realized_exp_pessimistic": r["realized_exp_pessimistic"],
                "dir_bar_exp": r["dir_bar_exp"],
                "hold_bar_exp": r["hold_bar_exp"],
                "directional_accuracy": r["directional_accuracy"],
                "breakeven_rt": r["breakeven_rt"],
                "daily_pnl": r["daily_pnl"],
            })
    if rows:
        with open(RESULTS_DIR / "per_fold_at_optimal.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Per-fold at optimal written to {RESULTS_DIR / 'per_fold_at_optimal.csv'}")


def write_cost_sensitivity_csv(all_results, metrics):
    """Write 3 cost scenarios at T*."""
    rows = []
    for geom_key in ["19_7", "10_5"]:
        m = metrics.get(geom_key, {})
        t_star = m.get("optimal_threshold")
        if t_star is None:
            continue

        cost_sens = m.get("cost_sensitivity_at_optimal", {})
        for scenario in ["optimistic", "base", "pessimistic"]:
            cs = cost_sens.get(scenario, {})
            rows.append({
                "geometry": GEOMETRIES[geom_key]["label"],
                "threshold": t_star,
                "scenario": scenario,
                "rt_cost": cs.get("rt_cost"),
                "mean_exp": cs.get("mean"),
                "fold1": cs.get("per_fold", [None])[0] if cs.get("per_fold") else None,
                "fold2": cs.get("per_fold", [None, None])[1] if cs.get("per_fold") and len(cs.get("per_fold", [])) > 1 else None,
                "fold3": cs.get("per_fold", [None, None, None])[2] if cs.get("per_fold") and len(cs.get("per_fold", [])) > 2 else None,
            })
    if rows:
        with open(RESULTS_DIR / "cost_sensitivity.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"Cost sensitivity written to {RESULTS_DIR / 'cost_sensitivity.csv'}")


def write_analysis(metrics, all_results):
    """Write analysis.md with all required sections."""
    m19 = metrics.get("19_7", {})
    m10 = metrics.get("10_5", {})
    sc = metrics.get("success_criteria", {})
    sanity = metrics.get("sanity_checks", {})
    p_dist = metrics.get("p_directional_distribution", {})

    lines = []
    lines.append("# Stage 1 Threshold Sweep — Analysis\n")

    # 1. Executive summary
    outcome = metrics.get("outcome", "?")
    outcome_desc = metrics.get("outcome_description", "")
    opt_t = m19.get("optimal_threshold", "N/A")
    opt_exp = m19.get("optimal_realized_expectancy", 0)
    opt_tr = m19.get("optimal_trade_rate", 0)

    lines.append("## Executive Summary\n")
    lines.append(f"**Outcome {outcome}:** {outcome_desc}\n")
    if opt_t is not None:
        lines.append(f"Optimal threshold T*={opt_t:.2f} at 19:7: realized exp **${opt_exp:.4f}**/trade, "
                      f"trade rate {opt_tr:.1%}.\n")
    lines.append("")

    # 2. P(directional) distribution — report FIRST per spec
    lines.append("## P(directional) Distribution (19:7)\n")
    if p_dist:
        lines.append(f"- N: {p_dist.get('n_total', 0):,}")
        lines.append(f"- Mean: {p_dist.get('mean', 0):.4f}, Std: {p_dist.get('std', 0):.4f}")
        lines.append(f"- Min: {p_dist.get('min', 0):.4f}, Max: {p_dist.get('max', 0):.4f}")
        pcts = p_dist.get("percentiles", {})
        lines.append(f"- Percentiles: p5={pcts.get('p5', 0):.4f}, p10={pcts.get('p10', 0):.4f}, "
                      f"p25={pcts.get('p25', 0):.4f}, p50={pcts.get('p50', 0):.4f}, "
                      f"p75={pcts.get('p75', 0):.4f}, p90={pcts.get('p90', 0):.4f}, "
                      f"p95={pcts.get('p95', 0):.4f}")
        lines.append(f"- Fraction within [0.45, 0.55]: **{p_dist.get('fraction_within_045_055', 0):.1%}**")
        frac_above = p_dist.get("fraction_above_threshold", {})
        lines.append(f"- Fraction above each threshold:")
        for t in THRESHOLDS:
            lines.append(f"  - T={t:.2f}: {frac_above.get(f'{t:.2f}', 0):.1%}")
        lines.append("")

        lines.append("### Histogram\n")
        lines.append("| Bin | Count | Fraction |")
        lines.append("|-----|-------|----------|")
        for h in p_dist.get("histogram", []):
            lines.append(f"| [{h['bin_low']:.2f}, {h['bin_high']:.2f}) | {h['count']:,} | {h['fraction']:.1%} |")
        lines.append("")

    # 3. Threshold curve table (19:7)
    lines.append("## Threshold Curve (19:7)\n")
    curve_19 = m19.get("threshold_curve", [])
    lines.append("| Threshold | Trade Rate | Hold Frac | Realized Exp | Dir-Bar Exp | Hold-Bar Exp | CV% | B/E RT | Daily PnL |")
    lines.append("|-----------|-----------|-----------|-------------|-------------|-------------|-----|--------|-----------|")
    for pt in curve_19:
        lines.append(f"| {pt['threshold']:.2f} | {pt['trade_rate_mean']:.1%} | "
                      f"{pt['hold_fraction_mean']:.1%} | ${pt['realized_exp_mean']:.4f} | "
                      f"${pt['dir_bar_exp_mean']:.4f} | ${pt['hold_bar_exp_mean']:.4f} | "
                      f"{pt['realized_exp_cv_pct']:.0f}% | ${pt['breakeven_rt_mean']:.2f} | "
                      f"${pt['daily_pnl_mean']:.2f} |")
    lines.append("")

    # 4. 10:5 control curve
    lines.append("## 10:5 Control Curve\n")
    curve_10 = m10.get("threshold_curve", [])
    lines.append("| Threshold | Trade Rate | Hold Frac | Realized Exp | Dir-Bar Exp |")
    lines.append("|-----------|-----------|-----------|-------------|-------------|")
    for pt in curve_10:
        lines.append(f"| {pt['threshold']:.2f} | {pt['trade_rate_mean']:.1%} | "
                      f"{pt['hold_fraction_mean']:.1%} | ${pt['realized_exp_mean']:.4f} | "
                      f"${pt['dir_bar_exp_mean']:.4f} |")
    lines.append("")

    # 5. Optimal threshold selection
    lines.append("## Optimal Threshold Selection (19:7)\n")
    bvo = metrics.get("baseline_vs_optimal", {})
    bl = bvo.get("baseline", {})
    opt = bvo.get("optimal", {})
    if bl and opt:
        lines.append("| Metric | Baseline (T=0.50) | Optimal (T=T*) | Delta |")
        lines.append("|--------|-------------------|----------------|-------|")
        for field, label, fmt in [
            ("realized_exp", "Realized exp", "${:.4f}"),
            ("trade_rate", "Trade rate", "{:.1%}"),
            ("hold_fraction", "Hold fraction", "{:.1%}"),
            ("per_fold_cv", "Per-fold CV", "{:.0f}%"),
            ("breakeven_rt", "Break-even RT", "${:.2f}"),
            ("daily_pnl", "Daily PnL", "${:.2f}"),
            ("dir_bar_exp", "Dir-bar exp", "${:.4f}"),
        ]:
            bv = bl.get(field, 0)
            ov = opt.get(field, 0)
            delta = ov - bv
            bv_str = fmt.format(bv)
            ov_str = fmt.format(ov)
            if "%" in fmt and "$" not in fmt:
                delta_str = f"{delta:+.1%}"
            elif "$" in fmt:
                delta_str = f"${delta:+.4f}" if ".4f" in fmt else f"${delta:+.2f}"
            else:
                delta_str = f"{delta:+.0f}%"
            lines.append(f"| {label} | {bv_str} | {ov_str} | {delta_str} |")
    lines.append("")

    # 6. Per-fold consistency at T*
    lines.append("## Per-Fold Consistency at T*\n")
    opt_per_fold = m19.get("optimal_per_fold", [])
    opt_n_trades = m19.get("optimal_n_trades_per_fold", [])
    if opt_per_fold:
        lines.append("| Fold | Realized Exp | Trades |")
        lines.append("|------|-------------|--------|")
        for i, (exp, nt) in enumerate(zip(opt_per_fold, opt_n_trades)):
            lines.append(f"| Fold {i+1} | ${exp:.4f} | {nt:,} |")
        lines.append(f"\n**CV at T*: {m19.get('optimal_realized_exp_cv_pct', 0):.0f}%** "
                      f"(baseline: {bl.get('per_fold_cv', 0):.0f}%)")
    lines.append("")

    # 7. Dir-bar quality check
    lines.append("## Directional-Bar Quality vs Threshold\n")
    dbq = metrics.get("dir_bar_quality_vs_threshold", [])
    lines.append("| Threshold | Dir-Bar Exp |")
    lines.append("|-----------|-------------|")
    for pt in dbq:
        lines.append(f"| {pt['threshold']:.2f} | ${pt['dir_bar_exp']:.4f} |")
    lines.append("")

    # 8. Pareto frontier
    lines.append("## Pareto Frontier (Expectancy x Trade Rate)\n")
    pareto = m19.get("pareto_frontier", [])
    lines.append(f"Thresholds on Pareto frontier: {pareto}")
    lines.append("")

    # 9. Cost sensitivity at T*
    lines.append("## Cost Sensitivity at T*\n")
    cost_sens = m19.get("cost_sensitivity_at_optimal", {})
    if cost_sens:
        lines.append("| Scenario | RT Cost | Mean Exp | Fold 1 | Fold 2 | Fold 3 |")
        lines.append("|----------|---------|----------|--------|--------|--------|")
        for scenario in ["optimistic", "base", "pessimistic"]:
            cs = cost_sens.get(scenario, {})
            pf = cs.get("per_fold", [0, 0, 0])
            lines.append(f"| {scenario} | ${cs.get('rt_cost', 0):.2f} | "
                          f"${cs.get('mean', 0):.4f} | "
                          f"${pf[0]:.4f} | ${pf[1]:.4f} | ${pf[2]:.4f} |")
    lines.append("")

    # 10. Already covered in #5 (comparison table)

    # 11. SC-1 through SC-4
    lines.append("## Success Criteria\n")
    for k, v in sc.items():
        status = "PASS" if v.get("pass") else "FAIL"
        lines.append(f"- **{k}**: {status} — {v['description']} (value: {v.get('value')})")
    lines.append("")

    # Sanity checks
    lines.append("## Sanity Checks\n")
    for k, v in sanity.items():
        status = "PASS" if v.get("pass") else "FAIL"
        lines.append(f"- **{k}**: {status} — {v['description']}")
    lines.append("")

    # 12. Outcome verdict
    lines.append(f"## Verdict: OUTCOME {outcome}\n")
    lines.append(outcome_desc)

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Analysis written to {RESULTS_DIR / 'analysis.md'}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    t0_global = time.time()
    set_seed(SEED)
    print("=" * 70)
    print("Stage 1 Threshold Sweep — Optimizing Hold-Bar Exposure")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Load data (both geometries)
    # ------------------------------------------------------------------
    print("\n[Step 0] Loading data...")
    data = {}
    for geom_key in ["19_7", "10_5"]:
        print(f"\n  Geometry {geom_key}:")
        features, labels, day_indices, unique_days, extra = load_geometry_data(geom_key)
        data[geom_key] = {
            "features": features,
            "labels": labels,
            "day_indices": day_indices,
            "unique_days": unique_days,
            "extra": extra,
        }

    # ------------------------------------------------------------------
    # Step 1: MVE (3 thresholds, Fold 1 only, 19:7)
    # ------------------------------------------------------------------
    print("\n[Step 1] MVE...")
    mve_results = run_mve(
        data["19_7"]["features"],
        data["19_7"]["labels"],
        data["19_7"]["day_indices"],
        data["19_7"]["extra"],
    )

    if mve_results.get("abort", False):
        print("\n** EXPERIMENT ABORTED AT MVE **")
        abort_metrics = {
            "experiment": "threshold-sweep",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": mve_results.get("abort_reason", "MVE failed"),
            "mve_gates": mve_results.get("gates", {}),
            "resource_usage": {
                "wall_clock_seconds": time.time() - t0_global,
                "gpu_hours": 0,
                "total_runs": 2,
            },
        }
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(abort_metrics, f, indent=2)
        return

    # ------------------------------------------------------------------
    # Step 2: Full protocol
    # ------------------------------------------------------------------
    print("\n[Step 2] Full protocol...")
    all_results, all_fold_data, aborted, abort_reason = run_full_protocol(data, t0_global)

    # ------------------------------------------------------------------
    # Step 3: Build metrics and write outputs
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t0_global
    metrics = build_metrics(all_results, all_fold_data, mve_results,
                            t0_global, elapsed_total, aborted, abort_reason)

    write_metrics(metrics)
    write_analysis(metrics, all_results)
    write_threshold_curve_csv(all_results)
    write_per_fold_at_optimal_csv(all_results, metrics)
    write_cost_sensitivity_csv(all_results, metrics)
    write_config()

    print(f"\n{'=' * 70}")
    print(f"COMPLETE — {elapsed_total:.1f}s wall clock")
    print(f"Outcome: {metrics.get('outcome', '?')} — {metrics.get('outcome_description', '')}")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
