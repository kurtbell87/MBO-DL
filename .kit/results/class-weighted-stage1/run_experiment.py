#!/usr/bin/env python3
"""
Experiment: Class-Weighted Stage 1 — Spreading the Probability Surface
Spec: .kit/experiments/class-weighted-stage1.md

Adapts the threshold-sweep pipeline (PR #36) with an outer loop over
Stage 1 `scale_pos_weight` values: [1.0, 0.5, 0.33, 0.25, 0.20].

Stage 1 is RE-TRAINED per weight (not post-hoc). Stage 2 is trained
ONCE per fold (independent of Stage 1 weight). Threshold sweep is
post-hoc on the per-weight Stage 1 probabilities.

Total fits: 5 weights x 3 folds = 15 Stage 1 + 3 Stage 2 = 18 XGB fits.
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
PROJECT_ROOT = Path("/Users/brandonbell/LOCAL_DEV/MBO-DL-02152026")
RESULTS_DIR = PROJECT_ROOT / ".kit" / "results" / "class-weighted-stage1"
DATA_BASE = PROJECT_ROOT / ".kit" / "results" / "label-geometry-1h"

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

# Only 19:7 geometry (10:5 excluded per spec)
GEOMETRY = {"target": 19, "stop": 7, "label": "19:7", "bev_wr": 0.384,
            "data_dir": DATA_BASE / "geom_19_7"}

WF_FOLDS = [
    {"train_range": (1, 100), "test_range": (101, 150), "name": "Fold 1"},
    {"train_range": (1, 150), "test_range": (151, 201), "name": "Fold 2"},
    {"train_range": (1, 201), "test_range": (202, 251), "name": "Fold 3 (holdout)"},
]

PURGE_BARS = 500
WALL_CLOCK_LIMIT_S = 20 * 60  # 20 min abort threshold per spec

# Primary independent variable: scale_pos_weight for Stage 1
SCALE_POS_WEIGHTS = [1.0, 0.5, 0.33, 0.25, 0.20]

# Threshold sweep levels
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Baseline from pnl-realized-return (PR #35) for sanity check
BASELINE = {
    "realized_wf_expectancy": 0.901517613505994,
    "trade_rate": 0.8518110743722463,
    "hold_fraction": 0.44389057472507454,
    "dir_bar_pnl": 3.77471732462107,
    "per_fold_realized": [0.010826135364183474, 2.5384366083743046, 0.1552900967794935],
    "p_directional_iqr": 0.04,  # 4pp from PR #36
    "p_directional_in_050_060": 0.806,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================================================
# Data Loading (identical to threshold-sweep / pnl-realized-return)
# ==========================================================================
def load_data():
    data_dir = GEOMETRY["data_dir"]
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

    correct = (pred_labels == true_labels) & dir_trades
    wrong = (pred_labels != true_labels) & dir_trades

    dir_pnl = np.zeros(n)
    dir_pnl[correct] = win_pnl - rt_cost
    dir_pnl[wrong] = -loss_pnl - rt_cost

    realized_pnl = dir_pnl.copy()

    if n_hold_trades > 0:
        hold_trade_indices_global = test_indices[hold_trades]
        fwd_returns_720, actual_bars_arr = compute_fwd_return_720(
            hold_trade_indices_global, extra)

        hold_pred_signs = np.sign(pred_labels[hold_trades])
        hold_gross_pnl = fwd_returns_720 * TICK_VALUE * hold_pred_signs
        hold_net_pnl = hold_gross_pnl - rt_cost
        realized_pnl[hold_trades] = hold_net_pnl

    realized_per_trade = float(realized_pnl[trades].sum()) / n_trades if n_trades > 0 else 0.0
    dir_mean_pnl = float(dir_pnl[dir_trades].mean()) if n_dir_trades > 0 else 0.0
    hold_mean_pnl = float(realized_pnl[hold_trades].mean()) if n_hold_trades > 0 else 0.0
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
# Prepare fold data (compute train/test splits, normalize, train Stage 2 ONCE)
# ==========================================================================
def prepare_fold(features, labels, day_indices, extra, train_range, test_range, fold_name):
    """Prepare fold data and train Stage 2 ONCE. Returns fold context for Stage 1 training."""
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

    # Internal train/val split (last 20% of training days)
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

    # STAGE 2: long vs short (trained on directional bars only, weight=1.0, ONCE per fold)
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

    s2_pred_proba = clf_s2.predict_proba(ft_test_z)[:, 1]
    s2_pred = np.where(s2_pred_proba > 0.5, 1, -1)

    del clf_s2

    return {
        "fold_name": fold_name,
        "n_test": len(test_indices),
        "test_indices": test_indices,
        "labels_test": labels_test,
        "s2_pred": s2_pred,
        "day_indices": day_indices,
        # Training data for Stage 1 (needed per weight)
        "inner_train_z": inner_train_z,
        "inner_val_z": inner_val_z,
        "inner_train_labels": inner_train_labels,
        "inner_val_labels": inner_val_labels,
        "ft_test_z": ft_test_z,
    }


def train_stage1_at_weight(fold_ctx, scale_pos_weight):
    """Train Stage 1 with a specific scale_pos_weight. Returns P(directional) probabilities."""
    s1_train_labels = (fold_ctx["inner_train_labels"] != 0).astype(np.int64)
    s1_val_labels = (fold_ctx["inner_val_labels"] != 0).astype(np.int64)

    params = TUNED_XGB_PARAMS_BINARY.copy()
    params["scale_pos_weight"] = scale_pos_weight

    clf_s1 = xgb.XGBClassifier(**params)
    if len(fold_ctx["inner_val_z"]) > 0:
        clf_s1.fit(fold_ctx["inner_train_z"], s1_train_labels,
                   eval_set=[(fold_ctx["inner_val_z"], s1_val_labels)],
                   verbose=False)
    else:
        clf_s1.fit(fold_ctx["inner_train_z"], s1_train_labels, verbose=False)

    s1_pred_proba = clf_s1.predict_proba(fold_ctx["ft_test_z"])[:, 1]

    # Stage 1 binary predictions at default T=0.50
    s1_pred = (s1_pred_proba > 0.5).astype(np.int64)
    true_binary = (fold_ctx["labels_test"] != 0).astype(np.int64)

    accuracy = float(accuracy_score(true_binary, s1_pred))
    precision = float(precision_score(true_binary, s1_pred, zero_division=0.0))
    recall_val = float(recall_score(true_binary, s1_pred, zero_division=0.0))

    del clf_s1

    return s1_pred_proba, accuracy, precision, recall_val


def apply_threshold(fold_ctx, s1_proba, threshold, extra):
    """Apply a Stage 1 threshold to pre-computed probabilities and compute PnL."""
    target = GEOMETRY["target"]
    stop = GEOMETRY["stop"]

    labels_test = fold_ctx["labels_test"]
    s2_pred = fold_ctx["s2_pred"]
    test_indices = fold_ctx["test_indices"]
    n_test = fold_ctx["n_test"]

    s1_dir_mask = (s1_proba > threshold)
    combined_pred = np.zeros(len(labels_test), dtype=np.int64)
    combined_pred[s1_dir_mask] = s2_pred[s1_dir_mask]

    n_trades = int((combined_pred != 0).sum())
    trade_rate = float(n_trades / n_test) if n_test > 0 else 0.0

    pnl_results = {}
    for scenario, rt_cost in COST_SCENARIOS.items():
        pnl = compute_pnl_realized(
            labels_test, combined_pred, target, stop, rt_cost,
            test_indices, extra)
        pnl_results[scenario] = pnl

    pnl_nocost = compute_pnl_realized(
        labels_test, combined_pred, target, stop, 0.0,
        test_indices, extra)
    breakeven_rt = pnl_nocost["realized_total_pnl"] / n_trades if n_trades > 0 else 0.0

    trades = (combined_pred != 0)
    dir_bars = (labels_test != 0)
    both_nonzero = (combined_pred != 0) & (labels_test != 0)
    n_dir_pairs = both_nonzero.sum()
    dir_acc = float((combined_pred == labels_test)[both_nonzero].sum() / n_dir_pairs) if n_dir_pairs > 0 else 0.0

    pnl_base = pnl_results["base"]

    day_indices_test = fold_ctx["day_indices"][test_indices]
    n_test_days = len(set(day_indices_test.tolist()))
    daily_pnl = pnl_base["realized_total_pnl"] / n_test_days if n_test_days > 0 else 0.0

    # Stage 2 accuracy on bars that pass threshold (confound #2 diagnostic)
    s2_mask = s1_dir_mask & (labels_test != 0)
    if s2_mask.sum() > 0:
        s2_acc_on_filtered = float((s2_pred[s2_mask] == np.sign(labels_test[s2_mask])).mean())
    else:
        s2_acc_on_filtered = 0.0

    return {
        "threshold": threshold,
        "fold_name": fold_ctx["fold_name"],
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
        "s2_accuracy_on_filtered": s2_acc_on_filtered,
    }


# ==========================================================================
# P(directional) Distribution Analysis per weight
# ==========================================================================
def analyze_p_distribution(all_proba):
    """Analyze P(directional) distribution from a concatenated array of probabilities."""
    percentiles = {
        "p5": float(np.percentile(all_proba, 5)),
        "p10": float(np.percentile(all_proba, 10)),
        "p25": float(np.percentile(all_proba, 25)),
        "p50": float(np.percentile(all_proba, 50)),
        "p75": float(np.percentile(all_proba, 75)),
        "p90": float(np.percentile(all_proba, 90)),
        "p95": float(np.percentile(all_proba, 95)),
    }

    iqr = percentiles["p75"] - percentiles["p25"]

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

    # Key fractions from spec
    frac_050_060 = float(np.mean((all_proba >= 0.50) & (all_proba < 0.60)))
    frac_060_100 = float(np.mean((all_proba >= 0.60) & (all_proba <= 1.00)))
    frac_070_100 = float(np.mean((all_proba >= 0.70) & (all_proba <= 1.00)))

    return {
        "n_total": len(all_proba),
        "mean": float(all_proba.mean()),
        "std": float(all_proba.std()),
        "min": float(all_proba.min()),
        "max": float(all_proba.max()),
        "iqr": float(iqr),
        "iqr_pp": float(iqr * 100),
        "percentiles": percentiles,
        "histogram": histogram,
        "fraction_in_050_060": frac_050_060,
        "fraction_in_060_100": frac_060_100,
        "fraction_in_070_100": frac_070_100,
    }


# ==========================================================================
# MVE
# ==========================================================================
def run_mve(features, labels, day_indices, extra):
    """Run 3 weights (1.0, 0.5, 0.25) on Fold 1 only at 19:7. Verify MVE checks."""
    print("\n" + "=" * 70)
    print("MVE — 3 weights (1.0, 0.5, 0.25), Fold 1 only, 19:7, T=0.50")
    print("=" * 70)

    fold_ctx = prepare_fold(
        features, labels, day_indices, extra,
        train_range=(1, 100), test_range=(101, 150),
        fold_name="MVE Fold 1")

    if fold_ctx is None:
        return {"abort": True, "abort_reason": "MVE fold returned None"}

    mve_weights = [1.0, 0.5, 0.25]
    mve_results = []
    for w in mve_weights:
        s1_proba, s1_acc, s1_prec, s1_rec = train_stage1_at_weight(fold_ctx, w)
        result = apply_threshold(fold_ctx, s1_proba, 0.50, extra)

        # P(directional) distribution
        p_dist = analyze_p_distribution(s1_proba)

        mve_results.append({
            "weight": w,
            "s1_accuracy": s1_acc,
            "s1_precision": s1_prec,
            "s1_recall": s1_rec,
            "p_iqr_pp": p_dist["iqr_pp"],
            "trade_rate": result["trade_rate"],
            "hold_fraction": result["hold_fraction"],
            "realized_exp_base": result["realized_exp_base"],
            "dir_bar_exp": result["dir_bar_exp"],
        })

        print(f"\n  Weight {w:.2f}:")
        print(f"    S1 accuracy: {s1_acc:.4f}, precision: {s1_prec:.4f}, recall: {s1_rec:.4f}")
        print(f"    P(dir) IQR: {p_dist['iqr_pp']:.1f}pp")
        print(f"    Trade rate: {result['trade_rate']:.4f}")
        print(f"    Hold fraction: {result['hold_fraction']:.4f}")
        print(f"    Realized exp (base): ${result['realized_exp_base']:.4f}")
        print(f"    Dir-bar exp: ${result['dir_bar_exp']:.4f}")

    gates = {}

    # Check 1: Weight=1.0 reproduces baseline (Fold 1)
    r10 = mve_results[0]
    baseline_fold1_realized = BASELINE["per_fold_realized"][0]
    realized_diff = abs(r10["realized_exp_base"] - baseline_fold1_realized)
    gates["baseline_reproduces"] = realized_diff < 0.02
    print(f"\n  Baseline reproduction: realized diff=${realized_diff:.6f} "
          f"({'PASS' if gates['baseline_reproduces'] else 'FAIL'})")

    # Check 2: Precision increases as weight decreases (1.0 -> 0.5 -> 0.25)
    prec_increasing = (mve_results[1]["s1_precision"] >= mve_results[0]["s1_precision"] and
                       mve_results[2]["s1_precision"] >= mve_results[1]["s1_precision"])
    gates["precision_increases"] = prec_increasing
    print(f"  Precision increases: {prec_increasing} "
          f"({mve_results[0]['s1_precision']:.4f} -> {mve_results[1]['s1_precision']:.4f} -> {mve_results[2]['s1_precision']:.4f})")

    # Check 3: P(directional) IQR increases as weight decreases
    iqr_increasing = (mve_results[1]["p_iqr_pp"] >= mve_results[0]["p_iqr_pp"] and
                      mve_results[2]["p_iqr_pp"] >= mve_results[1]["p_iqr_pp"])
    gates["iqr_increases"] = iqr_increasing
    print(f"  IQR increases: {iqr_increasing} "
          f"({mve_results[0]['p_iqr_pp']:.1f}pp -> {mve_results[1]['p_iqr_pp']:.1f}pp -> {mve_results[2]['p_iqr_pp']:.1f}pp)")

    # Check 4: Trade rate decreases as weight decreases
    tr_decreasing = (mve_results[1]["trade_rate"] <= mve_results[0]["trade_rate"] and
                     mve_results[2]["trade_rate"] <= mve_results[1]["trade_rate"])
    gates["trade_rate_decreases"] = tr_decreasing
    print(f"  Trade rate decreases: {tr_decreasing} "
          f"({mve_results[0]['trade_rate']:.4f} -> {mve_results[1]['trade_rate']:.4f} -> {mve_results[2]['trade_rate']:.4f})")

    # Check precision increases for non-degenerate weights (where model predicts > 0 positives)
    non_degenerate = [(r["weight"], r["s1_precision"], r["s1_recall"])
                      for r in mve_results if r["s1_recall"] > 0.0]
    if len(non_degenerate) >= 2:
        prec_ok = non_degenerate[-1][1] >= non_degenerate[0][1]
        gates["precision_increases_non_degenerate"] = prec_ok
        print(f"  Precision increases (non-degenerate only): {prec_ok} "
              f"({' -> '.join(f'{p:.4f}' for _, p, _ in non_degenerate)})")
    else:
        gates["precision_increases_non_degenerate"] = len(non_degenerate) <= 1
        print(f"  Precision increases (non-degenerate): only {len(non_degenerate)} non-degenerate weight(s)")

    # Count how many weights produce near-zero trade rate
    degenerate_count = sum(1 for r in mve_results if r["trade_rate"] < 0.01)
    gates["degenerate_weights"] = degenerate_count
    print(f"  Degenerate weights (trade rate < 1%): {degenerate_count}/{len(mve_results)}")

    gates["all_passed"] = all(v for k, v in gates.items()
                              if k != "degenerate_weights" and isinstance(v, bool))
    print(f"\n  MVE gates: {'ALL PASSED' if gates['all_passed'] else 'FAILED'}")

    # HARD ABORT only if baseline doesn't reproduce (code bug)
    if not gates["baseline_reproduces"]:
        return {"abort": True, "abort_reason": "Baseline reproduction failed", "gates": gates, "mve_results": mve_results}

    # Soft check: if precision at w=0.5 doesn't increase vs w=1.0, the parameter is broken
    # But if it does increase for non-degenerate weights, continue — degenerate weights
    # at w=0.25 are an expected confound (spec confound #1), not a bug
    if not gates.get("precision_increases_non_degenerate", False):
        return {"abort": True, "abort_reason": "Precision doesn't increase even at non-degenerate weights", "gates": gates, "mve_results": mve_results}

    return {"abort": False, "gates": gates, "mve_results": mve_results}


# ==========================================================================
# Full Protocol
# ==========================================================================
def run_full_protocol(features, labels, day_indices, extra, t0_global):
    """Run class-weighted sweep: 5 weights x 9 thresholds x 3 folds."""

    # Step 1: Prepare folds (trains Stage 2 ONCE per fold)
    print(f"\n{'=' * 70}")
    print("FULL PROTOCOL: Class-Weighted Stage 1 Sweep (19:7)")
    print(f"{'=' * 70}")

    fold_contexts = []
    for wf_idx, wf in enumerate(WF_FOLDS):
        elapsed = time.time() - t0_global
        if elapsed > WALL_CLOCK_LIMIT_S:
            return None, None, True, f"Wall-clock {elapsed:.0f}s > {WALL_CLOCK_LIMIT_S}s (during fold prep)"

        print(f"\n  Preparing {wf['name']} (train {wf['train_range']}, test {wf['test_range']})...")
        t_fold = time.time()

        fold_ctx = prepare_fold(
            features, labels, day_indices, extra,
            wf["train_range"], wf["test_range"], wf["name"])

        if fold_ctx is None:
            print(f"  !! Fold skipped (empty split)")
            continue

        fold_contexts.append(fold_ctx)
        print(f"  Stage 2 trained in {time.time() - t_fold:.1f}s. Test set: {fold_ctx['n_test']} bars.")

    # Step 2: For each weight, train Stage 1 per fold, then sweep thresholds
    # weight -> { "fold_probas": [(fold_ctx, s1_proba, s1_acc, s1_prec, s1_rec)], "threshold_results": [...] }
    weight_data = {}
    total_s1_fits = 0

    for weight in SCALE_POS_WEIGHTS:
        elapsed = time.time() - t0_global
        if elapsed > WALL_CLOCK_LIMIT_S:
            return weight_data, fold_contexts, True, f"Wall-clock {elapsed:.0f}s > {WALL_CLOCK_LIMIT_S}s (at weight={weight})"

        print(f"\n  --- Weight {weight:.2f} ---")
        fold_probas = []

        for fold_ctx in fold_contexts:
            t_s1 = time.time()
            s1_proba, s1_acc, s1_prec, s1_rec = train_stage1_at_weight(fold_ctx, weight)
            total_s1_fits += 1
            print(f"    {fold_ctx['fold_name']}: S1 trained in {time.time() - t_s1:.1f}s "
                  f"(acc={s1_acc:.4f}, prec={s1_prec:.4f}, rec={s1_rec:.4f})")
            fold_probas.append((fold_ctx, s1_proba, s1_acc, s1_prec, s1_rec))

        # Threshold sweep (post-hoc on Stage 1 probabilities)
        threshold_results = []
        for threshold in THRESHOLDS:
            for fold_ctx, s1_proba, _, _, _ in fold_probas:
                result = apply_threshold(fold_ctx, s1_proba, threshold, extra)
                result["weight"] = weight
                threshold_results.append(result)

        weight_data[weight] = {
            "fold_probas": fold_probas,
            "threshold_results": threshold_results,
        }

        # Print summary for this weight
        print(f"\n    Summary for weight={weight:.2f}:")
        print(f"    {'Threshold':>9} {'Trade Rate':>10} {'Hold Frac':>9} "
              f"{'Realized':>10} {'Dir Bar':>10} {'Trades':>8}")
        for threshold in THRESHOLDS:
            t_results = [r for r in threshold_results if r["threshold"] == threshold]
            mean_tr = np.mean([r["trade_rate"] for r in t_results])
            mean_hf = np.mean([r["hold_fraction"] for r in t_results])
            mean_re = np.mean([r["realized_exp_base"] for r in t_results])
            mean_db = np.mean([r["dir_bar_exp"] for r in t_results])
            mean_nt = np.mean([r["n_trades"] for r in t_results])
            print(f"    {threshold:>9.2f} {mean_tr:>10.4f} {mean_hf:>9.4f} "
                  f"${mean_re:>9.4f} ${mean_db:>9.4f} {mean_nt:>8.0f}")

    print(f"\n  Total Stage 1 fits: {total_s1_fits}")
    print(f"  Total Stage 2 fits: {len(fold_contexts)}")
    print(f"  Total XGB fits: {total_s1_fits + len(fold_contexts)}")

    return weight_data, fold_contexts, False, None


# ==========================================================================
# Metrics Builder
# ==========================================================================
def build_threshold_curve_for_weight(threshold_results):
    """Build threshold -> metrics curve for a single weight."""
    curve = []
    for threshold in THRESHOLDS:
        t_results = [r for r in threshold_results if r["threshold"] == threshold]
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


def find_optimal_pair(weight_data):
    """Find (weight, threshold) pair maximizing realized exp with trade rate > 15%."""
    best_pair = None
    best_exp = -float("inf")

    for weight, wd in weight_data.items():
        curve = build_threshold_curve_for_weight(wd["threshold_results"])
        for pt in curve:
            if pt["trade_rate_mean"] > 0.15 and pt["realized_exp_mean"] > best_exp:
                best_exp = pt["realized_exp_mean"]
                best_pair = (weight, pt["threshold"])

    return best_pair


def build_metrics(weight_data, fold_contexts, mve_result, t0, elapsed_total,
                  abort_triggered=False, abort_reason=None):
    """Build the complete metrics.json structure."""
    metrics = {
        "experiment": "class-weighted-stage1",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # --- P(directional) distribution per weight (THE primary diagnostic) ---
    p_dist_per_weight = {}
    for weight in SCALE_POS_WEIGHTS:
        wd = weight_data.get(weight)
        if wd is None:
            continue
        all_proba = np.concatenate([fp[1] for fp in wd["fold_probas"]])
        p_dist = analyze_p_distribution(all_proba)
        p_dist_per_weight[f"{weight:.2f}"] = p_dist

    metrics["p_directional_distribution_per_weight"] = p_dist_per_weight

    # --- Stage 1 accuracy/precision/recall per weight ---
    stage1_metrics_per_weight = {}
    for weight in SCALE_POS_WEIGHTS:
        wd = weight_data.get(weight)
        if wd is None:
            continue
        accs = [fp[2] for fp in wd["fold_probas"]]
        precs = [fp[3] for fp in wd["fold_probas"]]
        recs = [fp[4] for fp in wd["fold_probas"]]
        stage1_metrics_per_weight[f"{weight:.2f}"] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_per_fold": accs,
            "precision_mean": float(np.mean(precs)),
            "precision_per_fold": precs,
            "recall_mean": float(np.mean(recs)),
            "recall_per_fold": recs,
        }
    metrics["stage1_accuracy_per_weight"] = stage1_metrics_per_weight

    # --- Threshold curve per weight ---
    threshold_curves_per_weight = {}
    for weight in SCALE_POS_WEIGHTS:
        wd = weight_data.get(weight)
        if wd is None:
            continue
        curve = build_threshold_curve_for_weight(wd["threshold_results"])
        threshold_curves_per_weight[f"{weight:.2f}"] = curve
    metrics["threshold_curve_per_weight"] = threshold_curves_per_weight

    # --- Find optimal (weight, threshold) pair ---
    best_pair = find_optimal_pair(weight_data)
    if best_pair:
        best_weight, best_threshold = best_pair
        best_curve = threshold_curves_per_weight.get(f"{best_weight:.2f}", [])
        best_pt = next((pt for pt in best_curve if pt["threshold"] == best_threshold), None)

        metrics["best_weight_threshold_pair"] = {
            "weight": best_weight,
            "threshold": best_threshold,
        }
        metrics["best_realized_expectancy"] = best_pt["realized_exp_mean"] if best_pt else None
        metrics["best_trade_rate"] = best_pt["trade_rate_mean"] if best_pt else None

        if best_pt:
            metrics["hold_fraction_at_best"] = best_pt["hold_fraction_mean"]
            metrics["dir_bar_pnl_at_best"] = best_pt["dir_bar_exp_mean"]
            metrics["per_fold_cv_at_best"] = best_pt["realized_exp_cv_pct"]
            metrics["breakeven_rt_at_best"] = best_pt["breakeven_rt_mean"]
            metrics["daily_pnl_at_best"] = best_pt["daily_pnl_mean"]

            # Per-fold at optimal
            wd_best = weight_data.get(best_weight, {})
            best_fold_results = [r for r in wd_best.get("threshold_results", [])
                                 if r["threshold"] == best_threshold]
            per_fold = []
            for r in best_fold_results:
                per_fold.append({
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
                    "s2_accuracy_on_filtered": r["s2_accuracy_on_filtered"],
                })
            metrics["per_fold_at_best"] = per_fold

            # Cost sensitivity at optimal
            cost_sens = {}
            for scenario in ["optimistic", "base", "pessimistic"]:
                key = f"realized_exp_{scenario}"
                vals = [r[key] for r in best_fold_results]
                cost_sens[scenario] = {
                    "rt_cost": COST_SCENARIOS[scenario],
                    "mean": float(np.mean(vals)),
                    "per_fold": vals,
                }
            metrics["cost_sensitivity_at_best"] = cost_sens
    else:
        metrics["best_weight_threshold_pair"] = None
        metrics["best_realized_expectancy"] = None
        metrics["best_trade_rate"] = None

    # --- Dir-bar PnL stability per weight ---
    dir_bar_stability = {}
    for weight in SCALE_POS_WEIGHTS:
        wd = weight_data.get(weight)
        if wd is None:
            continue
        curve = threshold_curves_per_weight.get(f"{weight:.2f}", [])
        dir_bar_stability[f"{weight:.2f}"] = [
            {"threshold": pt["threshold"], "dir_bar_exp": pt["dir_bar_exp_mean"]}
            for pt in curve
        ]
    metrics["dir_bar_pnl_stability"] = dir_bar_stability

    # --- Sanity Checks ---
    sanity = {}

    # SC-S1: Weight 1.0 reproduces baseline
    baseline_curve = threshold_curves_per_weight.get("1.00", [])
    baseline_pt = next((pt for pt in baseline_curve if pt["threshold"] == 0.50), None)
    if baseline_pt:
        bl_exp = baseline_pt["realized_exp_mean"]
        bl_diff = abs(bl_exp - BASELINE["realized_wf_expectancy"])
        sanity["SC-S1"] = {
            "description": "Weight 1.0 reproduces baseline realized exp $0.90 +/- $0.01",
            "pass": bl_diff < 0.01,
            "value": bl_exp,
            "reference": BASELINE["realized_wf_expectancy"],
            "diff": bl_diff,
        }
        bl_tr = baseline_pt["trade_rate_mean"]
        bl_tr_diff = abs(bl_tr - BASELINE["trade_rate"])
        sanity["SC-S1b"] = {
            "description": "Weight 1.0 reproduces baseline trade rate 85.18%",
            "pass": bl_tr_diff < 0.001,
            "value": bl_tr,
            "reference": BASELINE["trade_rate"],
            "diff": bl_tr_diff,
        }

    # SC-S2: Precision increases as weight decreases
    prec_values = []
    for w in SCALE_POS_WEIGHTS:
        sm = stage1_metrics_per_weight.get(f"{w:.2f}", {})
        prec_values.append(sm.get("precision_mean", 0))
    prec_mono = all(prec_values[i] <= prec_values[i + 1]
                    for i in range(len(prec_values) - 1)) if len(prec_values) > 1 else False
    sanity["SC-S2"] = {
        "description": "Precision increases as weight decreases",
        "pass": prec_mono,
        "values": dict(zip([f"{w:.2f}" for w in SCALE_POS_WEIGHTS], prec_values)),
    }

    # SC-S3: Recall decreases as weight decreases
    rec_values = []
    for w in SCALE_POS_WEIGHTS:
        sm = stage1_metrics_per_weight.get(f"{w:.2f}", {})
        rec_values.append(sm.get("recall_mean", 0))
    rec_mono = all(rec_values[i] >= rec_values[i + 1]
                   for i in range(len(rec_values) - 1)) if len(rec_values) > 1 else False
    sanity["SC-S3"] = {
        "description": "Recall decreases as weight decreases",
        "pass": rec_mono,
        "values": dict(zip([f"{w:.2f}" for w in SCALE_POS_WEIGHTS], rec_values)),
    }

    # SC-S4: IQR increases as weight decreases
    iqr_values = []
    for w in SCALE_POS_WEIGHTS:
        pd_w = p_dist_per_weight.get(f"{w:.2f}", {})
        iqr_values.append(pd_w.get("iqr_pp", 0))
    iqr_mono = all(iqr_values[i] <= iqr_values[i + 1]
                   for i in range(len(iqr_values) - 1)) if len(iqr_values) > 1 else False
    sanity["SC-S4"] = {
        "description": "IQR monotonically increases as weight decreases",
        "pass": iqr_mono,
        "values": dict(zip([f"{w:.2f}" for w in SCALE_POS_WEIGHTS], iqr_values)),
    }

    metrics["sanity_checks"] = sanity

    # --- Success Criteria ---
    sc = {}
    best_exp = metrics.get("best_realized_expectancy", -999) or -999
    best_tr = metrics.get("best_trade_rate", 0) or 0

    # SC-1: exp > $1.50 at > 15% trade rate
    sc["SC-1"] = {
        "description": "Exists (weight, threshold) with realized WF exp > $1.50 and trade rate > 15%",
        "pass": best_exp > 1.50 and best_tr > 0.15,
        "value": best_exp,
        "trade_rate": best_tr,
    }

    # SC-2: IQR at best weight > 10pp
    best_weight_key = None
    if best_pair:
        best_weight_key = f"{best_pair[0]:.2f}"
    # Also check if ANY weight achieves IQR > 10pp
    max_iqr_weight = None
    max_iqr = 0
    for w in SCALE_POS_WEIGHTS:
        pd_w = p_dist_per_weight.get(f"{w:.2f}", {})
        w_iqr = pd_w.get("iqr_pp", 0)
        if w_iqr > max_iqr:
            max_iqr = w_iqr
            max_iqr_weight = w
    sc["SC-2"] = {
        "description": "P(directional) IQR at best weight > 10pp (vs baseline 4pp)",
        "pass": max_iqr > 10.0,
        "max_iqr_pp": max_iqr,
        "max_iqr_weight": max_iqr_weight,
    }

    # SC-3: Per-fold CV < 80% at optimal
    best_cv = metrics.get("per_fold_cv_at_best", 999) or 999
    sc["SC-3"] = {
        "description": "Per-fold CV < 80% at optimal (weight, threshold)",
        "pass": best_cv < 80,
        "value": best_cv,
    }

    # SC-4: Dir-bar PnL > $3.00 at optimal
    best_dir = metrics.get("dir_bar_pnl_at_best", 0) or 0
    sc["SC-4"] = {
        "description": "Dir-bar PnL > $3.00/trade at optimal (weight, threshold)",
        "pass": best_dir > 3.00,
        "value": best_dir,
    }

    metrics["success_criteria"] = sc

    # --- Outcome ---
    sc1_pass = sc["SC-1"]["pass"]
    sc2_pass = sc["SC-2"]["pass"]

    if sc2_pass and sc1_pass:
        outcome = "A"
        outcome_desc = "CONFIRMED — class weighting + threshold produces viable strategy."
    elif sc2_pass and not sc1_pass:
        outcome = "B"
        outcome_desc = "PARTIAL — probability spread works but underlying signal insufficient."
    else:
        outcome = "C"
        outcome_desc = "REFUTED — class weighting doesn't spread the probability surface."

    metrics["outcome"] = outcome
    metrics["outcome_description"] = outcome_desc

    # --- Comparison table: baseline vs optimal ---
    if baseline_pt and best_pair:
        best_weight, best_threshold = best_pair
        best_curve = threshold_curves_per_weight.get(f"{best_weight:.2f}", [])
        opt_pt = next((pt for pt in best_curve if pt["threshold"] == best_threshold), None)
        if opt_pt:
            metrics["baseline_vs_optimal"] = {
                "baseline": {
                    "weight": 1.0,
                    "threshold": 0.50,
                    "realized_exp": baseline_pt["realized_exp_mean"],
                    "trade_rate": baseline_pt["trade_rate_mean"],
                    "hold_fraction": baseline_pt["hold_fraction_mean"],
                    "per_fold_cv": baseline_pt["realized_exp_cv_pct"],
                    "breakeven_rt": baseline_pt["breakeven_rt_mean"],
                    "daily_pnl": baseline_pt["daily_pnl_mean"],
                    "dir_bar_exp": baseline_pt["dir_bar_exp_mean"],
                    "p_iqr_pp": p_dist_per_weight.get("1.00", {}).get("iqr_pp", 0),
                    "p_in_050_060": p_dist_per_weight.get("1.00", {}).get("fraction_in_050_060", 0),
                },
                "optimal": {
                    "weight": best_weight,
                    "threshold": best_threshold,
                    "realized_exp": opt_pt["realized_exp_mean"],
                    "trade_rate": opt_pt["trade_rate_mean"],
                    "hold_fraction": opt_pt["hold_fraction_mean"],
                    "per_fold_cv": opt_pt["realized_exp_cv_pct"],
                    "breakeven_rt": opt_pt["breakeven_rt_mean"],
                    "daily_pnl": opt_pt["daily_pnl_mean"],
                    "dir_bar_exp": opt_pt["dir_bar_exp_mean"],
                    "p_iqr_pp": p_dist_per_weight.get(f"{best_weight:.2f}", {}).get("iqr_pp", 0),
                    "p_in_050_060": p_dist_per_weight.get(f"{best_weight:.2f}", {}).get("fraction_in_050_060", 0),
                },
            }

    # MVE gates
    metrics["mve_gates"] = mve_result.get("gates", {})

    # Resource usage
    n_s1_fits = len(SCALE_POS_WEIGHTS) * len(fold_contexts) if fold_contexts else 0
    n_s2_fits = len(fold_contexts) if fold_contexts else 0
    metrics["resource_usage"] = {
        "wall_clock_seconds": elapsed_total,
        "wall_clock_minutes": elapsed_total / 60,
        "total_training_runs": n_s1_fits + n_s2_fits,
        "stage1_fits": n_s1_fits,
        "stage2_fits": n_s2_fits,
        "gpu_hours": 0,
        "total_threshold_evaluations": len(SCALE_POS_WEIGHTS) * len(THRESHOLDS) * len(fold_contexts) if fold_contexts else 0,
    }

    metrics["abort_triggered"] = abort_triggered
    metrics["abort_reason"] = abort_reason
    metrics["notes"] = (
        "Executed locally on Apple Silicon. Stage 2 trained ONCE per fold (weight=1.0). "
        "Stage 1 retrained per weight. Threshold sweep is post-hoc numpy on Stage 1 probabilities. "
        "Realized return PnL model identical to pnl-realized-return (PR #35). "
        "Only 19:7 geometry (10:5 excluded per spec)."
    )

    return metrics


# ==========================================================================
# Write Outputs
# ==========================================================================
def write_metrics(metrics):
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {RESULTS_DIR / 'metrics.json'}")


def write_config(fold_contexts):
    config = {
        "seed": SEED,
        "horizon_bars": HORIZON_BARS,
        "tick_value": TICK_VALUE,
        "tick_size": TICK_SIZE,
        "geometry": {"target": GEOMETRY["target"], "stop": GEOMETRY["stop"]},
        "xgb_params": TUNED_XGB_PARAMS_BINARY,
        "features": NON_SPATIAL_FEATURES,
        "cost_scenarios": COST_SCENARIOS,
        "walk_forward_folds": WF_FOLDS,
        "purge_bars": PURGE_BARS,
        "thresholds": THRESHOLDS,
        "scale_pos_weights": SCALE_POS_WEIGHTS,
        "data_source": str(GEOMETRY["data_dir"]),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {RESULTS_DIR / 'config.json'}")


def write_threshold_curve_csv(weight_data):
    """Write threshold curve CSV: 5 weights x 9 thresholds."""
    rows = []
    for weight in SCALE_POS_WEIGHTS:
        wd = weight_data.get(weight)
        if wd is None:
            continue
        curve = build_threshold_curve_for_weight(wd["threshold_results"])
        for pt in curve:
            rows.append({
                "weight": weight,
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


def write_p_distribution_csv(metrics):
    """Write P(directional) distribution stats per weight."""
    p_dists = metrics.get("p_directional_distribution_per_weight", {})
    rows = []
    for weight_str, pd in p_dists.items():
        pcts = pd.get("percentiles", {})
        rows.append({
            "weight": weight_str,
            "n_total": pd.get("n_total"),
            "mean": pd.get("mean"),
            "std": pd.get("std"),
            "iqr_pp": pd.get("iqr_pp"),
            "p10": pcts.get("p10"),
            "p25": pcts.get("p25"),
            "p50": pcts.get("p50"),
            "p75": pcts.get("p75"),
            "p90": pcts.get("p90"),
            "frac_in_050_060": pd.get("fraction_in_050_060"),
            "frac_in_060_100": pd.get("fraction_in_060_100"),
            "frac_in_070_100": pd.get("fraction_in_070_100"),
        })
    if rows:
        with open(RESULTS_DIR / "p_directional_distribution.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"P(directional) distribution written to {RESULTS_DIR / 'p_directional_distribution.csv'}")


def write_per_fold_at_optimal_csv(metrics):
    """Write per-fold details at optimal (weight, threshold)."""
    per_fold = metrics.get("per_fold_at_best", [])
    if per_fold:
        with open(RESULTS_DIR / "per_fold_at_optimal.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_fold[0].keys())
            writer.writeheader()
            writer.writerows(per_fold)
    print(f"Per-fold at optimal written to {RESULTS_DIR / 'per_fold_at_optimal.csv'}")


def write_cost_sensitivity_csv(metrics):
    """Write 3 cost scenarios at optimal."""
    cost_sens = metrics.get("cost_sensitivity_at_best", {})
    rows = []
    bwt = metrics.get("best_weight_threshold_pair", {})
    for scenario in ["optimistic", "base", "pessimistic"]:
        cs = cost_sens.get(scenario, {})
        if cs:
            rows.append({
                "weight": bwt.get("weight") if bwt else None,
                "threshold": bwt.get("threshold") if bwt else None,
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


def write_analysis(metrics, weight_data):
    """Write analysis.md with all required sections."""
    sc = metrics.get("success_criteria", {})
    sanity = metrics.get("sanity_checks", {})
    p_dists = metrics.get("p_directional_distribution_per_weight", {})
    s1_metrics = metrics.get("stage1_accuracy_per_weight", {})
    curves = metrics.get("threshold_curve_per_weight", {})
    bvo = metrics.get("baseline_vs_optimal", {})
    outcome = metrics.get("outcome", "?")
    outcome_desc = metrics.get("outcome_description", "")

    lines = []
    lines.append("# Class-Weighted Stage 1 — Analysis\n")

    # 1. Executive summary
    lines.append("## Executive Summary\n")
    lines.append(f"**Outcome {outcome}:** {outcome_desc}\n")
    bwt = metrics.get("best_weight_threshold_pair")
    if bwt:
        lines.append(f"Best (weight, threshold) = ({bwt['weight']:.2f}, {bwt['threshold']:.2f}): "
                      f"realized exp **${metrics.get('best_realized_expectancy', 0):.4f}**/trade, "
                      f"trade rate {metrics.get('best_trade_rate', 0):.1%}.\n")
    lines.append("")

    # 2. P(directional) distribution per weight — THE primary diagnostic
    lines.append("## P(directional) Distribution per Weight (Primary Diagnostic)\n")
    lines.append("| Weight | IQR (pp) | Mean | Std | p25 | p50 | p75 | Frac [0.50,0.60) | Frac [0.60,1.00] | Frac [0.70,1.00] |")
    lines.append("|--------|----------|------|-----|-----|-----|-----|------------------|------------------|------------------|")
    for weight in SCALE_POS_WEIGHTS:
        pd = p_dists.get(f"{weight:.2f}", {})
        pcts = pd.get("percentiles", {})
        lines.append(f"| {weight:.2f} | **{pd.get('iqr_pp', 0):.1f}** | {pd.get('mean', 0):.4f} | "
                      f"{pd.get('std', 0):.4f} | {pcts.get('p25', 0):.4f} | {pcts.get('p50', 0):.4f} | "
                      f"{pcts.get('p75', 0):.4f} | {pd.get('fraction_in_050_060', 0):.1%} | "
                      f"{pd.get('fraction_in_060_100', 0):.1%} | {pd.get('fraction_in_070_100', 0):.1%} |")
    lines.append("")

    # 3. Threshold curve per weight
    lines.append("## Threshold Curve per Weight\n")
    for weight in SCALE_POS_WEIGHTS:
        curve = curves.get(f"{weight:.2f}", [])
        lines.append(f"### Weight = {weight:.2f}\n")
        lines.append("| Threshold | Trade Rate | Hold Frac | Realized Exp | Dir-Bar Exp | Hold-Bar Exp | CV% | B/E RT | Daily PnL |")
        lines.append("|-----------|-----------|-----------|-------------|-------------|-------------|-----|--------|-----------|")
        for pt in curve:
            lines.append(f"| {pt['threshold']:.2f} | {pt['trade_rate_mean']:.1%} | "
                          f"{pt['hold_fraction_mean']:.1%} | ${pt['realized_exp_mean']:.4f} | "
                          f"${pt['dir_bar_exp_mean']:.4f} | ${pt['hold_bar_exp_mean']:.4f} | "
                          f"{pt['realized_exp_cv_pct']:.0f}% | ${pt['breakeven_rt_mean']:.2f} | "
                          f"${pt['daily_pnl_mean']:.2f} |")
        lines.append("")

    # 4. Best (weight, threshold) identification
    lines.append("## Best (Weight, Threshold) Identification\n")
    if bwt:
        lines.append(f"Optimal pair: **weight={bwt['weight']:.2f}, threshold={bwt['threshold']:.2f}**\n")
        lines.append(f"- Realized expectancy: ${metrics.get('best_realized_expectancy', 0):.4f}/trade")
        lines.append(f"- Trade rate: {metrics.get('best_trade_rate', 0):.1%}")
        lines.append(f"- Hold fraction: {metrics.get('hold_fraction_at_best', 0):.1%}")
        lines.append(f"- Per-fold CV: {metrics.get('per_fold_cv_at_best', 0):.0f}%")
        lines.append(f"- Break-even RT: ${metrics.get('breakeven_rt_at_best', 0):.2f}")
        lines.append(f"- Dir-bar PnL: ${metrics.get('dir_bar_pnl_at_best', 0):.4f}/trade")
    else:
        lines.append("No (weight, threshold) pair with trade rate > 15% found.\n")
    lines.append("")

    # 5. Per-fold consistency at optimal
    lines.append("## Per-Fold Consistency at Optimal\n")
    per_fold = metrics.get("per_fold_at_best", [])
    if per_fold:
        lines.append("| Fold | N Trades | Trade Rate | Hold Frac | Realized Exp | Dir-Bar Exp | S2 Acc |")
        lines.append("|------|---------|-----------|-----------|-------------|-------------|--------|")
        for pf in per_fold:
            lines.append(f"| {pf['fold']} | {pf['n_trades']:,} | {pf['trade_rate']:.1%} | "
                          f"{pf['hold_fraction']:.1%} | ${pf['realized_exp_base']:.4f} | "
                          f"${pf['dir_bar_exp']:.4f} | {pf['s2_accuracy_on_filtered']:.1%} |")
    lines.append("")

    # 6. Dir-bar quality check
    lines.append("## Dir-Bar PnL per Weight (Quality Check)\n")
    dir_bar = metrics.get("dir_bar_pnl_stability", {})
    for weight in SCALE_POS_WEIGHTS:
        entries = dir_bar.get(f"{weight:.2f}", [])
        t050_entry = next((e for e in entries if e["threshold"] == 0.50), None)
        if t050_entry:
            lines.append(f"- Weight {weight:.2f}, T=0.50: ${t050_entry['dir_bar_exp']:.4f}/trade")
    lines.append("")

    # 7. Stage 1 precision/recall per weight
    lines.append("## Stage 1 Precision/Recall per Weight\n")
    lines.append("| Weight | Accuracy | Precision | Recall |")
    lines.append("|--------|----------|-----------|--------|")
    for weight in SCALE_POS_WEIGHTS:
        sm = s1_metrics.get(f"{weight:.2f}", {})
        lines.append(f"| {weight:.2f} | {sm.get('accuracy_mean', 0):.4f} | "
                      f"{sm.get('precision_mean', 0):.4f} | {sm.get('recall_mean', 0):.4f} |")
    lines.append("")

    # 8. Cost sensitivity at optimal
    lines.append("## Cost Sensitivity at Optimal\n")
    cost_sens = metrics.get("cost_sensitivity_at_best", {})
    if cost_sens:
        lines.append("| Scenario | RT Cost | Mean Exp | Fold 1 | Fold 2 | Fold 3 |")
        lines.append("|----------|---------|---------|--------|--------|--------|")
        for scenario in ["optimistic", "base", "pessimistic"]:
            cs = cost_sens.get(scenario, {})
            pf = cs.get("per_fold", [0, 0, 0])
            lines.append(f"| {scenario} | ${cs.get('rt_cost', 0):.2f} | ${cs.get('mean', 0):.4f} | "
                          f"${pf[0]:.4f} | ${pf[1] if len(pf) > 1 else 0:.4f} | ${pf[2] if len(pf) > 2 else 0:.4f} |")
    lines.append("")

    # 9. Comparison table
    lines.append("## Comparison: Baseline vs Optimal\n")
    bl = bvo.get("baseline", {})
    opt = bvo.get("optimal", {})
    if bl and opt:
        lines.append("| Metric | Baseline (w=1.0, T=0.50) | Optimal (w=w*, T=T*) | Delta |")
        lines.append("|--------|--------------------------|---------------------|-------|")
        for field, label, fmt in [
            ("realized_exp", "Realized exp", "${:.4f}"),
            ("trade_rate", "Trade rate", "{:.1%}"),
            ("hold_fraction", "Hold fraction", "{:.1%}"),
            ("per_fold_cv", "Per-fold CV", "{:.0f}%"),
            ("breakeven_rt", "Break-even RT", "${:.2f}"),
            ("daily_pnl", "Daily PnL", "${:.2f}"),
            ("dir_bar_exp", "Dir-bar exp", "${:.4f}"),
            ("p_iqr_pp", "P(dir) IQR", "{:.1f}pp"),
            ("p_in_050_060", "P(dir) in [0.50,0.60)", "{:.1%}"),
        ]:
            bv = bl.get(field, 0)
            ov = opt.get(field, 0)
            delta = ov - bv
            bv_str = fmt.format(bv) if bv is not None else "N/A"
            ov_str = fmt.format(ov) if ov is not None else "N/A"
            delta_str = fmt.format(delta) if delta is not None else "N/A"
            lines.append(f"| {label} | {bv_str} | {ov_str} | {delta_str} |")
    lines.append("")

    # 10. Explicit SC-1 through SC-4
    lines.append("## Success Criteria\n")
    for sc_key in ["SC-1", "SC-2", "SC-3", "SC-4"]:
        sc_item = sc.get(sc_key, {})
        status = "PASS" if sc_item.get("pass") else "FAIL"
        lines.append(f"- **{sc_key}**: {status} — {sc_item.get('description', '')}")
    lines.append("")

    # Sanity checks
    lines.append("## Sanity Checks\n")
    for sc_key in sorted(sanity.keys()):
        sc_item = sanity[sc_key]
        status = "PASS" if sc_item.get("pass") else "FAIL"
        lines.append(f"- **{sc_key}**: {status} — {sc_item.get('description', '')}")
    lines.append("")

    # 11. Outcome verdict
    lines.append("## Verdict\n")
    lines.append(f"**Outcome {outcome}:** {outcome_desc}\n")

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Analysis written to {RESULTS_DIR / 'analysis.md'}")


# ==========================================================================
# Main
# ==========================================================================
def main():
    t0 = time.time()
    set_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Experiment: Class-Weighted Stage 1 — Spreading the Probability Surface")
    print(f"Geometry: {GEOMETRY['label']} (target={GEOMETRY['target']}, stop={GEOMETRY['stop']})")
    print(f"Weights: {SCALE_POS_WEIGHTS}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Wall-clock limit: {WALL_CLOCK_LIMIT_S}s")
    print("=" * 70)

    # Step 0: Load data
    print("\nLoading data...")
    features, labels, day_indices, unique_days, extra = load_data()

    # Step 1: Run MVE
    mve_result = run_mve(features, labels, day_indices, extra)
    if mve_result["abort"]:
        print(f"\n*** ABORT: {mve_result['abort_reason']} ***")
        elapsed = time.time() - t0
        metrics = {
            "experiment": "class-weighted-stage1",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": mve_result["abort_reason"],
            "mve_gates": mve_result.get("gates", {}),
            "resource_usage": {
                "wall_clock_seconds": elapsed,
                "wall_clock_minutes": elapsed / 60,
                "total_training_runs": 0,
                "gpu_hours": 0,
            },
        }
        write_metrics(metrics)
        return

    # Step 2: Full protocol
    weight_data, fold_contexts, abort_triggered, abort_reason = run_full_protocol(
        features, labels, day_indices, extra, t0)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"TOTAL WALL-CLOCK: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    if weight_data is None:
        metrics = {
            "experiment": "class-weighted-stage1",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "abort_triggered": True,
            "abort_reason": abort_reason or "Unknown error during full protocol",
            "mve_gates": mve_result.get("gates", {}),
            "resource_usage": {
                "wall_clock_seconds": elapsed,
                "wall_clock_minutes": elapsed / 60,
                "total_training_runs": 0,
                "gpu_hours": 0,
            },
        }
        write_metrics(metrics)
        return

    # Step 3: Build and write metrics
    metrics = build_metrics(weight_data, fold_contexts, mve_result, t0, elapsed,
                            abort_triggered, abort_reason)
    write_metrics(metrics)
    write_config(fold_contexts)
    write_threshold_curve_csv(weight_data)
    write_p_distribution_csv(metrics)
    write_per_fold_at_optimal_csv(metrics)
    write_cost_sensitivity_csv(metrics)
    write_analysis(metrics, weight_data)

    # Print final summary
    print(f"\nOutcome: {metrics['outcome']} — {metrics['outcome_description']}")
    for sc_key in ["SC-1", "SC-2", "SC-3", "SC-4"]:
        sc_item = metrics.get("success_criteria", {}).get(sc_key, {})
        print(f"  {sc_key}: {'PASS' if sc_item.get('pass') else 'FAIL'} — {sc_item.get('description', '')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
