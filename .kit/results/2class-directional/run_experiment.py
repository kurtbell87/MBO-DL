#!/usr/bin/env python3
"""
Experiment: 2-Class Directional — Two-Stage Reachability + Direction Pipeline
Spec: .kit/experiments/2class-directional.md

Two-stage XGBoost pipeline:
  Stage 1: binary "directional vs hold" (binary:logistic)
  Stage 2: binary "long vs short" on directional-only training bars (binary:logistic)
  Combined: Stage 1 predicts directional → Stage 2 assigns direction; else hold (pred=0)

2 geometries: 19:7 (primary), 10:5 (control)
Walk-forward: 3 expanding-window folds
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ==========================================================================
# Config
# ==========================================================================
SEED = 42
WORKTREE_ROOT = Path("/Users/brandonbell/LOCAL_DEV/MBO-DL-label-geom-p1")
RESULTS_DIR = WORKTREE_ROOT / ".kit" / "results" / "2class-directional"
DATA_BASE = WORKTREE_ROOT / ".kit" / "results" / "label-geometry-1h"

# Tuned XGBoost params — adapted for binary:logistic
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

# Non-spatial features (20 dimensions) — identical to all prior experiments
NON_SPATIAL_FEATURES = [
    "weighted_imbalance", "spread", "net_volume", "volume_imbalance",
    "trade_count", "avg_trade_size", "vwap_distance",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50", "high_low_range_50", "close_position",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "time_sin", "time_cos", "minutes_since_open",
]

# Transaction cost scenarios
COST_SCENARIOS = {"optimistic": 2.49, "base": 3.74, "pessimistic": 6.25}
TICK_VALUE = 1.25

# Geometries
GEOMETRIES = {
    "19_7": {"target": 19, "stop": 7, "label": "19:7", "bev_wr": 0.384,
             "data_dir": DATA_BASE / "geom_19_7"},
    "10_5": {"target": 10, "stop": 5, "label": "10:5", "bev_wr": 0.533,
             "data_dir": DATA_BASE / "geom_10_5"},
}

# Walk-forward folds (day_index based)
WF_FOLDS = [
    {"train_range": (1, 100), "test_range": (101, 150), "name": "Fold 1"},
    {"train_range": (1, 150), "test_range": (151, 201), "name": "Fold 2"},
    {"train_range": (1, 201), "test_range": (202, 251), "name": "Fold 3 (holdout)"},
]

# Purge (bars between train/test to prevent leakage)
PURGE_BARS = 500

# 3-class baseline results from label-geometry-1h walk-forward
THREE_CLASS_BASELINES = {
    "19_7": {
        "wf_mean_expectancy_base": 6.4156,
        "wf_mean_dir_accuracy": 0.5817,
        "wf_mean_dir_pred_rate": 0.00283,
        "wf_mean_label0_hit_rate": (0.3824 + 0.4444 + 0.9668) / 3,
        "cpcv_expectancy_base": 5.678,
    },
    "10_5": {
        "wf_mean_expectancy_base": -0.4989,
        "wf_mean_dir_accuracy": 0.5062,
        "wf_mean_dir_pred_rate": 0.9039,
        "wf_mean_label0_hit_rate": (0.3016 + 0.3030 + 0.3203) / 3,
        "cpcv_expectancy_base": -0.4897,
    },
}

# Wall-clock limit
WALL_CLOCK_LIMIT_S = 30 * 60  # 30 min abort


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def remap_feature_importance(importance_dict):
    """Remap XGBoost's 'f0', 'f1', ... keys to actual feature names."""
    remapped = {}
    for key, val in importance_dict.items():
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if idx < len(NON_SPATIAL_FEATURES):
                remapped[NON_SPATIAL_FEATURES[idx]] = val
            else:
                remapped[key] = val
        else:
            remapped[key] = val
    return remapped


# ==========================================================================
# Data Loading
# ==========================================================================
def load_geometry_data(geom_key):
    """Load all Parquet for one geometry, extract features + labels + day_index."""
    geom = GEOMETRIES[geom_key]
    data_dir = geom["data_dir"]

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files in {data_dir}")

    print(f"  Loading {len(parquet_files)} Parquet files from {data_dir.name}...")
    frames = []
    for pf in parquet_files:
        df = pl.read_parquet(pf)
        # Filter warmup bars
        if "is_warmup" in df.columns:
            df = df.filter(pl.col("is_warmup") == 0)
        frames.append(df)

    combined = pl.concat(frames)
    print(f"  Total rows after warmup filter: {len(combined)}")

    # Extract features
    features = combined.select(NON_SPATIAL_FEATURES).to_numpy().astype(np.float64)
    labels = combined["tb_label"].to_numpy().astype(np.float64).astype(np.int64)
    day_index = combined["day"].to_numpy() if "day" in combined.columns else None

    if day_index is None:
        raise ValueError("No 'day' column in Parquet — cannot do walk-forward splits")

    # Create sequential day index (1..N)
    unique_days = sorted(set(day_index.tolist()))
    day_map = {d: i + 1 for i, d in enumerate(unique_days)}
    day_indices = np.array([day_map[d] for d in day_index])

    # Verify schema
    assert features.shape[1] == 20, f"Expected 20 features, got {features.shape[1]}"
    assert set(np.unique(labels)).issubset({-1, 0, 1}), f"Unexpected labels: {np.unique(labels)}"

    # Class distribution
    n_total = len(labels)
    n_short = (labels == -1).sum()
    n_hold = (labels == 0).sum()
    n_long = (labels == 1).sum()
    n_dir = n_short + n_long
    print(f"  Classes: short={n_short} ({100*n_short/n_total:.1f}%), "
          f"hold={n_hold} ({100*n_hold/n_total:.1f}%), "
          f"long={n_long} ({100*n_long/n_total:.1f}%)")
    print(f"  Directional rate: {100*n_dir/n_total:.1f}%, "
          f"Long:Short ratio: {n_long/max(n_short,1):.3f}")
    print(f"  Day range: {min(day_indices)}-{max(day_indices)}, "
          f"Unique days: {len(unique_days)}")

    return features, labels, day_indices, unique_days


def apply_purge(train_indices, test_indices, day_indices, purge_bars):
    """Remove training bars within purge_bars of test set boundaries."""
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
# PnL and Metrics
# ==========================================================================
def compute_pnl(true_labels, pred_labels, target_ticks, stop_ticks, rt_cost):
    """PnL per observation with geometry-specific tick values."""
    win_pnl = target_ticks * TICK_VALUE
    loss_pnl = stop_ticks * TICK_VALUE
    pnl = np.zeros(len(true_labels))
    both_nonzero = (pred_labels != 0) & (true_labels != 0)
    correct = (pred_labels == true_labels) & both_nonzero
    wrong = (pred_labels != true_labels) & both_nonzero
    pnl[correct] = win_pnl - rt_cost
    pnl[wrong] = -loss_pnl - rt_cost
    return pnl


def compute_expectancy(pnl_array):
    """Per-trade expectancy and trade count."""
    trades = pnl_array[pnl_array != 0]
    if len(trades) == 0:
        return 0.0, 0.0, 0
    expectancy = float(trades.mean())
    gross_profit = float(trades[trades > 0].sum())
    gross_loss = float(abs(trades[trades < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return expectancy, pf, len(trades)


def compute_directional_accuracy(true_labels, pred_labels):
    """Correct predictions / bars where BOTH pred!=0 AND true!=0."""
    both_nonzero = (pred_labels != 0) & (true_labels != 0)
    n = both_nonzero.sum()
    if n == 0:
        return 0.0, 0
    correct = (pred_labels == true_labels) & both_nonzero
    return float(correct.sum() / n), int(n)


# ==========================================================================
# Two-Stage Training (single fold)
# ==========================================================================
def train_two_stage_fold(features, labels, day_indices,
                         train_range, test_range, geom_key, fold_name):
    """
    Train Stage 1 (directional vs hold) and Stage 2 (long vs short) for one fold.
    Returns combined predictions and per-stage metrics.
    """
    geom = GEOMETRIES[geom_key]
    target = geom["target"]
    stop = geom["stop"]

    # Split by day range
    train_mask = (day_indices >= train_range[0]) & (day_indices <= train_range[1])
    test_mask = (day_indices >= test_range[0]) & (day_indices <= test_range[1])

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    if len(test_indices) == 0 or len(train_indices) == 0:
        return None

    # Purge
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

    # Internal train/val split: last 20% of training days
    train_days_in_fold = sorted(set(day_indices[clean_train].tolist()))
    n_val_days = max(1, len(train_days_in_fold) // 5)
    val_day_set = set(train_days_in_fold[-n_val_days:])

    val_mask_inner = np.array([day_indices[i] in val_day_set for i in clean_train])
    train_mask_inner = ~val_mask_inner

    inner_train_z = ft_train_z[train_mask_inner]
    inner_val_z = ft_train_z[val_mask_inner]

    # Labels for all bars
    labels_train_all = labels[clean_train]
    labels_test = labels[test_indices]

    inner_train_labels = labels_train_all[train_mask_inner]
    inner_val_labels = labels_train_all[val_mask_inner]

    # =====================================================================
    # STAGE 1: Binary directional vs hold
    # =====================================================================
    t0_s1 = time.time()

    # Binary label: 1 = directional (tb_label != 0), 0 = hold (tb_label == 0)
    s1_train_labels = (inner_train_labels != 0).astype(np.int64)
    s1_val_labels = (inner_val_labels != 0).astype(np.int64)
    s1_test_labels = (labels_test != 0).astype(np.int64)

    clf_s1 = xgb.XGBClassifier(**TUNED_XGB_PARAMS_BINARY)
    if len(inner_val_z) > 0:
        clf_s1.fit(inner_train_z, s1_train_labels,
                   eval_set=[(inner_val_z, s1_val_labels)],
                   verbose=False)
    else:
        clf_s1.fit(inner_train_z, s1_train_labels,
                   verbose=False)

    s1_pred_proba = clf_s1.predict_proba(ft_test_z)[:, 1]  # P(directional)
    s1_pred = (s1_pred_proba > 0.5).astype(np.int64)  # 1=directional, 0=hold

    elapsed_s1 = time.time() - t0_s1

    # Stage 1 metrics
    s1_accuracy = float(accuracy_score(s1_test_labels, s1_pred))
    s1_precision_dir = float(precision_score(s1_test_labels, s1_pred, pos_label=1, zero_division=0))
    s1_recall_dir = float(recall_score(s1_test_labels, s1_pred, pos_label=1, zero_division=0))

    try:
        s1_n_est = clf_s1.best_iteration + 1
    except AttributeError:
        s1_n_est = TUNED_XGB_PARAMS_BINARY["n_estimators"]

    # Stage 1 feature importance
    s1_importance = remap_feature_importance(clf_s1.get_booster().get_score(importance_type="gain"))

    # =====================================================================
    # STAGE 2: Binary long vs short (trained on directional-only bars)
    # =====================================================================
    t0_s2 = time.time()

    # Filter training data to directional bars only
    dir_mask_train = (inner_train_labels != 0)
    dir_mask_val = (inner_val_labels != 0)

    s2_train_z = inner_train_z[dir_mask_train]
    s2_val_z = inner_val_z[dir_mask_val]

    # Binary label: 1 = long (tb_label == 1), 0 = short (tb_label == -1)
    s2_train_labels = (inner_train_labels[dir_mask_train] == 1).astype(np.int64)
    s2_val_labels = (inner_val_labels[dir_mask_val] == 1).astype(np.int64)

    clf_s2 = xgb.XGBClassifier(**TUNED_XGB_PARAMS_BINARY)
    if len(s2_val_z) > 0:
        clf_s2.fit(s2_train_z, s2_train_labels,
                   eval_set=[(s2_val_z, s2_val_labels)],
                   verbose=False)
    else:
        clf_s2.fit(s2_train_z, s2_train_labels,
                   verbose=False)

    # Predict on ALL test bars (filter at combination step)
    s2_pred_proba = clf_s2.predict_proba(ft_test_z)[:, 1]  # P(long)
    s2_pred = np.where(s2_pred_proba > 0.5, 1, -1)  # +1=long, -1=short

    elapsed_s2 = time.time() - t0_s2

    try:
        s2_n_est = clf_s2.best_iteration + 1
    except AttributeError:
        s2_n_est = TUNED_XGB_PARAMS_BINARY["n_estimators"]

    # Stage 2 accuracy on ALL truly-directional test bars
    dir_test_mask = (labels_test != 0)
    s2_labels_on_dir = (labels_test[dir_test_mask] == 1).astype(np.int64)
    s2_pred_on_dir = (s2_pred_proba[dir_test_mask] > 0.5).astype(np.int64)
    s2_acc_all_dir = float(accuracy_score(s2_labels_on_dir, s2_pred_on_dir)) if dir_test_mask.sum() > 0 else 0.0

    # Stage 2 feature importance
    s2_importance = remap_feature_importance(clf_s2.get_booster().get_score(importance_type="gain"))

    # =====================================================================
    # COMBINE: Stage 1 directional → Stage 2 direction, else 0
    # =====================================================================
    combined_pred = np.zeros(len(labels_test), dtype=np.int64)
    s1_dir_mask = (s1_pred == 1)
    combined_pred[s1_dir_mask] = s2_pred[s1_dir_mask]

    # Combined metrics
    n_test = len(labels_test)
    n_trades = int((combined_pred != 0).sum())
    trade_rate = float(n_trades / n_test) if n_test > 0 else 0.0

    dir_acc, n_dir_pairs = compute_directional_accuracy(labels_test, combined_pred)

    # label0_hit_rate: fraction of directional predictions hitting hold-labeled bars
    pred_dir = (combined_pred != 0)
    true_hold = (labels_test == 0)
    label0_hit_rate = float((pred_dir & true_hold).sum() / max(pred_dir.sum(), 1))

    # Stage 2 accuracy on Stage 1-FILTERED directional test bars (selection bias check)
    s1_filtered_and_dir = s1_dir_mask & dir_test_mask
    if s1_filtered_and_dir.sum() > 0:
        s2_filtered_labels = (labels_test[s1_filtered_and_dir] == 1).astype(np.int64)
        s2_filtered_preds = (s2_pred_proba[s1_filtered_and_dir] > 0.5).astype(np.int64)
        s2_acc_filtered_dir = float(accuracy_score(s2_filtered_labels, s2_filtered_preds))
    else:
        s2_acc_filtered_dir = 0.0

    # PnL for all cost scenarios
    pnl_results = {}
    for scenario, rt_cost in COST_SCENARIOS.items():
        pnl = compute_pnl(labels_test, combined_pred, target, stop, rt_cost)
        exp, pf, nt = compute_expectancy(pnl)
        # Daily PnL: total PnL / number of test trading days
        test_days = sorted(set(day_indices[test_indices].tolist()))
        n_test_days = len(test_days)
        total_pnl = float(pnl.sum())
        daily_pnl = total_pnl / n_test_days if n_test_days > 0 else 0.0
        pnl_results[scenario] = {
            "expectancy": exp,
            "profit_factor": pf,
            "n_trades": nt,
            "total_pnl": total_pnl,
            "daily_pnl": daily_pnl,
        }

    del clf_s1, clf_s2

    return {
        "fold_name": fold_name,
        "geom_key": geom_key,
        "train_range": train_range,
        "test_range": test_range,
        "n_train": len(clean_train),
        "n_test": n_test,
        "n_purged": n_purged,
        # Stage 1
        "stage1_accuracy": s1_accuracy,
        "stage1_precision_directional": s1_precision_dir,
        "stage1_recall_directional": s1_recall_dir,
        "stage1_n_estimators": s1_n_est,
        "stage1_elapsed_s": elapsed_s1,
        "stage1_importance": s1_importance,
        # Stage 2
        "stage2_acc_all_directional": s2_acc_all_dir,
        "stage2_acc_filtered_directional": s2_acc_filtered_dir,
        "stage2_n_estimators": s2_n_est,
        "stage2_elapsed_s": elapsed_s2,
        "stage2_importance": s2_importance,
        # Combined
        "trade_rate": trade_rate,
        "n_trades_base": pnl_results["base"]["n_trades"],
        "directional_accuracy": dir_acc,
        "n_directional_pairs": n_dir_pairs,
        "label0_hit_rate": label0_hit_rate,
        "pnl": pnl_results,
        "selection_bias_delta": s2_acc_all_dir - s2_acc_filtered_dir,
    }


# ==========================================================================
# MVE Gates
# ==========================================================================
def run_mve(features, labels, day_indices):
    """Run Minimum Viable Experiment gates at 19:7."""
    print("\n" + "=" * 70)
    print("MVE GATES — 19:7 Geometry")
    print("=" * 70)

    geom_key = "19_7"
    geom = GEOMETRIES[geom_key]

    # Simple 80/20 split by days
    max_day = day_indices.max()
    train_cutoff = int(max_day * 0.8)
    train_mask = day_indices <= train_cutoff
    test_mask = day_indices > train_cutoff

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    ft_train = features[train_idx].copy()
    ft_test = features[test_idx].copy()

    f_mean = np.nanmean(ft_train, axis=0)
    f_std = np.nanstd(ft_train, axis=0)
    f_std[f_std < 1e-10] = 1.0
    ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
    ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)

    labels_train = labels[train_idx]
    labels_test = labels[test_idx]

    # Internal val: last 20% of training days
    train_days = sorted(set(day_indices[train_idx].tolist()))
    n_val_days = max(1, len(train_days) // 5)
    val_day_set = set(train_days[-n_val_days:])
    val_mask_inner = np.array([day_indices[i] in val_day_set for i in train_idx])
    train_mask_inner = ~val_mask_inner

    inner_train_z = ft_train_z[train_mask_inner]
    inner_val_z = ft_train_z[val_mask_inner]
    inner_train_labels = labels_train[train_mask_inner]
    inner_val_labels = labels_train[val_mask_inner]

    results = {}

    # ---- Gate 1: Data loading ----
    print(f"\n  [Gate 1] Data: {len(features)} rows, 20 features, "
          f"tb_label in {set(np.unique(labels))}. PASS")
    results["gate1_data"] = True

    # ---- Gate 2: Stage 1 ----
    s1_train_labels = (inner_train_labels != 0).astype(np.int64)
    s1_val_labels = (inner_val_labels != 0).astype(np.int64)
    s1_test_labels = (labels_test != 0).astype(np.int64)

    majority_baseline = max(s1_test_labels.mean(), 1 - s1_test_labels.mean())

    clf_s1 = xgb.XGBClassifier(**TUNED_XGB_PARAMS_BINARY)
    clf_s1.fit(inner_train_z, s1_train_labels,
               eval_set=[(inner_val_z, s1_val_labels)],
               verbose=False)
    s1_pred = clf_s1.predict(ft_test_z)
    s1_acc = float(accuracy_score(s1_test_labels, s1_pred))

    both_classes = len(set(s1_pred.tolist())) > 1
    print(f"  [Gate 2] Stage 1 accuracy: {s1_acc:.4f} "
          f"(majority baseline: {majority_baseline:.4f}, "
          f"both classes predicted: {both_classes})")

    if s1_acc <= majority_baseline:
        print(f"  ** ABORT: Stage 1 accuracy {s1_acc:.4f} <= majority {majority_baseline:.4f}")
        results["gate2_stage1"] = False
        results["abort"] = True
        results["abort_reason"] = f"Stage 1 accuracy {s1_acc:.4f} <= majority baseline {majority_baseline:.4f}"
        return results

    results["gate2_stage1"] = True
    results["gate2_s1_accuracy"] = s1_acc
    print(f"  [Gate 2] PASS")

    # ---- Gate 3: Stage 2 ----
    dir_train_mask = (inner_train_labels != 0)
    dir_val_mask = (inner_val_labels != 0)

    s2_train_z = inner_train_z[dir_train_mask]
    s2_val_z = inner_val_z[dir_val_mask]
    s2_train_labels = (inner_train_labels[dir_train_mask] == 1).astype(np.int64)
    s2_val_labels = (inner_val_labels[dir_val_mask] == 1).astype(np.int64)

    clf_s2 = xgb.XGBClassifier(**TUNED_XGB_PARAMS_BINARY)
    clf_s2.fit(s2_train_z, s2_train_labels,
               eval_set=[(s2_val_z, s2_val_labels)],
               verbose=False)

    dir_test_mask = (labels_test != 0)
    s2_test_labels = (labels_test[dir_test_mask] == 1).astype(np.int64)
    s2_pred = clf_s2.predict(ft_test_z[dir_test_mask])
    s2_acc = float(accuracy_score(s2_test_labels, s2_pred))

    print(f"  [Gate 3] Stage 2 accuracy on directional test bars: {s2_acc:.4f}")
    if s2_acc <= 0.40:
        print(f"  ** ABORT: Stage 2 accuracy {s2_acc:.4f} <= 0.40")
        results["gate3_stage2"] = False
        results["abort"] = True
        results["abort_reason"] = f"Stage 2 accuracy {s2_acc:.4f} <= 0.40"
        return results

    results["gate3_stage2"] = True
    results["gate3_s2_accuracy"] = s2_acc
    print(f"  [Gate 3] PASS")

    # ---- Gate 4: Combined trade rate ----
    s1_pred_full = clf_s1.predict(ft_test_z)
    s2_pred_full_proba = clf_s2.predict_proba(ft_test_z)[:, 1]
    s2_pred_full = np.where(s2_pred_full_proba > 0.5, 1, -1)

    combined = np.zeros(len(labels_test), dtype=np.int64)
    s1_dir = (s1_pred_full == 1)
    combined[s1_dir] = s2_pred_full[s1_dir]

    trade_rate = float((combined != 0).sum() / len(combined))
    print(f"  [Gate 4] Combined trade rate: {trade_rate:.4f} ({100*trade_rate:.1f}%)")
    if trade_rate <= 0.05:
        print(f"  ** ABORT: Trade rate {trade_rate:.4f} <= 5%")
        results["gate4_trade_rate"] = False
        results["abort"] = True
        results["abort_reason"] = f"Combined trade rate {trade_rate:.4f} <= 5%"
        return results

    results["gate4_trade_rate"] = True
    results["gate4_combined_trade_rate"] = trade_rate
    print(f"  [Gate 4] PASS")

    results["abort"] = False
    print("\n  ** ALL MVE GATES PASSED **\n")

    del clf_s1, clf_s2
    return results


# ==========================================================================
# Main
# ==========================================================================
def main():
    t0_global = time.time()
    set_seed(SEED)
    print("=" * 70)
    print("2-Class Directional: Two-Stage Reachability + Direction Pipeline")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Load data for both geometries
    # ------------------------------------------------------------------
    print("\n[Step 0] Loading data...")
    data = {}
    for geom_key in ["19_7", "10_5"]:
        print(f"\n  Geometry {geom_key}:")
        features, labels, day_indices, unique_days = load_geometry_data(geom_key)
        data[geom_key] = {
            "features": features,
            "labels": labels,
            "day_indices": day_indices,
            "unique_days": unique_days,
        }

    # ------------------------------------------------------------------
    # Step 1: MVE Gates (19:7 only)
    # ------------------------------------------------------------------
    mve_results = run_mve(
        data["19_7"]["features"],
        data["19_7"]["labels"],
        data["19_7"]["day_indices"],
    )

    if mve_results.get("abort", False):
        print("\n** EXPERIMENT ABORTED AT MVE **")
        write_abort_metrics(mve_results, t0_global)
        return

    # ------------------------------------------------------------------
    # Step 2: Walk-Forward (both geometries)
    # ------------------------------------------------------------------
    all_wf_results = {}
    all_stage1_importance = {}
    all_stage2_importance = {}

    for geom_key in ["19_7", "10_5"]:
        print(f"\n{'=' * 70}")
        print(f"WALK-FORWARD: Geometry {GEOMETRIES[geom_key]['label']}")
        print(f"{'=' * 70}")

        wf_results = []
        s1_imp_accum = {}
        s2_imp_accum = {}

        for wf_idx, wf in enumerate(WF_FOLDS):
            # Check wall-clock
            elapsed = time.time() - t0_global
            if elapsed > WALL_CLOCK_LIMIT_S:
                print(f"\n** WALL-CLOCK ABORT: {elapsed:.0f}s > {WALL_CLOCK_LIMIT_S}s **")
                break

            print(f"\n  --- {wf['name']} (train {wf['train_range']}, test {wf['test_range']}) ---")

            fold_result = train_two_stage_fold(
                data[geom_key]["features"],
                data[geom_key]["labels"],
                data[geom_key]["day_indices"],
                wf["train_range"],
                wf["test_range"],
                geom_key,
                wf["name"],
            )

            if fold_result is None:
                print(f"  !! Fold skipped (empty split)")
                continue

            wf_results.append(fold_result)

            # Accumulate feature importance
            for feat, gain in fold_result["stage1_importance"].items():
                s1_imp_accum.setdefault(feat, []).append(gain)
            for feat, gain in fold_result["stage2_importance"].items():
                s2_imp_accum.setdefault(feat, []).append(gain)

            # Print fold summary
            print(f"  S1 acc={fold_result['stage1_accuracy']:.4f}, "
                  f"S1 prec={fold_result['stage1_precision_directional']:.4f}, "
                  f"S1 recall={fold_result['stage1_recall_directional']:.4f}")
            print(f"  S2 acc(all dir)={fold_result['stage2_acc_all_directional']:.4f}, "
                  f"S2 acc(filtered)={fold_result['stage2_acc_filtered_directional']:.4f}")
            print(f"  Combined: trade_rate={fold_result['trade_rate']:.4f}, "
                  f"dir_acc={fold_result['directional_accuracy']:.4f}, "
                  f"label0_hit={fold_result['label0_hit_rate']:.4f}")
            print(f"  Expectancy(base): ${fold_result['pnl']['base']['expectancy']:.4f}/trade, "
                  f"n_trades={fold_result['pnl']['base']['n_trades']}")

        all_wf_results[geom_key] = wf_results

        # Average feature importance
        s1_mean_imp = {f: float(np.mean(v)) for f, v in s1_imp_accum.items()}
        s2_mean_imp = {f: float(np.mean(v)) for f, v in s2_imp_accum.items()}
        all_stage1_importance[geom_key] = sorted(s1_mean_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        all_stage2_importance[geom_key] = sorted(s2_mean_imp.items(), key=lambda x: x[1], reverse=True)[:10]

    # ------------------------------------------------------------------
    # Step 3: Aggregate and compare
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("AGGREGATION AND COMPARISON")
    print(f"{'=' * 70}")

    metrics = build_metrics(all_wf_results, all_stage1_importance, all_stage2_importance,
                            mve_results, t0_global)

    # ------------------------------------------------------------------
    # Step 4: Write outputs
    # ------------------------------------------------------------------
    write_metrics(metrics)
    write_analysis(metrics, all_wf_results, all_stage1_importance, all_stage2_importance)
    write_walkforward_csv(all_wf_results)
    write_feature_importance_csv(all_stage1_importance, all_stage2_importance)
    write_cost_sensitivity_csv(all_wf_results)
    write_config()

    elapsed_total = time.time() - t0_global
    print(f"\n{'=' * 70}")
    print(f"COMPLETE — {elapsed_total:.1f}s wall clock")
    print(f"{'=' * 70}")


# ==========================================================================
# Metrics Builder
# ==========================================================================
def build_metrics(all_wf_results, s1_importance, s2_importance, mve_results, t0):
    """Build the complete metrics.json structure."""
    metrics = {
        "experiment": "2class-directional",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Per-geometry aggregation
    for geom_key in ["19_7", "10_5"]:
        geom = GEOMETRIES[geom_key]
        wf = all_wf_results.get(geom_key, [])
        if not wf:
            continue

        # Aggregate walk-forward
        exps_base = [f["pnl"]["base"]["expectancy"] for f in wf]
        dir_accs = [f["directional_accuracy"] for f in wf]
        trade_rates = [f["trade_rate"] for f in wf]
        l0_hits = [f["label0_hit_rate"] for f in wf]
        s1_accs = [f["stage1_accuracy"] for f in wf]
        s1_precs = [f["stage1_precision_directional"] for f in wf]
        s1_recalls = [f["stage1_recall_directional"] for f in wf]
        s2_all_accs = [f["stage2_acc_all_directional"] for f in wf]
        s2_filt_accs = [f["stage2_acc_filtered_directional"] for f in wf]
        daily_pnls = [f["pnl"]["base"]["daily_pnl"] for f in wf]
        selection_bias_deltas = [f["selection_bias_delta"] for f in wf]

        # Cost sensitivity
        cost_sensitivity = {}
        for scenario in COST_SCENARIOS:
            cs_exps = [f["pnl"][scenario]["expectancy"] for f in wf]
            cost_sensitivity[scenario] = {
                "mean_expectancy": float(np.mean(cs_exps)),
                "per_fold": cs_exps,
            }

        # Per-fold details
        per_fold = []
        for f in wf:
            per_fold.append({
                "fold": f["fold_name"],
                "n_trades": f["pnl"]["base"]["n_trades"],
                "expectancy_base": f["pnl"]["base"]["expectancy"],
                "directional_accuracy": f["directional_accuracy"],
                "trade_rate": f["trade_rate"],
                "label0_hit_rate": f["label0_hit_rate"],
                "stage1_accuracy": f["stage1_accuracy"],
                "stage1_precision": f["stage1_precision_directional"],
                "stage1_recall": f["stage1_recall_directional"],
                "stage2_acc_all_dir": f["stage2_acc_all_directional"],
                "stage2_acc_filtered": f["stage2_acc_filtered_directional"],
                "selection_bias_delta": f["selection_bias_delta"],
            })

        # 3-class baseline comparison
        baseline = THREE_CLASS_BASELINES.get(geom_key, {})
        delta_exp = float(np.mean(exps_base)) - baseline.get("wf_mean_expectancy_base", 0)
        delta_trade_rate = float(np.mean(trade_rates)) - baseline.get("wf_mean_dir_pred_rate", 0)
        delta_dir_acc = float(np.mean(dir_accs)) - baseline.get("wf_mean_dir_accuracy", 0)

        metrics[geom_key] = {
            "geometry": geom["label"],
            "target": geom["target"],
            "stop": geom["stop"],
            "bev_wr": geom["bev_wr"],
            # Primary metrics
            "two_stage_wf_expectancy": float(np.mean(exps_base)),
            "two_stage_wf_expectancy_std": float(np.std(exps_base)),
            "two_stage_dir_accuracy": float(np.mean(dir_accs)),
            "two_stage_dir_accuracy_std": float(np.std(dir_accs)),
            "two_stage_trade_rate": float(np.mean(trade_rates)),
            "two_stage_trade_rate_std": float(np.std(trade_rates)),
            "two_stage_label0_hit_rate": float(np.mean(l0_hits)),
            "two_stage_daily_pnl": float(np.mean(daily_pnls)),
            # Stage 1
            "stage1_binary_accuracy": float(np.mean(s1_accs)),
            "stage1_precision_directional": float(np.mean(s1_precs)),
            "stage1_recall_directional": float(np.mean(s1_recalls)),
            # Stage 2
            "stage2_binary_accuracy": float(np.mean(s2_all_accs)),
            "stage2_dir_acc_on_filtered": float(np.mean(s2_filt_accs)),
            "selection_bias_delta": float(np.mean(selection_bias_deltas)),
            # Feature importance
            "stage1_feature_importance": s1_importance.get(geom_key, []),
            "stage2_feature_importance": s2_importance.get(geom_key, []),
            # Cost sensitivity
            "cost_sensitivity": cost_sensitivity,
            # Per-fold
            "per_fold": per_fold,
            # Comparison to 3-class
            "comparison_3class_delta": {
                "expectancy_delta": delta_exp,
                "trade_rate_delta": delta_trade_rate,
                "dir_accuracy_delta": delta_dir_acc,
                "three_class_expectancy": baseline.get("wf_mean_expectancy_base", None),
                "three_class_trade_rate": baseline.get("wf_mean_dir_pred_rate", None),
                "three_class_dir_accuracy": baseline.get("wf_mean_dir_accuracy", None),
                "three_class_label0_hit_rate": baseline.get("wf_mean_label0_hit_rate", None),
            },
        }

    # Success criteria evaluation
    m_19_7 = metrics.get("19_7", {})
    sc1 = m_19_7.get("two_stage_wf_expectancy", -999) > 0.0
    sc2 = m_19_7.get("two_stage_dir_accuracy", 0) > 0.45
    sc3 = m_19_7.get("two_stage_trade_rate", 0) > 0.10
    sc4 = m_19_7.get("two_stage_label0_hit_rate", 1.0) < 0.50

    # Sanity checks
    sc_s1_19_7 = m_19_7.get("stage1_binary_accuracy", 0) > 0.526  # majority baseline 19:7
    m_10_5 = metrics.get("10_5", {})
    sc_s1_10_5 = m_10_5.get("stage1_binary_accuracy", 0) > 0.674  # majority baseline 10:5
    sc_s2 = m_10_5.get("stage2_binary_accuracy", 0) > 0.50
    sc_s3 = all(f["pnl"]["base"]["n_trades"] > 100 for f in all_wf_results.get("19_7", []))

    metrics["success_criteria"] = {
        "SC-1": {"description": "WF expectancy > $0.00 at 19:7 (base)", "pass": sc1,
                 "value": m_19_7.get("two_stage_wf_expectancy")},
        "SC-2": {"description": "Dir accuracy > 45% at 19:7", "pass": sc2,
                 "value": m_19_7.get("two_stage_dir_accuracy")},
        "SC-3": {"description": "Trade rate > 10% at 19:7", "pass": sc3,
                 "value": m_19_7.get("two_stage_trade_rate")},
        "SC-4": {"description": "label0_hit_rate < 50% at 19:7", "pass": sc4,
                 "value": m_19_7.get("two_stage_label0_hit_rate")},
    }

    metrics["sanity_checks"] = {
        "SC-S1_19_7": {"description": "Stage 1 acc > 52.6% at 19:7", "pass": sc_s1_19_7,
                       "value": m_19_7.get("stage1_binary_accuracy")},
        "SC-S1_10_5": {"description": "Stage 1 acc > 67.4% at 10:5", "pass": sc_s1_10_5,
                       "value": m_10_5.get("stage1_binary_accuracy")},
        "SC-S2": {"description": "Stage 2 acc > 50% at 10:5", "pass": sc_s2,
                  "value": m_10_5.get("stage2_binary_accuracy")},
        "SC-S3": {"description": "Per-fold trade count > 100 at 19:7", "pass": sc_s3},
    }

    # Determine outcome
    if sc1 and sc3:
        outcome = "A"
        outcome_desc = "CONFIRMED — positive expectancy + meaningful trade volume at 19:7"
    elif sc2 and not sc1:
        outcome = "B"
        outcome_desc = "PARTIAL — direction signal exists but economics negative"
    elif not sc2:
        outcome = "C"
        outcome_desc = "REFUTED — directional accuracy < 45% at 19:7"
    elif not sc3:
        outcome = "D"
        outcome_desc = "REFUTED — trade rate < 10% despite two-stage"
    else:
        outcome = "B"  # SC-2 passes, SC-1 fails
        outcome_desc = "PARTIAL — direction signal exists but economics negative"

    metrics["outcome"] = outcome
    metrics["outcome_description"] = outcome_desc

    # MVE
    metrics["mve_gates"] = {
        "gate1_data": mve_results.get("gate1_data"),
        "gate2_s1_accuracy": mve_results.get("gate2_s1_accuracy"),
        "gate3_s2_accuracy": mve_results.get("gate3_s2_accuracy"),
        "gate4_combined_trade_rate": mve_results.get("gate4_combined_trade_rate"),
        "all_passed": not mve_results.get("abort", True),
    }

    # Resource usage
    elapsed = time.time() - t0
    total_runs = sum(len(v) * 2 for v in all_wf_results.values()) + 2  # +2 for MVE
    metrics["resource_usage"] = {
        "wall_clock_seconds": elapsed,
        "wall_clock_minutes": elapsed / 60,
        "total_training_runs": total_runs,
        "gpu_hours": 0,
        "total_runs": total_runs,
    }
    metrics["abort_triggered"] = False
    metrics["abort_reason"] = None
    metrics["notes"] = (
        f"Executed locally on Apple Silicon. "
        f"2 geometries: 19:7 (primary), 10:5 (control). "
        f"Walk-forward: 3 expanding-window folds. "
        f"Wall clock: {elapsed/60:.1f} minutes. "
        f"Data from label-geometry-1h Parquet (3600s horizon). "
        f"Stage 1: binary:logistic (directional vs hold). "
        f"Stage 2: binary:logistic (long vs short, directional-only training). "
        f"COMPUTE_TARGET: local."
    )

    return metrics


def write_abort_metrics(mve_results, t0):
    """Write metrics.json for an aborted experiment."""
    elapsed = time.time() - t0
    metrics = {
        "experiment": "2class-directional",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "abort_triggered": True,
        "abort_reason": mve_results.get("abort_reason", "Unknown"),
        "mve_gates": mve_results,
        "resource_usage": {
            "wall_clock_seconds": elapsed,
            "wall_clock_minutes": elapsed / 60,
            "total_training_runs": 2,
            "gpu_hours": 0,
        },
        "notes": "Aborted at MVE stage.",
    }
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Abort metrics written to {out_path}")


# ==========================================================================
# Output Writers
# ==========================================================================
def write_metrics(metrics):
    out_path = RESULTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  metrics.json written to {out_path}")


def write_analysis(metrics, all_wf_results, s1_imp, s2_imp):
    """Write analysis.md with comparison tables and verdict."""
    m19 = metrics.get("19_7", {})
    m10 = metrics.get("10_5", {})
    sc = metrics.get("success_criteria", {})
    sanity = metrics.get("sanity_checks", {})

    lines = []
    lines.append("# 2-Class Directional — Analysis\n")
    lines.append(f"**Date:** {metrics['timestamp']}")
    lines.append(f"**Outcome:** {metrics['outcome']} — {metrics['outcome_description']}\n")

    # Executive summary
    lines.append("## Executive Summary\n")
    exp_19 = m19.get("two_stage_wf_expectancy", 0)
    tr_19 = m19.get("two_stage_trade_rate", 0)
    da_19 = m19.get("two_stage_dir_accuracy", 0)
    l0_19 = m19.get("two_stage_label0_hit_rate", 0)
    lines.append(
        f"Two-stage XGBoost at 19:7 achieves {100*tr_19:.1f}% trade rate "
        f"(vs 3-class 0.28%), directional accuracy {100*da_19:.1f}%, "
        f"and per-trade expectancy ${exp_19:.4f} (base costs $3.74 RT). "
        f"label0_hit_rate = {100*l0_19:.1f}%.\n"
    )

    # Comparison table
    lines.append("## Two-Stage vs 3-Class Comparison\n")
    lines.append("| Metric | 3-Class (19:7) | Two-Stage (19:7) | Delta | 3-Class (10:5) | Two-Stage (10:5) | Delta |")
    lines.append("|--------|---------------|-----------------|-------|---------------|-----------------|-------|")

    c19 = m19.get("comparison_3class_delta", {})
    c10 = m10.get("comparison_3class_delta", {})

    def fmt(v, pct=False, dollar=False):
        if v is None:
            return "N/A"
        if dollar:
            return f"${v:.4f}"
        if pct:
            return f"{100*v:.2f}%"
        return f"{v:.4f}"

    lines.append(f"| Trade rate | {fmt(c19.get('three_class_trade_rate'), pct=True)} | {fmt(m19.get('two_stage_trade_rate'), pct=True)} | {fmt(c19.get('trade_rate_delta'), pct=True)} | {fmt(c10.get('three_class_trade_rate'), pct=True)} | {fmt(m10.get('two_stage_trade_rate'), pct=True)} | {fmt(c10.get('trade_rate_delta'), pct=True)} |")
    lines.append(f"| Dir accuracy | {fmt(c19.get('three_class_dir_accuracy'), pct=True)} | {fmt(m19.get('two_stage_dir_accuracy'), pct=True)} | {fmt(c19.get('dir_accuracy_delta'), pct=True)} | {fmt(c10.get('three_class_dir_accuracy'), pct=True)} | {fmt(m10.get('two_stage_dir_accuracy'), pct=True)} | {fmt(c10.get('dir_accuracy_delta'), pct=True)} |")
    lines.append(f"| Expectancy (base) | {fmt(c19.get('three_class_expectancy'), dollar=True)} | {fmt(m19.get('two_stage_wf_expectancy'), dollar=True)} | {fmt(c19.get('expectancy_delta'), dollar=True)} | {fmt(c10.get('three_class_expectancy'), dollar=True)} | {fmt(m10.get('two_stage_wf_expectancy'), dollar=True)} | {fmt(c10.get('expectancy_delta'), dollar=True)} |")
    lines.append(f"| label0_hit_rate | {fmt(c19.get('three_class_label0_hit_rate'), pct=True)} | {fmt(m19.get('two_stage_label0_hit_rate'), pct=True)} | — | {fmt(c10.get('three_class_label0_hit_rate'), pct=True)} | {fmt(m10.get('two_stage_label0_hit_rate'), pct=True)} | — |")
    lines.append(f"| Daily PnL | — | {fmt(m19.get('two_stage_daily_pnl'), dollar=True)} | — | — | {fmt(m10.get('two_stage_daily_pnl'), dollar=True)} | — |")
    lines.append("")

    # Per-stage accuracy
    lines.append("## Per-Stage Accuracy Breakdown\n")
    lines.append("| Metric | 19:7 | 10:5 |")
    lines.append("|--------|------|------|")
    lines.append(f"| Stage 1 accuracy | {fmt(m19.get('stage1_binary_accuracy'), pct=True)} | {fmt(m10.get('stage1_binary_accuracy'), pct=True)} |")
    lines.append(f"| Stage 1 precision (directional) | {fmt(m19.get('stage1_precision_directional'), pct=True)} | {fmt(m10.get('stage1_precision_directional'), pct=True)} |")
    lines.append(f"| Stage 1 recall (directional) | {fmt(m19.get('stage1_recall_directional'), pct=True)} | {fmt(m10.get('stage1_recall_directional'), pct=True)} |")
    lines.append(f"| Stage 2 accuracy (all dir) | {fmt(m19.get('stage2_binary_accuracy'), pct=True)} | {fmt(m10.get('stage2_binary_accuracy'), pct=True)} |")
    lines.append(f"| Stage 2 accuracy (S1-filtered) | {fmt(m19.get('stage2_dir_acc_on_filtered'), pct=True)} | {fmt(m10.get('stage2_dir_acc_on_filtered'), pct=True)} |")
    lines.append(f"| Selection bias (all - filtered) | {fmt(m19.get('selection_bias_delta'), pct=True)} | {fmt(m10.get('selection_bias_delta'), pct=True)} |")
    lines.append("")

    # Selection bias check
    sb19 = abs(m19.get("selection_bias_delta", 0))
    sb10 = abs(m10.get("selection_bias_delta", 0))
    if sb19 > 0.03 or sb10 > 0.03:
        lines.append(f"**Selection bias flag:** |delta| > 3pp detected (19:7={100*sb19:.1f}pp, 10:5={100*sb10:.1f}pp).\n")
    else:
        lines.append(f"Selection bias within tolerance (19:7={100*sb19:.1f}pp, 10:5={100*sb10:.1f}pp < 3pp).\n")

    # Feature importance
    lines.append("## Feature Importance Decomposition\n")
    for geom_key, geom_label in [("19_7", "19:7"), ("10_5", "10:5")]:
        lines.append(f"### {geom_label}\n")
        lines.append("| Rank | Stage 1 (Reachability) | Gain | Stage 2 (Direction) | Gain |")
        lines.append("|------|----------------------|------|---------------------|------|")
        s1 = s1_imp.get(geom_key, [])
        s2 = s2_imp.get(geom_key, [])
        for i in range(10):
            s1_name = s1[i][0] if i < len(s1) else ""
            s1_gain = f"{s1[i][1]:.1f}" if i < len(s1) else ""
            s2_name = s2[i][0] if i < len(s2) else ""
            s2_gain = f"{s2[i][1]:.1f}" if i < len(s2) else ""
            lines.append(f"| {i+1} | {s1_name} | {s1_gain} | {s2_name} | {s2_gain} |")
        lines.append("")

    # Per-fold consistency
    lines.append("## Per-Fold Consistency\n")
    for geom_key, geom_label in [("19_7", "19:7"), ("10_5", "10:5")]:
        lines.append(f"### {geom_label}\n")
        lines.append("| Fold | Trades | Exp (base) | Dir Acc | Trade Rate | label0_hit | S1 Acc | S2 Acc (all) |")
        lines.append("|------|--------|------------|---------|------------|------------|--------|-------------|")
        for pf in metrics.get(geom_key, {}).get("per_fold", []):
            lines.append(
                f"| {pf['fold']} | {pf['n_trades']} | ${pf['expectancy_base']:.4f} | "
                f"{100*pf['directional_accuracy']:.1f}% | {100*pf['trade_rate']:.1f}% | "
                f"{100*pf['label0_hit_rate']:.1f}% | {100*pf['stage1_accuracy']:.1f}% | "
                f"{100*pf['stage2_acc_all_dir']:.1f}% |"
            )
        lines.append("")

    # Cost sensitivity
    lines.append("## Cost Sensitivity\n")
    lines.append("| Scenario | RT Cost | 19:7 Expectancy | 10:5 Expectancy |")
    lines.append("|----------|---------|----------------|----------------|")
    for scenario, rt_cost in COST_SCENARIOS.items():
        cs19 = m19.get("cost_sensitivity", {}).get(scenario, {}).get("mean_expectancy", 0)
        cs10 = m10.get("cost_sensitivity", {}).get(scenario, {}).get("mean_expectancy", 0)
        lines.append(f"| {scenario.capitalize()} | ${rt_cost:.2f} | ${cs19:.4f} | ${cs10:.4f} |")
    lines.append("")

    # Success criteria
    lines.append("## Success Criteria\n")
    for sc_key, sc_val in sc.items():
        status = "PASS" if sc_val["pass"] else "FAIL"
        lines.append(f"- **{sc_key}**: {sc_val['description']} — **{status}** (value: {sc_val.get('value')})")
    lines.append("")

    lines.append("## Sanity Checks\n")
    for sk, sv in sanity.items():
        status = "PASS" if sv["pass"] else "FAIL"
        val_str = f" (value: {sv.get('value')})" if sv.get("value") is not None else ""
        lines.append(f"- **{sk}**: {sv['description']} — **{status}**{val_str}")
    lines.append("")

    # Outcome verdict
    lines.append("## Outcome Verdict\n")
    lines.append(f"**{metrics['outcome']}** — {metrics['outcome_description']}\n")

    # Trade rate comparison headline
    lines.append("## Trade Rate Headline\n")
    three_class_tr = c19.get("three_class_trade_rate", 0)
    lines.append(f"3-class: {100*(three_class_tr or 0):.2f}% → Two-stage: {100*tr_19:.1f}% at 19:7")
    lines.append(f"(x{tr_19/(three_class_tr or 0.001):.0f} increase)\n")

    out_path = RESULTS_DIR / "analysis.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  analysis.md written to {out_path}")


def write_walkforward_csv(all_wf_results):
    """Write per-fold × per-geometry CSV."""
    out_path = RESULTS_DIR / "walkforward_results.csv"
    rows = []
    for geom_key, wf_list in all_wf_results.items():
        for f in wf_list:
            rows.append({
                "geometry": geom_key,
                "fold": f["fold_name"],
                "train_range": f"{f['train_range'][0]}-{f['train_range'][1]}",
                "test_range": f"{f['test_range'][0]}-{f['test_range'][1]}",
                "n_train": f["n_train"],
                "n_test": f["n_test"],
                "stage1_accuracy": f["stage1_accuracy"],
                "stage1_precision": f["stage1_precision_directional"],
                "stage1_recall": f["stage1_recall_directional"],
                "stage2_acc_all_dir": f["stage2_acc_all_directional"],
                "stage2_acc_filtered": f["stage2_acc_filtered_directional"],
                "trade_rate": f["trade_rate"],
                "directional_accuracy": f["directional_accuracy"],
                "label0_hit_rate": f["label0_hit_rate"],
                "expectancy_optimistic": f["pnl"]["optimistic"]["expectancy"],
                "expectancy_base": f["pnl"]["base"]["expectancy"],
                "expectancy_pessimistic": f["pnl"]["pessimistic"]["expectancy"],
                "n_trades": f["pnl"]["base"]["n_trades"],
                "daily_pnl_base": f["pnl"]["base"]["daily_pnl"],
                "selection_bias_delta": f["selection_bias_delta"],
                "stage1_n_estimators": f["stage1_n_estimators"],
                "stage2_n_estimators": f["stage2_n_estimators"],
                "stage1_elapsed_s": f["stage1_elapsed_s"],
                "stage2_elapsed_s": f["stage2_elapsed_s"],
            })

    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  walkforward_results.csv written to {out_path}")


def write_feature_importance_csv(s1_imp, s2_imp):
    """Write feature importance CSVs."""
    for stage, imp, filename in [
        ("stage1", s1_imp, "stage1_feature_importance.csv"),
        ("stage2", s2_imp, "stage2_feature_importance.csv"),
    ]:
        out_path = RESULTS_DIR / filename
        rows = []
        for geom_key, top10 in imp.items():
            for rank, (feat, gain) in enumerate(top10, 1):
                rows.append({
                    "geometry": geom_key,
                    "rank": rank,
                    "feature": feat,
                    "mean_gain": gain,
                })
        if rows:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        print(f"  {filename} written to {out_path}")


def write_cost_sensitivity_csv(all_wf_results):
    """Write cost sensitivity CSV."""
    out_path = RESULTS_DIR / "cost_sensitivity.csv"
    rows = []
    for geom_key, wf_list in all_wf_results.items():
        for scenario, rt_cost in COST_SCENARIOS.items():
            exps = [f["pnl"][scenario]["expectancy"] for f in wf_list]
            mean_exp = float(np.mean(exps)) if exps else 0
            rows.append({
                "geometry": geom_key,
                "scenario": scenario,
                "rt_cost": rt_cost,
                "mean_expectancy": mean_exp,
                "per_fold_expectancy": json.dumps(exps),
            })
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"  cost_sensitivity.csv written to {out_path}")


def write_config():
    """Write config.json for reproducibility."""
    config = {
        "seed": SEED,
        "xgb_params_binary": TUNED_XGB_PARAMS_BINARY,
        "features": NON_SPATIAL_FEATURES,
        "cost_scenarios": COST_SCENARIOS,
        "tick_value": TICK_VALUE,
        "geometries": {k: {"target": v["target"], "stop": v["stop"]} for k, v in GEOMETRIES.items()},
        "wf_folds": WF_FOLDS,
        "purge_bars": PURGE_BARS,
        "stage1_threshold": 0.5,
        "stage1_objective": "binary:logistic",
        "stage2_objective": "binary:logistic",
        "stage2_training_filter": "directional-only (tb_label != 0)",
        "data_source": "label-geometry-1h Parquet (3600s horizon)",
    }
    out_path = RESULTS_DIR / "config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config.json written to {out_path}")


if __name__ == "__main__":
    main()
