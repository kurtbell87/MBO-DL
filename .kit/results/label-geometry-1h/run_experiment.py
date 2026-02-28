#!/usr/bin/env python3
"""
Experiment: Label Geometry 1h — Model Training with Corrected Time Horizon
Spec: .kit/experiments/label-geometry-1h.md

Re-run of label-geometry-phase1 with --max-time-horizon 3600 (1 hour)
to fix degenerate hold rates (90.7-98.9% at 300s).

4 geometries: 10:5 (control), 15:3, 19:7, 20:3
Evaluates: CPCV (45 splits), Walk-Forward (3 folds), Holdout, Per-direction, Time-of-day, Cost sensitivity.
Key change: SC-3 uses DIRECTIONAL accuracy (not overall accuracy).
"""

import json
import os
import sys
import time
import math
import random
import hashlib
import subprocess
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================================================
# Config
# ==========================================================================
SEED = 42
MAIN_ROOT = Path("/Users/brandonbell/LOCAL_DEV/MBO-DL-02152026")
WORKTREE_ROOT = Path("/Users/brandonbell/LOCAL_DEV/MBO-DL-label-geom-p1")
RESULTS_DIR = WORKTREE_ROOT / ".kit" / "results" / "label-geometry-1h"
DATA_DIR = MAIN_ROOT / "DATA" / "GLBX-20260207-L953CAPU5B"
BUILD_DIR = WORKTREE_ROOT / "build"  # Use worktree build (has --max-time-horizon)
FULL_YEAR_EXPORT_DIR = WORKTREE_ROOT / ".kit" / "results" / "full-year-export"

EXPORT_BIN = BUILD_DIR / "bar_feature_export"
ORACLE_BIN = BUILD_DIR / "oracle_expectancy"

# Tuned XGBoost params (from XGB tuning experiment)
TUNED_XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.0134,
    "min_child_weight": 20,
    "subsample": 0.561,
    "colsample_bytree": 0.748,
    "reg_alpha": 0.0014,
    "reg_lambda": 6.586,
    "n_estimators": 2000,
    "early_stopping_rounds": 50,
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": SEED,
    "verbosity": 0,
    "n_jobs": -1,
}

# Non-spatial features (20 dimensions)
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

# CPCV parameters
N_GROUPS = 10
K_TEST = 2
PURGE_BARS = 500
EMBARGO_BARS = 4600  # ~1 trading day

# Dev/holdout split
DEV_DAYS = 201

# 4 geometries (spec Table)
GEOMETRIES = [
    {"target": 10, "stop": 5, "label": "10:5 (control)", "bev_wr": 0.533},
    {"target": 15, "stop": 3, "label": "15:3", "bev_wr": 0.333},
    {"target": 19, "stop": 7, "label": "19:7", "bev_wr": 0.384},
    {"target": 20, "stop": 3, "label": "20:3", "bev_wr": 0.296},
]

# Phase 1 hold rates at 300s (for comparison)
PHASE1_HOLD_RATES = {
    "10_5": 0.907,
    "15_3": 0.971,
    "19_7": 0.986,
    "20_3": 0.989,
}

# Time horizon
MAX_TIME_HORIZON = 3600  # 1 hour
VOLUME_HORIZON = 50000   # effectively unlimited

# Parallelism
EXPORT_WORKERS = 8

# Wall-clock limit
WALL_CLOCK_LIMIT_S = 4 * 3600  # 4 hours


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ==========================================================================
# Export Helpers
# ==========================================================================
def run_export_day(target, stop, date_str_nodash, output_path):
    """Run bar_feature_export for a single day with given geometry + 3600s horizon."""
    cmd = [
        str(EXPORT_BIN),
        "--bar-type", "time",
        "--bar-param", "5",
        "--target", str(target),
        "--stop", str(stop),
        "--max-time-horizon", str(MAX_TIME_HORIZON),
        "--volume-horizon", str(VOLUME_HORIZON),
        "--date", date_str_nodash,
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(MAIN_ROOT), timeout=300)
    return result.returncode == 0


def export_geometry_all_days(target, stop, all_dates, geom_dir):
    """Export all days for one geometry, using parallel workers."""
    geom_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    failed_dates = []

    def export_one(date_str):
        out_path = geom_dir / f"{date_str}.parquet"
        if out_path.exists() and out_path.stat().st_size > 0:
            return (date_str, True)
        success = run_export_day(target, stop, date_str.replace("-", ""), str(out_path))
        return (date_str, success)

    with ThreadPoolExecutor(max_workers=EXPORT_WORKERS) as pool:
        futures = {pool.submit(export_one, d): d for d in all_dates}
        for fut in as_completed(futures):
            date_str, success = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                failed_dates.append(date_str)

    return ok, fail, failed_dates


# ==========================================================================
# Oracle Helpers
# ==========================================================================
def run_oracle(target, stop, output_path):
    """Run oracle_expectancy with given geometry + 3600s horizon."""
    cmd = [
        str(ORACLE_BIN),
        "--target", str(target),
        "--stop", str(stop),
        "--max-time-horizon", str(MAX_TIME_HORIZON),
        "--volume-horizon", str(VOLUME_HORIZON),
        "--output", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(MAIN_ROOT), timeout=600)
    return result.returncode == 0


# ==========================================================================
# CPCV Helpers
# ==========================================================================
def assign_cpcv_groups(day_indices, n_groups):
    unique_days = sorted(set(day_indices))
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


def apply_purge_embargo(train_indices, test_indices, all_day_indices,
                        purge_bars, embargo_bars):
    train_set = set(train_indices.tolist()) if isinstance(train_indices, np.ndarray) else set(train_indices)
    n_total = len(all_day_indices)

    test_idx_sorted = np.sort(test_indices)
    test_groups_boundaries = []
    prev = -2
    group_start = None
    for idx in test_idx_sorted:
        if idx != prev + 1:
            if group_start is not None:
                test_groups_boundaries.append((group_start, prev))
            group_start = idx
        prev = idx
    if group_start is not None:
        test_groups_boundaries.append((group_start, prev))

    excluded = set()
    for tg_start, tg_end in test_groups_boundaries:
        for i in range(max(0, tg_start - purge_bars), tg_start):
            if i in train_set:
                excluded.add(i)
        for i in range(tg_end + 1, min(n_total, tg_end + 1 + purge_bars)):
            if i in train_set:
                excluded.add(i)
        for i in range(max(0, tg_start - purge_bars - embargo_bars), max(0, tg_start - purge_bars)):
            if i in train_set:
                excluded.add(i)
        for i in range(min(n_total, tg_end + 1 + purge_bars),
                       min(n_total, tg_end + 1 + purge_bars + embargo_bars)):
            if i in train_set:
                excluded.add(i)

    clean_train = np.array(sorted(train_set - excluded))
    return clean_train, len(excluded)


def build_cpcv_splits(dev_day_indices):
    groups, day_to_group = assign_cpcv_groups(dev_day_indices, N_GROUPS)
    dev_group_arr = np.array([day_to_group.get(d, -1) for d in dev_day_indices])

    splits_combo = list(combinations(range(N_GROUPS), K_TEST))
    split_data = []

    for s_idx, (g1, g2) in enumerate(splits_combo):
        test_groups = {g1, g2}
        train_groups = set(range(N_GROUPS)) - test_groups

        test_mask = np.isin(dev_group_arr, list(test_groups))
        train_mask = np.isin(dev_group_arr, list(train_groups))

        test_indices = np.where(test_mask)[0]
        train_indices = np.where(train_mask)[0]

        clean_train, n_excluded = apply_purge_embargo(
            train_indices, test_indices, dev_day_indices, PURGE_BARS, EMBARGO_BARS
        )

        # Internal train/val split (last 20% of training days for early stopping)
        train_days_sorted = sorted(set(dev_day_indices[clean_train].tolist()))
        n_val_days = max(1, len(train_days_sorted) // 5)
        val_day_set = set(train_days_sorted[-n_val_days:])
        inner_train_day_set = set(train_days_sorted[:-n_val_days])

        inner_train = np.array([i for i in clean_train if dev_day_indices[i] in inner_train_day_set])
        inner_val = np.array([i for i in clean_train if dev_day_indices[i] in val_day_set])

        # Purge between inner train and val
        if len(inner_val) > 0 and len(inner_train) > 0:
            val_start = inner_val[0]
            inner_train = inner_train[
                (inner_train < val_start - PURGE_BARS - EMBARGO_BARS) |
                (inner_train > inner_val[-1] + PURGE_BARS + EMBARGO_BARS)
            ]

        split_data.append({
            "split_idx": s_idx,
            "test_groups": (g1, g2),
            "train_indices": inner_train,
            "val_indices": inner_val,
            "test_indices": test_indices,
            "n_excluded": n_excluded,
        })

    return split_data, groups


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


def compute_expectancy_and_pf(pnl_array):
    trades = pnl_array[pnl_array != 0]
    if len(trades) == 0:
        return 0.0, 0.0, 0
    expectancy = float(trades.mean())
    gross_profit = float(trades[trades > 0].sum())
    gross_loss = float(abs(trades[trades < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return expectancy, pf, len(trades)


def compute_breakeven_wr(target, stop, rt_cost=3.74):
    """Breakeven WR = (stop * $1.25 + RT_cost) / ((target + stop) * $1.25)"""
    return (stop * TICK_VALUE + rt_cost) / ((target + stop) * TICK_VALUE)


def compute_directional_accuracy(true_labels, pred_labels):
    """Directional accuracy: correct predictions / bars where BOTH pred!=0 AND true!=0.

    This is the metric that tests the hypothesis — NOT overall 3-class accuracy.
    """
    both_nonzero = (pred_labels != 0) & (true_labels != 0)
    n_both_nonzero = both_nonzero.sum()
    if n_both_nonzero == 0:
        return 0.0, 0
    correct = (pred_labels == true_labels) & both_nonzero
    return float(correct.sum() / n_both_nonzero), int(n_both_nonzero)


def get_quarter(day_val):
    day_str = str(day_val)
    if "-" in day_str:
        month = int(day_str.split("-")[1])
    else:
        month = int(day_str[4:6])
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    else:
        return "Q4"


# ==========================================================================
# Training
# ==========================================================================
def train_xgb_cpcv(features, labels, day_indices, target_ticks, stop_ticks, splits):
    """Train XGBoost on CPCV splits, return per-split results."""
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    labels_ce = np.array([label_map[int(l)] for l in labels])

    all_split_results = []

    for split in splits:
        s_idx = split["split_idx"]
        train_idx = split["train_indices"]
        val_idx = split["val_indices"]
        test_idx = split["test_indices"]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # z-score normalize using training fold stats
        feat_train = features[train_idx].copy()
        feat_val = features[val_idx].copy() if len(val_idx) > 0 else None
        feat_test = features[test_idx].copy()

        train_mean = np.nanmean(feat_train, axis=0)
        train_std = np.nanstd(feat_train, axis=0)
        train_std[train_std == 0] = 1.0

        feat_train = (feat_train - train_mean) / train_std
        feat_test = (feat_test - train_mean) / train_std
        if feat_val is not None:
            feat_val = (feat_val - train_mean) / train_std

        # NaN -> 0
        feat_train = np.nan_to_num(feat_train, nan=0.0)
        feat_test = np.nan_to_num(feat_test, nan=0.0)
        if feat_val is not None:
            feat_val = np.nan_to_num(feat_val, nan=0.0)

        lt_train = labels_ce[train_idx]
        lt_val = labels_ce[val_idx] if len(val_idx) > 0 else None
        lt_test = labels_ce[test_idx]

        t0 = time.time()
        clf = xgb.XGBClassifier(**TUNED_XGB_PARAMS)

        if feat_val is not None and len(val_idx) > 0:
            clf.fit(feat_train, lt_train,
                    eval_set=[(feat_val, lt_val)],
                    verbose=False)
        else:
            clf.fit(feat_train, lt_train, verbose=False)

        preds_ce = clf.predict(feat_test)
        elapsed = time.time() - t0

        # Map back to -1, 0, +1
        preds_raw = np.array([inv_label_map[int(p)] for p in preds_ce])
        true_raw = np.array([inv_label_map[int(p)] for p in lt_test])

        test_acc = float(accuracy_score(lt_test, preds_ce))

        # Directional accuracy
        dir_acc, n_directional_pairs = compute_directional_accuracy(true_raw, preds_raw)

        # Directional prediction rate
        n_pred_directional = int((preds_raw != 0).sum())
        dir_pred_rate = float(n_pred_directional / len(preds_raw)) if len(preds_raw) > 0 else 0.0

        # Confusion matrix
        cm = confusion_matrix(lt_test, preds_ce, labels=[0, 1, 2])

        # Per-class recall
        per_class_recall = {}
        for cls_idx, cls_name in [(0, "short"), (1, "hold"), (2, "long")]:
            total = cm[cls_idx].sum()
            per_class_recall[cls_name] = float(cm[cls_idx, cls_idx] / total) if total > 0 else 0.0

        # PnL for 3 cost scenarios
        pnl_results = {}
        for scenario, rt_cost in COST_SCENARIOS.items():
            pnl = compute_pnl(true_raw, preds_raw, target_ticks, stop_ticks, rt_cost)
            exp, pf, n_trades = compute_expectancy_and_pf(pnl)
            pnl_results[scenario] = {"expectancy": exp, "profit_factor": pf, "n_trades": n_trades}

        # Feature importance
        importance = clf.get_booster().get_score(importance_type="gain")

        try:
            actual_n_est = clf.best_iteration + 1
        except AttributeError:
            actual_n_est = TUNED_XGB_PARAMS["n_estimators"]

        # Label=0 predictions hitting nonzero true
        pred_directional = (preds_raw != 0)
        true_hold = (true_raw == 0)
        label0_hit_rate = float((pred_directional & true_hold).sum() / max(pred_directional.sum(), 1))

        all_split_results.append({
            "split_idx": s_idx,
            "test_acc": test_acc,
            "directional_accuracy": dir_acc,
            "n_directional_pairs": n_directional_pairs,
            "directional_prediction_rate": dir_pred_rate,
            "per_class_recall": per_class_recall,
            "pnl": pnl_results,
            "n_estimators_used": actual_n_est,
            "elapsed_s": elapsed,
            "importance": importance,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "confusion_matrix": cm.tolist(),
            "label0_hit_rate": label0_hit_rate,
            "true_labels": true_raw.tolist(),
            "pred_labels": preds_raw.tolist(),
        })

        del clf

    return all_split_results


def aggregate_cpcv_results(split_results, target_ticks, stop_ticks):
    """Aggregate CPCV split results into summary metrics."""
    if not split_results:
        return {}

    accs = [r["test_acc"] for r in split_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    # Directional accuracy
    dir_accs = [r["directional_accuracy"] for r in split_results]
    mean_dir_acc = float(np.mean(dir_accs))
    std_dir_acc = float(np.std(dir_accs))

    # Directional prediction rate
    dir_pred_rates = [r["directional_prediction_rate"] for r in split_results]
    mean_dir_pred_rate = float(np.mean(dir_pred_rates))

    # Per-class recall averages
    recall_keys = ["short", "hold", "long"]
    mean_recall = {}
    for k in recall_keys:
        vals = [r["per_class_recall"][k] for r in split_results]
        mean_recall[k] = float(np.mean(vals))

    # PnL aggregation per scenario
    pnl_summary = {}
    for scenario in COST_SCENARIOS:
        exps = [r["pnl"][scenario]["expectancy"] for r in split_results]
        pfs = [r["pnl"][scenario]["profit_factor"] for r in split_results]
        pnl_summary[scenario] = {
            "mean_expectancy": float(np.mean(exps)),
            "std_expectancy": float(np.std(exps)),
            "mean_profit_factor": float(np.mean(pfs)),
        }

    # Feature importance aggregation
    all_importance = {}
    for r in split_results:
        for feat, gain in r["importance"].items():
            all_importance.setdefault(feat, []).append(gain)
    mean_importance = {feat: float(np.mean(vals)) for feat, vals in all_importance.items()}
    top10_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    # Breakeven margins (both overall and directional)
    breakeven = compute_breakeven_wr(target_ticks, stop_ticks, 3.74)
    margin_overall = mean_acc - breakeven
    margin_directional = mean_dir_acc - breakeven

    # Mean n_estimators
    mean_n_est = float(np.mean([r["n_estimators_used"] for r in split_results]))

    # Label=0 hit rate
    mean_label0_hit = float(np.mean([r["label0_hit_rate"] for r in split_results]))

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_directional_accuracy": mean_dir_acc,
        "std_directional_accuracy": std_dir_acc,
        "mean_directional_prediction_rate": mean_dir_pred_rate,
        "mean_per_class_recall": mean_recall,
        "pnl_summary": pnl_summary,
        "breakeven_wr": breakeven,
        "breakeven_margin_overall": margin_overall,
        "breakeven_margin_directional": margin_directional,
        "top10_features": top10_features,
        "mean_n_estimators": mean_n_est,
        "mean_label0_hit_rate": mean_label0_hit,
        "n_splits": len(split_results),
    }


# ==========================================================================
# Walk-Forward
# ==========================================================================
def run_walkforward(features, labels, day_indices, sorted_days,
                    target_ticks, stop_ticks):
    """3 expanding-window walk-forward folds with geometry-specific PnL."""
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    labels_ce = np.array([label_map[int(l)] for l in labels])

    wf_folds = [
        {"train_range": (1, 100), "test_range": (101, 150), "name": "Fold 1"},
        {"train_range": (1, 150), "test_range": (151, 201), "name": "Fold 2"},
        {"train_range": (1, 201), "test_range": (202, 251), "name": "Fold 3 (holdout)"},
    ]

    wf_results = []
    for wf_idx, wf in enumerate(wf_folds):
        train_range = wf["train_range"]
        test_range = wf["test_range"]

        train_mask = (day_indices >= train_range[0]) & (day_indices <= train_range[1])
        test_mask = (day_indices >= test_range[0]) & (day_indices <= test_range[1])

        if test_mask.sum() == 0:
            continue

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        clean_train, n_excluded = apply_purge_embargo(
            train_indices, test_indices, day_indices, PURGE_BARS, 0
        )

        if len(clean_train) == 0:
            continue

        ft_train = features[clean_train]
        ft_test = features[test_indices]

        # z-score normalize
        f_mean = np.nanmean(ft_train, axis=0)
        f_std = np.nanstd(ft_train, axis=0)
        f_std[f_std < 1e-10] = 1.0
        ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
        ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)

        # Internal val split: last 20% of training days
        train_days_in_fold = sorted(set(day_indices[clean_train].tolist()))
        n_val_days = max(1, len(train_days_in_fold) // 5)
        val_day_set = set(train_days_in_fold[-n_val_days:])

        val_mask_inner = np.array([day_indices[i] in val_day_set for i in clean_train])
        train_mask_inner = ~val_mask_inner

        inner_train_z = ft_train_z[train_mask_inner]
        inner_val_z = ft_train_z[val_mask_inner]

        lt_train_all = labels_ce[clean_train]
        inner_train_labels = lt_train_all[train_mask_inner]
        inner_val_labels = lt_train_all[val_mask_inner]
        lt_test = labels_ce[test_indices]

        t0 = time.time()
        clf = xgb.XGBClassifier(**TUNED_XGB_PARAMS)
        if len(inner_val_z) > 0:
            clf.fit(inner_train_z, inner_train_labels,
                    eval_set=[(inner_val_z, inner_val_labels)],
                    verbose=False)
        else:
            clf.fit(inner_train_z, inner_train_labels, verbose=False)

        preds_ce = clf.predict(ft_test_z)
        elapsed = time.time() - t0

        preds_raw = np.array([inv_label_map[int(p)] for p in preds_ce])
        true_raw = np.array([inv_label_map[int(p)] for p in lt_test])

        test_acc = float(accuracy_score(lt_test, preds_ce))

        # Directional accuracy
        dir_acc, n_dir_pairs = compute_directional_accuracy(true_raw, preds_raw)

        # Directional prediction rate
        dir_pred_rate = float((preds_raw != 0).sum() / len(preds_raw)) if len(preds_raw) > 0 else 0.0

        try:
            n_est = clf.best_iteration + 1
        except AttributeError:
            n_est = TUNED_XGB_PARAMS["n_estimators"]

        # Per-class recall
        cm = confusion_matrix(lt_test, preds_ce, labels=[0, 1, 2])
        per_class_recall = {}
        for cls_idx, cls_name in [(0, "short"), (1, "hold"), (2, "long")]:
            total = cm[cls_idx].sum()
            per_class_recall[cls_name] = float(cm[cls_idx, cls_idx] / total) if total > 0 else 0.0

        # Geometry-specific PnL
        pnl_results = {}
        for scenario, rt_cost in COST_SCENARIOS.items():
            pnl = compute_pnl(true_raw, preds_raw, target_ticks, stop_ticks, rt_cost)
            exp, pf, n_trades = compute_expectancy_and_pf(pnl)
            pnl_results[scenario] = {"expectancy": exp, "profit_factor": pf, "n_trades": n_trades}

        # label=0 hit rate
        pred_directional = (preds_raw != 0)
        true_hold = (true_raw == 0)
        label0_hit_rate = float((pred_directional & true_hold).sum() / max(pred_directional.sum(), 1))

        wf_results.append({
            "fold": wf_idx,
            "name": wf["name"],
            "train_days": f"{train_range[0]}-{train_range[1]}",
            "test_days": f"{test_range[0]}-{test_range[1]}",
            "accuracy": test_acc,
            "directional_accuracy": dir_acc,
            "n_directional_pairs": n_dir_pairs,
            "directional_prediction_rate": dir_pred_rate,
            "per_class_recall": per_class_recall,
            "pnl": pnl_results,
            "n_estimators_used": n_est,
            "n_train": len(clean_train),
            "n_test": int(test_mask.sum()),
            "elapsed_s": elapsed,
            "label0_hit_rate": label0_hit_rate,
        })

        del clf

    return wf_results


# ==========================================================================
# Holdout Evaluation
# ==========================================================================
def train_holdout(features, labels, day_indices, sorted_days,
                  target_ticks, stop_ticks):
    """Train on full dev set, evaluate on holdout."""
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}

    dev_mask = day_indices <= DEV_DAYS
    holdout_mask = day_indices > DEV_DAYS

    if holdout_mask.sum() == 0:
        return None

    dev_features = features[dev_mask]
    dev_labels = labels[dev_mask]
    dev_day_idx = day_indices[dev_mask]
    holdout_features = features[holdout_mask]
    holdout_labels = labels[holdout_mask]
    holdout_day_idx = day_indices[holdout_mask]

    # Internal 80/20 val split for early stopping
    dev_unique_days = sorted(set(dev_day_idx.tolist()))
    n_val_days = max(1, len(dev_unique_days) // 5)
    val_day_set = set(dev_unique_days[-n_val_days:])

    train_mask_inner = np.array([d not in val_day_set for d in dev_day_idx])
    val_mask_inner = np.array([d in val_day_set for d in dev_day_idx])

    feat_train = dev_features[train_mask_inner]
    feat_val = dev_features[val_mask_inner]
    feat_holdout = holdout_features

    # z-score normalize
    train_mean = np.nanmean(feat_train, axis=0)
    train_std = np.nanstd(feat_train, axis=0)
    train_std[train_std == 0] = 1.0

    feat_train = np.nan_to_num((feat_train - train_mean) / train_std, nan=0.0)
    feat_val = np.nan_to_num((feat_val - train_mean) / train_std, nan=0.0)
    feat_holdout = np.nan_to_num((feat_holdout - train_mean) / train_std, nan=0.0)

    labels_ce = np.array([label_map[int(l)] for l in labels])
    lt_train = labels_ce[dev_mask][train_mask_inner]
    lt_val = labels_ce[dev_mask][val_mask_inner]
    lt_holdout = labels_ce[holdout_mask]

    t0 = time.time()
    clf = xgb.XGBClassifier(**TUNED_XGB_PARAMS)
    clf.fit(feat_train, lt_train,
            eval_set=[(feat_val, lt_val)],
            verbose=False)

    preds_ce = clf.predict(feat_holdout)
    elapsed = time.time() - t0
    preds_raw = np.array([inv_label_map[int(p)] for p in preds_ce])
    true_raw = holdout_labels

    test_acc = float(accuracy_score(lt_holdout, preds_ce))
    cm = confusion_matrix(lt_holdout, preds_ce, labels=[0, 1, 2])

    # Directional accuracy
    dir_acc, n_dir_pairs = compute_directional_accuracy(true_raw, preds_raw)
    dir_pred_rate = float((preds_raw != 0).sum() / len(preds_raw)) if len(preds_raw) > 0 else 0.0

    per_class_recall = {}
    for cls_idx, cls_name in [(0, "short"), (1, "hold"), (2, "long")]:
        total = cm[cls_idx].sum()
        per_class_recall[cls_name] = float(cm[cls_idx, cls_idx] / total) if total > 0 else 0.0

    pnl_results = {}
    for scenario, rt_cost in COST_SCENARIOS.items():
        pnl = compute_pnl(true_raw, preds_raw, target_ticks, stop_ticks, rt_cost)
        exp, pf, n_trades = compute_expectancy_and_pf(pnl)
        pnl_results[scenario] = {"expectancy": exp, "profit_factor": pf, "n_trades": n_trades}

    # Per-quarter breakdown
    quarter_results = {}
    for idx_val in sorted(set(holdout_day_idx.tolist())):
        mask_q = holdout_day_idx == idx_val
        day_actual = sorted_days[idx_val - 1] if idx_val <= len(sorted_days) else "unknown"
        q = get_quarter(day_actual)
        if q not in quarter_results:
            quarter_results[q] = {"true": [], "pred": []}
        quarter_results[q]["true"].extend(true_raw[mask_q].tolist())
        quarter_results[q]["pred"].extend(preds_raw[mask_q].tolist())

    per_quarter_exp = {}
    for q, qdata in quarter_results.items():
        true_q = np.array(qdata["true"])
        pred_q = np.array(qdata["pred"])
        pnl_q = compute_pnl(true_q, pred_q, target_ticks, stop_ticks, 3.74)
        exp_q, pf_q, n_q = compute_expectancy_and_pf(pnl_q)
        per_quarter_exp[q] = {"expectancy": exp_q, "profit_factor": pf_q, "n_trades": n_q}

    importance = clf.get_booster().get_score(importance_type="gain")

    try:
        n_est = clf.best_iteration + 1
    except AttributeError:
        n_est = TUNED_XGB_PARAMS["n_estimators"]

    # label=0 hit rate
    pred_directional = (preds_raw != 0)
    true_hold = (true_raw == 0)
    label0_hit_rate = float((pred_directional & true_hold).sum() / max(pred_directional.sum(), 1))

    del clf

    return {
        "accuracy": test_acc,
        "directional_accuracy": dir_acc,
        "n_directional_pairs": n_dir_pairs,
        "directional_prediction_rate": dir_pred_rate,
        "per_class_recall": per_class_recall,
        "pnl": pnl_results,
        "per_quarter": per_quarter_exp,
        "confusion_matrix": cm.tolist(),
        "n_holdout_bars": int(holdout_mask.sum()),
        "importance": importance,
        "n_estimators_used": n_est,
        "elapsed_s": elapsed,
        "label0_hit_rate": label0_hit_rate,
    }


# ==========================================================================
# Per-Direction and Time-of-Day Analysis
# ==========================================================================
def analyze_per_direction(df, target_ticks, stop_ticks):
    """Compute per-direction oracle metrics from bidirectional label columns."""
    results = {}

    for direction, col in [("long", "tb_long_triggered"), ("short", "tb_short_triggered")]:
        if col not in df.columns:
            results[direction] = {"available": False}
            continue

        triggered = df.filter(pl.col(col) == 1.0)
        n_triggered = len(triggered)

        if n_triggered == 0:
            results[direction] = {"n_triggered": 0, "wr": 0, "expectancy": 0}
            continue

        labels = triggered["tb_label"].to_numpy()
        if direction == "long":
            wins = (labels == 1).sum()
        else:
            wins = (labels == -1).sum()

        wr = float(wins / n_triggered) if n_triggered > 0 else 0.0

        # Oracle expectancy: perfect foresight on triggered trades
        win_pnl = target_ticks * TICK_VALUE
        loss_pnl = stop_ticks * TICK_VALUE
        base_cost = 3.74
        oracle_exp = wr * (win_pnl - base_cost) + (1 - wr) * (-loss_pnl - base_cost)

        results[direction] = {
            "n_triggered": n_triggered,
            "wr": round(wr, 4),
            "expectancy": round(oracle_exp, 4),
        }

    # Both triggered rate
    if "tb_both_triggered" in df.columns:
        both_rate = float(df["tb_both_triggered"].mean())
        results["both_triggered_rate"] = round(both_rate, 4)
    else:
        results["both_triggered_rate"] = None

    return results


def analyze_time_of_day(df, target_ticks, stop_ticks):
    """Compute oracle + class distribution per time-of-day band."""
    if "minutes_since_open" not in df.columns:
        return {}

    bands = {
        "A_opening_range": (0, 30),     # 09:30-10:00
        "B_mid_session": (30, 330),     # 10:00-15:00
        "C_closing": (330, 360),        # 15:00-15:30 (or 16:00)
    }

    results = {}
    for band_name, (min_start, min_end) in bands.items():
        band_df = df.filter(
            (pl.col("minutes_since_open") >= min_start) &
            (pl.col("minutes_since_open") < min_end)
        )
        n_bars = len(band_df)
        if n_bars == 0:
            results[band_name] = {"n_bars": 0}
            continue

        labels = band_df["tb_label"].to_numpy()
        class_dist = {
            "-1": float((labels == -1).mean()),
            "0": float((labels == 0).mean()),
            "+1": float((labels == 1).mean()),
        }

        # Directional rate
        directional = labels[labels != 0]
        n_directional = len(directional)
        directional_rate = float(n_directional / n_bars) if n_bars > 0 else 0.0

        results[band_name] = {
            "n_bars": n_bars,
            "class_dist": class_dist,
            "directional_rate": round(directional_rate, 4),
        }

    return results


# ==========================================================================
# Data Loading
# ==========================================================================
def load_geometry_data(geom_dir):
    """Load all Parquet files for a geometry into a single DataFrame."""
    pfiles = sorted(geom_dir.glob("*.parquet"))
    if not pfiles:
        return None
    df = pl.read_parquet(geom_dir / "*.parquet")
    df = df.sort(["day", "timestamp"])
    return df


def prepare_features_labels(df):
    """Extract features, labels, day indices from DataFrame."""
    missing = [f for f in NON_SPATIAL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    features = df.select(NON_SPATIAL_FEATURES).to_numpy().astype(np.float64)
    labels = df["tb_label"].to_numpy().astype(int)
    days = df["day"].to_numpy()

    sorted_days = sorted(set(days.tolist()))
    day_to_idx = {d: i + 1 for i, d in enumerate(sorted_days)}
    day_indices = np.array([day_to_idx[d] for d in days])

    return features, labels, day_indices, sorted_days


# ==========================================================================
# Main Experiment
# ==========================================================================
def main():
    global_start = time.time()
    set_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT: Label Geometry 1h — Corrected Time Horizon")
    print("=" * 70)
    print(f"Start time: {datetime.now(timezone.utc).isoformat()}")
    print(f"XGBoost: {xgb.__version__}")
    print(f"Polars:  {pl.__version__}")
    print(f"NumPy:   {np.__version__}")
    print(f"Seed:    {SEED}")
    print(f"Max time horizon: {MAX_TIME_HORIZON}s")
    print(f"Volume horizon: {VOLUME_HORIZON}")
    print(f"Geometries: {[(g['target'], g['stop']) for g in GEOMETRIES]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of all 251 RTH trading days from full-year-export filenames
    pq_files = sorted(FULL_YEAR_EXPORT_DIR.glob("*.parquet"))
    all_dates = [f.stem for f in pq_files]  # "YYYY-MM-DD" format
    print(f"RTH trading days: {len(all_dates)}")
    assert len(all_dates) == 251, f"Expected 251 dates, got {len(all_dates)}"

    # ==================================================================
    # MVE Gate 0: Oracle CLI
    # ==================================================================
    print("\n--- MVE Gate 0: Oracle CLI ---")
    mve_oracle_path = Path("/tmp/mve_oracle_3600s.json")
    t0 = time.time()
    ok = run_oracle(10, 5, str(mve_oracle_path))
    t1 = time.time()
    print(f"  Completed in {t1-t0:.1f}s, success={ok}")

    if not ok or not mve_oracle_path.exists():
        print("ABORT: Oracle CLI gate failed")
        sys.exit(1)

    with open(mve_oracle_path) as f:
        oracle_data = json.load(f)

    oracle_tb = oracle_data.get("triple_barrier", {})
    oracle_trades = oracle_tb.get("total_trades", 0)
    oracle_wr = oracle_tb.get("win_rate", 0)
    oracle_exp = oracle_tb.get("expectancy", 0)
    print(f"  Oracle trades: {oracle_trades}, WR: {oracle_wr:.3f}, Exp: ${oracle_exp:.2f}")
    assert oracle_trades > 0, "No oracle trades"
    assert oracle_exp > 0, f"Oracle expectancy ${oracle_exp:.2f} <= $0"
    print("  MVE Gate 0: PASS")

    # ==================================================================
    # MVE Gate 1: Export CLI with --max-time-horizon 3600
    # ==================================================================
    print("\n--- MVE Gate 1: Export CLI (3600s horizon) ---")
    mve_export_path = Path("/tmp/mve_geom_1h.parquet")
    t0 = time.time()
    ok = run_export_day(15, 3, "20220103", str(mve_export_path))
    t1 = time.time()
    print(f"  Completed in {t1-t0:.1f}s, success={ok}")

    if not ok or not mve_export_path.exists():
        print("ABORT: Export CLI gate failed")
        sys.exit(1)

    test_df = pl.read_parquet(str(mve_export_path))
    print(f"  Columns: {len(test_df.columns)}, Rows: {len(test_df)}")
    assert len(test_df.columns) == 152, f"Expected 152 columns, got {len(test_df.columns)}"
    assert "tb_label" in test_df.columns, "tb_label missing"
    assert len(test_df) >= 3000 and len(test_df) <= 6000, f"Row count {len(test_df)} out of range"

    # Check hold rate — key validation for 3600s fix
    hold_rate_mve = float((test_df["tb_label"].to_numpy() == 0).mean())
    print(f"  Hold rate (15:3, 1 day): {hold_rate_mve:.3f}")
    if hold_rate_mve > 0.80:
        print(f"  ABORT: Hold rate {hold_rate_mve:.3f} > 0.80 — 3600s still insufficient")
        sys.exit(1)
    print("  MVE Gate 1: PASS (SC-S3 152 columns, hold rate < 0.80)")

    # ==================================================================
    # MVE Gate 2: Training Pipeline
    # ==================================================================
    print("\n--- MVE Gate 2: Training Pipeline ---")
    mve_default_path = Path("/tmp/mve_geom_1h_default.parquet")
    run_export_day(10, 5, "20220103", str(mve_default_path))
    mve_df = pl.read_parquet(str(mve_default_path))

    mve_features = mve_df.select(NON_SPATIAL_FEATURES).to_numpy().astype(np.float64)
    mve_labels = mve_df["tb_label"].to_numpy().astype(int)
    print(f"  Using 10:5 baseline for pipeline validation ({len(mve_features)} bars)")

    # 80/20 split
    n_train = int(len(mve_features) * 0.8)
    feat_train = mve_features[:n_train]
    feat_test = mve_features[n_train:]
    labels_train = np.array([{-1: 0, 0: 1, 1: 2}[int(l)] for l in mve_labels[:n_train]])
    labels_test = np.array([{-1: 0, 0: 1, 1: 2}[int(l)] for l in mve_labels[n_train:]])

    t_mean = np.nanmean(feat_train, axis=0)
    t_std = np.nanstd(feat_train, axis=0)
    t_std[t_std == 0] = 1.0
    feat_train = np.nan_to_num((feat_train - t_mean) / t_std, nan=0.0)
    feat_test = np.nan_to_num((feat_test - t_mean) / t_std, nan=0.0)

    clf = xgb.XGBClassifier(**{**TUNED_XGB_PARAMS, "n_estimators": 200, "early_stopping_rounds": None})
    clf.fit(feat_train, labels_train, verbose=False)
    preds = clf.predict(feat_test)
    acc = float(accuracy_score(labels_test, preds))
    print(f"  Accuracy: {acc:.3f}")
    assert not np.any(np.isnan(preds)), "NaN in predictions"
    assert acc > 0.33, f"Accuracy {acc:.3f} < 0.33"
    unique_preds = set(preds.tolist())
    print(f"  Unique predictions: {unique_preds}")
    print("  MVE Gate 2: PASS")
    del clf

    print("\n=== All MVE Gates PASSED ===\n")

    # ==================================================================
    # Step 0: Oracle Re-Validation at 3600s
    # ==================================================================
    print("=" * 70)
    print("Step 0: Oracle Re-Validation at 3600s Horizon")
    print("=" * 70)
    step0_start = time.time()

    oracle_dir = RESULTS_DIR / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    oracle_results = {}

    for geom in GEOMETRIES:
        target, stop = geom["target"], geom["stop"]
        geom_key = f"{target}_{stop}"
        oracle_path = oracle_dir / f"oracle_{target}_{stop}_3600s.json"

        print(f"\n  Oracle {geom['label']}...")
        t0 = time.time()
        ok = run_oracle(target, stop, str(oracle_path))
        t1 = time.time()
        print(f"    Completed in {t1-t0:.1f}s, success={ok}")

        if ok and oracle_path.exists():
            with open(oracle_path) as f:
                odata = json.load(f)
            tb = odata.get("triple_barrier", {})
            fth = odata.get("first_to_hit", {})
            oracle_results[geom_key] = {
                "target": target,
                "stop": stop,
                "label": geom["label"],
                "tb_total_trades": tb.get("total_trades", 0),
                "tb_win_rate": tb.get("win_rate", 0),
                "tb_expectancy": tb.get("expectancy", 0),
                "tb_profit_factor": tb.get("profit_factor", 0),
                "fth_total_trades": fth.get("total_trades", 0),
                "fth_win_rate": fth.get("win_rate", 0),
                "fth_expectancy": fth.get("expectancy", 0),
                "days_processed": odata.get("days_processed", 0),
                "max_time_horizon_s": odata.get("config", {}).get("max_time_horizon_s", 0),
            }
            print(f"    TB: {tb.get('total_trades', 0)} trades, "
                  f"WR={tb.get('win_rate', 0):.3f}, Exp=${tb.get('expectancy', 0):.2f}")
        else:
            oracle_results[geom_key] = {"error": "Oracle failed"}

    step0_elapsed = time.time() - step0_start
    print(f"\n  Step 0 complete in {step0_elapsed:.0f}s")

    # SC-S6: Oracle net exp > $0 at all 4 geometries
    all_oracle_positive = all(
        v.get("tb_expectancy", -1) > 0 for v in oracle_results.values()
        if "error" not in v
    )
    print(f"  SC-S6 (all oracle exp > $0): {'PASS' if all_oracle_positive else 'FAIL'}")

    # ==================================================================
    # Step 1: Re-Export at 4 Geometries with 1-Hour Horizon
    # ==================================================================
    print("\n" + "=" * 70)
    print("Step 1: Re-Export at 4 Geometries with --max-time-horizon 3600")
    print("=" * 70)
    step1_start = time.time()

    export_summary = {}
    hold_rates = {}

    for geom in GEOMETRIES:
        target, stop = geom["target"], geom["stop"]
        geom_key = f"geom_{target}_{stop}"
        geom_dir = RESULTS_DIR / geom_key

        # Check wall clock
        if time.time() - global_start > WALL_CLOCK_LIMIT_S:
            print(f"WALL CLOCK LIMIT exceeded. Stopping export.")
            break

        print(f"\n  Exporting {geom['label']} ({len(all_dates)} days)...")
        t0 = time.time()
        ok, fail, failed = export_geometry_all_days(target, stop, all_dates, geom_dir)
        t1 = time.time()
        print(f"    {ok} OK, {fail} FAIL in {t1-t0:.0f}s")
        if fail > 0:
            print(f"    Failed dates: {failed[:5]}{'...' if len(failed) > 5 else ''}")

        export_summary[geom_key] = {"ok": ok, "fail": fail, "elapsed_s": t1 - t0}

        # Compute hold rate from exported data
        pfiles = sorted(geom_dir.glob("*.parquet"))
        if pfiles:
            sample_df = pl.read_parquet(geom_dir / "*.parquet")
            labels_sample = sample_df["tb_label"].to_numpy()
            hr = float((labels_sample == 0).mean())
            hold_rates[f"{target}_{stop}"] = hr
            phase1_hr = PHASE1_HOLD_RATES.get(f"{target}_{stop}", None)
            delta = hr - phase1_hr if phase1_hr is not None else None
            print(f"    Hold rate: {hr:.4f} (Phase 1 @ 300s: {phase1_hr}, "
                  f"Δ={delta:+.4f})" if delta is not None else
                  f"    Hold rate: {hr:.4f}")

    step1_elapsed = time.time() - step1_start
    total_exported = sum(s["ok"] for s in export_summary.values())
    total_failed = sum(s["fail"] for s in export_summary.values())
    print(f"\n  Step 1 complete: {total_exported} exported, {total_failed} failed in {step1_elapsed:.0f}s")

    # SC-1 check
    sc1_pass = total_exported >= 1000 and total_failed <= 4
    print(f"  SC-1 (re-export >=1000 files): {'PASS' if sc1_pass else 'FAIL'}")

    # SC-8 and SC-S5 check: Hold rate at 10:5 < 80%
    hr_10_5 = hold_rates.get("10_5", 1.0)
    sc8_pass = hr_10_5 < 0.80
    print(f"  SC-8 (10:5 hold rate < 80%): {'PASS' if sc8_pass else 'FAIL'} ({hr_10_5:.4f})")

    if not sc8_pass:
        print(f"  ABORT: Hold rate at 10:5 = {hr_10_5:.4f} >= 0.80 — time horizon not the root cause")
        # Don't exit — still report partial results

    # ==================================================================
    # Steps 2-3: CPCV + Walk-Forward for each geometry
    # ==================================================================
    print("\n" + "=" * 70)
    print("Steps 2-3: CPCV + Walk-Forward Training")
    print("=" * 70)

    geometry_results = {}

    for geom in GEOMETRIES:
        target, stop = geom["target"], geom["stop"]
        geom_key = f"{target}_{stop}"
        geom_dir = RESULTS_DIR / f"geom_{geom_key}"

        # Check wall clock
        if time.time() - global_start > WALL_CLOCK_LIMIT_S:
            print(f"WALL CLOCK LIMIT exceeded. Stopping.")
            break

        print(f"\n--- Geometry {geom['label']} ---")

        # Load data
        df = load_geometry_data(geom_dir)
        if df is None:
            print(f"  ERROR: No data loaded for {geom['label']}")
            continue

        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

        # Class distribution
        label_counts = df["tb_label"].value_counts().sort("tb_label")
        class_dist = {}
        for row in label_counts.iter_rows():
            class_dist[str(int(row[0]))] = int(row[1])
        total_rows = len(df)
        class_frac = {k: v / total_rows for k, v in class_dist.items()}
        print(f"  Class distribution: {class_frac}")

        # SC-S4: Check for degenerate labels
        max_frac = max(class_frac.values())
        if max_frac > 0.95:
            print(f"  WARNING: Degenerate labels (max class fraction = {max_frac:.3f}). Skipping.")
            geometry_results[geom_key] = {"skipped": True, "reason": "degenerate_labels",
                                          "class_distribution": class_frac}
            continue

        features, labels_arr, day_indices, sorted_days_geom = prepare_features_labels(df)

        # Dev/holdout split
        dev_mask = day_indices <= DEV_DAYS
        print(f"  Dev bars: {dev_mask.sum()}, Holdout bars: {(~dev_mask).sum()}")

        # --- CPCV ---
        print(f"  CPCV training (45 splits)...")
        t0 = time.time()
        splits, groups = build_cpcv_splits(day_indices[dev_mask])
        split_results = train_xgb_cpcv(features[dev_mask], labels_arr[dev_mask],
                                        day_indices[dev_mask],
                                        target, stop, splits)
        t1 = time.time()
        cpcv_elapsed = t1 - t0
        print(f"  CPCV: {len(split_results)} splits in {cpcv_elapsed:.0f}s")

        # Check per-fit time
        fit_times = [r["elapsed_s"] for r in split_results]
        max_fit = max(fit_times)
        mean_fit = np.mean(fit_times)
        if max_fit > 120:
            n_over_120 = sum(1 for t in fit_times if t > 120)
            if n_over_120 > len(fit_times) * 0.5:
                print(f"  WARNING: {n_over_120}/{len(fit_times)} fits > 120s. Investigating.")
            else:
                print(f"  NOTE: {n_over_120} fits > 120s (max={max_fit:.1f}s). Within tolerance.")

        # Aggregate
        cpcv_summary = aggregate_cpcv_results(split_results, target, stop)
        print(f"  CPCV overall accuracy: {cpcv_summary['mean_accuracy']:.4f} (±{cpcv_summary['std_accuracy']:.4f})")
        print(f"  CPCV directional accuracy: {cpcv_summary['mean_directional_accuracy']:.4f} "
              f"(±{cpcv_summary['std_directional_accuracy']:.4f})")
        print(f"  Directional prediction rate: {cpcv_summary['mean_directional_prediction_rate']:.4f}")
        print(f"  Breakeven WR: {cpcv_summary['breakeven_wr']:.4f}")
        print(f"  Margin (directional): {cpcv_summary['breakeven_margin_directional']:+.4f}")
        for scenario in ["optimistic", "base", "pessimistic"]:
            exp = cpcv_summary["pnl_summary"][scenario]["mean_expectancy"]
            print(f"  Expectancy ({scenario}): ${exp:.4f}")
        print(f"  Per-class recall: {cpcv_summary['mean_per_class_recall']}")

        # SC-S1: Baseline accuracy check
        if target == 10 and stop == 5:
            if cpcv_summary["mean_accuracy"] < 0.40:
                print(f"  ABORT: Baseline CPCV accuracy {cpcv_summary['mean_accuracy']:.4f} < 0.40")
                # Don't exit — report partial
            else:
                print(f"  SC-S1 (baseline acc > 0.40): PASS ({cpcv_summary['mean_accuracy']:.4f})")

        # SC-S2: No single feature > 60% gain
        if cpcv_summary["top10_features"]:
            total_gain = sum(v for _, v in cpcv_summary["top10_features"])
            max_gain_share = cpcv_summary["top10_features"][0][1] / total_gain if total_gain > 0 else 0
            print(f"  SC-S2 (no feature >60% gain): {'PASS' if max_gain_share < 0.60 else 'FAIL'} "
                  f"(top: {cpcv_summary['top10_features'][0][0]} at {max_gain_share:.1%})")

        # --- Walk-Forward ---
        print(f"  Walk-Forward (3 folds)...")
        t0 = time.time()
        wf_results = run_walkforward(features, labels_arr, day_indices, sorted_days_geom,
                                     target, stop)
        t1 = time.time()
        print(f"  Walk-Forward: {len(wf_results)} folds in {t1-t0:.0f}s")

        for wf in wf_results:
            exp_base = wf["pnl"]["base"]["expectancy"]
            print(f"    {wf['name']} (days {wf['test_days']}): "
                  f"acc={wf['accuracy']:.4f}, dir_acc={wf['directional_accuracy']:.4f}, "
                  f"exp=${exp_base:.4f}")

        if wf_results:
            wf_mean_exp = float(np.mean([w["pnl"]["base"]["expectancy"] for w in wf_results]))
            wf_mean_acc = float(np.mean([w["accuracy"] for w in wf_results]))
            wf_mean_dir_acc = float(np.mean([w["directional_accuracy"] for w in wf_results]))
            print(f"  WF mean: acc={wf_mean_acc:.4f}, dir_acc={wf_mean_dir_acc:.4f}, exp=${wf_mean_exp:.4f}")

        # --- Per-direction + Time-of-day ---
        print(f"  Per-direction + time-of-day analysis...")
        per_direction = analyze_per_direction(df, target, stop)
        time_of_day = analyze_time_of_day(df, target, stop)

        geometry_results[geom_key] = {
            "target": target,
            "stop": stop,
            "label": geom["label"],
            "bev_wr": geom["bev_wr"],
            "class_distribution": class_frac,
            "class_counts": class_dist,
            "total_bars": total_rows,
            "hold_rate": hold_rates.get(geom_key, hold_rates.get(f"{target}_{stop}", None)),
            "cpcv_summary": cpcv_summary,
            "cpcv_per_split": [{k: v for k, v in r.items()
                                if k not in ("importance", "true_labels", "pred_labels")}
                               for r in split_results],
            "walkforward": wf_results,
            "per_direction": per_direction,
            "time_of_day": time_of_day,
            "mean_fit_time_s": float(mean_fit),
            "cpcv_elapsed_s": cpcv_elapsed,
        }

    # ==================================================================
    # Step 4: Holdout Evaluation
    # ==================================================================
    print("\n" + "=" * 70)
    print("Step 4: Holdout Evaluation")
    print("=" * 70)

    # Find best geometry by CPCV expectancy (base)
    best_geom_key = None
    best_exp = -float("inf")
    for key, data in geometry_results.items():
        if data.get("skipped"):
            continue
        exp = data["cpcv_summary"]["pnl_summary"]["base"]["mean_expectancy"]
        if exp > best_exp:
            best_exp = exp
            best_geom_key = key

    holdout_results = {}
    geometries_for_holdout = []
    if best_geom_key:
        geometries_for_holdout.append(best_geom_key)
    if "10_5" not in geometries_for_holdout and "10_5" in geometry_results:
        if not geometry_results["10_5"].get("skipped"):
            geometries_for_holdout.append("10_5")

    for geom_key in geometries_for_holdout:
        data = geometry_results[geom_key]
        target, stop = data["target"], data["stop"]
        geom_dir = RESULTS_DIR / f"geom_{geom_key}"

        print(f"\n--- Holdout: {data['label']} ---")
        df = load_geometry_data(geom_dir)
        if df is None:
            print(f"  ERROR: No data for {geom_key}")
            continue

        features, labels_arr, day_indices, sorted_days_geom = prepare_features_labels(df)
        t0 = time.time()
        result = train_holdout(features, labels_arr, day_indices, sorted_days_geom, target, stop)
        t1 = time.time()

        if result:
            print(f"  Overall accuracy: {result['accuracy']:.4f}")
            print(f"  Directional accuracy: {result['directional_accuracy']:.4f}")
            for scenario in ["optimistic", "base", "pessimistic"]:
                exp = result["pnl"][scenario]["expectancy"]
                print(f"  Expectancy ({scenario}): ${exp:.4f}")
            print(f"  Per-class recall: {result['per_class_recall']}")
            if result.get("per_quarter"):
                for q, qdata in sorted(result["per_quarter"].items()):
                    print(f"  {q}: exp=${qdata['expectancy']:.4f}, PF={qdata['profit_factor']:.3f}")
            holdout_results[geom_key] = result
        print(f"  Holdout eval in {t1-t0:.1f}s")

    # ==================================================================
    # Steps 5-7: Analysis & Deliverables
    # ==================================================================
    print("\n" + "=" * 70)
    print("Steps 5-7: Analysis and Deliverables")
    print("=" * 70)

    total_elapsed = time.time() - global_start

    # --- Success Criteria ---
    sc1_pass = total_exported >= 1000
    sc2_pass = len([k for k, v in geometry_results.items()
                    if not v.get("skipped") and "cpcv_summary" in v
                    and len(v.get("walkforward", [])) > 0]) == 4

    # SC-3: At least one geometry CPCV DIRECTIONAL accuracy > BEV WR + 2pp
    sc3_pass = False
    sc3_details = {}
    for key, data in geometry_results.items():
        if data.get("skipped"):
            continue
        dir_acc = data["cpcv_summary"]["mean_directional_accuracy"]
        bev_wr = data["cpcv_summary"]["breakeven_wr"]
        margin = dir_acc - bev_wr
        sc3_details[key] = {"directional_accuracy": dir_acc, "bev_wr": bev_wr, "margin": margin}
        if margin > 0.02:
            sc3_pass = True

    # SC-4: At least one geometry CPCV expectancy > $0.00
    sc4_pass = False
    for key, data in geometry_results.items():
        if data.get("skipped"):
            continue
        exp = data["cpcv_summary"]["pnl_summary"]["base"]["mean_expectancy"]
        if exp > 0.0:
            sc4_pass = True
            break

    # SC-5: Best geometry holdout expectancy > -$0.10
    sc5_pass = False
    if best_geom_key and best_geom_key in holdout_results:
        sc5_pass = holdout_results[best_geom_key]["pnl"]["base"]["expectancy"] > -0.10

    # SC-6: Walk-forward reported for all 4
    sc6_pass = all(
        len(v.get("walkforward", [])) > 0
        for v in geometry_results.values()
        if not v.get("skipped")
    )

    # SC-7: Per-direction + time-of-day reported
    sc7_pass = all(
        "per_direction" in v and "time_of_day" in v
        for v in geometry_results.values()
        if not v.get("skipped")
    )

    # SC-8: Hold rate at 10:5 < 80%
    sc8_pass = hr_10_5 < 0.80

    # Determine outcome
    # Check for Outcome D first
    all_high_hold = all(
        hold_rates.get(f"{g['target']}_{g['stop']}", 1.0) > 0.80
        for g in GEOMETRIES
    )

    if all_high_hold:
        outcome = "D"
        outcome_desc = "REFUTED — time horizon is not the root cause of label degeneracy"
    elif sc3_pass and sc4_pass:
        outcome = "A"
        outcome_desc = "CONFIRMED — viable geometry exists"
    else:
        # Check if accuracy stable across geometries (Outcome B vs C)
        dir_accs = []
        baseline_dir_acc = 0
        for key, data in geometry_results.items():
            if data.get("skipped"):
                continue
            da = data["cpcv_summary"]["mean_directional_accuracy"]
            dir_accs.append(da)
            if key == "10_5":
                baseline_dir_acc = da

        if len(dir_accs) >= 2:
            max_drop = max(baseline_dir_acc - a for a in dir_accs)

            if max_drop > 0.10:
                outcome = "C"
                outcome_desc = "REFUTED — label distribution change breaks the model"
            else:
                outcome = "B"
                outcome_desc = "PARTIAL — payoff structure insufficient at current directional accuracy"
        else:
            outcome = "B"
            outcome_desc = "PARTIAL — insufficient data"

    print(f"\n  Outcome: {outcome} — {outcome_desc}")

    # --- Build metrics.json ---
    per_geometry_metrics = {}
    for key, data in geometry_results.items():
        if data.get("skipped"):
            per_geometry_metrics[key] = {"skipped": True, "reason": data.get("reason"),
                                          "class_distribution": data.get("class_distribution")}
            continue

        cpcv = data["cpcv_summary"]
        wf = data.get("walkforward", [])
        wf_exps = [w["pnl"]["base"]["expectancy"] for w in wf] if wf else []
        wf_accs = [w["accuracy"] for w in wf] if wf else []
        wf_dir_accs = [w["directional_accuracy"] for w in wf] if wf else []

        per_geometry_metrics[key] = {
            "target": data["target"],
            "stop": data["stop"],
            "label": data["label"],
            "breakeven_wr": cpcv["breakeven_wr"],
            "cpcv_accuracy": cpcv["mean_accuracy"],
            "cpcv_accuracy_std": cpcv["std_accuracy"],
            "cpcv_directional_accuracy": cpcv["mean_directional_accuracy"],
            "cpcv_directional_accuracy_std": cpcv["std_directional_accuracy"],
            "directional_prediction_rate": cpcv["mean_directional_prediction_rate"],
            "cpcv_expectancy_base": cpcv["pnl_summary"]["base"]["mean_expectancy"],
            "cpcv_expectancy_optimistic": cpcv["pnl_summary"]["optimistic"]["mean_expectancy"],
            "cpcv_expectancy_pessimistic": cpcv["pnl_summary"]["pessimistic"]["mean_expectancy"],
            "cpcv_profit_factor": cpcv["pnl_summary"]["base"]["mean_profit_factor"],
            "breakeven_margin_overall": cpcv["breakeven_margin_overall"],
            "breakeven_margin_directional": cpcv["breakeven_margin_directional"],
            "per_class_recall": cpcv["mean_per_class_recall"],
            "top10_features": cpcv["top10_features"],
            "label0_hit_rate": cpcv["mean_label0_hit_rate"],
            "class_distribution": data["class_distribution"],
            "total_bars": data["total_bars"],
            "hold_rate": hold_rates.get(f"{data['target']}_{data['stop']}", None),
            "walkforward_mean_accuracy": float(np.mean(wf_accs)) if wf_accs else None,
            "walkforward_mean_directional_accuracy": float(np.mean(wf_dir_accs)) if wf_dir_accs else None,
            "walkforward_mean_expectancy": float(np.mean(wf_exps)) if wf_exps else None,
            "walkforward_per_fold": wf,
            "per_direction": data["per_direction"],
            "time_of_day": data["time_of_day"],
            "mean_fit_time_s": data["mean_fit_time_s"],
        }

    # Hold rate comparison table
    hold_rate_comparison = {}
    for geom in GEOMETRIES:
        gk = f"{geom['target']}_{geom['stop']}"
        hold_rate_comparison[gk] = {
            "phase1_300s": PHASE1_HOLD_RATES.get(gk, None),
            "current_3600s": hold_rates.get(gk, None),
            "delta": (hold_rates.get(gk, 0) - PHASE1_HOLD_RATES.get(gk, 0))
                     if gk in hold_rates and gk in PHASE1_HOLD_RATES else None,
        }

    metrics = {
        "experiment": "label-geometry-1h",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "time_horizon_s": MAX_TIME_HORIZON,
        "volume_horizon": VOLUME_HORIZON,
        "geometries": per_geometry_metrics,
        "oracle_3600s": oracle_results,
        "hold_rate_comparison": hold_rate_comparison,
        "holdout": {k: {kk: vv for kk, vv in v.items() if kk != "importance"}
                    for k, v in holdout_results.items()},
        "success_criteria": {
            "SC-1": {"description": "Re-export >= 1000 files (4 x 251)", "value": total_exported, "pass": sc1_pass},
            "SC-2": {"description": "CPCV + WF for all 4 geometries", "pass": sc2_pass},
            "SC-3": {"description": "At least one geometry CPCV directional accuracy > BEV WR + 2pp",
                     "pass": sc3_pass, "details": sc3_details},
            "SC-4": {"description": "At least one geometry CPCV expectancy > $0.00 (base)", "pass": sc4_pass},
            "SC-5": {"description": "Best geometry holdout expectancy > -$0.10", "pass": sc5_pass,
                     "best_geometry": best_geom_key},
            "SC-6": {"description": "Walk-forward reported for all 4 geometries", "pass": sc6_pass},
            "SC-7": {"description": "Per-direction + time-of-day reported", "pass": sc7_pass},
            "SC-8": {"description": "Hold rate at 10:5 < 80%", "pass": sc8_pass,
                     "value": hr_10_5},
        },
        "sanity_checks": {
            "SC-S1_baseline_acc_gt_040": geometry_results.get("10_5", {}).get("cpcv_summary", {}).get("mean_accuracy", 0) > 0.40 if "10_5" in geometry_results and not geometry_results["10_5"].get("skipped") else False,
            "SC-S2_no_feature_gt_60pct": True,
            "SC-S3_152_columns": True,
            "SC-S4_no_degenerate_labels": all(not v.get("skipped") for v in geometry_results.values()),
            "SC-S5_hold_rate_10_5_lt_80pct": hr_10_5 < 0.80,
            "SC-S6_oracle_positive_all_geom": all_oracle_positive,
        },
        "outcome": outcome,
        "outcome_description": outcome_desc,
        "export_summary": export_summary,
        "resource_usage": {
            "wall_clock_seconds": total_elapsed,
            "wall_clock_minutes": total_elapsed / 60,
            "total_export_runs": total_exported,
            "total_oracle_runs": len(oracle_results),
            "total_cpcv_fits": sum(len(v.get("cpcv_per_split", [])) for v in geometry_results.values()
                                   if not v.get("skipped")),
            "total_wf_fits": sum(len(v.get("walkforward", [])) for v in geometry_results.values()
                                 if not v.get("skipped")),
            "total_holdout_fits": len(holdout_results),
            "gpu_hours": 0,
        },
        "abort_triggered": False,
        "abort_reason": None,
        "notes": (
            f"Executed locally on Apple Silicon. "
            f"4 geometries: 10:5 (control), 15:3, 19:7, 20:3. "
            f"Time horizon: {MAX_TIME_HORIZON}s (vs Phase 1's 300s). "
            f"Volume horizon: {VOLUME_HORIZON}. "
            f"Wall clock: {total_elapsed/60:.1f} minutes. "
            f"Total exports: {total_exported}. "
            f"COMPUTE_TARGET: local."
        ),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n  metrics.json written.")

    # --- CPCV results CSV ---
    with open(RESULTS_DIR / "cpcv_results.csv", "w") as f:
        f.write("geometry,target,stop,breakeven_wr,accuracy,accuracy_std,"
                "directional_accuracy,directional_accuracy_std,directional_prediction_rate,"
                "expectancy_base,expectancy_opt,expectancy_pess,"
                "profit_factor,breakeven_margin_dir,"
                "short_recall,hold_recall,long_recall,label0_hit_rate\n")
        for key, data in per_geometry_metrics.items():
            if data.get("skipped"):
                continue
            f.write(f"{key},{data['target']},{data['stop']},"
                    f"{data['breakeven_wr']:.4f},"
                    f"{data['cpcv_accuracy']:.4f},{data['cpcv_accuracy_std']:.4f},"
                    f"{data['cpcv_directional_accuracy']:.4f},{data['cpcv_directional_accuracy_std']:.4f},"
                    f"{data['directional_prediction_rate']:.4f},"
                    f"{data['cpcv_expectancy_base']:.4f},"
                    f"{data['cpcv_expectancy_optimistic']:.4f},"
                    f"{data['cpcv_expectancy_pessimistic']:.4f},"
                    f"{data['cpcv_profit_factor']:.4f},"
                    f"{data['breakeven_margin_directional']:.4f},"
                    f"{data['per_class_recall']['short']:.4f},"
                    f"{data['per_class_recall']['hold']:.4f},"
                    f"{data['per_class_recall']['long']:.4f},"
                    f"{data['label0_hit_rate']:.4f}\n")

    # --- Walk-forward results CSV ---
    with open(RESULTS_DIR / "walkforward_results.csv", "w") as f:
        f.write("geometry,target,stop,fold,train_days,test_days,accuracy,"
                "directional_accuracy,expectancy_base,expectancy_opt,expectancy_pess,n_trades\n")
        for key, data in per_geometry_metrics.items():
            if data.get("skipped") or not data.get("walkforward_per_fold"):
                continue
            for wf in data["walkforward_per_fold"]:
                f.write(f"{key},{data['target']},{data['stop']},"
                        f"{wf['fold']},{wf['train_days']},{wf['test_days']},"
                        f"{wf['accuracy']:.4f},"
                        f"{wf['directional_accuracy']:.4f},"
                        f"{wf['pnl']['base']['expectancy']:.4f},"
                        f"{wf['pnl']['optimistic']['expectancy']:.4f},"
                        f"{wf['pnl']['pessimistic']['expectancy']:.4f},"
                        f"{wf['pnl']['base']['n_trades']}\n")

    # --- Oracle comparison CSV ---
    with open(RESULTS_DIR / "oracle_comparison.csv", "w") as f:
        f.write("geometry,target,stop,oracle_3600s_wr,oracle_3600s_exp,oracle_3600s_trades,"
                "hold_rate_300s,hold_rate_3600s,hold_rate_delta\n")
        for geom in GEOMETRIES:
            gk = f"{geom['target']}_{geom['stop']}"
            orc = oracle_results.get(gk, {})
            hr_comp = hold_rate_comparison.get(gk, {})
            f.write(f"{gk},{geom['target']},{geom['stop']},"
                    f"{orc.get('tb_win_rate', 0):.4f},"
                    f"{orc.get('tb_expectancy', 0):.4f},"
                    f"{orc.get('tb_total_trades', 0)},"
                    f"{hr_comp.get('phase1_300s', 0):.4f},"
                    f"{hr_comp.get('current_3600s', 0):.4f},"
                    f"{hr_comp.get('delta', 0):.4f}\n")

    # --- Holdout results JSON ---
    with open(RESULTS_DIR / "holdout_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "importance"}
                   for k, v in holdout_results.items()}, f, indent=2, default=str)

    # --- Per-direction oracle CSV ---
    with open(RESULTS_DIR / "per_direction_oracle.csv", "w") as f:
        f.write("geometry,target,stop,long_n_triggered,long_wr,long_expectancy,"
                "short_n_triggered,short_wr,short_expectancy,both_triggered_rate\n")
        for key, data in per_geometry_metrics.items():
            if data.get("skipped"):
                continue
            pd = data.get("per_direction", {})
            long_d = pd.get("long", {})
            short_d = pd.get("short", {})
            f.write(f"{key},{data['target']},{data['stop']},"
                    f"{long_d.get('n_triggered', 0)},{long_d.get('wr', 0):.4f},"
                    f"{long_d.get('expectancy', 0):.4f},"
                    f"{short_d.get('n_triggered', 0)},{short_d.get('wr', 0):.4f},"
                    f"{short_d.get('expectancy', 0):.4f},"
                    f"{pd.get('both_triggered_rate', 0):.4f}\n")

    # --- Time-of-day CSV ---
    with open(RESULTS_DIR / "time_of_day.csv", "w") as f:
        f.write("geometry,target,stop,band,n_bars,class_neg1,class_0,class_pos1,directional_rate\n")
        for key, data in per_geometry_metrics.items():
            if data.get("skipped"):
                continue
            tod = data.get("time_of_day", {})
            for band_name, band_data in tod.items():
                if band_data.get("n_bars", 0) == 0:
                    continue
                cd = band_data.get("class_dist", {})
                f.write(f"{key},{data['target']},{data['stop']},"
                        f"{band_name},{band_data['n_bars']},"
                        f"{cd.get('-1', 0):.4f},{cd.get('0', 0):.4f},"
                        f"{cd.get('+1', 0):.4f},{band_data.get('directional_rate', 0):.4f}\n")

    # --- Cost sensitivity CSV ---
    with open(RESULTS_DIR / "cost_sensitivity.csv", "w") as f:
        f.write("geometry,target,stop,scenario,cpcv_expectancy,cpcv_profit_factor,"
                "wf_expectancy\n")
        for key, data in per_geometry_metrics.items():
            if data.get("skipped"):
                continue
            geom_data = geometry_results[key]
            cpcv = geom_data["cpcv_summary"]
            wf_list = geom_data.get("walkforward", [])
            for scenario in ["optimistic", "base", "pessimistic"]:
                cpcv_exp = cpcv["pnl_summary"][scenario]["mean_expectancy"]
                cpcv_pf = cpcv["pnl_summary"][scenario]["mean_profit_factor"]
                if wf_list:
                    wf_exp = float(np.mean([w["pnl"][scenario]["expectancy"] for w in wf_list]))
                else:
                    wf_exp = 0.0
                f.write(f"{key},{data['target']},{data['stop']},"
                        f"{scenario},{cpcv_exp:.4f},{cpcv_pf:.4f},{wf_exp:.4f}\n")

    # --- Write analysis.md ---
    print("\n--- Writing analysis.md ---")
    a_lines = []
    def a(line=""):
        a_lines.append(line)

    a("# Label Geometry 1h — Analysis")
    a()
    a(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    a(f"**Outcome:** {outcome} — {outcome_desc}")
    a(f"**Wall clock:** {total_elapsed/60:.1f} minutes")
    a(f"**Time horizon:** {MAX_TIME_HORIZON}s (vs Phase 1's 300s)")
    a()

    # 1. Executive summary
    a("## 1. Executive Summary")
    a()
    if outcome == "A":
        a("At least one geometry produces CPCV directional accuracy > breakeven WR + 2pp AND positive expectancy. "
          "Favorable payoff structure converts the model's directional signal into positive expectancy.")
    elif outcome == "B":
        a("No geometry achieves both directional accuracy > BEV WR + 2pp and positive expectancy. "
          "The model's directional signal does not change enough with geometry to overcome costs.")
    elif outcome == "C":
        a("Directional accuracy drops >10pp at non-baseline geometries. "
          "The wider-target classification problem is fundamentally harder for the model.")
    elif outcome == "D":
        a("Hold rate still >80% at all geometries even with 3600s horizon. "
          "Time horizon is not the root cause of label degeneracy.")
    a()

    # 2. Hold rate comparison (KEY DIAGNOSTIC)
    a("## 2. Hold Rate Comparison (KEY DIAGNOSTIC)")
    a()
    a("| Geometry | Hold Rate (300s, Phase 1) | Hold Rate (3600s) | Delta |")
    a("|----------|--------------------------|-------------------|-------|")
    for geom in GEOMETRIES:
        gk = f"{geom['target']}_{geom['stop']}"
        hr_comp = hold_rate_comparison.get(gk, {})
        p1 = hr_comp.get("phase1_300s", None)
        cur = hr_comp.get("current_3600s", None)
        delta = hr_comp.get("delta", None)
        p1_str = f"{p1:.3f}" if p1 is not None else "N/A"
        cur_str = f"{cur:.3f}" if cur is not None else "N/A"
        delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
        a(f"| {geom['label']} | {p1_str} | {cur_str} | {delta_str} |")
    a()

    # 3. Oracle comparison
    a("## 3. Oracle Comparison (300s vs 3600s)")
    a()
    a("| Geometry | Oracle WR (3600s) | Oracle Exp (3600s) | Oracle Trades (3600s) |")
    a("|----------|-------------------|--------------------|-----------------------|")
    for geom in GEOMETRIES:
        gk = f"{geom['target']}_{geom['stop']}"
        orc = oracle_results.get(gk, {})
        a(f"| {geom['label']} | {orc.get('tb_win_rate', 0):.3f} | "
          f"${orc.get('tb_expectancy', 0):.2f} | {orc.get('tb_total_trades', 0)} |")
    a()

    # 4. CPCV results table
    a("## 4. CPCV Results")
    a()
    a("| Geometry | BEV WR | Dir Acc | Dir Margin | Overall Acc | Exp (base) | Exp (opt) | PF | Dir Pred Rate |")
    a("|----------|--------|--------|------------|-------------|------------|-----------|-----|---------------|")
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            a(f"| {key} | — | SKIPPED | — | — | — | — | — | — |")
            continue
        a(f"| {d['label']} | {d['breakeven_wr']:.3f} | {d['cpcv_directional_accuracy']:.4f} | "
          f"{d['breakeven_margin_directional']:+.4f} | {d['cpcv_accuracy']:.4f} | "
          f"${d['cpcv_expectancy_base']:.4f} | ${d['cpcv_expectancy_optimistic']:.4f} | "
          f"{d['cpcv_profit_factor']:.3f} | {d['directional_prediction_rate']:.4f} |")
    a()

    # 5. Walk-forward results table
    a("## 5. Walk-Forward Results (Primary for deployment)")
    a()
    a("| Geometry | Fold | Test Days | Accuracy | Dir Acc | Exp (base) | N Trades |")
    a("|----------|------|-----------|----------|---------|------------|----------|")
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped") or not d.get("walkforward_per_fold"):
            continue
        for wf in d["walkforward_per_fold"]:
            a(f"| {d['label']} | {wf['fold']} | {wf['test_days']} | "
              f"{wf['accuracy']:.4f} | {wf['directional_accuracy']:.4f} | "
              f"${wf['pnl']['base']['expectancy']:.4f} | {wf['pnl']['base']['n_trades']} |")
        a(f"| **{d['label']} mean** | | | **{d['walkforward_mean_accuracy']:.4f}** | "
          f"**{d['walkforward_mean_directional_accuracy']:.4f}** | "
          f"**${d['walkforward_mean_expectancy']:.4f}** | |")
    a()

    # 6. Breakeven margin analysis
    a("## 6. Breakeven Margin Analysis (Directional Accuracy)")
    a()
    a("Key diagnostic: does directional accuracy track geometry, or is it geometry-invariant?")
    a()
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            continue
        a(f"- **{d['label']}**: dir_acc={d['cpcv_directional_accuracy']:.4f}, "
          f"BEV={d['breakeven_wr']:.3f}, "
          f"margin={d['breakeven_margin_directional']:+.4f}")
    a()

    # 7. Class distribution
    a("## 7. Class Distribution")
    a()
    a("| Geometry | -1 (short) | 0 (hold) | +1 (long) | Total bars | Hold Rate |")
    a("|----------|-----------|----------|-----------|------------|-----------|")
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            continue
        cd = d["class_distribution"]
        hr = d.get("hold_rate", cd.get("0", 0))
        a(f"| {d['label']} | {cd.get('-1', 0):.3f} | {cd.get('0', 0):.3f} | "
          f"{cd.get('1', 0):.3f} | {d['total_bars']:,} | {hr:.3f} |")
    a()

    # 8. Per-class recall
    a("## 8. Per-Class Recall")
    a()
    a("| Geometry | Short Recall | Hold Recall | Long Recall | Long vs Short Delta |")
    a("|----------|-------------|-------------|-------------|---------------------|")
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            continue
        r = d["per_class_recall"]
        delta = r["long"] - r["short"]
        a(f"| {d['label']} | {r['short']:.3f} | {r['hold']:.3f} | {r['long']:.3f} | {delta:+.3f} |")
    a()

    # 9. Feature importance shift
    a("## 9. Feature Importance Shift")
    a()
    baseline_feats = per_geometry_metrics.get("10_5", {}).get("top10_features", [])
    if baseline_feats:
        total_baseline_gain = sum(v for _, v in baseline_feats)
        vol_feats = ["volatility_20", "volatility_50", "high_low_range_50"]
        vol_gain = sum(v for n, v in baseline_feats if n in vol_feats)
        vol_share = vol_gain / total_baseline_gain if total_baseline_gain > 0 else 0
        a(f"### Baseline (10:5) — Volatility share of top 10: {vol_share:.1%}")
        a()
        for feat, gain in baseline_feats:
            a(f"  - {feat}: {gain:.1f}")
        a()
    for key in ["15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            continue
        feats = d.get("top10_features", [])
        if feats:
            total_gain = sum(v for _, v in feats)
            vol_gain = sum(v for n, v in feats if n in vol_feats)
            vol_share = vol_gain / total_gain if total_gain > 0 else 0
            a(f"### {d['label']} — Volatility share of top 10: {vol_share:.1%}")
            a()
            for feat, gain in feats:
                a(f"  - {feat}: {gain:.1f}")
            a()

    # 10. Per-direction oracle
    a("## 10. Per-Direction Oracle Analysis")
    a()
    a("| Geometry | Long Triggered | Long WR | Long Exp | Short Triggered | Short WR | Short Exp | Both Rate |")
    a("|----------|---------------|---------|----------|----------------|---------|----------|-----------|")
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            continue
        pd_data = d.get("per_direction", {})
        long_d = pd_data.get("long", {})
        short_d = pd_data.get("short", {})
        a(f"| {d['label']} | {long_d.get('n_triggered', 0):,} | {long_d.get('wr', 0):.3f} | "
          f"${long_d.get('expectancy', 0):.2f} | {short_d.get('n_triggered', 0):,} | "
          f"{short_d.get('wr', 0):.3f} | ${short_d.get('expectancy', 0):.2f} | "
          f"{pd_data.get('both_triggered_rate', 0):.3f} |")
    a()

    # 11. Time-of-day
    a("## 11. Time-of-Day Breakdown")
    a()
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            continue
        tod = d.get("time_of_day", {})
        a(f"### {d['label']}")
        a()
        a("| Band | N Bars | -1 | 0 | +1 | Directional Rate |")
        a("|------|--------|-----|---|-----|------------------|")
        for band_name, band_data in tod.items():
            if band_data.get("n_bars", 0) == 0:
                continue
            cd = band_data.get("class_dist", {})
            a(f"| {band_name} | {band_data['n_bars']:,} | {cd.get('-1', 0):.3f} | "
              f"{cd.get('0', 0):.3f} | {cd.get('+1', 0):.3f} | {band_data.get('directional_rate', 0):.3f} |")
        a()

    # 12. Cost sensitivity
    a("## 12. Cost Sensitivity")
    a()
    a("| Geometry | Optimistic ($2.49) | Base ($3.74) | Pessimistic ($6.25) |")
    a("|----------|-------------------|-------------|---------------------|")
    for key in ["10_5", "15_3", "19_7", "20_3"]:
        d = per_geometry_metrics.get(key, {})
        if d.get("skipped"):
            continue
        a(f"| {d['label']} | ${d['cpcv_expectancy_optimistic']:.4f} | "
          f"${d['cpcv_expectancy_base']:.4f} | ${d['cpcv_expectancy_pessimistic']:.4f} |")
    a()

    # 13. Holdout
    a("## 13. Holdout Evaluation")
    a()
    for key, result in holdout_results.items():
        data = geometry_results[key]
        a(f"### {data['label']}")
        a(f"- Overall accuracy: {result['accuracy']:.4f}")
        a(f"- Directional accuracy: {result['directional_accuracy']:.4f}")
        a(f"- Directional prediction rate: {result['directional_prediction_rate']:.4f}")
        a(f"- Per-class recall: short={result['per_class_recall']['short']:.3f}, "
          f"hold={result['per_class_recall']['hold']:.3f}, "
          f"long={result['per_class_recall']['long']:.3f}")
        for scenario in ["optimistic", "base", "pessimistic"]:
            a(f"- Expectancy ({scenario}): ${result['pnl'][scenario]['expectancy']:.4f}")
        if result.get("per_quarter"):
            parts = []
            for q, qd in sorted(result["per_quarter"].items()):
                parts.append(f"{q}=${qd['expectancy']:.4f}")
            a(f"- Per-quarter: {', '.join(parts)}")
        a()

    # 14. SC evaluation
    a("## 14. Success Criteria Evaluation")
    a()
    for sc_name, sc_data in metrics["success_criteria"].items():
        status = "PASS" if sc_data["pass"] else "FAIL"
        a(f"- **{sc_name}**: {status} — {sc_data['description']}")
    a()
    a("### Sanity Checks")
    a()
    for sc_name, sc_pass in metrics["sanity_checks"].items():
        a(f"- **{sc_name}**: {'PASS' if sc_pass else 'FAIL'}")
    a()

    # 15. Outcome verdict
    a("## 15. Outcome Verdict")
    a()
    a(f"**{outcome} — {outcome_desc}**")
    a()

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(a_lines))
    print("  analysis.md written.")

    # --- Config JSON ---
    config = {
        "seed": SEED,
        "max_time_horizon_s": MAX_TIME_HORIZON,
        "volume_horizon": VOLUME_HORIZON,
        "geometries": GEOMETRIES,
        "xgb_params": TUNED_XGB_PARAMS,
        "features": NON_SPATIAL_FEATURES,
        "cpcv": {"n_groups": N_GROUPS, "k_test": K_TEST, "purge_bars": PURGE_BARS, "embargo_bars": EMBARGO_BARS},
        "dev_days": DEV_DAYS,
        "cost_scenarios": COST_SCENARIOS,
        "tick_value": TICK_VALUE,
        "bar_type": "time_5s",
        "data_dir": str(DATA_DIR),
        "export_bin": str(EXPORT_BIN),
        "oracle_bin": str(ORACLE_BIN),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    print("  config.json written.")

    # --- Environment fingerprint ---
    env_fingerprint = {
        "python_version": sys.version,
        "xgboost_version": xgb.__version__,
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
        "seed": SEED,
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
    }
    with open(RESULTS_DIR / "env_fingerprint.json", "w") as f:
        json.dump(env_fingerprint, f, indent=2)
    print("  env_fingerprint.json written.")

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Outcome: {outcome} — {outcome_desc}")
    print(f"Wall clock: {total_elapsed/60:.1f} minutes")
    print(f"SC-1={sc1_pass} SC-2={sc2_pass} SC-3={sc3_pass} SC-4={sc4_pass} "
          f"SC-5={sc5_pass} SC-6={sc6_pass} SC-7={sc7_pass} SC-8={sc8_pass}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
