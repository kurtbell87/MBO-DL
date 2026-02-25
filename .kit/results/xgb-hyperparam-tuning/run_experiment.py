#!/usr/bin/env python3
"""
Experiment: XGBoost Hyperparameter Tuning on Full-Year Data
Spec: .kit/experiments/xgb-hyperparam-tuning.md

Systematic hyperparameter tuning of XGBoost on 1.16M-bar full-year dataset.
Phase 1: Coarse random search (48 configs + 1 default) on 5-fold blocked CV.
Phase 2: Fine search (15 configs around top 5).
Phase 3: CPCV (N=10, k=2, 45 splits) for best tuned + default.
Phase 4: Holdout + walk-forward evaluation.
"""

import json
import os
import sys
import time
import math
import random
import hashlib
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from copy import deepcopy

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --- Config ---

SEED = 42
DATA_DIR = Path(".kit/results/full-year-export")
RESULTS_DIR = Path(".kit/results/xgb-hyperparam-tuning")

# Non-spatial features (20 dimensions, spec order)
NON_SPATIAL_FEATURES = [
    "weighted_imbalance", "spread", "net_volume", "volume_imbalance",
    "trade_count", "avg_trade_size", "vwap_distance",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50", "high_low_range_50", "close_position",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "time_sin", "time_cos", "minutes_since_open",
]

# Default XGBoost params (from E2E CNN / 9B baseline)
DEFAULT_XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softmax",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": SEED,
    "verbosity": 0,
    "n_jobs": -1,
}

# Tuned config template (with early stopping)
TUNED_TEMPLATE = {
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

# Transaction cost scenarios
COST_SCENARIOS = {
    "optimistic": 2.49,
    "base": 3.74,
    "pessimistic": 6.25,
}

# PnL parameters
TARGET_TICKS = 10
STOP_TICKS = 5
TICK_VALUE = 1.25
WIN_PNL = TARGET_TICKS * TICK_VALUE   # $12.50
LOSS_PNL = STOP_TICKS * TICK_VALUE    # $6.25

# CPCV parameters
N_GROUPS = 10
K_TEST = 2
PURGE_BARS = 500
EMBARGO_BARS = 4600  # ~1 trading day

# Dev/holdout split
DEV_DAYS = 201
HOLDOUT_START = 202

# Search budget
N_COARSE = 48
N_FINE_PER_TOP = 3
N_TOP = 5

# --- Seed ---

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# --- Search Space ---

def sample_random_config(rng):
    """Sample one random hyperparameter config from the search distributions."""
    config = {
        "max_depth": rng.choice([3, 4, 5, 6, 7, 8, 10]),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.005), np.log(0.3)))),
        "min_child_weight": rng.choice([1, 3, 5, 10, 20, 50]),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "reg_alpha": float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(10.0)))),
    }
    return config


def generate_neighbors(config, n_neighbors=3):
    """Generate neighbors by perturbing one hyperparameter at a time."""
    neighbors = []
    rng = np.random.RandomState(SEED + hash(str(config)) % 10000)

    # Discrete param grids
    max_depth_grid = [3, 4, 5, 6, 7, 8, 10]
    min_child_weight_grid = [1, 3, 5, 10, 20, 50]

    params_to_perturb = list(config.keys())
    rng.shuffle(params_to_perturb)

    for param in params_to_perturb[:n_neighbors]:
        neighbor = config.copy()

        if param == "max_depth":
            idx = max_depth_grid.index(config["max_depth"])
            candidates = []
            if idx > 0:
                candidates.append(max_depth_grid[idx - 1])
            if idx < len(max_depth_grid) - 1:
                candidates.append(max_depth_grid[idx + 1])
            if candidates:
                neighbor["max_depth"] = rng.choice(candidates)

        elif param == "min_child_weight":
            idx = min_child_weight_grid.index(config["min_child_weight"])
            candidates = []
            if idx > 0:
                candidates.append(min_child_weight_grid[idx - 1])
            if idx < len(min_child_weight_grid) - 1:
                candidates.append(min_child_weight_grid[idx + 1])
            if candidates:
                neighbor["min_child_weight"] = rng.choice(candidates)

        elif param in ("learning_rate", "reg_alpha", "reg_lambda"):
            # ±20% in log space
            factor = np.exp(rng.uniform(-0.2, 0.2))
            neighbor[param] = float(config[param] * factor)
            # Clip to search range
            if param == "learning_rate":
                neighbor[param] = float(np.clip(neighbor[param], 0.005, 0.3))
            elif param == "reg_alpha":
                neighbor[param] = float(np.clip(neighbor[param], 1e-3, 10.0))
            elif param == "reg_lambda":
                neighbor[param] = float(np.clip(neighbor[param], 0.1, 10.0))

        elif param in ("subsample", "colsample_bytree"):
            # ±20% absolute
            delta = rng.uniform(-0.2, 0.2) * config[param]
            neighbor[param] = float(np.clip(config[param] + delta, 0.5, 1.0))

        neighbors.append(neighbor)

    return neighbors


# --- PnL and Metrics ---

def compute_pnl(true_labels, pred_labels, rt_cost):
    """PnL per observation. pred=0 or true=0: $0. pred=true (both nonzero): win. else: loss."""
    pnl = np.zeros(len(true_labels))
    # Vectorized
    both_nonzero = (pred_labels != 0) & (true_labels != 0)
    correct = (pred_labels == true_labels) & both_nonzero
    wrong = (pred_labels != true_labels) & both_nonzero
    pnl[correct] = WIN_PNL - rt_cost
    pnl[wrong] = -LOSS_PNL - rt_cost
    return pnl


def compute_expectancy_and_pf(pnl_array):
    trades = pnl_array[pnl_array != 0]
    if len(trades) == 0:
        return 0.0, 0.0, 0, 0.0, 0.0
    expectancy = float(trades.mean())
    gross_profit = float(trades[trades > 0].sum())
    gross_loss = float(abs(trades[trades < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return expectancy, pf, len(trades), gross_profit, gross_loss


# --- CV Helpers ---

def assign_cpcv_groups(day_indices, n_groups):
    """Assign days to sequential groups."""
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
    """Remove training observations near test boundaries."""
    train_set = set(train_indices.tolist()) if isinstance(train_indices, np.ndarray) else set(train_indices)
    n_total = len(all_day_indices)

    # Find contiguous test-group boundaries
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
        # Purge before test start
        for i in range(max(0, tg_start - purge_bars), tg_start):
            if i in train_set:
                excluded.add(i)
        # Purge after test end
        for i in range(tg_end + 1, min(n_total, tg_end + 1 + purge_bars)):
            if i in train_set:
                excluded.add(i)
        # Embargo before test start (beyond purge)
        for i in range(max(0, tg_start - purge_bars - embargo_bars), max(0, tg_start - purge_bars)):
            if i in train_set:
                excluded.add(i)
        # Embargo after test end (beyond purge)
        for i in range(min(n_total, tg_end + 1 + purge_bars),
                       min(n_total, tg_end + 1 + purge_bars + embargo_bars)):
            if i in train_set:
                excluded.add(i)

    clean_train = np.array(sorted(train_set - excluded))
    return clean_train, len(excluded)


def build_5fold_blocked_cv(dev_day_indices, n_folds=5):
    """
    Build 5-fold blocked time-series CV on dev set.
    Block 1: days 1-40, Block 2: days 41-80, etc.
    Each fold: test on one block, train on remaining 4.
    """
    unique_days = sorted(set(dev_day_indices.tolist()))
    n_days = len(unique_days)
    days_per_block = n_days // n_folds
    remainder = n_days % n_folds

    blocks = []
    start = 0
    for b in range(n_folds):
        size = days_per_block + (1 if b < remainder else 0)
        block_days = set(unique_days[start:start + size])
        blocks.append(block_days)
        start += size

    folds = []
    for test_block_idx in range(n_folds):
        test_day_set = blocks[test_block_idx]
        train_day_set = set()
        for b in range(n_folds):
            if b != test_block_idx:
                train_day_set.update(blocks[b])

        test_indices = np.array([i for i, d in enumerate(dev_day_indices) if d in test_day_set])
        train_indices = np.array([i for i, d in enumerate(dev_day_indices) if d in train_day_set])

        # Apply purge and embargo
        clean_train, n_excluded = apply_purge_embargo(
            train_indices, test_indices, dev_day_indices, PURGE_BARS, EMBARGO_BARS
        )

        # Internal val split: last 10% of training days for early stopping
        train_days_sorted = sorted(set(dev_day_indices[clean_train].tolist()))
        n_val_days = max(1, len(train_days_sorted) // 10)
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

        folds.append({
            "fold_idx": test_block_idx,
            "train_indices": inner_train,
            "val_indices": inner_val,
            "test_indices": test_indices,
            "n_excluded": n_excluded,
        })

    return folds


def build_cpcv_splits(dev_day_indices):
    """Build all 45 CPCV splits (N=10, k=2)."""
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


def cpcv_path_mapping(n_groups, k_test):
    """Map C(n,k) splits to phi(n,k) non-overlapping backtest paths."""
    splits = list(combinations(range(n_groups), k_test))
    n_splits = len(splits)
    n_paths = n_splits * k_test // n_groups

    paths = []
    used_splits = set()

    for _ in range(n_paths):
        path = []
        covered = set()
        for s_idx, split in enumerate(splits):
            if s_idx in used_splits:
                continue
            if not covered.intersection(split):
                path.append(s_idx)
                covered.update(split)
                if len(covered) == n_groups:
                    break
        if len(covered) == n_groups:
            for s_idx in path:
                used_splits.add(s_idx)
            paths.append(path)
        else:
            break

    return paths, splits


# --- Training ---

def train_xgb_default(feat_train, labels_train, feat_test, labels_test):
    """Train XGBoost with default params (no early stopping, n_estimators=500)."""
    clf = xgb.XGBClassifier(**DEFAULT_XGB_PARAMS)
    clf.fit(feat_train, labels_train)
    preds = clf.predict(feat_test)
    train_preds = clf.predict(feat_train)
    test_acc = float(accuracy_score(labels_test, preds))
    train_acc = float(accuracy_score(labels_train, train_preds))
    importance = clf.get_booster().get_score(importance_type="gain")
    del clf
    return preds, test_acc, train_acc, importance, 500


def train_xgb_tuned(config, feat_train, labels_train, feat_val, labels_val,
                     feat_test, labels_test):
    """Train XGBoost with tuned config (early stopping on val set)."""
    params = {**TUNED_TEMPLATE}
    params.update(config)

    clf = xgb.XGBClassifier(**params)
    clf.fit(
        feat_train, labels_train,
        eval_set=[(feat_val, labels_val)],
        verbose=False,
    )

    preds = clf.predict(feat_test)
    train_preds = clf.predict(feat_train)
    test_acc = float(accuracy_score(labels_test, preds))
    train_acc = float(accuracy_score(labels_train, train_preds))
    importance = clf.get_booster().get_score(importance_type="gain")
    try:
        actual_n_estimators = clf.best_iteration + 1
    except AttributeError:
        actual_n_estimators = params["n_estimators"]
    del clf
    return preds, test_acc, train_acc, importance, actual_n_estimators


def eval_config_5fold(config, folds, dev_feat_z_folds, dev_labels, is_default=False):
    """Evaluate a single config across 5-fold blocked CV. Returns per-fold results."""
    fold_results = []
    for fold in folds:
        ft_train = dev_feat_z_folds[fold["fold_idx"]]["train"]
        ft_val = dev_feat_z_folds[fold["fold_idx"]]["val"]
        ft_test = dev_feat_z_folds[fold["fold_idx"]]["test"]

        lt_train = dev_labels[fold["train_indices"]]
        lt_val = dev_labels[fold["val_indices"]]
        lt_test = dev_labels[fold["test_indices"]]

        t0 = time.time()
        if is_default:
            preds, test_acc, train_acc, importance, n_est = train_xgb_default(
                ft_train, lt_train, ft_test, lt_test
            )
        else:
            preds, test_acc, train_acc, importance, n_est = train_xgb_tuned(
                config, ft_train, lt_train, ft_val, lt_val, ft_test, lt_test
            )
        elapsed = time.time() - t0

        fold_results.append({
            "fold_idx": fold["fold_idx"],
            "test_acc": test_acc,
            "train_acc": train_acc,
            "n_estimators_used": n_est,
            "elapsed_s": elapsed,
            "importance": importance,
        })

    mean_acc = np.mean([r["test_acc"] for r in fold_results])
    std_acc = np.std([r["test_acc"] for r in fold_results])
    mean_train_acc = np.mean([r["train_acc"] for r in fold_results])
    mean_n_est = np.mean([r["n_estimators_used"] for r in fold_results])

    return {
        "mean_acc": float(mean_acc),
        "std_acc": float(std_acc),
        "mean_train_acc": float(mean_train_acc),
        "mean_n_estimators": float(mean_n_est),
        "per_fold": fold_results,
    }


# --- Quarter Assignment ---

def get_quarter(day_val):
    """Map day value (YYYYMMDD int or 'YYYY-MM-DD' string) to quarter Q1-Q4."""
    day_str = str(day_val)
    if "-" in day_str:
        month = int(day_str.split("-")[1])
    else:
        # YYYYMMDD integer format
        month = int(day_str[4:6])
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    else:
        return "Q4"


# --- Main ---

def main():
    start_time = time.time()
    set_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT: XGBoost Hyperparameter Tuning (Full-Year CPCV)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # Phase 0: Data Loading and Preprocessing
    # ======================================================================
    print("\n--- Phase 0: Data Loading ---")
    print(f"XGBoost: {xgb.__version__}")
    print(f"Polars:  {pl.__version__}")
    print(f"NumPy:   {np.__version__}")
    print(f"Seed:    {SEED}")

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Parquet files found: {len(parquet_files)}")

    df = pl.read_parquet(DATA_DIR / "*.parquet")
    df = df.sort(["day", "timestamp"])
    print(f"Total rows: {len(df)}, Columns: {len(df.columns)}")

    # MVE Gate: data loading
    assert len(df) >= 1_150_000, f"ABORT: Expected >=1,150,000 rows, got {len(df)}"

    # Verify all 20 features
    missing_feats = [f for f in NON_SPATIAL_FEATURES if f not in df.columns]
    assert len(missing_feats) == 0, f"ABORT: Missing features: {missing_feats}"
    assert "tb_label" in df.columns, "ABORT: tb_label column missing"
    assert "day" in df.columns, "ABORT: day column missing"

    # Check 3 classes
    unique_labels = sorted(df["tb_label"].unique().to_list())
    assert len(unique_labels) == 3, f"ABORT: Expected 3 classes, got {unique_labels}"

    # Sort days chronologically
    sorted_days = sorted(df["day"].unique().to_list())
    day_to_idx = {d: i + 1 for i, d in enumerate(sorted_days)}
    n_unique_days = len(sorted_days)
    assert n_unique_days >= 250, f"ABORT: Expected >=250 unique days, got {n_unique_days}"
    print(f"Unique days: {n_unique_days}")

    # Development vs holdout
    dev_days = sorted_days[:DEV_DAYS]
    holdout_days = sorted_days[DEV_DAYS:]
    print(f"Development days: {len(dev_days)} (days 1-{DEV_DAYS})")
    print(f"Holdout days: {len(holdout_days)} (days {HOLDOUT_START}-{n_unique_days})")

    day_arr = df["day"].to_numpy()
    day_idx_arr = np.array([day_to_idx[d] for d in day_arr])

    dev_mask = np.isin(day_arr, dev_days)
    holdout_mask = np.isin(day_arr, holdout_days)
    print(f"Dev bars: {dev_mask.sum()}, Holdout bars: {holdout_mask.sum()}")

    # Labels (keep as -1, 0, +1 for XGBoost — map to 0, 1, 2 for multi:softmax)
    tb_raw = df["tb_label"].to_numpy().astype(int)
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    labels_ce = np.array([label_map[l] for l in tb_raw])

    dev_labels_raw = tb_raw[dev_mask]
    holdout_labels_raw = tb_raw[holdout_mask]
    label_dist = {
        "dev": {"-1": int((dev_labels_raw == -1).sum()), "0": int((dev_labels_raw == 0).sum()), "+1": int((dev_labels_raw == 1).sum())},
        "holdout": {"-1": int((holdout_labels_raw == -1).sum()), "0": int((holdout_labels_raw == 0).sum()), "+1": int((holdout_labels_raw == 1).sum())},
    }
    print(f"Label distribution (dev): {label_dist['dev']}")
    print(f"Label distribution (holdout): {label_dist['holdout']}")

    # Extract feature matrix
    feat_matrix = df[NON_SPATIAL_FEATURES].to_numpy().astype(np.float64)
    feat_nan_rate = float(np.isnan(feat_matrix).sum()) / feat_matrix.size
    print(f"Feature NaN rate: {feat_nan_rate:.6f}")

    # Dev/holdout feature+label arrays
    dev_feat = feat_matrix[dev_mask]
    dev_labels = labels_ce[dev_mask]
    dev_day_indices = day_idx_arr[dev_mask]
    dev_day_strings = day_arr[dev_mask]

    holdout_feat = feat_matrix[holdout_mask]
    holdout_labels = labels_ce[holdout_mask]
    holdout_day_strings = day_arr[holdout_mask]

    print(f"\nMVE Gate: Data loading PASSED")
    print(f"  Rows: {len(df)}, Days: {n_unique_days}, Features: {len(NON_SPATIAL_FEATURES)}, Classes: {len(unique_labels)}")

    # ======================================================================
    # Phase 0.5: Build CV Structures
    # ======================================================================
    print("\n--- Building CV Structures ---")

    # 5-fold blocked CV for search
    folds_5 = build_5fold_blocked_cv(dev_day_indices, n_folds=5)
    for f in folds_5:
        print(f"  Fold {f['fold_idx']}: train={len(f['train_indices'])}, val={len(f['val_indices'])}, test={len(f['test_indices'])}, excluded={f['n_excluded']}")

    # Pre-normalize features per fold (z-score using train stats only)
    dev_feat_z_folds = {}
    for fold in folds_5:
        ft_train = dev_feat[fold["train_indices"]]
        f_mean = np.nanmean(ft_train, axis=0)
        f_std = np.nanstd(ft_train, axis=0)
        f_std[f_std < 1e-10] = 1.0

        dev_feat_z_folds[fold["fold_idx"]] = {
            "train": np.nan_to_num((dev_feat[fold["train_indices"]] - f_mean) / f_std, nan=0.0),
            "val": np.nan_to_num((dev_feat[fold["val_indices"]] - f_mean) / f_std, nan=0.0),
            "test": np.nan_to_num((dev_feat[fold["test_indices"]] - f_mean) / f_std, nan=0.0),
            "mean": f_mean,
            "std": f_std,
        }

    # ======================================================================
    # MVE: Baseline Reproduction
    # ======================================================================
    print("\n--- MVE: Baseline Reproduction ---")

    # Simple 80/20 time-series split of dev set
    split_point = int(len(dev_feat) * 0.8)
    mve_train_feat = dev_feat[:split_point]
    mve_test_feat = dev_feat[split_point:]
    mve_train_labels = dev_labels[:split_point]
    mve_test_labels = dev_labels[split_point:]

    # Z-score
    f_mean = np.nanmean(mve_train_feat, axis=0)
    f_std = np.nanstd(mve_train_feat, axis=0)
    f_std[f_std < 1e-10] = 1.0
    mve_train_z = np.nan_to_num((mve_train_feat - f_mean) / f_std, nan=0.0)
    mve_test_z = np.nan_to_num((mve_test_feat - f_mean) / f_std, nan=0.0)

    clf = xgb.XGBClassifier(**DEFAULT_XGB_PARAMS)
    clf.fit(mve_train_z, mve_train_labels)
    mve_preds = clf.predict(mve_test_z)
    mve_acc = float(accuracy_score(mve_test_labels, mve_preds))
    print(f"  MVE baseline accuracy: {mve_acc:.4f} (expected >=0.40)")
    assert mve_acc >= 0.40, f"ABORT: MVE baseline accuracy {mve_acc:.4f} < 0.40"
    print(f"  MVE Gate: Baseline reproduction PASSED")
    del clf

    # ======================================================================
    # MVE: Early Stopping Test
    # ======================================================================
    print("\n--- MVE: Early Stopping Test ---")

    # Reserve last 10% of training days as validation
    val_split = int(split_point * 0.9)
    mve_inner_train = mve_train_z[:val_split]
    mve_inner_val = mve_train_z[val_split:]
    mve_inner_train_labels = mve_train_labels[:val_split]
    mve_inner_val_labels = mve_train_labels[val_split:]

    es_config = {
        "max_depth": 4,
        "learning_rate": 0.01,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    es_params = {**TUNED_TEMPLATE, **es_config}
    clf_es = xgb.XGBClassifier(**es_params)
    clf_es.fit(
        mve_inner_train, mve_inner_train_labels,
        eval_set=[(mve_inner_val, mve_inner_val_labels)],
        verbose=False,
    )

    try:
        es_n_est = clf_es.best_iteration + 1
    except AttributeError:
        es_n_est = 2000
    es_preds = clf_es.predict(mve_test_z)
    es_acc = float(accuracy_score(mve_test_labels, es_preds))
    print(f"  Early stopping triggered at {es_n_est} trees (max 2000)")
    print(f"  Early stopping test accuracy: {es_acc:.4f} (expected >0.33)")
    assert es_n_est < 2000, f"ABORT: Early stopping did not trigger (used all 2000 trees)"
    assert es_acc > 0.33, f"ABORT: Early stopping accuracy {es_acc:.4f} <= 0.33"
    print(f"  MVE Gate: Early stopping test PASSED")
    del clf_es

    print(f"\n*** ALL MVE GATES PASSED ***")

    # ======================================================================
    # Phase 1: Coarse Random Search (48 configs + 1 default)
    # ======================================================================
    print("\n--- Phase 1: Coarse Random Search ---")

    rng = np.random.RandomState(SEED)

    # Generate 48 random configs
    random_configs = []
    for i in range(N_COARSE):
        random_configs.append(sample_random_config(rng))

    # All configs: default (#0) + 48 random
    all_configs = [None] + random_configs  # None = default config
    search_results = []

    nan_count = 0
    for cfg_idx, cfg in enumerate(all_configs):
        is_default = (cfg is None)
        t0 = time.time()

        try:
            result = eval_config_5fold(
                cfg, folds_5, dev_feat_z_folds, dev_labels, is_default=is_default
            )
        except Exception as e:
            print(f"  Config {cfg_idx}: FAILED ({e})")
            nan_count += 1
            search_results.append({
                "config_idx": cfg_idx,
                "config": cfg if cfg else "default",
                "mean_acc": 0.0,
                "std_acc": 0.0,
                "mean_train_acc": 0.0,
                "mean_n_estimators": 0,
                "error": str(e),
            })
            continue

        elapsed = time.time() - t0

        search_results.append({
            "config_idx": cfg_idx,
            "config": cfg if cfg else "default",
            "mean_acc": result["mean_acc"],
            "std_acc": result["std_acc"],
            "mean_train_acc": result["mean_train_acc"],
            "mean_n_estimators": result["mean_n_estimators"],
            "per_fold": result["per_fold"],
        })

        label = "DEFAULT" if is_default else f"config_{cfg_idx}"
        if cfg_idx % 10 == 0 or is_default:
            print(f"  [{cfg_idx:2d}/{len(all_configs)-1}] {label}: mean_acc={result['mean_acc']:.4f} "
                  f"(std={result['std_acc']:.4f}), train_acc={result['mean_train_acc']:.4f}, "
                  f"n_est={result['mean_n_estimators']:.0f}, {elapsed:.1f}s")

    # Abort checks
    if nan_count > len(all_configs) * 0.1:
        print(f"ABORT: >10% configs failed ({nan_count}/{len(all_configs)})")
        sys.exit(1)

    default_result = search_results[0]
    default_acc = default_result["mean_acc"]
    print(f"\n  Default config 5-fold CV accuracy: {default_acc:.4f}")

    if default_acc < 0.43:
        print(f"  *** ABORT: Default baseline {default_acc:.4f} < 0.43 (expected ~0.449) ***")
        sys.exit(1)

    all_accs = [r["mean_acc"] for r in search_results if "error" not in r]
    if max(all_accs) < 0.40:
        print(f"  *** ABORT: All configs < 0.40 accuracy ***")
        sys.exit(1)

    # Rank by mean accuracy
    ranked = sorted([r for r in search_results if "error" not in r], key=lambda x: x["mean_acc"], reverse=True)

    print(f"\n  Phase 1 Results:")
    print(f"  Default accuracy: {default_acc:.4f}")
    print(f"  Best accuracy: {ranked[0]['mean_acc']:.4f} (config_idx={ranked[0]['config_idx']})")
    print(f"  Worst accuracy: {ranked[-1]['mean_acc']:.4f}")
    print(f"  Configs above default: {sum(1 for r in ranked if r['mean_acc'] > default_acc)}")

    # Top 5 configs (excluding default if it's in top 5 — we want to search around non-default)
    top_5 = []
    for r in ranked:
        if r["config"] != "default" and len(top_5) < N_TOP:
            top_5.append(r)

    print(f"\n  Top 5 configs for fine search:")
    for i, t in enumerate(top_5):
        print(f"    #{i+1}: config_idx={t['config_idx']}, acc={t['mean_acc']:.4f}, "
              f"config={t['config']}")

    # Save coarse search results
    coarse_csv_data = []
    for r in search_results:
        row = {
            "config_idx": r["config_idx"],
            "is_default": r["config"] == "default",
            "mean_acc": r["mean_acc"],
            "std_acc": r["std_acc"],
            "mean_train_acc": r["mean_train_acc"],
            "mean_n_estimators": r["mean_n_estimators"],
        }
        if isinstance(r["config"], dict):
            for k, v in r["config"].items():
                row[k] = v
        coarse_csv_data.append(row)

    with open(RESULTS_DIR / "coarse_search.csv", "w") as f:
        if coarse_csv_data:
            keys = list(coarse_csv_data[0].keys())
            f.write(",".join(keys) + "\n")
            for row in coarse_csv_data:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    # ======================================================================
    # Phase 2: Fine Search (15 configs around top 5)
    # ======================================================================
    print("\n--- Phase 2: Fine Search ---")

    fine_configs = []
    for top_r in top_5:
        neighbors = generate_neighbors(top_r["config"], n_neighbors=N_FINE_PER_TOP)
        fine_configs.extend(neighbors)

    fine_results = []
    for fi, cfg in enumerate(fine_configs):
        t0 = time.time()
        try:
            result = eval_config_5fold(cfg, folds_5, dev_feat_z_folds, dev_labels, is_default=False)
        except Exception as e:
            print(f"  Fine config {fi}: FAILED ({e})")
            fine_results.append({
                "config_idx": fi + len(all_configs),
                "config": cfg,
                "mean_acc": 0.0,
                "std_acc": 0.0,
                "mean_train_acc": 0.0,
                "mean_n_estimators": 0,
                "error": str(e),
            })
            continue

        elapsed = time.time() - t0
        fine_results.append({
            "config_idx": fi + len(all_configs),
            "config": cfg,
            "mean_acc": result["mean_acc"],
            "std_acc": result["std_acc"],
            "mean_train_acc": result["mean_train_acc"],
            "mean_n_estimators": result["mean_n_estimators"],
            "per_fold": result["per_fold"],
        })

        if fi % 5 == 0:
            print(f"  [Fine {fi:2d}/{len(fine_configs)-1}]: acc={result['mean_acc']:.4f} "
                  f"(std={result['std_acc']:.4f}), {elapsed:.1f}s")

    # Save fine search results
    fine_csv_data = []
    for r in fine_results:
        row = {
            "config_idx": r["config_idx"],
            "mean_acc": r["mean_acc"],
            "std_acc": r["std_acc"],
            "mean_train_acc": r["mean_train_acc"],
            "mean_n_estimators": r["mean_n_estimators"],
        }
        if isinstance(r["config"], dict):
            for k, v in r["config"].items():
                row[k] = v
        fine_csv_data.append(row)

    with open(RESULTS_DIR / "fine_search.csv", "w") as f:
        if fine_csv_data:
            keys = list(fine_csv_data[0].keys())
            f.write(",".join(keys) + "\n")
            for row in fine_csv_data:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    # Combine all 64 configs and rank
    all_search_results = search_results + fine_results
    all_valid = [r for r in all_search_results if "error" not in r]
    all_ranked = sorted(all_valid, key=lambda x: x["mean_acc"], reverse=True)

    # Best config overall
    best_overall = all_ranked[0]
    best_config = best_overall["config"]
    best_search_acc = best_overall["mean_acc"]

    print(f"\n  All 64 configs ranked. Best: config_idx={best_overall['config_idx']}, acc={best_search_acc:.4f}")
    if isinstance(best_config, dict):
        print(f"  Best config params: {best_config}")
    else:
        print(f"  Best config: DEFAULT")

    # SC-4 check: at least 3 configs outperform default
    configs_above_default = sum(1 for r in all_ranked if r["mean_acc"] > default_acc and r["config"] != "default")
    print(f"  Configs above default (SC-4): {configs_above_default}")

    # Search landscape stats
    acc_values = [r["mean_acc"] for r in all_valid]
    search_landscape = {
        "min": float(min(acc_values)),
        "max": float(max(acc_values)),
        "mean": float(np.mean(acc_values)),
        "std": float(np.std(acc_values)),
        "p25": float(np.percentile(acc_values, 25)),
        "p50": float(np.percentile(acc_values, 50)),
        "p75": float(np.percentile(acc_values, 75)),
        "n_above_default": configs_above_default,
        "default_acc": default_acc,
    }
    print(f"  Search landscape: min={search_landscape['min']:.4f}, "
          f"max={search_landscape['max']:.4f}, std={search_landscape['std']:.4f}")

    # ======================================================================
    # Phase 3: CPCV Final Evaluation (best tuned vs default)
    # ======================================================================
    print("\n--- Phase 3: CPCV Final Evaluation ---")

    cpcv_splits, cpcv_groups = build_cpcv_splits(dev_day_indices)
    print(f"  CPCV splits: {len(cpcv_splits)}")

    # Evaluate both configs: best tuned and default
    configs_to_eval = {}
    if isinstance(best_config, dict):
        configs_to_eval["tuned"] = best_config
    else:
        # Best is default — still run both, tuned = second-best non-default
        # Find best non-default
        for r in all_ranked:
            if isinstance(r["config"], dict):
                configs_to_eval["tuned"] = r["config"]
                best_search_acc = r["mean_acc"]
                break
        if "tuned" not in configs_to_eval:
            # All configs are somehow default?? Use the default itself
            configs_to_eval["tuned"] = {
                "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10,
                "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            }
    configs_to_eval["default"] = "default"

    cpcv_results = {}

    for config_name, config in configs_to_eval.items():
        is_default = (config == "default")
        print(f"\n  Evaluating {config_name} on 45 CPCV splits...")

        split_results = []
        for s_idx, sd in enumerate(cpcv_splits):
            # Z-score normalization using training fold stats
            ft_train = dev_feat[sd["train_indices"]]
            f_mean = np.nanmean(ft_train, axis=0)
            f_std = np.nanstd(ft_train, axis=0)
            f_std[f_std < 1e-10] = 1.0

            ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
            ft_val_z = np.nan_to_num((dev_feat[sd["val_indices"]] - f_mean) / f_std, nan=0.0)
            ft_test_z = np.nan_to_num((dev_feat[sd["test_indices"]] - f_mean) / f_std, nan=0.0)

            lt_train = dev_labels[sd["train_indices"]]
            lt_val = dev_labels[sd["val_indices"]]
            lt_test = dev_labels[sd["test_indices"]]

            t0 = time.time()
            if is_default:
                preds, test_acc, train_acc, importance, n_est = train_xgb_default(
                    ft_train_z, lt_train, ft_test_z, lt_test
                )
            else:
                preds, test_acc, train_acc, importance, n_est = train_xgb_tuned(
                    config, ft_train_z, lt_train, ft_val_z, lt_val, ft_test_z, lt_test
                )
            elapsed = time.time() - t0

            # Map predictions back to original labels for PnL
            true_orig = np.array([inv_label_map[l] for l in lt_test])
            pred_orig = np.array([inv_label_map[p] for p in preds])

            # PnL under all cost scenarios
            pnl_by_cost = {}
            for scenario, rt_cost in COST_SCENARIOS.items():
                pnl = compute_pnl(true_orig, pred_orig, rt_cost)
                exp, pf, n_trades, gp, gl = compute_expectancy_and_pf(pnl)
                pnl_by_cost[scenario] = {
                    "expectancy": exp, "profit_factor": pf,
                    "n_trades": n_trades, "gross_profit": gp, "gross_loss": gl,
                }

            # Confusion matrix
            cm = confusion_matrix(lt_test, preds, labels=[0, 1, 2])

            # Quarter info for test bars
            test_days = dev_day_strings[sd["test_indices"]]
            quarters = [get_quarter(str(d)) for d in test_days]

            split_results.append({
                "split_idx": s_idx,
                "test_groups": sd["test_groups"],
                "test_acc": test_acc,
                "train_acc": train_acc,
                "n_estimators_used": n_est,
                "elapsed_s": elapsed,
                "true_labels": true_orig.tolist(),
                "pred_labels": pred_orig.tolist(),
                "pnl_by_cost": pnl_by_cost,
                "confusion_matrix": cm.tolist(),
                "importance": importance,
                "quarters": quarters,
            })

            if (s_idx + 1) % 10 == 0 or s_idx == 0:
                print(f"    Split {s_idx:2d}: acc={test_acc:.4f}, exp_base=${pnl_by_cost['base']['expectancy']:.4f}, {elapsed:.1f}s")

        # Aggregate CPCV results
        accs = [r["test_acc"] for r in split_results]
        train_accs = [r["train_acc"] for r in split_results]
        n_ests = [r["n_estimators_used"] for r in split_results]

        # Pool all predictions for per-class metrics
        all_true = []
        all_pred = []
        for r in split_results:
            all_true.extend(r["true_labels"])
            all_pred.extend(r["pred_labels"])
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        # Per-class recall
        per_class_recall = {}
        for cls in [-1, 0, 1]:
            cls_mask = all_true == cls
            if cls_mask.sum() > 0:
                per_class_recall[str(cls)] = float((all_pred[cls_mask] == cls).sum() / cls_mask.sum())

        # Pooled PnL
        pooled_pnl_by_cost = {}
        for scenario, rt_cost in COST_SCENARIOS.items():
            pnl = compute_pnl(all_true, all_pred, rt_cost)
            exp, pf, n_trades, gp, gl = compute_expectancy_and_pf(pnl)
            pooled_pnl_by_cost[scenario] = {
                "expectancy": exp, "profit_factor": pf,
                "n_trades": n_trades, "gross_profit": gp, "gross_loss": gl,
            }

        # Per-split expectancy for std calculation
        per_split_exp = [r["pnl_by_cost"]["base"]["expectancy"] for r in split_results]

        # Aggregate feature importance
        agg_importance = {}
        for r in split_results:
            for feat, gain in r["importance"].items():
                agg_importance[feat] = agg_importance.get(feat, 0.0) + gain
        total_gain = sum(agg_importance.values())
        top_10_importance = sorted(agg_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        # Per-quarter expectancy
        quarter_pnl = {"Q1": [], "Q2": [], "Q3": [], "Q4": []}
        for r in split_results:
            true_orig = np.array(r["true_labels"])
            pred_orig = np.array(r["pred_labels"])
            qs = r["quarters"]
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                q_mask = np.array([qq == q for qq in qs])
                if q_mask.sum() > 0:
                    q_pnl = compute_pnl(true_orig[q_mask], pred_orig[q_mask], COST_SCENARIOS["base"])
                    q_trades = q_pnl[q_pnl != 0]
                    if len(q_trades) > 0:
                        quarter_pnl[q].append(float(q_trades.mean()))

        per_quarter_exp = {q: float(np.mean(vals)) if vals else 0.0 for q, vals in quarter_pnl.items()}

        # label=0 trade fraction
        directional_preds = np.abs(all_pred) == 1
        if directional_preds.sum() > 0:
            label0_frac = float((all_true[directional_preds] == 0).sum() / directional_preds.sum())
        else:
            label0_frac = 0.0

        # Backtest paths for PBO
        paths, _ = cpcv_path_mapping(N_GROUPS, K_TEST)
        if len(paths) < 9:
            paths = [[i] for i in range(45)]

        path_accs = []
        path_exps = []
        for path_splits in paths:
            path_true = []
            path_pred = []
            for si in path_splits:
                path_true.extend(split_results[si]["true_labels"])
                path_pred.extend(split_results[si]["pred_labels"])
            path_true = np.array(path_true)
            path_pred = np.array(path_pred)
            # Map back to CE labels for accuracy
            path_true_ce = np.array([label_map[l] for l in path_true])
            path_pred_ce = np.array([label_map[p] for p in path_pred])
            path_accs.append(float(accuracy_score(path_true_ce, path_pred_ce)))
            path_pnl = compute_pnl(path_true, path_pred, COST_SCENARIOS["base"])
            exp, _, _, _, _ = compute_expectancy_and_pf(path_pnl)
            path_exps.append(exp)

        # PBO (probability of backtest overfitting)
        n_negative = sum(1 for e in path_exps if e < 0)
        pbo = n_negative / len(path_exps) if path_exps else 1.0

        cpcv_results[config_name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_train_accuracy": float(np.mean(train_accs)),
            "mean_expectancy_base": pooled_pnl_by_cost["base"]["expectancy"],
            "std_expectancy": float(np.std(per_split_exp)),
            "per_class_recall": per_class_recall,
            "profit_factor": pooled_pnl_by_cost["base"]["profit_factor"],
            "cost_sensitivity": pooled_pnl_by_cost,
            "per_quarter_expectancy": per_quarter_exp,
            "top_10_importance": [(f, float(g)) for f, g in top_10_importance],
            "mean_n_estimators": float(np.mean(n_ests)),
            "path_accuracies": path_accs,
            "path_expectancies": path_exps,
            "pbo": pbo,
            "label0_trade_fraction": label0_frac,
            "per_split": [{
                "split_idx": r["split_idx"],
                "test_groups": r["test_groups"],
                "test_acc": r["test_acc"],
                "train_acc": r["train_acc"],
                "n_estimators_used": r["n_estimators_used"],
                "expectancy_base": r["pnl_by_cost"]["base"]["expectancy"],
            } for r in split_results],
        }

        print(f"\n  {config_name} CPCV summary:")
        print(f"    Mean accuracy: {np.mean(accs):.4f} (std={np.std(accs):.4f})")
        print(f"    Mean expectancy (base): ${pooled_pnl_by_cost['base']['expectancy']:.4f}")
        print(f"    Profit factor: {pooled_pnl_by_cost['base']['profit_factor']:.4f}")
        print(f"    Per-class recall: {per_class_recall}")
        print(f"    Per-quarter exp: {per_quarter_exp}")
        print(f"    PBO: {pbo:.4f}")
        print(f"    Label0 trade fraction: {label0_frac:.4f}")

    # Save CPCV results
    cpcv_save = {}
    for name, res in cpcv_results.items():
        cpcv_save[name] = {k: v for k, v in res.items() if k != "per_split"}
        cpcv_save[name]["per_split"] = res["per_split"]
    with open(RESULTS_DIR / "cpcv_results.json", "w") as f:
        json.dump(cpcv_save, f, indent=2, default=str)

    # ======================================================================
    # Phase 4: Holdout Evaluation + Walk-Forward
    # ======================================================================
    print("\n--- Phase 4: Holdout Evaluation ---")

    holdout_results = {}
    tuned_config = configs_to_eval["tuned"]

    for config_name in ["tuned", "default"]:
        is_default = (config_name == "default")
        cfg = "default" if is_default else tuned_config

        # Train on ALL dev days
        f_mean = np.nanmean(dev_feat, axis=0)
        f_std = np.nanstd(dev_feat, axis=0)
        f_std[f_std < 1e-10] = 1.0

        dev_feat_z = np.nan_to_num((dev_feat - f_mean) / f_std, nan=0.0)
        holdout_feat_z = np.nan_to_num((holdout_feat - f_mean) / f_std, nan=0.0)

        if is_default:
            preds, test_acc, train_acc, importance, n_est = train_xgb_default(
                dev_feat_z, dev_labels, holdout_feat_z, holdout_labels
            )
        else:
            # Internal 80/20 val split for early stopping
            val_split_idx = int(len(dev_feat_z) * 0.8)
            inner_train = dev_feat_z[:val_split_idx]
            inner_val = dev_feat_z[val_split_idx:]
            inner_train_labels = dev_labels[:val_split_idx]
            inner_val_labels = dev_labels[val_split_idx:]

            preds, test_acc, train_acc, importance, n_est = train_xgb_tuned(
                cfg, inner_train, inner_train_labels,
                inner_val, inner_val_labels,
                holdout_feat_z, holdout_labels
            )

        # Map back to original labels
        true_orig = np.array([inv_label_map[l] for l in holdout_labels])
        pred_orig = np.array([inv_label_map[p] for p in preds])

        # PnL
        pnl_by_cost = {}
        for scenario, rt_cost in COST_SCENARIOS.items():
            pnl = compute_pnl(true_orig, pred_orig, rt_cost)
            exp, pf, n_trades, gp, gl = compute_expectancy_and_pf(pnl)
            pnl_by_cost[scenario] = {
                "expectancy": exp, "profit_factor": pf,
                "n_trades": n_trades, "gross_profit": gp, "gross_loss": gl,
            }

        # Confusion matrix
        cm = confusion_matrix(holdout_labels, preds, labels=[0, 1, 2])

        # Per-class recall
        per_class_recall = {}
        for cls in [-1, 0, 1]:
            cls_mask = true_orig == cls
            if cls_mask.sum() > 0:
                per_class_recall[str(cls)] = float((pred_orig[cls_mask] == cls).sum() / cls_mask.sum())

        holdout_results[config_name] = {
            "accuracy": test_acc,
            "train_accuracy": train_acc,
            "n_estimators_used": n_est,
            "expectancy_base": pnl_by_cost["base"]["expectancy"],
            "profit_factor": pnl_by_cost["base"]["profit_factor"],
            "cost_sensitivity": pnl_by_cost,
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm.tolist(),
        }

        print(f"\n  {config_name} holdout:")
        print(f"    Accuracy: {test_acc:.4f}")
        print(f"    Expectancy (base): ${pnl_by_cost['base']['expectancy']:.4f}")
        print(f"    Profit factor: {pnl_by_cost['base']['profit_factor']:.4f}")
        print(f"    Per-class recall: {per_class_recall}")

    # Save holdout results
    with open(RESULTS_DIR / "holdout_results.json", "w") as f:
        json.dump(holdout_results, f, indent=2, default=str)

    # Walk-forward (best config only)
    print("\n--- Walk-Forward Validation ---")

    wf_folds = [
        {"train_day_range": (1, 100), "test_day_range": (101, 140)},
        {"train_day_range": (1, 140), "test_day_range": (141, 180)},
        {"train_day_range": (1, 180), "test_day_range": (181, 201)},
    ]

    wf_results = []
    for wf_idx, wf in enumerate(wf_folds):
        train_range = wf["train_day_range"]
        test_range = wf["test_day_range"]

        train_mask = (dev_day_indices >= train_range[0]) & (dev_day_indices <= train_range[1])
        test_mask = (dev_day_indices >= test_range[0]) & (dev_day_indices <= test_range[1])

        # Apply purge between train and test
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        clean_train, _ = apply_purge_embargo(
            train_indices, test_indices, dev_day_indices, PURGE_BARS, 0  # No embargo for WF
        )

        ft_train = dev_feat[clean_train]
        ft_test = dev_feat[test_indices]
        lt_train = dev_labels[clean_train]
        lt_test = dev_labels[test_indices]

        f_mean = np.nanmean(ft_train, axis=0)
        f_std = np.nanstd(ft_train, axis=0)
        f_std[f_std < 1e-10] = 1.0
        ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
        ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)

        # Internal val split
        val_split = int(len(ft_train_z) * 0.9)
        inner_train_z = ft_train_z[:val_split]
        inner_val_z = ft_train_z[val_split:]
        inner_train_labels = lt_train[:val_split]
        inner_val_labels = lt_train[val_split:]

        preds, test_acc, train_acc, importance, n_est = train_xgb_tuned(
            tuned_config, inner_train_z, inner_train_labels,
            inner_val_z, inner_val_labels,
            ft_test_z, lt_test
        )

        true_orig = np.array([inv_label_map[l] for l in lt_test])
        pred_orig = np.array([inv_label_map[p] for p in preds])
        pnl = compute_pnl(true_orig, pred_orig, COST_SCENARIOS["base"])
        exp, pf, n_trades, _, _ = compute_expectancy_and_pf(pnl)

        wf_results.append({
            "fold": wf_idx,
            "train_days": f"{train_range[0]}-{train_range[1]}",
            "test_days": f"{test_range[0]}-{test_range[1]}",
            "accuracy": test_acc,
            "expectancy_base": exp,
            "n_estimators_used": n_est,
            "n_trades": n_trades,
        })

        print(f"  WF fold {wf_idx}: train days {train_range[0]}-{train_range[1]}, "
              f"test days {test_range[0]}-{test_range[1]}: acc={test_acc:.4f}, exp=${exp:.4f}")

    wf_mean_acc = float(np.mean([r["accuracy"] for r in wf_results]))
    wf_mean_exp = float(np.mean([r["expectancy_base"] for r in wf_results]))
    print(f"\n  Walk-forward mean accuracy: {wf_mean_acc:.4f}")
    print(f"  Walk-forward mean expectancy: ${wf_mean_exp:.4f}")

    # Save walk-forward results
    with open(RESULTS_DIR / "walkforward_results.csv", "w") as f:
        f.write("fold,train_days,test_days,accuracy,expectancy_base,n_estimators_used,n_trades\n")
        for r in wf_results:
            f.write(f"{r['fold']},{r['train_days']},{r['test_days']},{r['accuracy']:.6f},"
                    f"{r['expectancy_base']:.6f},{r['n_estimators_used']},{r['n_trades']}\n")

    # ======================================================================
    # Metrics Collection and Output
    # ======================================================================
    print("\n--- Metrics Collection ---")

    total_elapsed = time.time() - start_time

    # Determine best config for reporting
    tuned_cpcv = cpcv_results["tuned"]
    default_cpcv = cpcv_results["default"]

    # Success criteria evaluation
    sc1_pass = tuned_cpcv["mean_accuracy"] >= 0.469
    sc2_pass = tuned_cpcv["mean_expectancy_base"] >= 0.0
    sc3_pass = holdout_results["tuned"]["accuracy"] >= 0.441
    sc4_pass = configs_above_default >= 3
    sc5_pass = tuned_cpcv["std_accuracy"] < 0.05

    # Sanity checks
    sanity_train_gt_test = tuned_cpcv["mean_train_accuracy"] > tuned_cpcv["mean_accuracy"]

    tuned_top10 = tuned_cpcv["top_10_importance"]
    total_top10_gain = sum(g for _, g in tuned_top10)
    tuned_agg_importance = {}
    for r in cpcv_results["tuned"]["per_split"]:
        pass  # Already aggregated
    max_single_feat_share = (tuned_top10[0][1] / sum(g for _, g in tuned_top10)) if tuned_top10 else 0.0
    sanity_no_single_feat_dominant = max_single_feat_share < 0.50

    sanity_holdout_within_5pp = abs(holdout_results["tuned"]["accuracy"] - tuned_cpcv["mean_accuracy"]) < 0.05

    # Early stopping triggers (>= 90% of CPCV fits)
    es_triggered_count = sum(1 for r in cpcv_results["tuned"]["per_split"] if r["n_estimators_used"] < 2000)
    sanity_early_stopping = es_triggered_count >= len(cpcv_results["tuned"]["per_split"]) * 0.9

    # Default reproduces ~0.449 ± 2pp
    sanity_default_reproduces = abs(default_cpcv["mean_accuracy"] - 0.449) <= 0.02

    # Determine outcome
    if sc1_pass and sc2_pass:
        outcome = "A"
        outcome_desc = "CONFIRMED — tuning closes the breakeven gap"
    elif sc1_pass and not sc2_pass:
        outcome = "B"
        outcome_desc = "PARTIAL — accuracy improves but costs still dominate"
    elif sc4_pass and not sc1_pass:
        outcome = "C"
        outcome_desc = "MARGINAL — some configs better, but <2pp gain"
    else:
        outcome = "D"
        outcome_desc = "REFUTED — default params are at or near optimal"

    # Feature importance comparison
    feature_importance = {
        "tuned": {f: float(g) for f, g in tuned_cpcv["top_10_importance"]},
        "default": {f: float(g) for f, g in default_cpcv["top_10_importance"]},
    }
    with open(RESULTS_DIR / "feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)

    # Cost sensitivity
    cost_sensitivity = {
        "tuned": tuned_cpcv["cost_sensitivity"],
        "default": default_cpcv["cost_sensitivity"],
    }
    with open(RESULTS_DIR / "cost_sensitivity.json", "w") as f:
        json.dump(cost_sensitivity, f, indent=2)

    # Long vs short recall
    long_recall_tuned = tuned_cpcv["per_class_recall"].get("1", 0.0)
    short_recall_tuned = tuned_cpcv["per_class_recall"].get("-1", 0.0)
    long_recall_default = default_cpcv["per_class_recall"].get("1", 0.0)
    short_recall_default = default_cpcv["per_class_recall"].get("-1", 0.0)

    # Breakeven cost for tuned model
    # Find RT cost where expectancy = 0 by interpolation
    breakeven_rt = None
    opt_exp = tuned_cpcv["cost_sensitivity"]["optimistic"]["expectancy"]
    base_exp = tuned_cpcv["cost_sensitivity"]["base"]["expectancy"]
    pess_exp = tuned_cpcv["cost_sensitivity"]["pessimistic"]["expectancy"]

    if opt_exp > 0 and base_exp < 0:
        # Linear interpolation between optimistic and base
        breakeven_rt = 2.49 + (3.74 - 2.49) * opt_exp / (opt_exp - base_exp)
    elif base_exp > 0 and pess_exp < 0:
        breakeven_rt = 3.74 + (6.25 - 3.74) * base_exp / (base_exp - pess_exp)
    elif opt_exp > 0 and base_exp > 0 and pess_exp > 0:
        breakeven_rt = float("inf")  # Profitable at all cost levels

    # Metrics JSON
    best_config_report = configs_to_eval["tuned"] if isinstance(configs_to_eval["tuned"], dict) else DEFAULT_XGB_PARAMS.copy()

    metrics = {
        "experiment": "xgb-hyperparam-tuning",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "cpcv_mean_accuracy": default_cpcv["mean_accuracy"],
            "cpcv_mean_expectancy_base": default_cpcv["mean_expectancy_base"],
            "cpcv_accuracy_std": default_cpcv["std_accuracy"],
            "holdout_accuracy": holdout_results["default"]["accuracy"],
            "holdout_expectancy_base": holdout_results["default"]["expectancy_base"],
            "profit_factor": default_cpcv["profit_factor"],
            "per_class_recall": default_cpcv["per_class_recall"],
            "per_quarter_expectancy": default_cpcv["per_quarter_expectancy"],
            "top_10_importance": default_cpcv["top_10_importance"],
            "pbo": default_cpcv["pbo"],
        },
        "treatment": {
            "best_config": best_config_report,
            "search_best_cv_accuracy": best_search_acc,
            "cpcv_mean_accuracy": tuned_cpcv["mean_accuracy"],
            "cpcv_mean_expectancy_base": tuned_cpcv["mean_expectancy_base"],
            "cpcv_accuracy_std": tuned_cpcv["std_accuracy"],
            "cpcv_expectancy_std": tuned_cpcv["std_expectancy"],
            "holdout_accuracy": holdout_results["tuned"]["accuracy"],
            "holdout_expectancy_base": holdout_results["tuned"]["expectancy_base"],
            "profit_factor": tuned_cpcv["profit_factor"],
            "per_class_recall": tuned_cpcv["per_class_recall"],
            "per_quarter_expectancy": tuned_cpcv["per_quarter_expectancy"],
            "top_10_importance": tuned_cpcv["top_10_importance"],
            "best_n_estimators_mean": tuned_cpcv["mean_n_estimators"],
            "pbo": tuned_cpcv["pbo"],
            "walkforward_accuracy": wf_mean_acc,
            "walkforward_expectancy": wf_mean_exp,
            "long_recall_vs_short": {
                "tuned_long": long_recall_tuned,
                "tuned_short": short_recall_tuned,
                "default_long": long_recall_default,
                "default_short": short_recall_default,
            },
            "cost_sensitivity": cost_sensitivity,
            "breakeven_rt_cost": breakeven_rt,
            "label0_trade_fraction": tuned_cpcv["label0_trade_fraction"],
        },
        "search_landscape": {
            "total_configs": len(all_valid),
            "configs_above_default": configs_above_default,
            "accuracy_distribution": search_landscape,
            "all_configs_accuracy": [{"config_idx": r["config_idx"], "mean_acc": r["mean_acc"]} for r in all_ranked],
        },
        "sanity_checks": {
            "train_acc_gt_test_acc": sanity_train_gt_test,
            "no_single_feature_gt_50pct_gain": sanity_no_single_feat_dominant,
            "holdout_within_5pp_of_cpcv": sanity_holdout_within_5pp,
            "early_stopping_triggers_90pct": sanity_early_stopping,
            "default_reproduces_baseline": sanity_default_reproduces,
            "early_stopping_trigger_rate": f"{es_triggered_count}/{len(cpcv_results['tuned']['per_split'])}",
            "max_single_feature_gain_share": max_single_feat_share,
            "holdout_cpcv_delta": abs(holdout_results["tuned"]["accuracy"] - tuned_cpcv["mean_accuracy"]),
            "default_cpcv_accuracy": default_cpcv["mean_accuracy"],
        },
        "success_criteria": {
            "SC-1": {
                "description": "CPCV mean accuracy >= 0.469",
                "value": tuned_cpcv["mean_accuracy"],
                "threshold": 0.469,
                "pass": sc1_pass,
            },
            "SC-2": {
                "description": "CPCV mean expectancy >= $0.00 (base costs)",
                "value": tuned_cpcv["mean_expectancy_base"],
                "threshold": 0.0,
                "pass": sc2_pass,
            },
            "SC-3": {
                "description": "Holdout accuracy >= 0.441",
                "value": holdout_results["tuned"]["accuracy"],
                "threshold": 0.441,
                "pass": sc3_pass,
            },
            "SC-4": {
                "description": "At least 3 search configs outperform default on 5-fold CV",
                "value": configs_above_default,
                "threshold": 3,
                "pass": sc4_pass,
            },
            "SC-5": {
                "description": "Best config CPCV accuracy std < 0.05",
                "value": tuned_cpcv["std_accuracy"],
                "threshold": 0.05,
                "pass": sc5_pass,
            },
        },
        "outcome": outcome,
        "outcome_description": outcome_desc,
        "resource_usage": {
            "wall_clock_seconds": total_elapsed,
            "wall_clock_minutes": total_elapsed / 60,
            "total_training_runs": len(all_valid) * 5 + len(fine_configs) * 5 + 45 * 2 + 5 + 3,
            "total_configs_evaluated": len(all_valid) + len(fine_configs),
        },
        "abort_triggered": False,
        "abort_reason": None,
        "notes": (
            f"Executed locally on Apple Silicon. "
            f"Default config {config_name} 5-fold CV accuracy: {default_acc:.4f}. "
            f"Best search config 5-fold CV accuracy: {best_search_acc:.4f}. "
            f"Wall clock: {total_elapsed/60:.1f} minutes."
        ),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Outcome: {outcome} — {outcome_desc}")
    print(f"\nSuccess Criteria:")
    for sc, details in metrics["success_criteria"].items():
        status = "PASS" if details["pass"] else "FAIL"
        print(f"  {sc}: {status} ({details['description']}: {details['value']:.4f} vs {details['threshold']})")
    print(f"\nSanity Checks:")
    for check, val in metrics["sanity_checks"].items():
        print(f"  {check}: {val}")
    print(f"\nDefault CPCV: acc={default_cpcv['mean_accuracy']:.4f}, exp=${default_cpcv['mean_expectancy_base']:.4f}")
    print(f"Tuned CPCV:   acc={tuned_cpcv['mean_accuracy']:.4f}, exp=${tuned_cpcv['mean_expectancy_base']:.4f}")
    print(f"Tuned holdout: acc={holdout_results['tuned']['accuracy']:.4f}, exp=${holdout_results['tuned']['expectancy_base']:.4f}")
    print(f"Walk-forward: acc={wf_mean_acc:.4f}, exp=${wf_mean_exp:.4f}")
    print(f"\nBest config: {best_config_report}")
    print(f"Wall clock: {total_elapsed/60:.1f} minutes")
    print(f"\nMetrics written to: {RESULTS_DIR / 'metrics.json'}")

    # ======================================================================
    # Write analysis.md
    # ======================================================================
    print("\n--- Writing analysis.md ---")

    a_lines = []
    def a(line=""):
        a_lines.append(line)

    a("# XGBoost Hyperparameter Tuning — Analysis")
    a()
    a(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    a(f"**Outcome:** {outcome} — {outcome_desc}")
    a()

    a("## 1. Executive Summary")
    a()
    if outcome == "A":
        a(f"Systematic hyperparameter tuning improved XGBoost 3-class accuracy from {default_cpcv['mean_accuracy']:.4f} "
          f"to {tuned_cpcv['mean_accuracy']:.4f} ({(tuned_cpcv['mean_accuracy'] - default_cpcv['mean_accuracy'])*100:.1f}pp gain) "
          f"and pushed CPCV mean expectancy to ${tuned_cpcv['mean_expectancy_base']:.4f}/trade under base costs. "
          f"Both SC-1 and SC-2 pass — tuning closes the breakeven gap.")
    elif outcome == "B":
        a(f"Tuning improved accuracy from {default_cpcv['mean_accuracy']:.4f} to {tuned_cpcv['mean_accuracy']:.4f} "
          f"({(tuned_cpcv['mean_accuracy'] - default_cpcv['mean_accuracy'])*100:.1f}pp gain), "
          f"but expectancy remains negative at ${tuned_cpcv['mean_expectancy_base']:.4f}/trade. "
          f"Accuracy improves but transaction costs still dominate.")
    elif outcome == "C":
        a(f"Some configs outperform default ({configs_above_default} configs above baseline), "
          f"but the best achieves only {tuned_cpcv['mean_accuracy']:.4f} vs {default_cpcv['mean_accuracy']:.4f} "
          f"({(tuned_cpcv['mean_accuracy'] - default_cpcv['mean_accuracy'])*100:.1f}pp). "
          f"Default params are near-optimal; tuning is not the bottleneck.")
    else:
        a(f"No config outperforms the default ({configs_above_default} configs above baseline, "
          f"threshold: 3). Default accuracy {default_cpcv['mean_accuracy']:.4f} is at or near optimal "
          f"for this feature set. Tuning is not the path forward.")
    a()

    a("## 2. Search Landscape")
    a()
    a(f"Total configs evaluated: {len(all_valid)} (49 coarse + {len(fine_configs)} fine)")
    a(f"- Min accuracy: {search_landscape['min']:.4f}")
    a(f"- Max accuracy: {search_landscape['max']:.4f}")
    a(f"- Mean accuracy: {search_landscape['mean']:.4f}")
    a(f"- Std accuracy: {search_landscape['std']:.4f}")
    a(f"- Configs above default: {configs_above_default}")
    a(f"- Default accuracy: {default_acc:.4f}")
    a()

    a("## 3. Best Tuned Hyperparameters vs Default")
    a()
    a("| Parameter | Default | Tuned |")
    a("|-----------|---------|-------|")
    default_params = {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10,
                      "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
                      "n_estimators": 500}
    for param in ["max_depth", "learning_rate", "min_child_weight", "subsample",
                   "colsample_bytree", "reg_alpha", "reg_lambda"]:
        def_val = default_params.get(param, "—")
        tun_val = best_config_report.get(param, "—")
        if isinstance(tun_val, float):
            tun_val = f"{tun_val:.6f}"
        if isinstance(def_val, float):
            def_val = f"{def_val:.6f}"
        a(f"| {param} | {def_val} | {tun_val} |")
    a(f"| n_estimators | 500 (fixed) | 2000 (early stopping, mean used: {tuned_cpcv['mean_n_estimators']:.0f}) |")
    a()

    a("## 4. CPCV Comparison: Best Tuned vs Default")
    a()
    a("| Metric | Default | Tuned | Delta |")
    a("|--------|---------|-------|-------|")
    a(f"| CPCV Mean Accuracy | {default_cpcv['mean_accuracy']:.4f} | {tuned_cpcv['mean_accuracy']:.4f} | {(tuned_cpcv['mean_accuracy']-default_cpcv['mean_accuracy'])*100:+.1f}pp |")
    a(f"| CPCV Accuracy Std | {default_cpcv['std_accuracy']:.4f} | {tuned_cpcv['std_accuracy']:.4f} | — |")
    a(f"| Expectancy (base) | ${default_cpcv['mean_expectancy_base']:.4f} | ${tuned_cpcv['mean_expectancy_base']:.4f} | ${tuned_cpcv['mean_expectancy_base']-default_cpcv['mean_expectancy_base']:+.4f} |")
    a(f"| Profit Factor | {default_cpcv['profit_factor']:.4f} | {tuned_cpcv['profit_factor']:.4f} | {tuned_cpcv['profit_factor']-default_cpcv['profit_factor']:+.4f} |")
    a(f"| Short (-1) Recall | {short_recall_default:.4f} | {short_recall_tuned:.4f} | {short_recall_tuned-short_recall_default:+.4f} |")
    a(f"| Hold (0) Recall | {default_cpcv['per_class_recall'].get('0', 0):.4f} | {tuned_cpcv['per_class_recall'].get('0', 0):.4f} | — |")
    a(f"| Long (+1) Recall | {long_recall_default:.4f} | {long_recall_tuned:.4f} | {long_recall_tuned-long_recall_default:+.4f} |")
    a(f"| PBO | {default_cpcv['pbo']:.4f} | {tuned_cpcv['pbo']:.4f} | — |")
    a()

    a("## 5. Per-Quarter Expectancy Breakdown")
    a()
    a("| Quarter | Default | Tuned |")
    a("|---------|---------|-------|")
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        def_q = default_cpcv["per_quarter_expectancy"].get(q, 0.0)
        tun_q = tuned_cpcv["per_quarter_expectancy"].get(q, 0.0)
        a(f"| {q} | ${def_q:.4f} | ${tun_q:.4f} |")
    a()

    a("## 6. Feature Importance Comparison (Top 10 by Gain)")
    a()
    a("### Default Config")
    a("| Rank | Feature | Gain |")
    a("|------|---------|------|")
    for i, (feat, gain) in enumerate(default_cpcv["top_10_importance"]):
        a(f"| {i+1} | {feat} | {gain:.1f} |")
    a()
    a("### Tuned Config")
    a("| Rank | Feature | Gain |")
    a("|------|---------|------|")
    for i, (feat, gain) in enumerate(tuned_cpcv["top_10_importance"]):
        a(f"| {i+1} | {feat} | {gain:.1f} |")
    a()

    a("## 7. Walk-Forward vs CPCV Consistency")
    a()
    a(f"- CPCV mean accuracy (tuned): {tuned_cpcv['mean_accuracy']:.4f}")
    a(f"- Walk-forward mean accuracy: {wf_mean_acc:.4f}")
    wf_divergence = abs(wf_mean_acc - tuned_cpcv["mean_accuracy"])
    a(f"- Divergence: {wf_divergence*100:.1f}pp ({'<5pp OK' if wf_divergence < 0.05 else '>5pp WARNING — CPCV temporal mixing may bias results'})")
    a()
    a("| WF Fold | Train Days | Test Days | Accuracy | Expectancy |")
    a("|---------|------------|-----------|----------|------------|")
    for r in wf_results:
        a(f"| {r['fold']} | {r['train_days']} | {r['test_days']} | {r['accuracy']:.4f} | ${r['expectancy_base']:.4f} |")
    a()

    a("## 8. Holdout Evaluation (SACRED — one shot)")
    a()
    a("| Metric | Default | Tuned | Delta |")
    a("|--------|---------|-------|-------|")
    a(f"| Accuracy | {holdout_results['default']['accuracy']:.4f} | {holdout_results['tuned']['accuracy']:.4f} | {(holdout_results['tuned']['accuracy']-holdout_results['default']['accuracy'])*100:+.1f}pp |")
    a(f"| Expectancy (base) | ${holdout_results['default']['expectancy_base']:.4f} | ${holdout_results['tuned']['expectancy_base']:.4f} | ${holdout_results['tuned']['expectancy_base']-holdout_results['default']['expectancy_base']:+.4f} |")
    a(f"| Profit Factor | {holdout_results['default']['profit_factor']:.4f} | {holdout_results['tuned']['profit_factor']:.4f} | — |")
    a()

    a("## 9. Cost Sensitivity")
    a()
    a("| Scenario | RT Cost | Default Exp | Tuned Exp | Default PF | Tuned PF |")
    a("|----------|---------|-------------|-----------|------------|----------|")
    for scenario in ["optimistic", "base", "pessimistic"]:
        rt = COST_SCENARIOS[scenario]
        def_exp = default_cpcv["cost_sensitivity"][scenario]["expectancy"]
        tun_exp = tuned_cpcv["cost_sensitivity"][scenario]["expectancy"]
        def_pf = default_cpcv["cost_sensitivity"][scenario]["profit_factor"]
        tun_pf = tuned_cpcv["cost_sensitivity"][scenario]["profit_factor"]
        a(f"| {scenario.capitalize()} | ${rt:.2f} | ${def_exp:.4f} | ${tun_exp:.4f} | {def_pf:.4f} | {tun_pf:.4f} |")
    if breakeven_rt is not None:
        a(f"\nBreakeven RT cost for tuned model: ${breakeven_rt:.2f}")
    a()

    a("## 10. Success Criteria Pass/Fail")
    a()
    for sc, details in metrics["success_criteria"].items():
        status = "PASS" if details["pass"] else "FAIL"
        a(f"- **{sc}**: {status} — {details['description']} (value: {details['value']:.4f}, threshold: {details['threshold']})")
    a(f"- **Sanity checks**: {'All passed' if all([sanity_train_gt_test, sanity_no_single_feat_dominant, sanity_holdout_within_5pp, sanity_early_stopping, sanity_default_reproduces]) else 'Some failed — see metrics.json'}")
    a()
    a(f"**Outcome: {outcome} — {outcome_desc}**")
    a()

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(a_lines))

    print(f"analysis.md written to: {RESULTS_DIR / 'analysis.md'}")
    print(f"\nDone. Total wall clock: {total_elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
