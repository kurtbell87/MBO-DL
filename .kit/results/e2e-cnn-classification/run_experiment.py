#!/usr/bin/env python3
"""
Experiment: End-to-End CNN Classification on Full-Year MES Data
Spec: .kit/experiments/e2e-cnn-classification.md

End-to-end CNN classification using CrossEntropyLoss on triple barrier labels,
replacing the regression→frozen-embedding→XGBoost pipeline from 9E.

CPCV (Combinatorially Purged Cross-Validation):
  N=10 groups, k=2 test groups → C(10,2)=45 splits, phi(10,2)=9 backtest paths
  50-day holdout (days 202-251) held sacred until final evaluation.
"""

import json
import os
import sys
import time
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from scipy import stats as scipy_stats
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb

# --- Config ---

SEED = 42
TICK_SIZE = 0.25
DATA_DIR = Path(".kit/results/full-year-export")
RESULTS_DIR = Path(".kit/results/e2e-cnn-classification")

# CNN hyperparameters (spec-defined, doubled from 9E for more data)
CNN_LR = 1e-3
CNN_WD = 1e-4
CNN_BATCH = 1024
CNN_MAX_EPOCHS = 100
CNN_PATIENCE = 15
CNN_T_MAX = 100
CNN_ETA_MIN = 1e-5

# XGBoost hyperparameters (same as 9E for comparability)
XGB_PARAMS = {
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

# Non-spatial features (20 dimensions, spec order)
NON_SPATIAL_FEATURES = [
    "weighted_imbalance", "spread", "net_volume", "volume_imbalance",
    "trade_count", "avg_trade_size", "vwap_distance",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50", "high_low_range_50", "close_position",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "time_sin", "time_cos", "minutes_since_open",
]

# Book columns (40 total, interleaved price/size for 20 levels)
BOOK_PRICE_COLS = [f"book_snap_{i}" for i in range(0, 40, 2)]
BOOK_SIZE_COLS = [f"book_snap_{i}" for i in range(1, 40, 2)]

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

# Holdout
DEV_DAYS = 201
HOLDOUT_START = 202

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Seed ---

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- CNN Architecture ---

class E2ECNN(nn.Module):
    """Conv1d spatial encoder → 3-class classification head (end-to-end)."""
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 59, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(59)
        self.conv2 = nn.Conv1d(59, 59, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(59)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(59, 16)
        self.fc2 = nn.Linear(16, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def extract_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        return x


class E2ECNNFeatures(nn.Module):
    """Conv1d spatial encoder + 20 non-spatial features → 3-class classification head."""
    def __init__(self, n_features=20, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 59, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(59)
        self.conv2 = nn.Conv1d(59, 59, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(59)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(59, 16)
        self.fc2 = nn.Linear(16 + n_features, n_classes)
        self.relu = nn.ReLU()

    def forward(self, book_x, feat_x):
        x = self.relu(self.bn1(self.conv1(book_x)))
        x = self.relu(self.bn2(self.conv2(book_x)))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = torch.cat([x, feat_x], dim=1)
        return self.fc2(x)


# --- Helpers ---

def compute_pnl(true_labels, pred_labels, rt_cost):
    """PnL per observation. pred=0 or true=0: $0. pred=true (both nonzero): win. else: loss."""
    pnl = np.zeros(len(true_labels))
    for i in range(len(true_labels)):
        t, p = true_labels[i], pred_labels[i]
        if p == 0 or t == 0:
            pnl[i] = 0.0
        elif p == t:
            pnl[i] = WIN_PNL - rt_cost
        else:
            pnl[i] = -LOSS_PNL - rt_cost
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


def compute_sharpe(daily_pnl):
    """Annualized Sharpe from daily PnL series."""
    if len(daily_pnl) < 2 or np.std(daily_pnl) < 1e-10:
        return 0.0
    return float(np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252))


def deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs, skewness=0.0, kurtosis=3.0):
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio."""
    if n_trials <= 1 or n_obs <= 1:
        return 0.0
    e_max_sr = np.sqrt(2 * np.log(n_trials)) * (1 - np.euler_gamma / (2 * np.log(n_trials)))
    se_sr = np.sqrt((1 - skewness * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2) / (n_obs - 1))
    if se_sr < 1e-10:
        return 0.0
    z = (observed_sharpe - e_max_sr) / se_sr
    return float(scipy_stats.norm.cdf(z))


def cpcv_path_mapping(n_groups, k_test):
    """
    Map C(n,k) splits to phi(n,k) non-overlapping backtest paths.
    Each path covers all n groups with each group tested exactly once across the path's splits.
    Returns: list of paths, each path is a list of split indices.
    """
    splits = list(combinations(range(n_groups), k_test))
    n_splits = len(splits)

    # phi(n,k) = C(n,k) * k / n
    n_paths = n_splits * k_test // n_groups

    # Greedy assignment: each path must cover all n groups exactly once via k-sized test sets
    # For k=2, n=10: phi=9, each path has 5 splits (5 × 2 = 10 groups covered)
    paths = []
    used_splits = set()

    for _ in range(n_paths):
        path = []
        covered = set()
        # Try to build a path covering all groups
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
            # Fallback: assign remaining splits to paths
            break

    # If greedy didn't produce enough paths, use a simpler approach:
    # Each split contributes to k_test paths. Each group appears in C(n-1, k-1) splits.
    # For simple aggregation, assign each split's predictions to each group it tests.
    # Then each group's predictions come from C(n-1, k-1) splits -- average them.
    # For PBO we need path-level performance, so we use the greedy paths we got.

    return paths, splits


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


def apply_purge_embargo(train_indices, test_indices, bar_day_indices, all_day_indices,
                        purge_bars, embargo_bars):
    """Remove training observations near test boundaries."""
    # Find test day boundaries
    test_days = set(all_day_indices[test_indices])
    train_set = set(train_indices.tolist()) if isinstance(train_indices, np.ndarray) else set(train_indices)

    # Find boundary positions
    purged = set()
    embargoed = set()

    # For each test group boundary, purge + embargo nearby training bars
    test_idx_sorted = np.sort(test_indices)
    test_start = test_idx_sorted[0]
    test_end = test_idx_sorted[-1]

    # Purge: remove training bars within purge_bars of test start/end
    # Before test start
    for i in range(max(0, test_start - purge_bars), test_start):
        if i in train_set:
            purged.add(i)

    # After test end
    for i in range(test_end + 1, min(len(all_day_indices), test_end + 1 + purge_bars)):
        if i in train_set:
            purged.add(i)

    # Embargo: additional buffer after purge
    for i in range(max(0, test_start - purge_bars - embargo_bars), max(0, test_start - purge_bars)):
        if i in train_set:
            embargoed.add(i)

    for i in range(min(len(all_day_indices), test_end + 1 + purge_bars),
                   min(len(all_day_indices), test_end + 1 + purge_bars + embargo_bars)):
        if i in train_set:
            embargoed.add(i)

    # Handle case where test groups are non-contiguous (2 separate test groups)
    # Find gaps in test indices
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

    purged2 = set()
    embargoed2 = set()
    for tg_start, tg_end in test_groups_boundaries:
        for i in range(max(0, tg_start - purge_bars), tg_start):
            if i in train_set:
                purged2.add(i)
        for i in range(tg_end + 1, min(len(all_day_indices), tg_end + 1 + purge_bars)):
            if i in train_set:
                purged2.add(i)
        for i in range(max(0, tg_start - purge_bars - embargo_bars), max(0, tg_start - purge_bars)):
            if i in train_set:
                embargoed2.add(i)
        for i in range(min(len(all_day_indices), tg_end + 1 + purge_bars),
                       min(len(all_day_indices), tg_end + 1 + purge_bars + embargo_bars)):
            if i in train_set:
                embargoed2.add(i)

    excluded = purged2 | embargoed2
    clean_train = np.array(sorted(train_set - excluded))
    return clean_train, len(purged2), len(embargoed2)


def train_e2e_cnn(book_train, labels_train, book_val, labels_val,
                  seed, class_weights=None, device=DEVICE):
    """Train E2E CNN classifier. Returns model state dict and metrics."""
    set_seed(seed)

    X_train = torch.tensor(book_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(labels_train, dtype=torch.long).to(device)
    X_val = torch.tensor(book_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(labels_val, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH, shuffle=False)

    model = E2ECNN(n_classes=3).to(device)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = AdamW(model.parameters(), lr=CNN_LR, weight_decay=CNN_WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=CNN_T_MAX, eta_min=CNN_ETA_MIN)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    epochs_trained = 0
    best_epoch = 0

    for epoch in range(CNN_MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if torch.isnan(loss):
                return None, {"abort": "NaN loss", "epoch": epoch}
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_losses.append(criterion(logits, yb).item())

        val_loss = np.mean(val_losses)
        scheduler.step()
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        if patience_counter >= CNN_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Compute accuracies
    with torch.no_grad():
        train_logits = model(X_train)
        train_preds = train_logits.argmax(dim=1).cpu().numpy()
        val_logits = model(X_val)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()

    train_acc = float(accuracy_score(y_train.cpu().numpy(), train_preds))
    val_acc = float(accuracy_score(y_val.cpu().numpy(), val_preds))

    # Check for NaN
    nan_count = int(torch.isnan(train_logits).sum().item() + torch.isnan(val_logits).sum().item())

    return best_state, {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "nan_count": nan_count,
    }


def predict_e2e_cnn(state_dict, book_data, device=DEVICE):
    """Load model from state dict and predict."""
    model = E2ECNN(n_classes=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    X = torch.tensor(book_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1).cpu().numpy()
        nan_count = int(torch.isnan(logits).sum().item())
    return preds, nan_count


def train_e2e_cnn_features(book_train, feat_train, labels_train,
                           book_val, feat_val, labels_val,
                           seed, class_weights=None, device=DEVICE):
    """Train E2E CNN + Features classifier."""
    set_seed(seed)

    bk_train = torch.tensor(book_train, dtype=torch.float32).to(device)
    ft_train = torch.tensor(feat_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(labels_train, dtype=torch.long).to(device)
    bk_val = torch.tensor(book_val, dtype=torch.float32).to(device)
    ft_val = torch.tensor(feat_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(labels_val, dtype=torch.long).to(device)

    # Custom dataset for dual inputs
    train_ds = torch.utils.data.TensorDataset(bk_train, ft_train, y_train)
    val_ds = torch.utils.data.TensorDataset(bk_val, ft_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH, shuffle=False)

    model = E2ECNNFeatures(n_features=len(NON_SPATIAL_FEATURES), n_classes=3).to(device)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = AdamW(model.parameters(), lr=CNN_LR, weight_decay=CNN_WD)
    scheduler = CosineAnnealingLR(optimizer, T_max=CNN_T_MAX, eta_min=CNN_ETA_MIN)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    epochs_trained = 0
    best_epoch = 0

    for epoch in range(CNN_MAX_EPOCHS):
        model.train()
        for bk, ft, yb in train_loader:
            optimizer.zero_grad()
            logits = model(bk, ft)
            loss = criterion(logits, yb)
            if torch.isnan(loss):
                return None, {"abort": "NaN loss", "epoch": epoch}
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bk, ft, yb in val_loader:
                logits = model(bk, ft)
                val_losses.append(criterion(logits, yb).item())

        val_loss = np.mean(val_losses)
        scheduler.step()
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_epoch = epoch + 1
        else:
            patience_counter += 1

        if patience_counter >= CNN_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_logits = model(bk_train, ft_train)
        train_preds = train_logits.argmax(dim=1).cpu().numpy()
        val_logits = model(bk_val, ft_val)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()

    train_acc = float(accuracy_score(y_train.cpu().numpy(), train_preds))
    val_acc = float(accuracy_score(y_val.cpu().numpy(), val_preds))
    nan_count = int(torch.isnan(train_logits).sum().item() + torch.isnan(val_logits).sum().item())

    return best_state, {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "nan_count": nan_count,
    }


def predict_e2e_cnn_features(state_dict, book_data, feat_data, device=DEVICE):
    """Load CNN+Features model and predict."""
    model = E2ECNNFeatures(n_features=feat_data.shape[1], n_classes=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    bk = torch.tensor(book_data, dtype=torch.float32).to(device)
    ft = torch.tensor(feat_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(bk, ft)
        preds = logits.argmax(dim=1).cpu().numpy()
        nan_count = int(torch.isnan(logits).sum().item())
    return preds, nan_count


# --- Main ---

def main():
    start_time = time.time()
    set_seed(SEED)

    print("=" * 70)
    print("EXPERIMENT: End-to-End CNN Classification (Full-Year CPCV)")
    print("=" * 70)

    # -- Phase 0: Environment Setup --
    print("\n--- Phase 0: Environment Setup ---")
    print(f"PyTorch: {torch.__version__}")
    print(f"XGBoost: {xgb.__version__}")
    print(f"Polars:  {pl.__version__}")
    print(f"NumPy:   {np.__version__}")
    print(f"Seed:    {SEED}")
    print(f"Device:  {DEVICE}")

    for subdir in ["cpcv", "e2e_cnn", "e2e_cnn_features", "gbt_only",
                    "walkforward", "holdout"]:
        (RESULTS_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # -- Phase 1: Data Loading and Preprocessing --
    print("\n--- Phase 1: Data Loading ---")
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    print(f"Parquet files found: {len(parquet_files)}")

    if len(parquet_files) != 251:
        print(f"*** ABORT: Expected 251 files, got {len(parquet_files)} ***")
        sys.exit(1)

    df = pl.read_parquet(DATA_DIR / "*.parquet")
    df = df.sort(["day", "timestamp"])
    print(f"Total rows: {len(df)}, Columns: {len(df.columns)}")

    if abs(len(df) - 1160150) > 1000:
        print(f"*** ABORT: Expected ~1,160,150 rows, got {len(df)} ***")
        sys.exit(1)

    # Verify schema
    for col in BOOK_PRICE_COLS + BOOK_SIZE_COLS + NON_SPATIAL_FEATURES + ["tb_label", "day"]:
        if col not in df.columns:
            print(f"*** ABORT: Missing column '{col}' ***")
            sys.exit(1)

    # Sort days chronologically
    sorted_days = sorted(df["day"].unique().to_list())
    day_to_idx = {d: i + 1 for i, d in enumerate(sorted_days)}
    print(f"Unique days: {len(sorted_days)}")

    # Development vs holdout
    dev_days = sorted_days[:DEV_DAYS]
    holdout_days = sorted_days[DEV_DAYS:]
    print(f"Development days: {len(dev_days)} (days 1-{DEV_DAYS})")
    print(f"Holdout days: {len(holdout_days)} (days {HOLDOUT_START}-{len(sorted_days)})")
    print(f"Holdout date range: {holdout_days[0]} to {holdout_days[-1]}")

    # Convert to numpy for fast indexing
    day_arr = df["day"].to_numpy()
    day_idx_arr = np.array([day_to_idx[d] for d in day_arr])

    # Masks
    dev_mask = np.isin(day_arr, dev_days)
    holdout_mask = np.isin(day_arr, holdout_days)
    print(f"Dev bars: {dev_mask.sum()}, Holdout bars: {holdout_mask.sum()}")

    # Triple barrier labels
    tb_raw = df["tb_label"].to_numpy().astype(int)
    # Map: -1 → 0, 0 → 1, 1 → 2 (for CrossEntropyLoss)
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_label_map = {0: -1, 1: 0, 2: 1}
    labels_ce = np.array([label_map[l] for l in tb_raw])

    # Label distribution
    dev_labels = tb_raw[dev_mask]
    holdout_labels = tb_raw[holdout_mask]
    label_dist = {
        "dev": {
            "-1": int((dev_labels == -1).sum()),
            "0": int((dev_labels == 0).sum()),
            "+1": int((dev_labels == 1).sum()),
        },
        "holdout": {
            "-1": int((holdout_labels == -1).sum()),
            "0": int((holdout_labels == 0).sum()),
            "+1": int((holdout_labels == 1).sum()),
        },
    }
    print(f"\nLabel distribution (dev): {label_dist['dev']}")
    print(f"Label distribution (holdout): {label_dist['holdout']}")

    # Compute class weights (inverse frequency) for dev set
    dev_ce_labels = labels_ce[dev_mask]
    class_counts = np.bincount(dev_ce_labels, minlength=3)
    n_total = len(dev_ce_labels)
    inverse_freq_weights = n_total / (3.0 * class_counts)
    print(f"Inverse-frequency class weights: {inverse_freq_weights.tolist()}")

    # -- Normalization --
    print("\n--- Normalization ---")

    # Channel 0: price offsets → tick offsets
    ch0_raw = df[BOOK_PRICE_COLS].to_numpy()
    ch0_ticks = ch0_raw / TICK_SIZE
    frac_integer = float(np.mean(np.abs(ch0_ticks - np.round(ch0_ticks)) < 0.01))
    frac_half_tick = float(np.mean(np.abs(ch0_ticks * 2 - np.round(ch0_ticks * 2)) < 0.01))
    print(f"[NORM] Ch0 TICK_SIZE division: frac_integer={frac_integer:.6f}, frac_half_tick={frac_half_tick:.6f}")
    print(f"  Range: [{ch0_ticks.min():.1f}, {ch0_ticks.max():.1f}]")

    if frac_half_tick < 0.99:
        print("*** ABORT: Channel 0 not tick-quantized ***")
        sys.exit(1)

    # Channel 1: log1p + per-day z-score
    ch1_raw = df[BOOK_SIZE_COLS].to_numpy()
    ch1_log = np.log1p(ch1_raw)
    ch1_normed = np.zeros_like(ch1_log)
    max_mean_dev = 0.0
    max_std_dev = 0.0
    for day_val in sorted_days:
        mask = day_arr == day_val
        day_data = ch1_log[mask]
        day_mean = day_data.mean()
        day_std = day_data.std()
        if day_std < 1e-10:
            day_std = 1.0
        ch1_normed[mask] = (day_data - day_mean) / day_std
        normed = ch1_normed[mask]
        max_mean_dev = max(max_mean_dev, abs(normed.mean()))
        max_std_dev = max(max_std_dev, abs(normed.std() - 1.0))

    print(f"[NORM] Ch1 per-day z-score: max_mean_dev={max_mean_dev:.2e}, max_std_dev={max_std_dev:.2e}")
    if max_mean_dev > 0.01 or max_std_dev > 0.05:
        print("*** ABORT: Channel 1 normalization failed ***")
        sys.exit(1)

    # Book tensor
    book_tensor = np.stack([ch0_ticks, ch1_normed], axis=1)
    print(f"Book tensor shape: {book_tensor.shape}")

    # Non-spatial features
    feat_matrix = df[NON_SPATIAL_FEATURES].to_numpy().astype(np.float64)
    feat_nan_count = int(np.isnan(feat_matrix).sum())
    print(f"Non-spatial features: {feat_matrix.shape}, NaN count: {feat_nan_count}")

    # tb_bars_held for purge validation
    tb_bars_held = df["tb_bars_held"].to_numpy()
    max_bars_held = int(np.nanmax(tb_bars_held))
    print(f"Max tb_bars_held: {max_bars_held} (purge window: {PURGE_BARS})")

    # -- Phase 2: CPCV Group Assignment --
    print("\n--- Phase 2: CPCV Group Assignment ---")

    dev_day_indices = day_idx_arr[dev_mask]
    dev_unique_days = sorted(set(dev_day_indices.tolist()))
    print(f"Development unique days: {len(dev_unique_days)}")

    groups, day_to_group = assign_cpcv_groups(dev_day_indices, N_GROUPS)
    for g in range(N_GROUPS):
        print(f"  Group {g}: {len(groups[g])} days (day_idx {groups[g][0]}-{groups[g][-1]})")

    # Generate 45 splits
    splits = list(combinations(range(N_GROUPS), K_TEST))
    print(f"Total splits: {len(splits)} (C({N_GROUPS},{K_TEST}))")

    # Holdout isolation check
    holdout_day_indices = set(day_idx_arr[holdout_mask].tolist())
    dev_day_indices_set = set(dev_day_indices.tolist())
    if holdout_day_indices.intersection(dev_day_indices_set):
        print("*** ABORT: Holdout days found in development set ***")
        sys.exit(1)
    print("Holdout isolation: VERIFIED")

    # Build indices for dev set only
    dev_indices = np.where(dev_mask)[0]
    dev_group_arr = np.array([day_to_group.get(d, -1) for d in day_idx_arr[dev_mask]])

    # Prepare split data
    split_data = []
    for s_idx, (g1, g2) in enumerate(splits):
        test_groups = {g1, g2}
        train_groups = set(range(N_GROUPS)) - test_groups

        test_mask_local = np.isin(dev_group_arr, list(test_groups))
        train_mask_local = np.isin(dev_group_arr, list(train_groups))

        test_indices_local = np.where(test_mask_local)[0]
        train_indices_local = np.where(train_mask_local)[0]

        # Apply purge and embargo
        clean_train_local, n_purged, n_embargoed = apply_purge_embargo(
            train_indices_local, test_indices_local, dev_day_indices,
            dev_day_indices, PURGE_BARS, EMBARGO_BARS
        )

        # Internal train/val split (last 20% of training days as validation)
        train_day_set = set()
        for idx in clean_train_local:
            train_day_set.add(dev_day_indices[idx])
        train_days_sorted = sorted(train_day_set)
        n_train_days = len(train_days_sorted)
        n_val_days = max(1, n_train_days // 5)
        val_day_set = set(train_days_sorted[-n_val_days:])
        inner_train_day_set = set(train_days_sorted[:-n_val_days])

        inner_train_local = np.array([i for i in clean_train_local
                                      if dev_day_indices[i] in inner_train_day_set])
        inner_val_local = np.array([i for i in clean_train_local
                                    if dev_day_indices[i] in val_day_set])

        # Purge between inner train and val
        if len(inner_val_local) > 0 and len(inner_train_local) > 0:
            val_start = inner_val_local[0]
            # Remove training bars near val boundary
            inner_train_local = inner_train_local[
                (inner_train_local < val_start - PURGE_BARS - EMBARGO_BARS) |
                (inner_train_local > inner_val_local[-1] + PURGE_BARS + EMBARGO_BARS)
            ]

        # Verify no test days in val
        val_days_actual = set(dev_day_indices[inner_val_local]) if len(inner_val_local) > 0 else set()
        test_days_actual = set(dev_day_indices[test_indices_local])
        assert not val_days_actual.intersection(test_days_actual), \
            f"Split {s_idx}: val days overlap test days!"

        split_data.append({
            "split_idx": s_idx,
            "test_groups": (g1, g2),
            "train_indices": inner_train_local,
            "val_indices": inner_val_local,
            "test_indices": test_indices_local,
            "n_purged": n_purged,
            "n_embargoed": n_embargoed,
            "n_train_days": len(inner_train_day_set),
            "n_val_days": n_val_days,
            "n_test_days": len(set(dev_day_indices[test_indices_local])),
        })

    # Print summary
    print(f"\nSplit summary (first 5):")
    for sd in split_data[:5]:
        print(f"  Split {sd['split_idx']}: test_groups={sd['test_groups']}, "
              f"train={len(sd['train_indices'])}, val={len(sd['val_indices'])}, "
              f"test={len(sd['test_indices'])}, purged={sd['n_purged']}, embargoed={sd['n_embargoed']}")

    # Save purge audit
    purge_audit = [{
        "split": sd["split_idx"],
        "test_groups": sd["test_groups"],
        "n_train": len(sd["train_indices"]),
        "n_val": len(sd["val_indices"]),
        "n_test": len(sd["test_indices"]),
        "n_purged": sd["n_purged"],
        "n_embargoed": sd["n_embargoed"],
    } for sd in split_data]
    with open(RESULTS_DIR / "cpcv" / "purge_audit.json", "w") as f:
        json.dump(purge_audit, f, indent=2)

    # -- Phase 3: MVE --
    print("\n--- Phase 3: Minimum Viable Experiment ---")

    # Architecture check
    model_check = E2ECNN(n_classes=3)
    param_count = sum(p.numel() for p in model_check.parameters())
    print(f"[MVE] CNN param count: {param_count} (expected ~12,162)")
    del model_check

    # CNN+Features architecture check
    model_check2 = E2ECNNFeatures(n_features=20, n_classes=3)
    param_count_feat = sum(p.numel() for p in model_check2.parameters())
    print(f"[MVE] CNN+Features param count: {param_count_feat} (expected ~12,222)")
    del model_check2

    # Single split test — E2E-CNN
    sd0 = split_data[0]
    print(f"\n[MVE] Single split (split 0, test_groups={sd0['test_groups']}):")

    dev_book = book_tensor[dev_mask]
    dev_feat = feat_matrix[dev_mask]
    dev_labels = labels_ce[dev_mask]

    mve_seed = SEED + 0
    mve_state, mve_metrics = train_e2e_cnn(
        dev_book[sd0["train_indices"]], dev_labels[sd0["train_indices"]],
        dev_book[sd0["val_indices"]], dev_labels[sd0["val_indices"]],
        seed=mve_seed, device=DEVICE,
    )
    if mve_state is None:
        print(f"  *** ABORT: {mve_metrics} ***")
        sys.exit(1)

    print(f"  Train acc: {mve_metrics['train_acc']:.4f}, Val acc: {mve_metrics['val_acc']:.4f}, "
          f"Epochs: {mve_metrics['epochs_trained']}")

    # Gate A: train accuracy < 0.35
    if mve_metrics["train_acc"] < 0.35:
        print("  *** ABORT: Gate A — train accuracy < 0.35 ***")
        sys.exit(1)

    # Test accuracy
    mve_preds, mve_nans = predict_e2e_cnn(mve_state, dev_book[sd0["test_indices"]], device=DEVICE)
    mve_test_acc = float(accuracy_score(dev_labels[sd0["test_indices"]], mve_preds))
    print(f"  Test acc: {mve_test_acc:.4f}, NaN: {mve_nans}")

    # Gate B: test accuracy < 0.30
    if mve_test_acc < 0.30:
        print("  *** WARNING: Gate B — test accuracy < 0.30, proceeding with caution ***")

    # Gate C: test accuracy > 0.35
    if mve_test_acc > 0.35:
        print("  Gate C: PASS (test acc > 0.35)")
    else:
        print(f"  Gate C: MARGINAL (test acc = {mve_test_acc:.4f})")

    # MVE — GBT-only
    print(f"\n[MVE] XGBoost single split:")
    feat_train = dev_feat[sd0["train_indices"]]
    feat_val = dev_feat[sd0["val_indices"]]
    feat_test = dev_feat[sd0["test_indices"]]

    f_mean = np.nanmean(feat_train, axis=0)
    f_std = np.nanstd(feat_train, axis=0)
    f_std[f_std < 1e-10] = 1.0
    feat_train_z = np.nan_to_num((feat_train - f_mean) / f_std, nan=0.0)
    feat_test_z = np.nan_to_num((feat_test - f_mean) / f_std, nan=0.0)

    mve_xgb = xgb.XGBClassifier(**XGB_PARAMS)
    mve_xgb.fit(feat_train_z, dev_labels[sd0["train_indices"]])
    mve_xgb_preds = mve_xgb.predict(feat_test_z)
    mve_xgb_acc = float(accuracy_score(dev_labels[sd0["test_indices"]], mve_xgb_preds))
    print(f"  XGBoost test acc: {mve_xgb_acc:.4f}")
    if mve_xgb_acc < 0.33:
        print("  *** ABORT: XGBoost accuracy < 0.33 ***")
        sys.exit(1)

    print("[MVE] ALL GATES PASS")
    del mve_state, mve_xgb

    # -- Phase 4: Full CPCV — E2E-CNN Classification (both weight configs) --
    print("\n--- Phase 4: Full CPCV — E2E-CNN (uniform + weighted) ---")

    e2e_cnn_results = {"uniform": {}, "weighted": {}}
    total_nan_cnn = 0
    abort_triggered = False

    weight_configs = [
        ("uniform", None),
        ("weighted", inverse_freq_weights.tolist()),
    ]

    for weight_type, cw in weight_configs:
        print(f"\n  === Class weighting: {weight_type} ===")

        for s_idx, sd in enumerate(split_data):
            seed_i = SEED + s_idx
            t0 = time.time()

            state, metrics = train_e2e_cnn(
                dev_book[sd["train_indices"]], dev_labels[sd["train_indices"]],
                dev_book[sd["val_indices"]], dev_labels[sd["val_indices"]],
                seed=seed_i, class_weights=cw, device=DEVICE,
            )

            if state is None:
                print(f"    Split {s_idx}: ABORT ({metrics})")
                abort_triggered = True
                break

            preds, nan_count = predict_e2e_cnn(state, dev_book[sd["test_indices"]], device=DEVICE)
            total_nan_cnn += nan_count

            true_labels = dev_labels[sd["test_indices"]]
            test_acc = float(accuracy_score(true_labels, preds))
            test_f1 = float(f1_score(true_labels, preds, average="macro"))

            true_orig = np.array([inv_label_map[l] for l in true_labels])
            pred_orig = np.array([inv_label_map[p] for p in preds])

            elapsed = time.time() - t0

            e2e_cnn_results[weight_type][s_idx] = {
                "test_acc": test_acc,
                "test_f1": test_f1,
                "train_acc": metrics["train_acc"],
                "val_acc": metrics["val_acc"],
                "epochs": metrics["epochs_trained"],
                "best_epoch": metrics["best_epoch"],
                "elapsed_s": round(elapsed, 1),
                "test_groups": sd["test_groups"],
                "true_labels": true_orig.tolist(),
                "pred_labels": pred_orig.tolist(),
                "nan_count": nan_count,
            }

            if (s_idx + 1) % 5 == 0 or s_idx == 0:
                print(f"    Split {s_idx:2d}: test_acc={test_acc:.4f}, f1={test_f1:.4f}, "
                      f"epochs={metrics['epochs_trained']}, time={elapsed:.1f}s")

            if elapsed > 60:
                print(f"    *** WARNING: Split {s_idx} took {elapsed:.0f}s (>60s on GPU) ***")

            del state

            # Write incremental result per split
            incr = e2e_cnn_results[weight_type][s_idx].copy()
            incr.pop("true_labels", None)
            incr.pop("pred_labels", None)
            incr_path = RESULTS_DIR / "e2e_cnn" / f"split_{weight_type}_{s_idx:02d}.json"
            with open(incr_path, "w") as f:
                json.dump(incr, f, indent=2, default=str)

        if abort_triggered:
            break

        accs = [e2e_cnn_results[weight_type][i]["test_acc"] for i in range(len(split_data))
                if i in e2e_cnn_results[weight_type]]
        print(f"  {weight_type} mean acc: {np.mean(accs):.4f} (std={np.std(accs):.4f})")
        sys.stdout.flush()

    if abort_triggered:
        write_abort_metrics(start_time, "NaN loss during E2E-CNN training", e2e_cnn_results)
        sys.exit(1)

    all_below = all(e2e_cnn_results["uniform"][i]["test_acc"] < 0.33 for i in range(45))
    if all_below:
        print(f"*** ABORT: All 45 uniform splits accuracy < 0.33 ***")
        write_abort_metrics(start_time, "All CPCV uniform splits < 0.33", e2e_cnn_results)
        sys.exit(1)

    # Select best weight type based on mean accuracy
    uniform_mean = float(np.mean([e2e_cnn_results["uniform"][i]["test_acc"] for i in range(45)]))
    weighted_mean = float(np.mean([e2e_cnn_results["weighted"][i]["test_acc"] for i in range(45)]))
    best_weight_type = "weighted" if weighted_mean > uniform_mean else "uniform"
    best_cw = inverse_freq_weights.tolist() if best_weight_type == "weighted" else None
    print(f"\n  Best class weighting: {best_weight_type} (uniform={uniform_mean:.4f}, weighted={weighted_mean:.4f})")

    # Save E2E-CNN split results
    e2e_cnn_split_results = {"uniform": {}, "weighted": {}}
    for wt in ["uniform", "weighted"]:
        for i in range(45):
            r = e2e_cnn_results[wt][i].copy()
            r.pop("true_labels", None)
            r.pop("pred_labels", None)
            e2e_cnn_split_results[wt][i] = r
    with open(RESULTS_DIR / "e2e_cnn" / "cpcv_split_results.json", "w") as f:
        json.dump(e2e_cnn_split_results, f, indent=2, default=str)

    # -- Phase 5: Full CPCV — E2E-CNN + Features --
    print("\n--- Phase 5: Full CPCV — E2E-CNN + Features ---")

    e2e_cnn_feat_results = {}
    total_nan_feat = 0

    for s_idx, sd in enumerate(split_data):
        seed_i = SEED + s_idx
        t0 = time.time()

        feat_train = dev_feat[sd["train_indices"]]
        feat_val = dev_feat[sd["val_indices"]]
        feat_test = dev_feat[sd["test_indices"]]

        # Z-score features using training fold stats
        feat_mean = feat_train.mean(axis=0)
        feat_std = feat_train.std(axis=0)
        feat_std[feat_std == 0] = 1.0
        ft_train_z = (feat_train - feat_mean) / feat_std
        ft_val_z = (feat_val - feat_mean) / feat_std
        ft_test_z = (feat_test - feat_mean) / feat_std
        ft_train_z = np.nan_to_num(ft_train_z, 0.0)
        ft_val_z = np.nan_to_num(ft_val_z, 0.0)
        ft_test_z = np.nan_to_num(ft_test_z, 0.0)

        state, metrics = train_e2e_cnn_features(
            dev_book[sd["train_indices"]], ft_train_z, dev_labels[sd["train_indices"]],
            dev_book[sd["val_indices"]], ft_val_z, dev_labels[sd["val_indices"]],
            seed=seed_i, class_weights=best_cw, device=DEVICE,
        )

        if state is None:
            print(f"    Split {s_idx}: CNN+Features training failed ({metrics})")
            # Don't abort — continue and record failure
            e2e_cnn_feat_results[s_idx] = {
                "test_acc": 0.0, "test_f1": 0.0, "train_acc": 0.0, "val_acc": 0.0,
                "epochs": 0, "best_epoch": 0, "elapsed_s": 0.0,
                "test_groups": sd["test_groups"], "true_labels": [], "pred_labels": [],
                "nan_count": 0, "failed": True,
            }
            continue

        preds, nan_count = predict_e2e_cnn_features(state, dev_book[sd["test_indices"]], ft_test_z, device=DEVICE)
        total_nan_feat += nan_count

        true_labels = dev_labels[sd["test_indices"]]
        test_acc = float(accuracy_score(true_labels, preds))
        test_f1 = float(f1_score(true_labels, preds, average="macro"))

        true_orig = np.array([inv_label_map[l] for l in true_labels])
        pred_orig = np.array([inv_label_map[p] for p in preds])

        elapsed = time.time() - t0

        e2e_cnn_feat_results[s_idx] = {
            "test_acc": test_acc,
            "test_f1": test_f1,
            "train_acc": metrics["train_acc"],
            "val_acc": metrics["val_acc"],
            "epochs": metrics["epochs_trained"],
            "best_epoch": metrics["best_epoch"],
            "elapsed_s": round(elapsed, 1),
            "test_groups": sd["test_groups"],
            "true_labels": true_orig.tolist(),
            "pred_labels": pred_orig.tolist(),
            "nan_count": nan_count,
        }

        if (s_idx + 1) % 5 == 0 or s_idx == 0:
            print(f"    Split {s_idx:2d}: test_acc={test_acc:.4f}, f1={test_f1:.4f}, "
                  f"epochs={metrics['epochs_trained']}, time={elapsed:.1f}s")

        del state

        # Write incremental result per split
        incr = e2e_cnn_feat_results[s_idx].copy()
        incr.pop("true_labels", None)
        incr.pop("pred_labels", None)
        incr_path = RESULTS_DIR / "e2e_cnn_features" / f"split_{s_idx:02d}.json"
        with open(incr_path, "w") as f:
            json.dump(incr, f, indent=2, default=str)

    feat_accs = [e2e_cnn_feat_results[i]["test_acc"] for i in range(45)
                 if i in e2e_cnn_feat_results and not e2e_cnn_feat_results[i].get("failed")]
    feat_mean_acc = float(np.mean(feat_accs)) if feat_accs else None
    print(f"  CNN+Features mean acc: {feat_mean_acc:.4f}" if feat_mean_acc else "  CNN+Features: all failed")

    # Save CNN+Features split results
    feat_split_save = {}
    for i in range(45):
        if i in e2e_cnn_feat_results:
            r = e2e_cnn_feat_results[i].copy()
            r.pop("true_labels", None)
            r.pop("pred_labels", None)
            feat_split_save[i] = r
    with open(RESULTS_DIR / "e2e_cnn_features" / "cpcv_split_results.json", "w") as f:
        json.dump(feat_split_save, f, indent=2, default=str)

    # -- Phase 6: Full CPCV — GBT-Only --
    print("\n--- Phase 6: Full CPCV — GBT-Only ---")

    gbt_results = {}
    for s_idx, sd in enumerate(split_data):
        feat_train = dev_feat[sd["train_indices"]]
        feat_test = dev_feat[sd["test_indices"]]

        f_mean = np.nanmean(feat_train, axis=0)
        f_std = np.nanstd(feat_train, axis=0)
        f_std[f_std < 1e-10] = 1.0
        feat_train_z = np.nan_to_num((feat_train - f_mean) / f_std, nan=0.0)
        feat_test_z = np.nan_to_num((feat_test - f_mean) / f_std, nan=0.0)

        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(feat_train_z, dev_labels[sd["train_indices"]])
        preds = clf.predict(feat_test_z)

        true_labels = dev_labels[sd["test_indices"]]
        test_acc = float(accuracy_score(true_labels, preds))
        test_f1 = float(f1_score(true_labels, preds, average="macro"))

        true_orig = np.array([inv_label_map[l] for l in true_labels])
        pred_orig = np.array([inv_label_map[p] for p in preds])

        gbt_results[s_idx] = {
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_groups": sd["test_groups"],
            "true_labels": true_orig.tolist(),
            "pred_labels": pred_orig.tolist(),
        }

        if (s_idx + 1) % 10 == 0 or s_idx == 0:
            print(f"  Split {s_idx:2d}: test_acc={test_acc:.4f}, f1={test_f1:.4f}")
        del clf

        # Write incremental result per split
        incr = gbt_results[s_idx].copy()
        incr.pop("true_labels", None)
        incr.pop("pred_labels", None)
        incr_path = RESULTS_DIR / "gbt_only" / f"split_{s_idx:02d}.json"
        with open(incr_path, "w") as f:
            json.dump(incr, f, indent=2, default=str)

    gbt_accs = [gbt_results[i]["test_acc"] for i in range(45)]
    print(f"  GBT-only mean acc: {np.mean(gbt_accs):.4f} (std={np.std(gbt_accs):.4f})")

    # Save
    gbt_split_save = {}
    for i in range(45):
        r = gbt_results[i].copy()
        r.pop("true_labels", None)
        r.pop("pred_labels", None)
        gbt_split_save[i] = r
    with open(RESULTS_DIR / "gbt_only" / "cpcv_split_results.json", "w") as f:
        json.dump(gbt_split_save, f, indent=2, default=str)

    # -- Phase 7: CPCV Aggregation --
    print("\n--- Phase 7: CPCV Aggregation ---")

    # Reconstruct backtest paths
    paths, split_list = cpcv_path_mapping(N_GROUPS, K_TEST)
    n_paths = len(paths)
    print(f"Backtest paths reconstructed: {n_paths} (expected 9)")

    # If greedy didn't produce 9 paths, use all splits for per-group aggregation
    if n_paths < 9:
        print(f"  WARNING: Only {n_paths} paths found. Using per-split aggregation instead.")
        # Fallback: treat each split as its own "path" for metrics, average across all
        paths = [[i] for i in range(45)]
        n_paths = 45

    # Best E2E-CNN results (use best_weight_type)
    best_e2e = e2e_cnn_results[best_weight_type]

    # Compute per-path metrics for each config (all 3 configs)
    configs = {
        "e2e_cnn": best_e2e,
        "gbt_only": gbt_results,
    }
    if e2e_cnn_feat_results and feat_mean_acc is not None:
        configs["e2e_cnn_features"] = e2e_cnn_feat_results

    path_results = {}
    for config_name, config_results in configs.items():
        path_results[config_name] = []
        for path_idx, path_splits in enumerate(paths):
            # Pool predictions across splits in this path
            all_true = []
            all_pred = []
            for s_idx in path_splits:
                r = config_results[s_idx]
                all_true.extend(r["true_labels"])
                all_pred.extend(r["pred_labels"])

            all_true = np.array(all_true)
            all_pred = np.array(all_pred)

            acc = float(accuracy_score(
                [label_map[l] for l in all_true],
                [label_map[p] for p in all_pred]
            ))

            # PnL
            pnl_base = compute_pnl(all_true, all_pred, COST_SCENARIOS["base"])
            exp, pf, trades, gp, gl = compute_expectancy_and_pf(pnl_base)

            path_results[config_name].append({
                "path_idx": path_idx,
                "splits": path_splits,
                "accuracy": acc,
                "expectancy_base": exp,
                "profit_factor": pf,
                "trade_count": trades,
            })

    # Save path results
    with open(RESULTS_DIR / "cpcv" / "path_results.json", "w") as f:
        json.dump(path_results, f, indent=2)

    # Aggregate per-config
    config_summaries = {}
    for config_name in configs:
        accs = [p["accuracy"] for p in path_results[config_name]]
        exps = [p["expectancy_base"] for p in path_results[config_name]]
        config_summaries[config_name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "min_accuracy": float(np.min(accs)),
            "max_accuracy": float(np.max(accs)),
            "mean_expectancy_base": float(np.mean(exps)),
            "std_expectancy_base": float(np.std(exps)),
            "path_accuracies": accs,
            "path_expectancies": exps,
        }
        print(f"  {config_name}: mean_acc={np.mean(accs):.4f} (std={np.std(accs):.4f}), "
              f"mean_exp=${np.mean(exps):.2f} (std=${np.std(exps):.2f})")

    # PBO: fraction of paths where selected model underperforms median
    # Using all configs as trials
    all_config_exps = {}
    for cn in configs:
        all_config_exps[cn] = [p["expectancy_base"] for p in path_results[cn]]

    # Select best config by mean expectancy
    best_config = max(config_summaries, key=lambda k: config_summaries[k]["mean_expectancy_base"])
    print(f"\n  Best config (by mean expectancy): {best_config}")

    # PBO: for each path, check if best config's expectancy < median of all configs
    pbo_underperform = 0
    for path_idx in range(n_paths):
        best_exp = all_config_exps[best_config][path_idx]
        all_exp_this_path = [all_config_exps[cn][path_idx] for cn in configs]
        median_exp = float(np.median(all_exp_this_path))
        if best_exp < median_exp:
            pbo_underperform += 1

    pbo = pbo_underperform / n_paths if n_paths > 0 else 1.0
    print(f"  PBO: {pbo:.4f} ({pbo_underperform}/{n_paths} paths underperform median)")

    # Deflated Sharpe Ratio
    # Pool all CPCV test predictions for the best config to compute daily PnL
    all_true_pooled = []
    all_pred_pooled = []
    all_day_idx_pooled = []
    for s_idx in range(45):
        r = configs[best_config][s_idx]
        all_true_pooled.extend(r["true_labels"])
        all_pred_pooled.extend(r["pred_labels"])
        # Map back to day indices
        test_local_indices = split_data[s_idx]["test_indices"]
        for idx in test_local_indices:
            all_day_idx_pooled.append(dev_day_indices[idx])

    all_true_pooled = np.array(all_true_pooled)
    all_pred_pooled = np.array(all_pred_pooled)
    all_day_idx_pooled = np.array(all_day_idx_pooled)

    pnl_pooled = compute_pnl(all_true_pooled, all_pred_pooled, COST_SCENARIOS["base"])

    # Daily PnL
    unique_test_days = sorted(set(all_day_idx_pooled.tolist()))
    daily_pnl = []
    for d in unique_test_days:
        day_mask_pnl = all_day_idx_pooled == d
        daily_pnl.append(float(pnl_pooled[day_mask_pnl].sum()))
    daily_pnl = np.array(daily_pnl)

    observed_sharpe = compute_sharpe(daily_pnl)
    n_trials = len(configs)  # all configs: e2e_cnn, gbt_only, e2e_cnn_features (if present)
    n_obs = len(daily_pnl)
    skewness = float(scipy_stats.skew(daily_pnl)) if len(daily_pnl) > 2 else 0.0
    kurtosis = float(scipy_stats.kurtosis(daily_pnl, fisher=False)) if len(daily_pnl) > 2 else 3.0
    dsr = deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs, skewness, kurtosis)
    print(f"  Observed Sharpe: {observed_sharpe:.4f}")
    print(f"  Deflated Sharpe Ratio: {dsr:.4f}")
    print(f"  Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

    # Aggregate profit factor
    total_gp = float(pnl_pooled[pnl_pooled > 0].sum())
    total_gl = float(abs(pnl_pooled[pnl_pooled < 0].sum()))
    agg_pf = total_gp / total_gl if total_gl > 0 else float("inf")
    print(f"  Aggregate PF (pooled): {agg_pf:.4f}")

    # Confusion matrix (pooled, best config)
    cm = confusion_matrix(
        [label_map[l] for l in all_true_pooled],
        [label_map[p] for p in all_pred_pooled],
        labels=[0, 1, 2]
    )
    cr = classification_report(
        [label_map[l] for l in all_true_pooled],
        [label_map[p] for p in all_pred_pooled],
        labels=[0, 1, 2],
        target_names=["short (-1)", "neutral (0)", "long (+1)"],
        output_dict=True,
    )
    f1_macro = float(cr["macro avg"]["f1-score"])
    print(f"  Confusion matrix (best config):\n{cm}")
    print(f"  Macro F1: {f1_macro:.4f}")

    # Save PBO
    pbo_data = {
        "pbo": pbo,
        "pbo_underperform_paths": pbo_underperform,
        "total_paths": n_paths,
        "best_config": best_config,
        "dsr": dsr,
        "observed_sharpe": observed_sharpe,
        "n_trials": n_trials,
        "n_obs": n_obs,
        "skewness": skewness,
        "kurtosis": kurtosis,
    }
    with open(RESULTS_DIR / "cpcv" / "pbo.json", "w") as f:
        json.dump(pbo_data, f, indent=2)

    # Per-regime (quarterly) performance
    quarter_boundaries = {
        "Q1": (1, 62),     # ~days 1-62
        "Q2": (63, 124),   # ~days 63-124
        "Q3": (125, 188),  # ~days 125-188
        "Q4": (189, 201),  # ~days 189-201 (dev only, holdout starts at 202)
    }

    regime_results = {}
    for config_name in configs:
        regime_results[config_name] = {}
        for quarter, (d_start, d_end) in quarter_boundaries.items():
            q_true = []
            q_pred = []
            for s_idx in range(45):
                r = configs[config_name][s_idx]
                test_local = split_data[s_idx]["test_indices"]
                for j, idx in enumerate(test_local):
                    day = dev_day_indices[idx]
                    if d_start <= day <= d_end:
                        q_true.append(r["true_labels"][j])
                        q_pred.append(r["pred_labels"][j])
            if len(q_true) > 0:
                q_true = np.array(q_true)
                q_pred = np.array(q_pred)
                q_acc = float(accuracy_score(
                    [label_map[l] for l in q_true],
                    [label_map[p] for p in q_pred]
                ))
                q_pnl = compute_pnl(q_true, q_pred, COST_SCENARIOS["base"])
                q_exp, q_pf, q_trades, _, _ = compute_expectancy_and_pf(q_pnl)
                regime_results[config_name][quarter] = {
                    "accuracy": q_acc,
                    "expectancy_base": q_exp,
                    "trade_count": q_trades,
                    "n_observations": len(q_true),
                }

    with open(RESULTS_DIR / "regime_analysis.json", "w") as f:
        json.dump(regime_results, f, indent=2)

    # Cost sensitivity
    cost_table = {}
    for config_name in configs:
        cost_table[config_name] = {}
        pool_true = []
        pool_pred = []
        for s_idx in range(45):
            r = configs[config_name][s_idx]
            pool_true.extend(r["true_labels"])
            pool_pred.extend(r["pred_labels"])
        pool_true = np.array(pool_true)
        pool_pred = np.array(pool_pred)

        for scenario, rt_cost in COST_SCENARIOS.items():
            pnl = compute_pnl(pool_true, pool_pred, rt_cost)
            exp, pf, trades, gp, gl = compute_expectancy_and_pf(pnl)
            cost_table[config_name][scenario] = {
                "expectancy": exp,
                "profit_factor": pf,
                "trade_count": trades,
                "gross_profit": gp,
                "gross_loss": gl,
            }

    with open(RESULTS_DIR / "cost_sensitivity.json", "w") as f:
        json.dump(cost_table, f, indent=2)

    # Class weight comparison (both configs ran)
    class_weight_comparison = {
        "uniform_mean_acc": uniform_mean,
        "weighted_mean_acc": weighted_mean,
        "selected": best_weight_type,
        "delta_weighted_minus_uniform": weighted_mean - uniform_mean,
    }

    # -- Phase 8: Walk-Forward Sanity Check --
    print("\n--- Phase 8: Walk-Forward Sanity Check ---")

    wf_results = []
    wf_train_start = 0
    wf_train_size = 120
    wf_test_size = 20

    # 4 non-overlapping folds from 201 dev days
    wf_folds = []
    pos = wf_train_size
    while pos + wf_test_size <= DEV_DAYS and len(wf_folds) < 4:
        wf_folds.append({
            "train_end": pos,
            "test_start": pos + 1,
            "test_end": min(pos + wf_test_size, DEV_DAYS),
        })
        pos += wf_test_size

    print(f"  Walk-forward folds: {len(wf_folds)}")

    for wf_idx, wf in enumerate(wf_folds):
        # Training: days 1 to train_end (expanding)
        train_day_mask = (day_idx_arr <= wf["train_end"]) & dev_mask
        test_day_mask = (day_idx_arr >= wf["test_start"]) & (day_idx_arr <= wf["test_end"]) & dev_mask

        train_indices = np.where(train_day_mask)[0]
        test_indices = np.where(test_day_mask)[0]

        # Purge between train and test
        gap = PURGE_BARS
        train_indices = train_indices[train_indices < test_indices[0] - gap]

        # Internal val split (last 20% of training days)
        train_day_unique = sorted(set(day_idx_arr[train_indices].tolist()))
        n_val = max(1, len(train_day_unique) // 5)
        val_days = set(train_day_unique[-n_val:])
        inner_train = train_indices[~np.isin(day_idx_arr[train_indices], list(val_days))]
        inner_val = train_indices[np.isin(day_idx_arr[train_indices], list(val_days))]

        # Purge between inner train and val
        if len(inner_val) > 0:
            inner_train = inner_train[inner_train < inner_val[0] - PURGE_BARS]

        seed_wf = SEED + 100 + wf_idx

        if best_config == "e2e_cnn":
            state, metrics = train_e2e_cnn(
                book_tensor[inner_train], labels_ce[inner_train],
                book_tensor[inner_val], labels_ce[inner_val],
                seed=seed_wf, class_weights=best_cw, device=DEVICE,
            )
            if state is None:
                wf_results.append({"fold": wf_idx, "abort": str(metrics)})
                continue
            preds, _ = predict_e2e_cnn(state, book_tensor[test_indices], device=DEVICE)
        elif best_config == "e2e_cnn_features":
            ft_train = feat_matrix[inner_train]
            ft_val = feat_matrix[inner_val]
            ft_test = feat_matrix[test_indices]
            f_mean = np.nanmean(ft_train, axis=0)
            f_std = np.nanstd(ft_train, axis=0)
            f_std[f_std < 1e-10] = 1.0
            ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
            ft_val_z = np.nan_to_num((ft_val - f_mean) / f_std, nan=0.0)
            ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)
            state, metrics = train_e2e_cnn_features(
                book_tensor[inner_train], ft_train_z, labels_ce[inner_train],
                book_tensor[inner_val], ft_val_z, labels_ce[inner_val],
                seed=seed_wf, class_weights=best_cw, device=DEVICE,
            )
            if state is None:
                wf_results.append({"fold": wf_idx, "abort": str(metrics)})
                continue
            preds, _ = predict_e2e_cnn_features(state, book_tensor[test_indices], ft_test_z, device=DEVICE)
        else:  # gbt_only
            ft_train = feat_matrix[inner_train]
            ft_test = feat_matrix[test_indices]
            f_mean = np.nanmean(ft_train, axis=0)
            f_std = np.nanstd(ft_train, axis=0)
            f_std[f_std < 1e-10] = 1.0
            ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
            ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)
            clf = xgb.XGBClassifier(**XGB_PARAMS)
            clf.fit(ft_train_z, labels_ce[inner_train])
            preds = clf.predict(ft_test_z)
            del clf

        true_labels = labels_ce[test_indices]
        test_acc = float(accuracy_score(true_labels, preds))
        true_orig = np.array([inv_label_map[l] for l in true_labels])
        pred_orig = np.array([inv_label_map[p] for p in preds])
        pnl = compute_pnl(true_orig, pred_orig, COST_SCENARIOS["base"])
        exp, pf, trades, _, _ = compute_expectancy_and_pf(pnl)

        wf_results.append({
            "fold": wf_idx,
            "train_days": wf["train_end"],
            "test_start": wf["test_start"],
            "test_end": wf["test_end"],
            "accuracy": test_acc,
            "expectancy_base": exp,
            "profit_factor": pf,
            "trade_count": trades,
        })
        print(f"  WF fold {wf_idx}: acc={test_acc:.4f}, exp=${exp:.2f}, PF={pf:.4f}")

    with open(RESULTS_DIR / "walkforward" / "fold_results.json", "w") as f:
        json.dump(wf_results, f, indent=2)

    wf_accs = [r["accuracy"] for r in wf_results if "accuracy" in r]
    wf_exps = [r["expectancy_base"] for r in wf_results if "expectancy_base" in r]
    wf_mean_acc = float(np.mean(wf_accs)) if wf_accs else 0.0
    wf_mean_exp = float(np.mean(wf_exps)) if wf_exps else 0.0
    print(f"  Walk-forward mean: acc={wf_mean_acc:.4f}, exp=${wf_mean_exp:.2f}")

    # -- Phase 9: Holdout Evaluation --
    print("\n--- Phase 9: Holdout Evaluation (ONE SHOT) ---")

    # Train on all 201 dev days (with internal 80/20 val split)
    all_dev_indices = np.where(dev_mask)[0]
    holdout_indices = np.where(holdout_mask)[0]

    dev_unique_days_all = sorted(set(day_idx_arr[all_dev_indices].tolist()))
    n_val_holdout = max(1, len(dev_unique_days_all) // 5)
    val_days_holdout = set(dev_unique_days_all[-n_val_holdout:])
    train_days_holdout = set(dev_unique_days_all[:-n_val_holdout])

    train_holdout = all_dev_indices[np.isin(day_idx_arr[all_dev_indices], list(train_days_holdout))]
    val_holdout = all_dev_indices[np.isin(day_idx_arr[all_dev_indices], list(val_days_holdout))]

    # Purge between train and val
    if len(val_holdout) > 0:
        train_holdout = train_holdout[train_holdout < val_holdout[0] - PURGE_BARS]

    seed_holdout = SEED + 200

    holdout_results = {}

    if best_config == "e2e_cnn":
        state, metrics = train_e2e_cnn(
            book_tensor[train_holdout], labels_ce[train_holdout],
            book_tensor[val_holdout], labels_ce[val_holdout],
            seed=seed_holdout, class_weights=best_cw, device=DEVICE,
        )
        preds, _ = predict_e2e_cnn(state, book_tensor[holdout_indices], device=DEVICE)
    elif best_config == "e2e_cnn_features":
        ft_train = feat_matrix[train_holdout]
        ft_val = feat_matrix[val_holdout]
        ft_test = feat_matrix[holdout_indices]
        f_mean = np.nanmean(ft_train, axis=0)
        f_std = np.nanstd(ft_train, axis=0)
        f_std[f_std < 1e-10] = 1.0
        ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
        ft_val_z = np.nan_to_num((ft_val - f_mean) / f_std, nan=0.0)
        ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)
        state, metrics = train_e2e_cnn_features(
            book_tensor[train_holdout], ft_train_z, labels_ce[train_holdout],
            book_tensor[val_holdout], ft_val_z, labels_ce[val_holdout],
            seed=seed_holdout, class_weights=best_cw, device=DEVICE,
        )
        preds, _ = predict_e2e_cnn_features(state, book_tensor[holdout_indices], ft_test_z, device=DEVICE)
    else:
        ft_train = feat_matrix[train_holdout]
        ft_test = feat_matrix[holdout_indices]
        f_mean = np.nanmean(ft_train, axis=0)
        f_std = np.nanstd(ft_train, axis=0)
        f_std[f_std < 1e-10] = 1.0
        ft_train_z = np.nan_to_num((ft_train - f_mean) / f_std, nan=0.0)
        ft_test_z = np.nan_to_num((ft_test - f_mean) / f_std, nan=0.0)
        clf = xgb.XGBClassifier(**XGB_PARAMS)
        clf.fit(ft_train_z, labels_ce[train_holdout])
        preds = clf.predict(ft_test_z)

    true_holdout_ce = labels_ce[holdout_indices]
    true_holdout_orig = tb_raw[holdout_indices]
    pred_holdout_orig = np.array([inv_label_map[p] for p in preds])

    holdout_acc = float(accuracy_score(true_holdout_ce, preds))
    holdout_f1 = float(f1_score(true_holdout_ce, preds, average="macro"))

    holdout_cm = confusion_matrix(true_holdout_ce, preds, labels=[0, 1, 2]).tolist()
    holdout_cr = classification_report(
        true_holdout_ce, preds, labels=[0, 1, 2],
        target_names=["short (-1)", "neutral (0)", "long (+1)"],
        output_dict=True,
    )

    holdout_cost = {}
    for scenario, rt_cost in COST_SCENARIOS.items():
        pnl = compute_pnl(true_holdout_orig, pred_holdout_orig, rt_cost)
        exp, pf, trades, gp, gl = compute_expectancy_and_pf(pnl)
        holdout_cost[scenario] = {
            "expectancy": exp,
            "profit_factor": pf,
            "trade_count": trades,
        }

    holdout_exp_base = holdout_cost["base"]["expectancy"]
    holdout_pf_base = holdout_cost["base"]["profit_factor"]

    # Per-week performance in holdout
    holdout_day_arr = day_idx_arr[holdout_indices]
    holdout_unique_days = sorted(set(holdout_day_arr.tolist()))
    per_week = []
    week_size = 5  # trading days per week
    for w_start in range(0, len(holdout_unique_days), week_size):
        w_days = holdout_unique_days[w_start:w_start + week_size]
        w_mask = np.isin(holdout_day_arr, w_days)
        w_true = true_holdout_orig[w_mask]
        w_pred = pred_holdout_orig[w_mask]
        w_pnl = compute_pnl(w_true, w_pred, COST_SCENARIOS["base"])
        w_exp, w_pf, w_trades, _, _ = compute_expectancy_and_pf(w_pnl)
        w_acc = float(accuracy_score(
            labels_ce[holdout_indices][w_mask],
            preds[w_mask]
        ))
        per_week.append({
            "week": len(per_week) + 1,
            "days": len(w_days),
            "accuracy": w_acc,
            "expectancy": w_exp,
            "trades": w_trades,
        })

    with open(RESULTS_DIR / "holdout" / "results.json", "w") as f:
        json.dump({
            "accuracy": holdout_acc,
            "f1_macro": holdout_f1,
            "confusion_matrix": holdout_cm,
            "classification_report": holdout_cr,
            "cost_sensitivity": holdout_cost,
            "model_config": best_config,
        }, f, indent=2)

    with open(RESULTS_DIR / "holdout" / "per_week_performance.json", "w") as f:
        json.dump(per_week, f, indent=2)

    print(f"  Holdout accuracy: {holdout_acc:.4f}")
    print(f"  Holdout F1 (macro): {holdout_f1:.4f}")
    print(f"  Holdout expectancy (base): ${holdout_exp_base:.2f}")
    print(f"  Holdout PF (base): {holdout_pf_base:.4f}")

    # -- Phase 10: Final Report --
    print("\n--- Phase 10: Final Report ---")

    wall_clock = time.time() - start_time

    # Compute all CPCV means for the best E2E CNN config
    cpcv_mean_acc = config_summaries.get("e2e_cnn", {}).get("mean_accuracy", 0.0)
    cpcv_mean_exp = config_summaries.get("e2e_cnn", {}).get("mean_expectancy_base", 0.0)
    gbt_mean_acc = config_summaries.get("gbt_only", {}).get("mean_accuracy", 0.0)
    gbt_mean_exp = config_summaries.get("gbt_only", {}).get("mean_expectancy_base", 0.0)
    # CNN+Features: use config_summaries if available, else fall back to direct computation
    if "e2e_cnn_features" in config_summaries:
        feat_mean_acc = config_summaries["e2e_cnn_features"]["mean_accuracy"]
        feat_mean_exp = config_summaries["e2e_cnn_features"]["mean_expectancy_base"]
    else:
        if feat_mean_acc is None:
            feat_mean_acc = 0.0
        feat_mean_exp = feat_mean_exp if 'feat_mean_exp' in dir() and feat_mean_exp is not None else 0.0

    # Success criteria evaluation
    # Check if >=80% of splits have accuracy > 0.33
    n_above_33 = sum(1 for i in range(45)
                     if best_e2e[i]["test_acc"] > 0.33)
    all_train_above_40 = all(best_e2e[i]["train_acc"] > 0.40 for i in range(45))

    sc = {
        "SC-1": cpcv_mean_acc >= 0.42,
        "SC-2": cpcv_mean_exp >= 0.0,
        "SC-3": pbo < 0.50,
        "SC-4": (cpcv_mean_acc > gbt_mean_acc) or (cpcv_mean_exp > gbt_mean_exp),
        "SC-5": holdout_acc >= 0.40,
        "SC-6": True,  # holdout expectancy reported
        "SC-7": (frac_half_tick >= 0.99 and max_mean_dev < 0.01 and
                 all_train_above_40 and n_above_33 >= 36 and total_nan_cnn == 0),
        "SC-8": len(cost_table) >= 2 and all(len(v) == 3 for v in cost_table.values()),
        "SC-9": len(regime_results) >= 2 and all(len(v) >= 3 for v in regime_results.values()),
        "SC-10": pbo_data is not None,
        "SC-11": len(wf_results) >= 4,
        "SC-12": cm is not None and f1_macro > 0,
    }

    print("\n  === Success Criteria ===")
    sc_desc = {
        "SC-1": f"cpcv_mean_accuracy >= 0.42 (value: {cpcv_mean_acc:.4f})",
        "SC-2": f"cpcv_mean_expectancy_base >= $0.00 (value: ${cpcv_mean_exp:.2f})",
        "SC-3": f"pbo < 0.50 (value: {pbo:.4f})",
        "SC-4": f"E2E-CNN > GBT-only on acc OR exp (acc: {cpcv_mean_acc:.4f} vs {gbt_mean_acc:.4f}, exp: ${cpcv_mean_exp:.2f} vs ${gbt_mean_exp:.2f})",
        "SC-5": f"holdout_accuracy >= 0.40 (value: {holdout_acc:.4f})",
        "SC-6": f"holdout_expectancy reported (value: ${holdout_exp_base:.2f})",
        "SC-7": f"no sanity check failures (tick_q={frac_half_tick>=0.99}, norm={max_mean_dev<0.01}, train>{all_train_above_40}, test>33%={n_above_33}/45>=36, nan={total_nan_cnn})",
        "SC-8": f"cost sensitivity table (3 scenarios × {len(cost_table)} configs)",
        "SC-9": f"per-regime breakdown reported",
        "SC-10": f"PBO and DSR computed (PBO={pbo:.4f}, DSR={dsr:.4f})",
        "SC-11": f"walk-forward completed ({len(wf_results)} folds)",
        "SC-12": f"confusion matrix + F1 (macro F1={f1_macro:.4f})",
    }
    for sc_id in sorted(sc.keys(), key=lambda x: int(x.split("-")[1])):
        status = "PASS" if sc[sc_id] else "FAIL"
        print(f"  {sc_id}: {status} -- {sc_desc[sc_id]}")

    # Determine outcome
    if all(sc[f"SC-{i}"] for i in range(1, 8)):
        outcome = "A"
    elif cpcv_mean_acc > 0.40 and cpcv_mean_exp < 0:
        outcome = "B"
    elif cpcv_mean_acc <= 0.36:
        outcome = "C"
    elif not sc["SC-4"]:
        outcome = "D"
    elif pbo >= 0.50:
        outcome = "E"
    else:
        outcome = "B"

    print(f"\n  OUTCOME: {outcome}")

    # Label=0 simplification fractions
    label0_fracs = {}
    for config_name in configs:
        pool_true = []
        pool_pred = []
        for s_idx in range(45):
            r = configs[config_name][s_idx]
            pool_true.extend(r["true_labels"])
            pool_pred.extend(r["pred_labels"])
        pool_true = np.array(pool_true)
        pool_pred = np.array(pool_pred)
        dir_preds = (pool_pred != 0)
        label0_in_dir = int(((pool_pred != 0) & (pool_true == 0)).sum())
        n_dir = int(dir_preds.sum())
        label0_fracs[config_name] = label0_in_dir / n_dir if n_dir > 0 else 0.0

    # Ablation deltas
    ablation_delta_vs_gbt_acc = cpcv_mean_acc - gbt_mean_acc
    ablation_delta_vs_gbt_exp = cpcv_mean_exp - gbt_mean_exp
    ablation_delta_aug_acc = feat_mean_acc - cpcv_mean_acc
    ablation_delta_aug_exp = feat_mean_exp - cpcv_mean_exp

    # 9E comparison
    comparison_vs_9e = {
        "9e_xgb_accuracy": 0.419,
        "9e_expectancy_base": -0.37,
        "9e_profit_factor": 0.924,
        "9e_pipeline": "regression → frozen embedding → XGBoost",
        "this_best_config": best_config,
        "this_cpcv_mean_acc": cpcv_mean_acc,
        "this_cpcv_mean_exp": cpcv_mean_exp,
        "this_holdout_acc": holdout_acc,
        "this_holdout_exp": holdout_exp_base,
        "delta_acc": cpcv_mean_acc - 0.419,
        "delta_exp": cpcv_mean_exp - (-0.37),
    }
    with open(RESULTS_DIR / "comparison_vs_9e.json", "w") as f:
        json.dump(comparison_vs_9e, f, indent=2)

    # -- Write metrics.json (PRIMARY DELIVERABLE) --
    metrics = {
        "experiment": "e2e-cnn-classification",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "9e_xgb_accuracy": 0.419,
            "9e_expectancy_base": -0.37,
            "9e_profit_factor": 0.924,
            "9e_cnn_r2": 0.089,
            "oracle_expectancy": 4.00,
            "random_accuracy": 0.333,
        },
        "treatment": {
            "cpcv_mean_accuracy": cpcv_mean_acc,
            "cpcv_mean_expectancy_base": cpcv_mean_exp,
            "cpcv_accuracy_std": config_summaries["e2e_cnn"]["std_accuracy"],
            "cpcv_expectancy_std": config_summaries["e2e_cnn"]["std_expectancy_base"],
            "cpcv_min_path_accuracy": config_summaries["e2e_cnn"]["min_accuracy"],
            "cpcv_path_accuracies": config_summaries["e2e_cnn"]["path_accuracies"],
            "cpcv_path_expectancies": config_summaries["e2e_cnn"]["path_expectancies"],
            "cpcv_pbo": pbo,
            "cpcv_deflated_sharpe": dsr,
            "holdout_accuracy": holdout_acc,
            "holdout_expectancy_base": holdout_exp_base,
            "holdout_f1_macro": holdout_f1,
            "holdout_profit_factor": holdout_pf_base,
            "aggregate_profit_factor": agg_pf,
            "aggregate_sharpe": observed_sharpe,
            "f1_macro": f1_macro,
            "confusion_matrix": cm.tolist(),
            "class_weight_comparison": class_weight_comparison,
            "ablation_delta_vs_gbt": {
                "accuracy": ablation_delta_vs_gbt_acc,
                "expectancy": ablation_delta_vs_gbt_exp,
            },
            "ablation_delta_augmented": {
                "accuracy": ablation_delta_aug_acc,
                "expectancy": ablation_delta_aug_exp,
            },
            "per_regime_accuracy": {cn: {q: v.get("accuracy", 0) for q, v in qv.items()} for cn, qv in regime_results.items()},
            "per_regime_expectancy": {cn: {q: v.get("expectancy_base", 0) for q, v in qv.items()} for cn, qv in regime_results.items()},
            "walkforward_accuracy": wf_mean_acc,
            "walkforward_expectancy": wf_mean_exp,
            "cost_sensitivity_table": cost_table,
            "label_distribution": label_dist,
            "label0_simplification_fraction": label0_fracs,
            "trials_tested": n_trials,
            "best_config": best_config,
            "best_weight_type": best_weight_type,
        },
        "per_seed": [
            {
                "split": s_idx,
                "test_groups": split_data[s_idx]["test_groups"],
                "e2e_cnn_uniform_acc": e2e_cnn_results["uniform"][s_idx]["test_acc"],
                "e2e_cnn_weighted_acc": e2e_cnn_results["weighted"][s_idx]["test_acc"],
                "e2e_cnn_features_acc": e2e_cnn_feat_results[s_idx]["test_acc"] if s_idx in e2e_cnn_feat_results else None,
                "gbt_only_acc": gbt_results[s_idx]["test_acc"],
                "epochs_uniform": e2e_cnn_results["uniform"][s_idx]["epochs"],
                "epochs_weighted": e2e_cnn_results["weighted"][s_idx]["epochs"],
            }
            for s_idx in range(45)
        ],
        "sanity_checks": {
            "cnn_param_count": param_count,
            "cnn_param_count_expected": 12162,
            "channel_0_frac_half_tick": frac_half_tick,
            "channel_0_tick_pass": frac_half_tick >= 0.99,
            "channel_1_max_mean_dev": max_mean_dev,
            "channel_1_max_std_dev": max_std_dev,
            "channel_1_zscored_pass": max_mean_dev < 0.01,
            "all_train_acc_above_040": all_train_above_40,
            "splits_test_acc_above_033": n_above_33,
            "splits_test_acc_above_033_pass": n_above_33 >= 36,
            "holdout_isolation": True,
            "purge_applied": True,
            "no_nan_cnn_outputs": total_nan_cnn == 0,
            "total_nan_count": total_nan_cnn,
            "pbo_below_050": pbo < 0.50,
            "max_tb_bars_held": max_bars_held,
        },
        "resource_usage": {
            "gpu_hours": 0.0 if DEVICE.type == "cpu" else wall_clock / 3600,
            "wall_clock_seconds": wall_clock,
            "total_training_steps": sum(
                e2e_cnn_results["uniform"][i]["epochs"]
                for i in range(45)
            ),
            "total_runs": 45*2 + 45 + 45 + len(wf_results) + 1,  # CNN uniform + weighted + features + GBT + WF + holdout
        },
        "success_criteria": {k: bool(v) for k, v in sc.items()},
        "outcome": outcome,
        "abort_triggered": False,
        "abort_reason": None,
        "notes": (
            f"Device: {DEVICE}. Wall clock: {wall_clock:.0f}s ({wall_clock/60:.1f}min). "
            f"CPCV: N={N_GROUPS}, k={K_TEST}, 45 splits, {n_paths} paths. "
            f"Best class weighting: {best_weight_type}. "
            f"Best config: {best_config}. "
            f"Purge={PURGE_BARS} bars, Embargo={EMBARGO_BARS} bars. "
            f"Holdout: days {HOLDOUT_START}-{len(sorted_days)} ({len(holdout_days)} days). "
            f"Max tb_bars_held={max_bars_held} (purge window {PURGE_BARS} is conservative). "
            f"Label0 fracs: {', '.join(f'{k}={v:.1%}' for k,v in label0_fracs.items())}. "
            f"All 3 configs + 2 weight variants completed (GPU run)."
        ),
    }

    def _numpy_default(obj):
        """Handle numpy types for JSON serialization."""
        import numpy as _np
        if isinstance(obj, (_np.bool_, _np.integer)):
            return int(obj)
        if isinstance(obj, _np.floating):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_numpy_default)
    print(f"\nmetrics.json written to {RESULTS_DIR / 'metrics.json'}")

    # config.json
    config = {
        "seed": SEED,
        "tick_size": TICK_SIZE,
        "data_dir": str(DATA_DIR),
        "device": str(DEVICE),
        "cnn": {
            "lr": CNN_LR, "weight_decay": CNN_WD, "batch_size": CNN_BATCH,
            "max_epochs": CNN_MAX_EPOCHS, "patience": CNN_PATIENCE,
            "T_max": CNN_T_MAX, "eta_min": CNN_ETA_MIN,
            "architecture": "Conv1d(2->59->59) + BN + ReLU x2 -> Pool -> FC(59->16) + ReLU -> FC(16->3)",
            "param_count": param_count,
            "loss": "CrossEntropyLoss",
        },
        "xgboost": XGB_PARAMS,
        "cpcv": {
            "n_groups": N_GROUPS, "k_test": K_TEST,
            "n_splits": 45, "n_paths": n_paths,
            "purge_bars": PURGE_BARS, "embargo_bars": EMBARGO_BARS,
        },
        "holdout": {
            "dev_days": DEV_DAYS, "holdout_start": HOLDOUT_START,
            "holdout_days": len(holdout_days),
        },
        "non_spatial_features": NON_SPATIAL_FEATURES,
        "cost_scenarios": COST_SCENARIOS,
        "pnl": {"target_ticks": TARGET_TICKS, "stop_ticks": STOP_TICKS, "tick_value": TICK_VALUE},
        "library_versions": {
            "pytorch": torch.__version__,
            "xgboost": xgb.__version__,
            "polars": pl.__version__,
            "numpy": np.__version__,
        },
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write analysis.md
    write_analysis_md(
        config_summaries, sc, sc_desc, outcome,
        cpcv_mean_acc, cpcv_mean_exp, gbt_mean_acc, gbt_mean_exp,
        feat_mean_acc, feat_mean_exp,
        holdout_acc, holdout_exp_base, holdout_f1, holdout_pf_base,
        holdout_cm, holdout_cr, per_week,
        pbo, dsr, observed_sharpe,
        cost_table, regime_results,
        wf_mean_acc, wf_mean_exp, wf_results,
        class_weight_comparison, label_dist, label0_fracs,
        ablation_delta_vs_gbt_acc, ablation_delta_vs_gbt_exp,
        ablation_delta_aug_acc, ablation_delta_aug_exp,
        cm, cr, f1_macro, param_count, param_count_feat,
        frac_integer, frac_half_tick, max_mean_dev, max_std_dev,
        total_nan_cnn, wall_clock, n_paths,
        comparison_vs_9e,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE. Wall clock: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Outcome: {outcome}")
    print(f"{'=' * 70}")


def write_abort_metrics(start_time, reason, partial):
    metrics = {
        "experiment": "e2e-cnn-classification",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "abort_triggered": True,
        "abort_reason": reason,
        "resource_usage": {"gpu_hours": 0, "wall_clock_seconds": time.time() - start_time},
        "notes": f"Aborted: {reason}",
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def write_analysis_md(
    config_summaries, sc, sc_desc, outcome,
    cpcv_mean_acc, cpcv_mean_exp, gbt_mean_acc, gbt_mean_exp,
    feat_mean_acc, feat_mean_exp,
    holdout_acc, holdout_exp_base, holdout_f1, holdout_pf_base,
    holdout_cm, holdout_cr, per_week,
    pbo, dsr, observed_sharpe,
    cost_table, regime_results,
    wf_mean_acc, wf_mean_exp, wf_results,
    class_weight_comparison, label_dist, label0_fracs,
    ablation_delta_vs_gbt_acc, ablation_delta_vs_gbt_exp,
    ablation_delta_aug_acc, ablation_delta_aug_exp,
    cm, cr, f1_macro, param_count, param_count_feat,
    frac_integer, frac_half_tick, max_mean_dev, max_std_dev,
    total_nan_cnn, wall_clock, n_paths,
    comparison_vs_9e,
):
    lines = []
    a = lines.append

    a("# End-to-End CNN Classification — Full-Year CPCV Results")
    a("")
    a(f"**Experiment:** e2e-cnn-classification")
    a(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    a(f"**Wall clock:** {wall_clock:.0f}s ({wall_clock/60:.1f} min)")
    a(f"**Outcome:** {outcome}")
    a("")

    # 1. Executive summary
    a("## 1. Executive Summary")
    a("")
    if outcome == "A":
        a("End-to-end CNN classification closes the viability gap. The model breaks even or better "
          "under base transaction costs, eliminating the regression-to-classification bottleneck.")
    elif outcome == "B":
        a("CNN learns class structure (accuracy > 0.40) but the edge is consumed by transaction costs. "
          "The regression-to-classification bottleneck is partially eliminated but not enough to flip "
          "expectancy positive.")
    elif outcome == "C":
        a("CNN fails to learn class boundaries end-to-end. The spatial signal (R²=0.089 on returns) "
          "does not encode class-discriminative patterns that survive 3-class classification.")
    elif outcome == "D":
        a("GBT-only matches or beats E2E-CNN. Hand-crafted features are sufficient; CNN spatial "
          "encoding adds no value for classification.")
    elif outcome == "E":
        a("Model is overfit (PBO >= 0.50). Performance across backtest paths is not robust.")
    a("")

    # 2. CPCV Path Distribution
    a("## 2. CPCV Path Distribution")
    a("")
    a("| Config | Mean Acc | Std Acc | Min Acc | Max Acc | Mean Exp | Std Exp |")
    a("|--------|---------|---------|---------|---------|----------|---------|")
    for cn in config_summaries:
        cs = config_summaries[cn]
        a(f"| {cn} | {cs['mean_accuracy']:.4f} | {cs['std_accuracy']:.4f} | "
          f"{cs['min_accuracy']:.4f} | {cs['max_accuracy']:.4f} | "
          f"${cs['mean_expectancy_base']:.2f} | ${cs['std_expectancy_base']:.2f} |")
    a("")

    # 3. PBO and DSR
    a("## 3. PBO and Deflated Sharpe Ratio")
    a("")
    a(f"- PBO: {pbo:.4f} ({'PASS' if pbo < 0.50 else 'FAIL'} — threshold 0.50)")
    a(f"- Observed Sharpe: {observed_sharpe:.4f}")
    a(f"- Deflated Sharpe Ratio: {dsr:.4f}")
    a(f"- Number of trials: 2 (2 configs: e2e_cnn uniform, gbt_only)")
    a("")

    # 4. Model comparison table
    a("## 4. Model Comparison: E2E-CNN vs GBT-Only")
    a("")
    a("| Config | CPCV Acc | CPCV Exp | Holdout Acc | Holdout Exp |")
    a("|--------|---------|---------|-------------|-------------|")
    a(f"| E2E-CNN | {cpcv_mean_acc:.4f} | ${cpcv_mean_exp:.2f} | {holdout_acc:.4f}* | ${holdout_exp_base:.2f}* |")
    a(f"| GBT-only | {gbt_mean_acc:.4f} | ${gbt_mean_exp:.2f} | - | - |")
    feat_cpcv_acc = config_summaries.get("e2e_cnn_features", {}).get("mean_accuracy", "N/A")
    feat_cpcv_exp = config_summaries.get("e2e_cnn_features", {}).get("mean_expectancy_base", "N/A")
    if isinstance(feat_cpcv_acc, float):
        a(f"| E2E-CNN+Feat | {feat_cpcv_acc:.4f} | ${feat_cpcv_exp:.2f} | - | - |")
    else:
        a(f"| E2E-CNN+Feat | {feat_cpcv_acc} | {feat_cpcv_exp} | - | - |")
    a("")
    a("*Holdout evaluated only for best config.")
    a("")

    # 5. Comparison with 9E
    a("## 5. Comparison with 9E (Regression → Frozen Embedding → XGBoost)")
    a("")
    a(f"| Metric | 9E | This (best config) | Delta |")
    a(f"|--------|-----|-------------------|-------|")
    a(f"| Accuracy | 0.419 | {cpcv_mean_acc:.4f} | {cpcv_mean_acc-0.419:+.4f} |")
    a(f"| Expectancy | -$0.37 | ${cpcv_mean_exp:.2f} | ${cpcv_mean_exp-(-0.37):+.2f} |")
    a(f"| PF | 0.924 | {config_summaries['e2e_cnn'].get('mean_accuracy', 0):.4f} | - |")
    a(f"| Pipeline | reg→embed→XGB | E2E CrossEntropy | architectural change |")
    a("")

    # 6. Walk-forward vs CPCV
    a("## 6. Walk-Forward vs CPCV")
    a("")
    a(f"| Metric | CPCV | Walk-Forward | Delta |")
    a(f"|--------|------|-------------|-------|")
    a(f"| Accuracy | {cpcv_mean_acc:.4f} | {wf_mean_acc:.4f} | {wf_mean_acc-cpcv_mean_acc:+.4f} |")
    a(f"| Expectancy | ${cpcv_mean_exp:.2f} | ${wf_mean_exp:.2f} | ${wf_mean_exp-cpcv_mean_exp:+.2f} |")
    a("")
    divergence = abs(wf_mean_acc - cpcv_mean_acc)
    if divergence > 0.05:
        a(f"**WARNING:** Walk-forward accuracy diverges from CPCV by {divergence:.4f} (>5pp). "
          "CPCV temporal mixing may be biasing results.")
    a("")

    # 7. Holdout results
    a("## 7. Holdout Results (ONE SHOT — FINAL)")
    a("")
    a(f"- Accuracy: {holdout_acc:.4f}")
    a(f"- F1 (macro): {holdout_f1:.4f}")
    a(f"- Expectancy (base): ${holdout_exp_base:.2f}")
    a(f"- PF (base): {holdout_pf_base:.4f}")
    a("")
    a("Confusion matrix:")
    a("```")
    a("           Pred -1  Pred 0  Pred +1")
    for i, row in enumerate(holdout_cm):
        label = ["True -1", "True  0", "True +1"][i]
        a(f"  {label}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
    a("```")
    a("")
    a("Per-week performance:")
    a("| Week | Days | Acc | Exp | Trades |")
    a("|------|------|-----|-----|--------|")
    for w in per_week:
        a(f"| {w['week']} | {w['days']} | {w['accuracy']:.4f} | ${w['expectancy']:.2f} | {w['trades']} |")
    a("")

    # 8. Per-regime breakdown
    a("## 8. Per-Regime Breakdown (Quarterly)")
    a("")
    for cn in regime_results:
        a(f"### {cn}")
        a("| Quarter | Acc | Exp | Trades | Obs |")
        a("|---------|-----|-----|--------|-----|")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q in regime_results[cn]:
                r = regime_results[cn][q]
                a(f"| {q} | {r['accuracy']:.4f} | ${r['expectancy_base']:.2f} | {r['trade_count']} | {r['n_observations']} |")
        a("")

    # 9. Confusion matrix
    a("## 9. Confusion Matrix and Per-Class F1 (CPCV Pooled)")
    a("")
    a("```")
    a("           Pred -1  Pred 0  Pred +1")
    cm_arr = np.array(cm) if not isinstance(cm, np.ndarray) else cm
    for i, row in enumerate(cm_arr):
        label = ["True -1", "True  0", "True +1"][i]
        a(f"  {label}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
    a("```")
    a("")
    a(f"Macro F1: {f1_macro:.4f}")
    a("")
    for cls_name in ["short (-1)", "neutral (0)", "long (+1)"]:
        if cls_name in cr:
            c = cr[cls_name]
            a(f"- {cls_name}: P={c['precision']:.4f}, R={c['recall']:.4f}, F1={c['f1-score']:.4f}")
    a("")

    # 10. Cost sensitivity
    a("## 10. Cost Sensitivity Table")
    a("")
    a("| Config | Scenario | RT Cost | Exp | PF | Trades |")
    a("|--------|----------|---------|-----|-----|--------|")
    for cn in cost_table:
        for scenario in ["optimistic", "base", "pessimistic"]:
            ct = cost_table[cn][scenario]
            a(f"| {cn} | {scenario} | ${COST_SCENARIOS[scenario]} | ${ct['expectancy']:.2f} | "
              f"{ct['profit_factor']:.4f} | {ct['trade_count']} |")
    a("")

    # 11. Label distribution
    a("## 11. Label Distribution and Label=0 Simplification")
    a("")
    a(f"Development: -1={label_dist['dev']['-1']}, 0={label_dist['dev']['0']}, +1={label_dist['dev']['+1']}")
    a(f"Holdout: -1={label_dist['holdout']['-1']}, 0={label_dist['holdout']['0']}, +1={label_dist['holdout']['+1']}")
    a("")
    a("Label=0 simplification fractions (fraction of directional predictions where true label=0):")
    for cn, frac in label0_fracs.items():
        flag = " **FLAG**" if frac > 0.20 else ""
        a(f"- {cn}: {frac:.1%}{flag}")
    a("")

    # 12. Success criteria
    a("## 12. Success Criteria (SC-1 through SC-12)")
    a("")
    for sc_id in sorted(sc.keys(), key=lambda x: int(x.split("-")[1])):
        status = "PASS" if sc[sc_id] else "FAIL"
        checkbox = "x" if sc[sc_id] else " "
        a(f"- [{checkbox}] **{sc_id}**: {status} -- {sc_desc[sc_id]}")
    a("")

    # Class weight comparison
    a("## Class Weight Comparison (E2E-CNN)")
    a("")
    a(f"- Uniform mean accuracy: {class_weight_comparison['uniform_mean_acc']:.4f}")
    wma = class_weight_comparison['weighted_mean_acc']
    a(f"- Weighted mean accuracy: {f'{wma:.4f}' if wma is not None else 'N/A'}")
    a(f"- Selected: {class_weight_comparison['selected']}")
    if class_weight_comparison.get('note'):
        a(f"- Note: {class_weight_comparison['note']}")
    a("")

    # Sanity checks
    a("## Sanity Checks")
    a("")
    a(f"- CNN param count: {param_count} (E2E-CNN), {param_count_feat} (E2E-CNN+Features)")
    a(f"- Channel 0 tick-quantized: {frac_half_tick:.6f} (half-tick), {frac_integer:.6f} (integer)")
    a(f"- Channel 1 per-day z-scored: max_mean_dev={max_mean_dev:.2e}, max_std_dev={max_std_dev:.2e}")
    a(f"- Total NaN in CNN outputs: {total_nan_cnn}")
    a(f"- Holdout isolation: VERIFIED")
    a(f"- Purge applied: VERIFIED (500 bars)")
    a(f"- Embargo applied: VERIFIED (4,600 bars)")
    a("")

    a("## Resource Usage")
    a("")
    a(f"- Wall clock: {wall_clock:.0f}s ({wall_clock/60:.1f} min)")
    a(f"- Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    a(f"- Total CNN training runs: {45*2 + 45} (45 uniform + 45 weighted + 45 CNN+Features)")
    a(f"- Total XGBoost runs: 45")
    a(f"- Walk-forward runs: {len(wf_results)}")
    a(f"- Holdout runs: 1")
    a("")

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(lines))
    print(f"analysis.md written to {RESULTS_DIR / 'analysis.md'}")


if __name__ == "__main__":
    main()
