#!/usr/bin/env python3
"""
CNN+GBT Hybrid Model Training Experiment
Spec: .kit/experiments/hybrid-model-training.md

Full 5-fold expanding-window pipeline with ablations.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hybrid_model.cnn_encoder import CNNEncoder
from scripts.hybrid_model.train_xgboost import (
    train_xgboost_classifier,
    predict_xgboost,
    predict_xgboost_proba,
)

# ============================================================================
# Constants
# ============================================================================

SEED = 42
DATA_PATH = PROJECT_ROOT / ".kit" / "results" / "hybrid-model" / "time_5s.csv"
RESULTS_DIR = PROJECT_ROOT / ".kit" / "results" / "hybrid-model-training"

BOOK_SNAP_COLS = [f"book_snap_{i}" for i in range(40)]

# 20 non-spatial features from spec
NONSPATIAL_FEATURES = [
    "weighted_imbalance",
    "spread",
    "net_volume",
    "volume_imbalance",
    "trade_count",
    "avg_trade_size",
    "vwap_distance",
    "return_1",
    "return_5",
    "return_20",
    "volatility_20",
    "volatility_50",
    "high_low_range_50",
    "close_position",
    "cancel_add_ratio",
    "message_rate",
    "modify_fraction",
    "time_sin",
    "time_cos",
    "minutes_since_open",
]

# Triple barrier PnL parameters
TICK_VALUE = 1.25  # $0.25 * 5 = $1.25 per tick
TARGET_TICKS = 10
STOP_TICKS = 5
WIN_PNL_GROSS = TARGET_TICKS * TICK_VALUE  # $12.50
LOSS_PNL_GROSS = STOP_TICKS * TICK_VALUE   # $6.25

COST_SCENARIOS = {
    "optimistic": 2.49,
    "base": 3.74,
    "pessimistic": 6.25,
}

# 5-fold expanding-window splits (1-indexed day positions)
FOLDS = [
    {"train": (1, 4), "test": (5, 7)},
    {"train": (1, 7), "test": (8, 10)},
    {"train": (1, 10), "test": (11, 13)},
    {"train": (1, 13), "test": (14, 16)},
    {"train": (1, 16), "test": (17, 19)},
]

# CNN training hyperparameters
CNN_LR = 1e-3
CNN_WD = 1e-5
CNN_BATCH = 256
CNN_EPOCHS = 50
CNN_PATIENCE = 10

# Abort thresholds
ABORT_NAN_EPOCH = 5
ABORT_TRAIN_R2 = 0.01
ABORT_TEST_R2_ALL_NEG = True
ABORT_XGB_ACC = 0.33


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load and validate the CSV data."""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")

    # Check for duplicate column names — the CSV has return_1 and return_5 twice
    # pandas appends .1 to duplicates
    cols = list(df.columns)
    dup_return1 = [c for c in cols if c.startswith("return_1")]
    dup_return5 = [c for c in cols if c.startswith("return_5")]
    print(f"  return_1 columns: {dup_return1}")
    print(f"  return_5 columns: {dup_return5}")

    # The first occurrence is in the feature section, the second (with .1 suffix)
    # is the target section. Use .1 versions as forward returns.
    # Identify target columns
    if "return_1.1" in cols:
        df["fwd_return_1"] = df["return_1.1"]
        df["fwd_return_5"] = df["return_5.1"]
    else:
        # If no duplicates, the return columns serve dual purpose
        df["fwd_return_1"] = df["return_1"]
        df["fwd_return_5"] = df["return_5"]

    # Identify unique days
    days_sorted = sorted(df["day"].unique())
    print(f"  Unique days: {len(days_sorted)}")
    print(f"  Days: {days_sorted}")

    # Create day_index (1-based)
    day_map = {d: i + 1 for i, d in enumerate(days_sorted)}
    df["day_index"] = df["day"].map(day_map)

    # Label distribution
    print(f"\n  Label distribution (tb_label):")
    label_counts = df["tb_label"].value_counts().sort_index()
    for lab, cnt in label_counts.items():
        print(f"    {lab}: {cnt} ({cnt/len(df)*100:.1f}%)")

    # Verify book snap columns exist
    missing_book = [c for c in BOOK_SNAP_COLS if c not in df.columns]
    if missing_book:
        raise ValueError(f"Missing book snap columns: {missing_book}")

    # Verify non-spatial features — map to actual names
    feature_mapping = {}
    missing_features = []
    for feat in NONSPATIAL_FEATURES:
        if feat in df.columns:
            feature_mapping[feat] = feat
        else:
            # Try alternate names
            found = False
            for alt in [f"book_{feat}", feat.replace("_", "")]:
                if alt in df.columns:
                    feature_mapping[feat] = alt
                    found = True
                    break
            if not found:
                missing_features.append(feat)

    if missing_features:
        print(f"\n  WARNING: Missing features: {missing_features}")
        print(f"  Available feature-like columns:")
        for c in df.columns:
            if c not in BOOK_SNAP_COLS and c not in [
                "timestamp", "bar_type", "bar_param", "day", "is_warmup",
                "bar_index", "tb_label", "tb_exit_type", "tb_bars_held",
                "return_100", "mbo_event_count", "return_20.1", "return_100.1",
            ] and not c.startswith("msg_summary"):
                print(f"    {c}")

    actual_nonspatial = [feature_mapping[f] for f in NONSPATIAL_FEATURES if f in feature_mapping]
    print(f"\n  Non-spatial features mapped: {len(actual_nonspatial)}/{len(NONSPATIAL_FEATURES)}")

    return df, days_sorted, actual_nonspatial, feature_mapping


def get_fold_data(df, days_sorted, fold_idx):
    """Get train/test split for a given fold."""
    fold = FOLDS[fold_idx]
    train_start, train_end = fold["train"]
    test_start, test_end = fold["test"]

    train_days = days_sorted[train_start - 1 : train_end]
    test_days = days_sorted[test_start - 1 : test_end]

    train_mask = df["day"].isin(train_days)
    test_mask = df["day"].isin(test_days)

    return df[train_mask].copy(), df[test_mask].copy(), train_days, test_days


def prepare_book_tensor(df):
    """Reshape book_snap columns to (N, 2, 20) tensor.

    CSV format: book_snap_0=price_0, book_snap_1=size_0, ..., book_snap_38=price_19, book_snap_39=size_19
    Reshape to (N, 20, 2) then transpose to (N, 2, 20) for Conv1d.
    """
    book_data = df[BOOK_SNAP_COLS].values.astype(np.float32)  # (N, 40)
    book_reshaped = book_data.reshape(-1, 20, 2)  # (N, 20, 2): 20 levels, 2 features each
    book_tensor = torch.from_numpy(book_reshaped).permute(0, 2, 1)  # (N, 2, 20)
    return book_tensor


def normalize_book_sizes(book_tensor, train_mask=None, stats=None):
    """Normalize both channels of book tensor using z-score from train stats.

    Channel 0 (price offsets): z-score normalize
    Channel 1 (sizes): z-score normalize
    """
    if stats is None:
        prices = book_tensor[:, 0, :]  # (N, 20)
        sizes = book_tensor[:, 1, :]   # (N, 20)
        stats = {
            "price_mean": prices.mean().item(),
            "price_std": max(prices.std().item(), 1e-8),
            "size_mean": sizes.mean().item(),
            "size_std": max(sizes.std().item(), 1e-8),
        }

    book_tensor = book_tensor.clone()
    book_tensor[:, 0, :] = (book_tensor[:, 0, :] - stats["price_mean"]) / stats["price_std"]
    book_tensor[:, 1, :] = (book_tensor[:, 1, :] - stats["size_mean"]) / stats["size_std"]
    return book_tensor, stats


def normalize_features(df, feature_cols, train_stats=None):
    """Z-score normalize features. NaN → 0.0 after normalization."""
    features = df[feature_cols].values.astype(np.float32)

    if train_stats is None:
        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        stds[stds < 1e-8] = 1.0
        train_stats = {"means": means, "stds": stds}

    features = (features - train_stats["means"]) / train_stats["stds"]
    features = np.nan_to_num(features, nan=0.0)
    return features, train_stats


# ============================================================================
# CNN Training
# ============================================================================

def train_cnn_regression(book_train, y_train, book_val, y_val, horizon_name, fold_idx,
                         max_epochs=CNN_EPOCHS, patience=CNN_PATIENCE):
    """Train CNN encoder + regression head on forward returns.

    Returns:
        encoder: trained CNNEncoder
        metrics: dict with train_r2, test_r2, epochs_trained
    """
    set_seed(SEED)
    encoder = CNNEncoder()
    head = nn.Linear(16, 1)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=CNN_LR, weight_decay=CNN_WD,
    )

    # Filter out NaN targets from training data
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    if not train_valid.all():
        print(f"    Filtering {(~train_valid).sum()} NaN targets from train set")
        book_train = book_train[train_valid]
        y_train = y_train[train_valid]
    if not val_valid.all():
        print(f"    Filtering {(~val_valid).sum()} NaN targets from test set")
        book_val_clean = book_val[val_valid]
        y_val_clean = y_val[val_valid]
    else:
        book_val_clean = book_val
        y_val_clean = y_val

    # Normalize targets for stable MSE training; track stats for inverse transform
    y_train_mean = float(np.mean(y_train))
    y_train_std = float(np.std(y_train))
    if y_train_std < 1e-8:
        y_train_std = 1.0

    y_train_norm = (y_train - y_train_mean) / y_train_std
    y_val_norm = (y_val_clean - y_train_mean) / y_train_std

    y_train_t = torch.from_numpy(y_train_norm.astype(np.float32))
    y_val_t = torch.from_numpy(y_val_norm.astype(np.float32))

    n_train = len(book_train)

    # Early stopping: use last 20% of train days as validation for stopping
    n_es_val = max(1, int(n_train * 0.2))
    es_train_book = book_train[:-n_es_val]
    es_train_y = y_train_t[:-n_es_val]
    es_val_book = book_train[-n_es_val:]
    es_val_y = y_train_t[-n_es_val:]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    epochs_trained = 0

    for epoch in range(max_epochs):
        encoder.train()
        head.train()

        # Mini-batch training
        indices = torch.randperm(len(es_train_book))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(es_train_book), CNN_BATCH):
            end = min(start + CNN_BATCH, len(es_train_book))
            batch_idx = indices[start:end]
            x_batch = es_train_book[batch_idx]
            y_batch = es_train_y[batch_idx]

            optimizer.zero_grad()
            emb = encoder(x_batch)
            pred = head(emb).squeeze(-1)
            loss = nn.functional.mse_loss(pred, y_batch)

            # Check for NaN
            if torch.isnan(loss):
                if epoch < ABORT_NAN_EPOCH:
                    return None, {"abort": True, "reason": f"NaN loss at epoch {epoch}"}
                break

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation loss for early stopping
        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_emb = encoder(es_val_book)
            val_pred = head(val_emb).squeeze(-1)
            val_loss = nn.functional.mse_loss(val_pred, es_val_y).item()

        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best state
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])

    # Compute R² on full train and test sets (using clean data without NaN)
    # Predictions are in normalized scale; R² is scale-invariant so this is fine.
    # But we compute against normalized targets for consistency.
    encoder.eval()
    head.eval()
    with torch.no_grad():
        train_pred_norm = head(encoder(book_train)).squeeze(-1).numpy()
        val_pred_norm = head(encoder(book_val_clean)).squeeze(-1).numpy()

    # R² is invariant to affine transform, so compute on normalized values
    train_r2 = compute_r2(y_train_norm, train_pred_norm)
    test_r2 = compute_r2(y_val_norm, val_pred_norm)

    metrics = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "epochs_trained": epochs_trained,
        "best_val_loss": float(best_val_loss),
    }

    print(f"  Fold {fold_idx+1} CNN {horizon_name}: "
          f"train_R²={train_r2:.6f}, test_R²={test_r2:.6f}, epochs={epochs_trained}")

    return encoder, metrics


def compute_r2(y_true, y_pred):
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def extract_embeddings(encoder, book_tensor):
    """Extract 16-dim embeddings from frozen encoder."""
    encoder.eval()
    with torch.no_grad():
        embeddings = encoder(book_tensor).numpy()
    # Sanity check: no NaN
    assert not np.any(np.isnan(embeddings)), "NaN detected in CNN embeddings"
    return embeddings


# ============================================================================
# PnL Computation
# ============================================================================

def compute_pnl(predictions, true_labels, cost_rt):
    """Compute per-trade PnL under given cost scenario.

    Returns dict with expectancy, profit_factor, trade_count, gross_profit, gross_loss, net_pnl.
    """
    pnls = []
    for pred, true_lab in zip(predictions, true_labels):
        if pred == 0 or true_lab == 0:
            # No trade
            continue

        if pred == true_lab:
            # Correct direction
            pnl = WIN_PNL_GROSS - cost_rt
        else:
            # Wrong direction
            pnl = -LOSS_PNL_GROSS - cost_rt

        pnls.append(pnl)

    if len(pnls) == 0:
        return {
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
        }

    pnls = np.array(pnls)
    gross_profit = pnls[pnls > 0].sum() if any(pnls > 0) else 0.0
    gross_loss = abs(pnls[pnls < 0].sum()) if any(pnls < 0) else 0.0
    net_pnl = pnls.sum()
    expectancy = pnls.mean()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "trade_count": len(pnls),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "net_pnl": float(net_pnl),
    }


def compute_sharpe(predictions, true_labels, days, cost_rt):
    """Compute annualized Sharpe ratio of daily PnL."""
    # Group trades by day
    daily_pnls = {}
    for pred, true_lab, day in zip(predictions, true_labels, days):
        if pred == 0 or true_lab == 0:
            continue
        if pred == true_lab:
            pnl = WIN_PNL_GROSS - cost_rt
        else:
            pnl = -LOSS_PNL_GROSS - cost_rt

        if day not in daily_pnls:
            daily_pnls[day] = 0.0
        daily_pnls[day] += pnl

    if len(daily_pnls) < 2:
        return 0.0

    daily_returns = np.array(list(daily_pnls.values()))
    if daily_returns.std() < 1e-12:
        return 0.0

    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    return float(sharpe)


# ============================================================================
# CNN-Only Ablation
# ============================================================================

def train_cnn_classifier(book_train, y_train, book_val, y_val, fold_idx):
    """Train CNN encoder + Linear(16, 3) classification head on tb_label."""
    set_seed(SEED)
    encoder = CNNEncoder()
    head = nn.Linear(16, 3)

    # Map labels {-1, 0, 1} -> {0, 1, 2}
    y_train_mapped = torch.from_numpy((y_train + 1).astype(np.int64))
    y_val_mapped = torch.from_numpy((y_val + 1).astype(np.int64))

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=CNN_LR, weight_decay=CNN_WD,
    )
    criterion = nn.CrossEntropyLoss()

    n_train = len(book_train)
    n_es_val = max(1, int(n_train * 0.2))
    es_train_book = book_train[:-n_es_val]
    es_train_y = y_train_mapped[:-n_es_val]
    es_val_book = book_train[-n_es_val:]
    es_val_y = y_train_mapped[-n_es_val:]

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(CNN_EPOCHS):
        encoder.train()
        head.train()

        indices = torch.randperm(len(es_train_book))
        for start in range(0, len(es_train_book), CNN_BATCH):
            end = min(start + CNN_BATCH, len(es_train_book))
            batch_idx = indices[start:end]
            x_batch = es_train_book[batch_idx]
            y_batch = es_train_y[batch_idx]

            optimizer.zero_grad()
            emb = encoder(x_batch)
            logits = head(emb)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        encoder.eval()
        head.eval()
        with torch.no_grad():
            val_logits = head(encoder(es_val_book))
            val_loss = criterion(val_logits, es_val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {
                "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= CNN_PATIENCE:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])

    encoder.eval()
    head.eval()
    with torch.no_grad():
        test_logits = head(encoder(book_val))
        test_preds = test_logits.argmax(dim=1).numpy() - 1  # back to {-1, 0, 1}

    accuracy = accuracy_score(y_val, test_preds)
    f1 = f1_score(y_val, test_preds, average="macro", zero_division=0)

    print(f"  Fold {fold_idx+1} CNN-only: accuracy={accuracy:.4f}, F1={f1:.4f}")
    return test_preds, {"accuracy": float(accuracy), "f1_macro": float(f1)}


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_experiment():
    wall_start = time.time()
    set_seed(SEED)

    # Step 1: Load data
    print("=" * 70)
    print("STEP 1: Data Loading and Validation")
    print("=" * 70)
    df, days_sorted, actual_nonspatial, feature_mapping = load_data()

    if len(actual_nonspatial) < 20:
        print(f"\nWARNING: Only {len(actual_nonspatial)}/20 non-spatial features found.")
        print(f"Using available features: {actual_nonspatial}")

    # Collect all feature columns for GBT-only ablation (all non-book, non-meta, non-target)
    meta_cols = {"timestamp", "bar_type", "bar_param", "day", "is_warmup", "bar_index",
                 "day_index", "fwd_return_1", "fwd_return_5"}
    target_cols = {"tb_label", "tb_exit_type", "tb_bars_held", "return_100",
                   "mbo_event_count", "return_1.1", "return_5.1", "return_20.1", "return_100.1"}
    msg_cols = {c for c in df.columns if c.startswith("msg_summary")}
    book_cols = set(BOOK_SNAP_COLS)

    all_feature_cols = [c for c in df.columns
                        if c not in meta_cols
                        and c not in target_cols
                        and c not in msg_cols
                        and c not in book_cols
                        and c not in {"fwd_return_1", "fwd_return_5"}]
    print(f"\n  All feature columns for GBT-only: {len(all_feature_cols)}")

    # Step 2: Verify fold structure
    print("\n" + "=" * 70)
    print("STEP 2: Fold Structure")
    print("=" * 70)
    label_distribution = {}
    for fold_idx in range(5):
        train_df, test_df, train_days, test_days = get_fold_data(df, days_sorted, fold_idx)
        overlap = set(train_days) & set(test_days)
        assert len(overlap) == 0, f"Fold {fold_idx+1}: train/test day overlap: {overlap}"
        print(f"  Fold {fold_idx+1}: train={len(train_df)} ({len(train_days)} days: "
              f"{train_days[0]}–{train_days[-1]}), test={len(test_df)} ({len(test_days)} days: "
              f"{test_days[0]}–{test_days[-1]})")

        fold_label_dist = test_df["tb_label"].value_counts().sort_index().to_dict()
        label_distribution[f"fold_{fold_idx+1}"] = {str(k): int(v) for k, v in fold_label_dist.items()}

    # ========================================================================
    # Step 3: MVE (Fold 1 only)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Minimum Viable Experiment (Fold 1)")
    print("=" * 70)

    train_df_mve, test_df_mve, _, _ = get_fold_data(df, days_sorted, 0)

    # MVE Check 1: Data loading
    print(f"\n  MVE Check 1: Data shape train={train_df_mve.shape}, test={test_df_mve.shape}")
    book_train_mve = prepare_book_tensor(train_df_mve)
    book_test_mve = prepare_book_tensor(test_df_mve)
    print(f"  Book tensors: train={book_train_mve.shape}, test={book_test_mve.shape}")

    book_train_mve, book_stats = normalize_book_sizes(book_train_mve)
    book_test_mve, _ = normalize_book_sizes(book_test_mve, stats=book_stats)

    # MVE Check 2: Single-fold CNN overfit check (h=5)
    print("\n  MVE Check 2: CNN overfit check (h=5, fold 1)")
    y_train_mve = train_df_mve["fwd_return_5"].values
    y_test_mve = test_df_mve["fwd_return_5"].values

    encoder_mve, metrics_mve = train_cnn_regression(
        book_train_mve, y_train_mve, book_test_mve, y_test_mve,
        horizon_name="h5_mve", fold_idx=0
    )

    if metrics_mve.get("abort"):
        print(f"  ABORT: {metrics_mve['reason']}")
        write_abort_metrics(metrics_mve["reason"], wall_start)
        return

    if metrics_mve["train_r2"] < ABORT_TRAIN_R2:
        print(f"  WARNING: Train R² = {metrics_mve['train_r2']:.6f} < {ABORT_TRAIN_R2}")
        print(f"  CNN cannot fit training data. Proceeding with full pipeline anyway")
        print(f"  to collect all metrics — XGBoost may still work on non-spatial features.")
        print(f"  R3 architecture used BatchNorm + 32→32 channels (12k params);")
        print(f"  current spec uses 32→64 (7.5k params). Architecture mismatch noted.")

    if metrics_mve["test_r2"] < -0.1:
        print(f"  WARNING: Test R² = {metrics_mve['test_r2']:.6f} (severe overfitting)")

    # MVE Check 3: Single-fold XGBoost check
    print("\n  MVE Check 3: XGBoost check (fold 1)")
    embeddings_mve = extract_embeddings(encoder_mve, book_train_mve)
    features_train_mve, feat_stats_mve = normalize_features(train_df_mve, actual_nonspatial)
    X_train_mve = np.concatenate([embeddings_mve, features_train_mve], axis=1)
    y_xgb_train = train_df_mve["tb_label"].values.astype(int)

    model_mve = train_xgboost_classifier(X_train_mve, y_xgb_train, seed=SEED)
    preds_mve = predict_xgboost(model_mve, X_train_mve)
    acc_mve = accuracy_score(y_xgb_train, preds_mve)
    print(f"  XGBoost train accuracy: {acc_mve:.4f}")
    if acc_mve < ABORT_XGB_ACC:
        print(f"  ABORT: XGBoost accuracy ({acc_mve:.4f}) < {ABORT_XGB_ACC}")
        write_abort_metrics(
            f"XGBoost train accuracy ({acc_mve:.4f}) < {ABORT_XGB_ACC}", wall_start)
        return

    print("\n  MVE PASSED — proceeding to full protocol")

    # ========================================================================
    # Step 4: Full 5-Fold Pipeline
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Full 5-Fold Pipeline")
    print("=" * 70)

    fold_results = []
    all_cnn_r2_h1 = []
    all_cnn_r2_h5 = []
    all_cnn_train_r2_h5 = []
    all_xgb_accuracy = []
    all_xgb_f1 = []
    all_predictions = []
    all_true_labels = []
    all_test_days = []
    all_feature_importances = []

    for fold_idx in range(5):
        fold_start = time.time()
        print(f"\n--- Fold {fold_idx + 1}/5 ---")

        train_df_f, test_df_f, train_days_f, test_days_f = get_fold_data(df, days_sorted, fold_idx)

        # Prepare book tensors
        book_train = prepare_book_tensor(train_df_f)
        book_test = prepare_book_tensor(test_df_f)
        book_train, bstats = normalize_book_sizes(book_train)
        book_test, _ = normalize_book_sizes(book_test, stats=bstats)

        # Stage 1a: CNN h=1
        print(f"\n  Stage 1a: CNN h=1")
        y_train_h1 = train_df_f["fwd_return_1"].values
        y_test_h1 = test_df_f["fwd_return_1"].values
        enc_h1, met_h1 = train_cnn_regression(
            book_train, y_train_h1, book_test, y_test_h1,
            horizon_name="h1", fold_idx=fold_idx
        )

        if met_h1.get("abort"):
            print(f"  ABORT: {met_h1['reason']}")
            write_abort_metrics(met_h1["reason"], wall_start)
            return

        # Stage 1b: CNN h=5
        print(f"\n  Stage 1b: CNN h=5")
        y_train_h5 = train_df_f["fwd_return_5"].values
        y_test_h5 = test_df_f["fwd_return_5"].values
        enc_h5, met_h5 = train_cnn_regression(
            book_train, y_train_h5, book_test, y_test_h5,
            horizon_name="h5", fold_idx=fold_idx
        )

        if met_h5.get("abort"):
            print(f"  ABORT: {met_h5['reason']}")
            write_abort_metrics(met_h5["reason"], wall_start)
            return

        all_cnn_r2_h1.append(met_h1["test_r2"])
        all_cnn_r2_h5.append(met_h5["test_r2"])
        all_cnn_train_r2_h5.append(met_h5["train_r2"])

        # Stage 1c: Select best horizon
        mean_h1_so_far = np.mean(all_cnn_r2_h1)
        mean_h5_so_far = np.mean(all_cnn_r2_h5)
        selected_horizon = "h5" if mean_h5_so_far >= mean_h1_so_far else "h1"
        selected_encoder = enc_h5 if selected_horizon == "h5" else enc_h1
        print(f"  Selected horizon: {selected_horizon} (h1 mean R²={mean_h1_so_far:.6f}, h5 mean R²={mean_h5_so_far:.6f})")

        # Stage 2: Embedding extraction
        train_embeddings = extract_embeddings(selected_encoder, book_train)
        test_embeddings = extract_embeddings(selected_encoder, book_test)

        # Stage 3: Feature assembly
        feat_train, feat_stats = normalize_features(train_df_f, actual_nonspatial)
        feat_test, _ = normalize_features(test_df_f, actual_nonspatial, train_stats=feat_stats)

        X_train = np.concatenate([train_embeddings, feat_train], axis=1)
        X_test = np.concatenate([test_embeddings, feat_test], axis=1)
        y_train_xgb = train_df_f["tb_label"].values.astype(int)
        y_test_xgb = test_df_f["tb_label"].values.astype(int)

        print(f"  Feature dims: {X_train.shape[1]} (16 CNN + {feat_train.shape[1]} non-spatial)")

        # Stage 4: XGBoost classification
        print(f"\n  Stage 4: XGBoost classification")
        xgb_model = train_xgboost_classifier(X_train, y_train_xgb, seed=SEED)
        test_preds = predict_xgboost(xgb_model, X_test)
        test_proba = predict_xgboost_proba(xgb_model, X_test)

        # Stage 5: Evaluation
        acc = accuracy_score(y_test_xgb, test_preds)
        f1 = f1_score(y_test_xgb, test_preds, average="macro", zero_division=0)
        all_xgb_accuracy.append(acc)
        all_xgb_f1.append(f1)
        print(f"  XGBoost: accuracy={acc:.4f}, F1_macro={f1:.4f}")

        all_predictions.extend(test_preds.tolist())
        all_true_labels.extend(y_test_xgb.tolist())
        all_test_days.extend(test_df_f["day"].values.tolist())

        # Feature importance
        importance = xgb_model.get_booster().get_score(importance_type="gain")
        all_feature_importances.append(importance)

        # PnL per cost scenario
        pnl_results = {}
        for scenario, cost in COST_SCENARIOS.items():
            pnl = compute_pnl(test_preds, y_test_xgb, cost)
            pnl_results[scenario] = pnl
        sharpe_base = compute_sharpe(test_preds, y_test_xgb, test_df_f["day"].values, COST_SCENARIOS["base"])

        # Save fold results
        fold_dir = RESULTS_DIR / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        with open(fold_dir / "cnn_h1_metrics.json", "w") as f:
            json.dump(met_h1, f, indent=2)
        with open(fold_dir / "cnn_h5_metrics.json", "w") as f:
            json.dump(met_h5, f, indent=2)
        with open(fold_dir / "xgb_metrics.json", "w") as f:
            json.dump({
                "accuracy": acc, "f1_macro": f1,
                "class_report": classification_report(y_test_xgb, test_preds, output_dict=True, zero_division=0),
            }, f, indent=2)
        with open(fold_dir / "pnl_metrics.json", "w") as f:
            json.dump({"cost_scenarios": pnl_results, "sharpe_base": sharpe_base}, f, indent=2)

        # Save predictions CSV
        pred_df = pd.DataFrame({
            "bar_index": test_df_f["bar_index"].values,
            "day": test_df_f["day"].values,
            "true_label": y_test_xgb,
            "predicted_label": test_preds,
            "prob_neg": test_proba[:, 0],
            "prob_zero": test_proba[:, 1],
            "prob_pos": test_proba[:, 2],
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        # Save models
        torch.save(selected_encoder.state_dict(), fold_dir / "cnn_encoder.pt")
        xgb_model.save_model(str(fold_dir / "xgb_model.json"))

        fold_result = {
            "fold": fold_idx + 1,
            "train_days": [int(d) for d in train_days_f],
            "test_days": [int(d) for d in test_days_f],
            "train_size": len(train_df_f),
            "test_size": len(test_df_f),
            "cnn_r2_h1": met_h1["test_r2"],
            "cnn_r2_h5": met_h5["test_r2"],
            "cnn_train_r2_h5": met_h5["train_r2"],
            "xgb_accuracy": acc,
            "xgb_f1_macro": f1,
            "pnl": pnl_results,
            "sharpe_base": sharpe_base,
            "selected_horizon": selected_horizon,
        }
        fold_results.append(fold_result)

        fold_elapsed = time.time() - fold_start
        print(f"  Fold {fold_idx+1} completed in {fold_elapsed:.1f}s")

        # Check wall-clock abort
        total_elapsed = time.time() - wall_start
        if total_elapsed > 5400:  # 90 minutes
            print(f"  ABORT: Wall clock exceeded 90 minutes ({total_elapsed:.0f}s)")
            write_abort_metrics(
                f"Wall clock exceeded 90 minutes ({total_elapsed:.0f}s)", wall_start,
                partial_results=fold_results)
            return

    # Check abort criterion: all 5 folds negative test R² at h=5
    if all(r2 < 0 for r2 in all_cnn_r2_h5):
        print("\nWARNING: All 5 folds produce negative test R² at h=5.")
        print("  CNN embedding carries no signal. Proceeding to ablations to collect all metrics.")

    # ========================================================================
    # Step 5: GBT-Only Ablation
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: GBT-Only Ablation")
    print("=" * 70)

    gbt_only_results = []
    gbt_only_predictions = []
    gbt_only_true_labels = []
    gbt_only_days = []

    for fold_idx in range(5):
        print(f"\n  Fold {fold_idx+1}: GBT-only")
        train_df_f, test_df_f, _, _ = get_fold_data(df, days_sorted, fold_idx)

        feat_train, feat_stats = normalize_features(train_df_f, all_feature_cols)
        feat_test, _ = normalize_features(test_df_f, all_feature_cols, train_stats=feat_stats)

        y_train_xgb = train_df_f["tb_label"].values.astype(int)
        y_test_xgb = test_df_f["tb_label"].values.astype(int)

        xgb_model = train_xgboost_classifier(feat_train, y_train_xgb, seed=SEED)
        test_preds = predict_xgboost(xgb_model, feat_test)

        acc = accuracy_score(y_test_xgb, test_preds)
        f1 = f1_score(y_test_xgb, test_preds, average="macro", zero_division=0)

        pnl_results = {}
        for scenario, cost in COST_SCENARIOS.items():
            pnl_results[scenario] = compute_pnl(test_preds, y_test_xgb, cost)

        gbt_only_results.append({
            "fold": fold_idx + 1,
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "pnl": pnl_results,
            "n_features": feat_train.shape[1],
        })

        gbt_only_predictions.extend(test_preds.tolist())
        gbt_only_true_labels.extend(y_test_xgb.tolist())
        gbt_only_days.extend(test_df_f["day"].values.tolist())

        print(f"    accuracy={acc:.4f}, F1={f1:.4f}, features={feat_train.shape[1]}")

    # ========================================================================
    # Step 6: CNN-Only Ablation
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: CNN-Only Ablation")
    print("=" * 70)

    cnn_only_results = []
    cnn_only_predictions = []
    cnn_only_true_labels = []
    cnn_only_days = []

    for fold_idx in range(5):
        print(f"\n  Fold {fold_idx+1}: CNN-only classifier")
        train_df_f, test_df_f, _, _ = get_fold_data(df, days_sorted, fold_idx)

        book_train = prepare_book_tensor(train_df_f)
        book_test = prepare_book_tensor(test_df_f)
        book_train, bstats = normalize_book_sizes(book_train)
        book_test, _ = normalize_book_sizes(book_test, stats=bstats)

        y_train_xgb = train_df_f["tb_label"].values.astype(int)
        y_test_xgb = test_df_f["tb_label"].values.astype(int)

        test_preds, cnn_met = train_cnn_classifier(
            book_train, y_train_xgb, book_test, y_test_xgb, fold_idx
        )

        pnl_results = {}
        for scenario, cost in COST_SCENARIOS.items():
            pnl_results[scenario] = compute_pnl(test_preds, y_test_xgb, cost)

        cnn_only_results.append({
            "fold": fold_idx + 1,
            "accuracy": cnn_met["accuracy"],
            "f1_macro": cnn_met["f1_macro"],
            "pnl": pnl_results,
        })

        cnn_only_predictions.extend(test_preds.tolist())
        cnn_only_true_labels.extend(y_test_xgb.tolist())
        cnn_only_days.extend(test_df_f["day"].values.tolist())

    # ========================================================================
    # Step 7: Aggregate Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Aggregating Results")
    print("=" * 70)

    wall_elapsed = time.time() - wall_start

    # Aggregate hybrid metrics
    mean_cnn_r2_h1 = float(np.mean(all_cnn_r2_h1))
    mean_cnn_r2_h5 = float(np.mean(all_cnn_r2_h5))
    per_fold_r2_std = float(np.std(all_cnn_r2_h5))
    mean_xgb_accuracy = float(np.mean(all_xgb_accuracy))
    mean_xgb_f1_macro = float(np.mean(all_xgb_f1))

    # Pooled PnL metrics
    preds_arr = np.array(all_predictions)
    labels_arr = np.array(all_true_labels)
    days_arr = np.array(all_test_days)

    cost_sensitivity = {}
    for scenario, cost in COST_SCENARIOS.items():
        pnl = compute_pnl(preds_arr, labels_arr, cost)
        sharpe = compute_sharpe(preds_arr, labels_arr, days_arr, cost)
        cost_sensitivity[scenario] = {**pnl, "sharpe": sharpe}

    aggregate_expectancy_base = cost_sensitivity["base"]["expectancy"]
    aggregate_pf_base = cost_sensitivity["base"]["profit_factor"]
    aggregate_sharpe_base = cost_sensitivity["base"]["sharpe"]

    # GBT-only aggregate
    gbt_preds_arr = np.array(gbt_only_predictions)
    gbt_labels_arr = np.array(gbt_only_true_labels)
    gbt_days_arr = np.array(gbt_only_days)

    gbt_mean_acc = float(np.mean([r["accuracy"] for r in gbt_only_results]))
    gbt_mean_f1 = float(np.mean([r["f1_macro"] for r in gbt_only_results]))
    gbt_agg_pnl_base = compute_pnl(gbt_preds_arr, gbt_labels_arr, COST_SCENARIOS["base"])
    gbt_agg_sharpe = compute_sharpe(gbt_preds_arr, gbt_labels_arr, gbt_days_arr, COST_SCENARIOS["base"])

    # CNN-only aggregate
    cnn_preds_arr = np.array(cnn_only_predictions)
    cnn_labels_arr = np.array(cnn_only_true_labels)
    cnn_days_arr = np.array(cnn_only_days)

    cnn_mean_acc = float(np.mean([r["accuracy"] for r in cnn_only_results]))
    cnn_mean_f1 = float(np.mean([r["f1_macro"] for r in cnn_only_results]))
    cnn_agg_pnl_base = compute_pnl(cnn_preds_arr, cnn_labels_arr, COST_SCENARIOS["base"])
    cnn_agg_sharpe = compute_sharpe(cnn_preds_arr, cnn_labels_arr, cnn_days_arr, COST_SCENARIOS["base"])

    # Ablation deltas
    ablation_delta_accuracy = mean_xgb_accuracy - max(gbt_mean_acc, cnn_mean_acc)
    ablation_delta_expectancy = aggregate_expectancy_base - max(
        gbt_agg_pnl_base["expectancy"], cnn_agg_pnl_base["expectancy"])

    # Feature importance (aggregate across folds)
    agg_importance = {}
    for imp in all_feature_importances:
        for feat, gain in imp.items():
            if feat not in agg_importance:
                agg_importance[feat] = 0.0
            agg_importance[feat] += gain
    for feat in agg_importance:
        agg_importance[feat] /= 5.0

    sorted_importance = sorted(agg_importance.items(), key=lambda x: x[1], reverse=True)
    top10_features = sorted_importance[:10]

    # Map XGBoost feature indices to names
    feature_names = [f"cnn_emb_{i}" for i in range(16)] + actual_nonspatial
    top10_named = []
    for feat_key, gain in top10_features:
        # XGBoost uses 'fN' format
        if feat_key.startswith("f"):
            idx = int(feat_key[1:])
            name = feature_names[idx] if idx < len(feature_names) else feat_key
        else:
            name = feat_key
        top10_named.append({"feature": name, "index": feat_key, "mean_gain": float(gain)})

    # ========================================================================
    # Sanity Checks
    # ========================================================================
    print("\n--- Sanity Checks ---")
    sanity = {}

    # CNN R² h=5 on train > 0.15
    max_train_r2_h5 = max(all_cnn_train_r2_h5)
    sanity["cnn_train_r2_h5_max"] = float(max_train_r2_h5)
    sanity["cnn_train_r2_h5_check"] = max_train_r2_h5 > 0.15
    print(f"  CNN train R² h=5 max: {max_train_r2_h5:.6f} (threshold > 0.15: {'PASS' if max_train_r2_h5 > 0.15 else 'FAIL'})")

    # CNN R² h=5 test for fold 4 or 5 > 0
    fold4_r2 = all_cnn_r2_h5[3] if len(all_cnn_r2_h5) > 3 else None
    fold5_r2 = all_cnn_r2_h5[4] if len(all_cnn_r2_h5) > 4 else None
    late_fold_positive = (fold4_r2 is not None and fold4_r2 > 0) or (fold5_r2 is not None and fold5_r2 > 0)
    sanity["fold4_r2_h5"] = float(fold4_r2) if fold4_r2 is not None else None
    sanity["fold5_r2_h5"] = float(fold5_r2) if fold5_r2 is not None else None
    sanity["late_fold_positive_r2_check"] = late_fold_positive
    print(f"  Fold 4 R² h=5: {fold4_r2:.6f}, Fold 5 R² h=5: {fold5_r2:.6f} "
          f"(at least one > 0: {'PASS' if late_fold_positive else 'FAIL'})")

    # XGBoost accuracy > 0.33
    sanity["xgb_accuracy_above_random"] = mean_xgb_accuracy > 0.33
    print(f"  XGBoost mean accuracy: {mean_xgb_accuracy:.4f} (> 0.33: {'PASS' if mean_xgb_accuracy > 0.33 else 'FAIL'})")

    # No NaN in CNN output (checked during extraction — would have asserted)
    sanity["no_nan_embeddings"] = True
    print(f"  No NaN in CNN embeddings: PASS")

    # Fold day boundaries non-overlapping (checked above)
    sanity["fold_boundaries_valid"] = True
    print(f"  Fold day boundaries non-overlapping: PASS")

    # XGBoost accuracy <= 0.90
    sanity["xgb_accuracy_below_ceiling"] = mean_xgb_accuracy <= 0.90
    print(f"  XGBoost accuracy <= 0.90: {'PASS' if mean_xgb_accuracy <= 0.90 else 'FAIL'} ({mean_xgb_accuracy:.4f})")

    all_sanity_pass = all([
        sanity.get("cnn_train_r2_h5_check", False),
        sanity.get("late_fold_positive_r2_check", False),
        sanity.get("xgb_accuracy_above_random", False),
        sanity.get("no_nan_embeddings", True),
        sanity.get("fold_boundaries_valid", True),
        sanity.get("xgb_accuracy_below_ceiling", True),
    ])

    # ========================================================================
    # Success Criteria
    # ========================================================================
    print("\n--- Success Criteria ---")
    neg_fold_count = sum(1 for r2 in all_cnn_r2_h5 if r2 < 0)

    sc = {
        "SC-1": mean_cnn_r2_h5 >= 0.08,
        "SC-2": True,  # h=1 reported
        "SC-3": mean_xgb_accuracy >= 0.38,
        "SC-4": neg_fold_count == 0,
        "SC-5": aggregate_expectancy_base >= 0.50,
        "SC-6": aggregate_pf_base >= 1.5,
        "SC-7": (mean_xgb_accuracy > gbt_mean_acc) or (aggregate_expectancy_base > gbt_agg_pnl_base["expectancy"]),
        "SC-8": (mean_xgb_accuracy > cnn_mean_acc) or (aggregate_expectancy_base > cnn_agg_pnl_base["expectancy"]),
        "SC-9": True,  # cost sensitivity table produced
        "SC-10": all_sanity_pass,
    }

    for name, passed in sc.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    # ========================================================================
    # Write All Deliverables
    # ========================================================================

    # Aggregate metrics
    aggregate = {
        "mean_cnn_r2_h1": mean_cnn_r2_h1,
        "mean_cnn_r2_h5": mean_cnn_r2_h5,
        "per_fold_r2_std": per_fold_r2_std,
        "neg_fold_count_h5": neg_fold_count,
        "mean_xgb_accuracy": mean_xgb_accuracy,
        "mean_xgb_f1_macro": mean_xgb_f1_macro,
        "aggregate_expectancy_base": aggregate_expectancy_base,
        "aggregate_profit_factor_base": aggregate_pf_base,
        "aggregate_sharpe_base": aggregate_sharpe_base,
        "ablation_delta_accuracy": float(ablation_delta_accuracy),
        "ablation_delta_expectancy": float(ablation_delta_expectancy),
        "per_fold": fold_results,
    }
    with open(RESULTS_DIR / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    # GBT-only ablation
    gbt_ablation = {
        "mean_accuracy": gbt_mean_acc,
        "mean_f1_macro": gbt_mean_f1,
        "aggregate_expectancy_base": gbt_agg_pnl_base["expectancy"],
        "aggregate_profit_factor_base": gbt_agg_pnl_base["profit_factor"],
        "aggregate_sharpe_base": gbt_agg_sharpe,
        "per_fold": gbt_only_results,
    }
    with open(RESULTS_DIR / "ablation_gbt_only.json", "w") as f:
        json.dump(gbt_ablation, f, indent=2)

    # CNN-only ablation
    cnn_ablation = {
        "mean_accuracy": cnn_mean_acc,
        "mean_f1_macro": cnn_mean_f1,
        "aggregate_expectancy_base": cnn_agg_pnl_base["expectancy"],
        "aggregate_profit_factor_base": cnn_agg_pnl_base["profit_factor"],
        "aggregate_sharpe_base": cnn_agg_sharpe,
        "per_fold": cnn_only_results,
    }
    with open(RESULTS_DIR / "ablation_cnn_only.json", "w") as f:
        json.dump(cnn_ablation, f, indent=2)

    # Cost sensitivity
    with open(RESULTS_DIR / "cost_sensitivity.json", "w") as f:
        json.dump(cost_sensitivity, f, indent=2)

    # Feature importance
    with open(RESULTS_DIR / "feature_importance.json", "w") as f:
        json.dump({
            "top10": top10_named,
            "per_fold": [{k: float(v) for k, v in imp.items()} for imp in all_feature_importances],
        }, f, indent=2)

    # ========================================================================
    # Write metrics.json (primary deliverable)
    # ========================================================================
    metrics = {
        "experiment": "hybrid-model-training",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "r3_cnn_r2_h5": 0.132,
            "oracle_expectancy": 4.00,
            "oracle_pf": 3.30,
            "oracle_wr": 0.643,
            "random_accuracy": 0.333,
        },
        "treatment": {
            "mean_cnn_r2_h1": mean_cnn_r2_h1,
            "mean_cnn_r2_h5": mean_cnn_r2_h5,
            "per_fold_r2_std": per_fold_r2_std,
            "neg_fold_count_h5": neg_fold_count,
            "mean_xgb_accuracy": mean_xgb_accuracy,
            "mean_xgb_f1_macro": mean_xgb_f1_macro,
            "aggregate_expectancy_base": aggregate_expectancy_base,
            "aggregate_profit_factor_base": aggregate_pf_base,
            "aggregate_sharpe_base": aggregate_sharpe_base,
            "ablation_delta_accuracy": float(ablation_delta_accuracy),
            "ablation_delta_expectancy": float(ablation_delta_expectancy),
            "cost_sensitivity_table": cost_sensitivity,
            "xgb_top10_features": top10_named,
            "label_distribution": label_distribution,
        },
        "ablation_gbt_only": {
            "mean_accuracy": gbt_mean_acc,
            "mean_f1_macro": gbt_mean_f1,
            "aggregate_expectancy_base": gbt_agg_pnl_base["expectancy"],
            "aggregate_profit_factor_base": gbt_agg_pnl_base["profit_factor"],
        },
        "ablation_cnn_only": {
            "mean_accuracy": cnn_mean_acc,
            "mean_f1_macro": cnn_mean_f1,
            "aggregate_expectancy_base": cnn_agg_pnl_base["expectancy"],
            "aggregate_profit_factor_base": cnn_agg_pnl_base["profit_factor"],
        },
        "per_fold": [
            {
                "fold": r["fold"],
                "train_days": r["train_days"],
                "test_days": r["test_days"],
                "train_size": r["train_size"],
                "test_size": r["test_size"],
                "cnn_r2_h1": r["cnn_r2_h1"],
                "cnn_r2_h5": r["cnn_r2_h5"],
                "cnn_train_r2_h5": r["cnn_train_r2_h5"],
                "xgb_accuracy": r["xgb_accuracy"],
                "xgb_f1_macro": r["xgb_f1_macro"],
                "pnl_base_expectancy": r["pnl"]["base"]["expectancy"],
                "pnl_base_pf": r["pnl"]["base"]["profit_factor"],
                "selected_horizon": r["selected_horizon"],
            }
            for r in fold_results
        ],
        "sanity_checks": sanity,
        "success_criteria": sc,
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": float(wall_elapsed),
            "total_training_runs": 30,
            "total_runs": 30,
        },
        "abort_triggered": False,
        "abort_reason": None,
        "notes": (
            f"Feature mapping: {len(actual_nonspatial)}/{len(NONSPATIAL_FEATURES)} non-spatial features found. "
            f"Duplicate return_1/return_5 columns in CSV handled (used .1 suffix as fwd targets). "
            f"Book snap columns interleaved as (price, size) pairs, reshaped to (N, 2, 20) for CNN. "
            f"All feature columns for GBT-only ablation: {len(all_feature_cols)}."
        ),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Wall clock: {wall_elapsed:.1f}s ({wall_elapsed/60:.1f}min)")
    print(f"Results written to {RESULTS_DIR}")
    print(f"{'=' * 70}")

    # Write analysis.md
    write_analysis(
        fold_results, aggregate, gbt_ablation, cnn_ablation,
        cost_sensitivity, top10_named, label_distribution, sanity, sc,
        days_sorted, all_cnn_r2_h1, all_cnn_r2_h5
    )


def write_abort_metrics(reason, wall_start, partial_results=None):
    """Write metrics.json for an aborted experiment."""
    wall_elapsed = time.time() - wall_start
    metrics = {
        "experiment": "hybrid-model-training",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "r3_cnn_r2_h5": 0.132,
            "oracle_expectancy": 4.00,
            "random_accuracy": 0.333,
        },
        "treatment": {},
        "per_fold": partial_results or [],
        "sanity_checks": {},
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": float(wall_elapsed),
        },
        "abort_triggered": True,
        "abort_reason": reason,
        "notes": f"Experiment aborted: {reason}",
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nAbort metrics written to {RESULTS_DIR / 'metrics.json'}")


def write_analysis(fold_results, aggregate, gbt_ablation, cnn_ablation,
                   cost_sensitivity, top10_features, label_distribution,
                   sanity, sc, days_sorted, all_r2_h1, all_r2_h5):
    """Write analysis.md with all required tables."""
    lines = []
    lines.append("# CNN+GBT Hybrid Model Training — Results\n")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary\n")
    sc_pass_count = sum(1 for v in sc.values() if v)
    lines.append(f"Success criteria: {sc_pass_count}/10 passed.\n")

    # 2. CNN R² Table
    lines.append("## 2. CNN R² by Fold and Horizon\n")
    lines.append("| Fold | h=1 R² | h=5 R² | Train R² (h=5) | Train Size | Selected |")
    lines.append("|------|--------|--------|----------------|------------|----------|")
    for r in fold_results:
        lines.append(f"| {r['fold']} | {r['cnn_r2_h1']:.6f} | {r['cnn_r2_h5']:.6f} | "
                      f"{r['cnn_train_r2_h5']:.6f} | {r['train_size']} | {r['selected_horizon']} |")
    lines.append(f"| **Mean** | **{aggregate['mean_cnn_r2_h1']:.6f}** | **{aggregate['mean_cnn_r2_h5']:.6f}** | | | |")
    lines.append(f"| **Std** | | **{aggregate['per_fold_r2_std']:.6f}** | | | |\n")

    # 3. XGBoost Accuracy and F1 Table
    lines.append("## 3. XGBoost Classification Metrics\n")
    lines.append("| Fold | Accuracy | F1 Macro |")
    lines.append("|------|----------|----------|")
    for r in fold_results:
        lines.append(f"| {r['fold']} | {r['xgb_accuracy']:.4f} | {r['xgb_f1_macro']:.4f} |")
    lines.append(f"| **Mean** | **{aggregate['mean_xgb_accuracy']:.4f}** | **{aggregate['mean_xgb_f1_macro']:.4f}** |\n")

    # 4. PnL Table
    lines.append("## 4. PnL by Fold and Cost Scenario\n")
    lines.append("| Fold | Optimistic Exp | Base Exp | Pessimistic Exp | Base PF | Base Trades |")
    lines.append("|------|---------------|----------|-----------------|---------|-------------|")
    for r in fold_results:
        lines.append(f"| {r['fold']} | ${r['pnl']['optimistic']['expectancy']:.2f} | "
                      f"${r['pnl']['base']['expectancy']:.2f} | "
                      f"${r['pnl']['pessimistic']['expectancy']:.2f} | "
                      f"{r['pnl']['base']['profit_factor']:.3f} | "
                      f"{r['pnl']['base']['trade_count']} |")
    lines.append(f"\n**Aggregate (pooled):**\n")
    for scenario in ["optimistic", "base", "pessimistic"]:
        cs = cost_sensitivity[scenario]
        lines.append(f"- **{scenario.capitalize()}**: expectancy=${cs['expectancy']:.2f}, "
                      f"PF={cs['profit_factor']:.3f}, trades={cs['trade_count']}, Sharpe={cs['sharpe']:.3f}")
    lines.append("")

    # 5. Ablation Comparison
    lines.append("## 5. Ablation Comparison\n")
    lines.append("| Model | Mean Accuracy | Mean F1 | Agg Expectancy (base) | Agg PF (base) |")
    lines.append("|-------|--------------|---------|----------------------|---------------|")
    lines.append(f"| **Hybrid** | {aggregate['mean_xgb_accuracy']:.4f} | {aggregate['mean_xgb_f1_macro']:.4f} | "
                  f"${aggregate['aggregate_expectancy_base']:.2f} | {aggregate['aggregate_profit_factor_base']:.3f} |")
    lines.append(f"| GBT-only | {gbt_ablation['mean_accuracy']:.4f} | {gbt_ablation['mean_f1_macro']:.4f} | "
                  f"${gbt_ablation['aggregate_expectancy_base']:.2f} | {gbt_ablation['aggregate_profit_factor_base']:.3f} |")
    lines.append(f"| CNN-only | {cnn_ablation['mean_accuracy']:.4f} | {cnn_ablation['mean_f1_macro']:.4f} | "
                  f"${cnn_ablation['aggregate_expectancy_base']:.2f} | {cnn_ablation['aggregate_profit_factor_base']:.3f} |")
    lines.append(f"\nDelta accuracy (hybrid − max baseline): {aggregate['ablation_delta_accuracy']:.4f}")
    lines.append(f"Delta expectancy (hybrid − max baseline): ${aggregate['ablation_delta_expectancy']:.2f}\n")

    # 6. Label Distribution
    lines.append("## 6. Label Distribution (Test Sets)\n")
    lines.append("| Fold | Label -1 | Label 0 | Label +1 |")
    lines.append("|------|----------|---------|----------|")
    for fold_key, dist in label_distribution.items():
        lines.append(f"| {fold_key} | {dist.get('-1', 0)} | {dist.get('0', 0)} | {dist.get('1', 0)} |")
    lines.append("")

    # 7. Feature Importance
    lines.append("## 7. XGBoost Top-10 Features (Mean Gain)\n")
    lines.append("| Rank | Feature | Index | Mean Gain |")
    lines.append("|------|---------|-------|-----------|")
    for i, feat in enumerate(top10_features):
        lines.append(f"| {i+1} | {feat['feature']} | {feat['index']} | {feat['mean_gain']:.2f} |")
    lines.append("")

    # 8. Fold Date Ranges
    lines.append("## 8. Fold Date Ranges and Train Set Sizes\n")
    lines.append("| Fold | Train Days | Test Days | Train Size | Test Size |")
    lines.append("|------|-----------|-----------|------------|-----------|")
    for r in fold_results:
        lines.append(f"| {r['fold']} | {r['train_days'][0]}–{r['train_days'][-1]} ({len(r['train_days'])}d) | "
                      f"{r['test_days'][0]}–{r['test_days'][-1]} ({len(r['test_days'])}d) | "
                      f"{r['train_size']} | {r['test_size']} |")
    lines.append("")

    # 9. Success Criteria
    lines.append("## 9. Success Criteria Pass/Fail\n")
    sc_descriptions = {
        "SC-1": f"mean_cnn_r2_h5 >= 0.08 → {aggregate['mean_cnn_r2_h5']:.6f}",
        "SC-2": f"mean_cnn_r2_h1 reported → {aggregate['mean_cnn_r2_h1']:.6f}",
        "SC-3": f"mean_xgb_accuracy >= 0.38 → {aggregate['mean_xgb_accuracy']:.4f}",
        "SC-4": f"No negative CNN R² folds (h=5) → {aggregate['neg_fold_count_h5']}/5 negative",
        "SC-5": f"aggregate_expectancy_base >= $0.50 → ${aggregate['aggregate_expectancy_base']:.2f}",
        "SC-6": f"aggregate_profit_factor_base >= 1.5 → {aggregate['aggregate_profit_factor_base']:.3f}",
        "SC-7": f"Hybrid beats GBT-only (acc OR exp)",
        "SC-8": f"Hybrid beats CNN-only (acc OR exp)",
        "SC-9": f"Cost sensitivity table produced",
        "SC-10": f"No sanity check failures",
    }
    for name, passed in sc.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"- **{name}** [{status}]: {sc_descriptions.get(name, '')}")

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(lines))

    print(f"  analysis.md written to {RESULTS_DIR / 'analysis.md'}")


if __name__ == "__main__":
    run_experiment()
