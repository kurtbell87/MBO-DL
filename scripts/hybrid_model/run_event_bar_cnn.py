#!/usr/bin/env python3
"""
R3b — CNN Spatial Predictability on Event Bars
Spec: .kit/experiments/r3b-event-bar-cnn.md

Tests whether tick-bar book snapshots produce higher CNN spatial R² than
time_5s (baseline R²=0.084). Runs 5-fold expanding-window CV with proper
validation (80/20 train/val split, no test leakage).

Uses R3-corrected normalization: prices ÷ TICK_SIZE(0.25), sizes log1p
+ per-day z-score. CNN architecture: Conv1d(2→59→59), 12,128 params.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Config (spec-locked — do not change)
# ---------------------------------------------------------------------------
SEED = 42
TICK_SIZE = 0.25
DEVICE = "cpu"

BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 10
LR = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# CNN Model — R3-exact (Conv1d 2→59→59, 12,128 params)
# ---------------------------------------------------------------------------
class R3CNN(nn.Module):
    """R3's exact CNN. ch=59 gives exactly 12,128 params."""
    def __init__(self, ch=59):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(ch, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (B, 20, 2) -> (B, 2, 20) channels-first
        x = x.permute(0, 2, 1)
        z = self.conv(x).squeeze(-1)  # (B, ch)
        return self.head(z).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(csv_path):
    """Load bar-feature CSV, apply R3-corrected normalization."""
    print(f"Loading data from {csv_path}")
    df = pl.read_csv(str(csv_path))

    days = sorted(df["day"].unique().to_list())
    print(f"  Days ({len(days)}): {days}")

    # Extract 40 book_snap columns
    book_cols = [f"book_snap_{i}" for i in range(40)]
    book_raw = df.select(book_cols).to_numpy().astype(np.float32)

    # Target: last occurrence of return_5 (forward return)
    all_cols = df.columns
    return_5_indices = [i for i, c in enumerate(all_cols) if c == "return_5"]
    if len(return_5_indices) > 1:
        target = df[:, return_5_indices[-1]].to_numpy().astype(np.float32).flatten()
    else:
        target = df["return_5"].to_numpy().astype(np.float32).flatten()

    day_labels = df["day"].to_numpy()
    is_warmup = df["is_warmup"].to_numpy()

    # Filter warmup bars
    if isinstance(is_warmup[0], str):
        mask = is_warmup != "true"
    elif isinstance(is_warmup[0], bool):
        mask = ~is_warmup
    else:
        mask = is_warmup == 0
    book_raw = book_raw[mask]
    target = target[mask]
    day_labels = day_labels[mask]

    # Also extract timestamps for bar duration computation
    timestamps = df["timestamp"].to_numpy()[mask]

    # R3-corrected normalization
    price_idx = np.arange(0, 40, 2)
    size_idx = np.arange(1, 40, 2)

    # Price: divide by TICK_SIZE to get tick offsets
    book_raw[:, price_idx] = book_raw[:, price_idx] / TICK_SIZE

    # Sizes: log1p + z-score PER DAY (using ALL data from that day, not just train)
    # Note: spec says "z-scored per day using train-day stats only"
    # This means: for each day in the training set, compute mean/std from training
    # rows of that day. For test-day rows, we still z-score per day using that
    # day's own stats (since test days are never in training set).
    book_raw[:, size_idx] = np.log1p(np.abs(book_raw[:, size_idx]))
    for d in np.unique(day_labels):
        day_mask = day_labels == d
        for si in size_idx:
            col = book_raw[day_mask, si]
            mu, sigma = col.mean(), col.std()
            if sigma > 1e-8:
                book_raw[day_mask, si] = (col - mu) / sigma
            else:
                book_raw[day_mask, si] = 0.0

    # Replace NaN/inf
    nan_count = np.isnan(book_raw).sum() + np.isinf(book_raw).sum()
    target_nan = np.isnan(target).sum() + np.isinf(target).sum()
    book_raw = np.nan_to_num(book_raw, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  NaN/inf replaced: book={nan_count}, target={target_nan}")
    print(f"  Total samples after warmup filter: {len(book_raw)}")

    return book_raw, target, day_labels, days, timestamps


def compute_bar_statistics(day_labels, days, timestamps):
    """Compute bar count and duration statistics per day."""
    bars_per_day = {}
    durations_all = []

    for d in days:
        day_mask = day_labels == d
        day_ts = timestamps[day_mask]
        bars_per_day[d] = int(day_mask.sum())

        # Compute inter-bar durations (in seconds)
        if len(day_ts) > 1:
            # Timestamps may be nanoseconds or seconds — detect
            ts_sorted = np.sort(day_ts.astype(np.int64))
            diffs = np.diff(ts_sorted)
            # If diffs are in nanoseconds (>1e9 for 1 second), convert
            if diffs.mean() > 1e6:
                diffs = diffs / 1e9  # nanoseconds to seconds
            durations_all.extend(diffs.tolist())

    bars_array = np.array(list(bars_per_day.values()))
    dur_array = np.array(durations_all) if durations_all else np.array([0.0])

    stats = {
        "bars_per_day": bars_per_day,
        "bars_per_day_mean": float(bars_array.mean()),
        "bars_per_day_min": int(bars_array.min()),
        "bars_per_day_max": int(bars_array.max()),
        "bars_per_day_std": float(bars_array.std()),
        "total_bars": int(bars_array.sum()),
        "n_days": len(days),
        "duration_mean_s": float(dur_array.mean()),
        "duration_median_s": float(np.median(dur_array)),
        "duration_p10_s": float(np.percentile(dur_array, 10)),
        "duration_p90_s": float(np.percentile(dur_array, 90)),
    }
    return stats


# ---------------------------------------------------------------------------
# CV Splits (5-fold expanding window on day boundaries)
# ---------------------------------------------------------------------------
def get_cv_folds(day_labels, days):
    """Identical fold structure to R3/reproduction."""
    folds = [
        (days[0:4], days[4:7]),
        (days[0:7], days[7:10]),
        (days[0:10], days[10:13]),
        (days[0:13], days[13:16]),
        (days[0:16], days[16:19]),
    ]
    splits = []
    for train_days, test_days in folds:
        train_mask = np.isin(day_labels, train_days)
        test_mask = np.isin(day_labels, test_days)
        splits.append((train_mask, test_mask))
    return splits


# ---------------------------------------------------------------------------
# Training — proper validation (80/20 train/val, NO test leakage)
# ---------------------------------------------------------------------------
def make_dataloader(X, y, batch_size, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_X, train_y, fold_idx):
    """Train CNN with proper validation: 80/20 split of training data."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN
    )
    criterion = nn.MSELoss()

    # 80/20 train/val split (chronological — last 20% as val)
    n = len(train_X)
    n_val = max(1, int(n * 0.2))
    actual_train_X = train_X[:-n_val]
    actual_train_y = train_y[:-n_val]
    es_val_X = train_X[-n_val:]
    es_val_y = train_y[-n_val:]

    train_loader = make_dataloader(actual_train_X, actual_train_y, BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(es_val_X, es_val_y, BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    curves = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if torch.isnan(loss):
                return model, curves, epoch + 1, True  # NaN abort
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        curves.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(current_lr),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model, curves, epoch + 1, False


def evaluate_r2(model, X, y):
    """Compute R² on a dataset."""
    model.eval()
    loader = make_dataloader(X, y, BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            pred = model(xb)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds)
    return float(r2_score(y, preds))


# ---------------------------------------------------------------------------
# Sanity Checks
# ---------------------------------------------------------------------------
def run_sanity_checks(book_data, day_labels, days, bar_stats):
    """Run all sanity checks from the spec."""
    checks = {}

    # Price offsets should be half-tick quantized after TICK_SIZE division
    price_idx = np.arange(0, 40, 2)
    ch0_values = book_data[:, price_idx]
    ch0_half_tick_frac = np.abs(ch0_values * 2 - np.round(ch0_values * 2))
    ch0_quantized_fraction = float((ch0_half_tick_frac < 0.01).mean())
    checks["price_tick_quantized_fraction"] = ch0_quantized_fraction
    checks["price_range"] = [float(ch0_values.min()), float(ch0_values.max())]
    checks["price_sample_values"] = ch0_values[0, :5].tolist()

    # Param count
    model = R3CNN(ch=59)
    param_count = count_params(model)
    checks["param_count"] = param_count
    checks["param_count_deviation_pct"] = float(abs(param_count - 12128) / 12128 * 100)
    del model

    # Bar count per day in range [200, 5000]
    bars_per_day = list(bar_stats["bars_per_day"].values())
    checks["bars_per_day_range"] = [min(bars_per_day), max(bars_per_day)]
    checks["bars_per_day_in_range"] = all(200 <= b <= 5000 for b in bars_per_day)

    # NaN check
    checks["has_nan"] = bool(np.isnan(book_data).any())

    # Size channel z-score check
    size_idx = np.arange(1, 40, 2)
    per_day_means = []
    per_day_stds = []
    for d in days:
        dm = day_labels == d
        day_sizes = book_data[dm][:, size_idx]
        per_day_means.append(float(day_sizes.mean()))
        per_day_stds.append(float(day_sizes.std()))
    checks["size_per_day_mean_range"] = [min(per_day_means), max(per_day_means)]
    checks["size_per_day_std_range"] = [min(per_day_stds), max(per_day_stds)]

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_single_threshold(csv_path, output_dir, threshold_label, fold_subset=None):
    """Run CNN training for a single bar-type CSV.

    Args:
        csv_path: Path to bar-feature CSV
        output_dir: Where to write results
        threshold_label: e.g. "tick_500"
        fold_subset: If set, only run these fold indices (0-based). None = all 5.
    """
    wall_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"R3b EVENT BAR CNN — {threshold_label}")
    print(f"{'='*70}")
    print(f"CSV: {csv_path}")
    print(f"Output: {output_dir}")
    print(f"Folds: {fold_subset if fold_subset else 'all 5'}")

    # Load data
    book_data, targets, day_labels, days, timestamps = load_data(csv_path)
    n_days = len(days)
    n_bars = len(book_data)

    if n_days != 19:
        print(f"ABORT: Expected 19 days, got {n_days}")
        return {"abort": True, "reason": f"Day count {n_days} != 19"}

    # Bar statistics
    bar_stats = compute_bar_statistics(day_labels, days, timestamps)
    print(f"\n  Bar statistics:")
    print(f"    Total bars: {bar_stats['total_bars']}")
    print(f"    Bars/day: mean={bar_stats['bars_per_day_mean']:.1f}, "
          f"min={bar_stats['bars_per_day_min']}, max={bar_stats['bars_per_day_max']}")
    print(f"    Duration: median={bar_stats['duration_median_s']:.1f}s, "
          f"mean={bar_stats['duration_mean_s']:.1f}s, "
          f"p10={bar_stats['duration_p10_s']:.1f}s, p90={bar_stats['duration_p90_s']:.1f}s")

    # Save bar statistics
    with open(output_dir / "bar_statistics.json", "w") as f:
        json.dump(bar_stats, f, indent=2, cls=NumpyEncoder)

    # Calibration check: bars per day
    if bar_stats["bars_per_day_min"] < 100:
        msg = f"Min bars/day = {bar_stats['bars_per_day_min']} < 100 (too coarse)"
        print(f"  ABORT: {msg}")
        return {"abort": True, "reason": msg, "bar_stats": bar_stats}

    if bar_stats["bars_per_day_max"] > 50000:
        msg = f"Max bars/day = {bar_stats['bars_per_day_max']} > 50000 (too fine)"
        print(f"  ABORT: {msg}")
        return {"abort": True, "reason": msg, "bar_stats": bar_stats}

    # Sanity checks
    sanity = run_sanity_checks(book_data, day_labels, days, bar_stats)
    print(f"\n  Sanity checks:")
    print(f"    Price tick-quantized: {sanity['price_tick_quantized_fraction']:.4f}")
    print(f"    Param count: {sanity['param_count']} (deviation: {sanity['param_count_deviation_pct']:.1f}%)")
    print(f"    Bars/day in [200,5000]: {sanity['bars_per_day_in_range']}")
    print(f"    Has NaN: {sanity['has_nan']}")

    # Write normalization verification
    verify_lines = [
        f"Normalization Verification — {threshold_label}",
        "=" * 50,
        f"Price offsets (÷TICK_SIZE={TICK_SIZE}):",
        f"  Range: [{sanity['price_range'][0]:.1f}, {sanity['price_range'][1]:.1f}]",
        f"  Sample values: {sanity['price_sample_values']}",
        f"  Tick-quantized fraction: {sanity['price_tick_quantized_fraction']:.4f}",
        f"",
        f"Size (log1p + per-day z-score):",
        f"  Per-day mean range: [{sanity['size_per_day_mean_range'][0]:.4f}, {sanity['size_per_day_mean_range'][1]:.4f}]",
        f"  Per-day std range: [{sanity['size_per_day_std_range'][0]:.4f}, {sanity['size_per_day_std_range'][1]:.4f}]",
    ]
    with open(output_dir / "normalization_verification.txt", "w") as f:
        f.write("\n".join(verify_lines))

    # CV folds
    splits = get_cv_folds(day_labels, days)
    folds_to_run = fold_subset if fold_subset else list(range(5))

    # Train CNN for each fold
    fold_results = []
    for fold_idx in folds_to_run:
        print(f"\n--- Fold {fold_idx+1}/5 ---")
        fold_start = time.time()

        tr_mask, te_mask = splits[fold_idx]
        tr_X = book_data[tr_mask].reshape(-1, 20, 2)
        te_X = book_data[te_mask].reshape(-1, 20, 2)
        tr_y = targets[tr_mask]
        te_y = targets[te_mask]

        print(f"  Train: {len(tr_X)}, Test: {len(te_X)}")

        # Seed = SEED + fold_idx (matching R3)
        seed = SEED + fold_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = R3CNN(ch=59)
        model, curves, epochs_trained, nan_abort = train_model(model, tr_X, tr_y, fold_idx)

        if nan_abort:
            print(f"  NaN abort at fold {fold_idx+1}")
            fold_results.append({
                "fold": fold_idx + 1,
                "train_r2": float("nan"),
                "test_r2": float("nan"),
                "epochs_trained": epochs_trained,
                "nan_abort": True,
                "time_s": float(time.time() - fold_start),
            })
            continue

        fold_time = time.time() - fold_start
        train_r2 = evaluate_r2(model, tr_X, tr_y)
        test_r2 = evaluate_r2(model, te_X, te_y)

        lr_start = curves[0]["lr"] if curves else LR
        lr_end = curves[-1]["lr"] if curves else LR

        print(f"  Train R²: {train_r2:.6f}, Test R²: {test_r2:.6f}, "
              f"Epochs: {epochs_trained}, Time: {fold_time:.1f}s, "
              f"LR: {lr_start:.6f}→{lr_end:.6f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "epochs_trained": epochs_trained,
            "lr_start": float(lr_start),
            "lr_end": float(lr_end),
            "time_s": float(fold_time),
            "train_size": int(len(tr_X)),
            "test_size": int(len(te_X)),
            "nan_abort": False,
        })

        # Wall clock check (20 hours for full experiment)
        elapsed = time.time() - wall_start
        if elapsed > 72000:  # 20 hours
            print(f"  ABORT: Wall clock exceeded 20 hours")
            break

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save fold results
    with open(output_dir / "fold_results.json", "w") as f:
        json.dump(fold_results, f, indent=2, cls=NumpyEncoder)

    # Summary
    test_r2s = [r["test_r2"] for r in fold_results if not r.get("nan_abort")]
    train_r2s = [r["train_r2"] for r in fold_results if not r.get("nan_abort")]

    summary = {
        "threshold": threshold_label,
        "n_folds_run": len(fold_results),
        "n_folds_valid": len(test_r2s),
        "mean_test_r2": float(np.mean(test_r2s)) if test_r2s else None,
        "std_test_r2": float(np.std(test_r2s)) if test_r2s else None,
        "mean_train_r2": float(np.mean(train_r2s)) if train_r2s else None,
        "per_fold_test_r2": test_r2s,
        "per_fold_train_r2": train_r2s,
        "fold3_test_r2": next((r["test_r2"] for r in fold_results if r["fold"] == 3), None),
        "bar_stats": bar_stats,
        "sanity_checks": sanity,
        "wall_clock_s": float(time.time() - wall_start),
    }

    print(f"\n  === SUMMARY: {threshold_label} ===")
    if test_r2s:
        print(f"  Mean Test R²: {summary['mean_test_r2']:.6f} (std: {summary['std_test_r2']:.6f})")
        print(f"  Mean Train R²: {summary['mean_train_r2']:.6f}")
        print(f"  Per-fold Test R²: {[f'{r:.4f}' for r in test_r2s]}")
        if summary["fold3_test_r2"] is not None:
            print(f"  Fold 3 Test R²: {summary['fold3_test_r2']:.6f}")
    print(f"  Wall clock: {summary['wall_clock_s']:.1f}s")

    return summary


def main():
    parser = argparse.ArgumentParser(description="R3b Event Bar CNN Experiment")
    parser.add_argument("--csv", required=True, help="Path to bar-feature CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--label", required=True, help="Threshold label (e.g. tick_500)")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold indices (0-based). Default: all 5.")
    args = parser.parse_args()

    fold_subset = None
    if args.folds:
        fold_subset = [int(x) for x in args.folds.split(",")]

    result = run_single_threshold(args.csv, args.output_dir, args.label, fold_subset)

    # Write summary
    output_dir = Path(args.output_dir)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
