#!/usr/bin/env python3
"""
R3b Genuine Tick Bars — CNN Spatial Signal on Trade-Event Tick Bars
Full protocol: calibration -> MVE -> full sweep -> analysis -> metrics

Spec: .kit/experiments/r3b-genuine-tick-bars.md
Output: .kit/results/r3b-genuine-tick-bars/

Uses R3-corrected normalization: prices / TICK_SIZE(0.25), sizes log1p
+ per-day z-score. CNN architecture: Conv1d(2->59->59), 12,128 params.
"""

import csv
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

# ============================================================================
# Config (spec-locked -- do not change)
# ============================================================================
TICK_SIZE = 0.25
SEED_BASE = 42
BATCH_SIZE = 512
MAX_EPOCHS = 50
PATIENCE = 10
LR = 1e-3
LR_MIN = 1e-5
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"

THRESHOLDS = [25, 50, 100, 250, 500, 1000, 2000, 5000]
VIABLE_MIN_BARS_PER_DAY = 100
MAX_SELECTED_THRESHOLDS = 4
WALL_CLOCK_BUDGET = 7200  # 2 hours

# Baseline from 9E
BASELINE_R2 = 0.089
BASELINE_PER_FOLD = [0.139, 0.086, -0.049, 0.131, 0.140]
BASELINE_BARS_PER_DAY = 4630

BAR_EXPORT_BIN = "./build/bar_feature_export"
RESULTS_DIR = Path(".kit/results/r3b-genuine-tick-bars")


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


# ============================================================================
# CNN Model -- R3-exact (Conv1d 2->59->59, 12,128 params)
# ============================================================================
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
        # x: (B, 20, 2) -> permute to (B, 2, 20) channels-first
        x = x.permute(0, 2, 1)
        z = self.conv(x).squeeze(-1)  # (B, ch)
        return self.head(z).squeeze(-1)  # (B,)


# ============================================================================
# Data Loading + R3-corrected normalization
# ============================================================================
def load_csv(csv_path):
    """Load bar-feature CSV, apply normalization, return processed data."""
    print(f"  Loading {csv_path}")
    df = pl.read_csv(str(csv_path))

    days = sorted(df["day"].unique().to_list())
    print(f"    Days ({len(days)}): {days[:5]}...{days[-3:]}" if len(days) > 8 else f"    Days ({len(days)}): {days}")

    # Extract 40 book_snap columns
    book_cols = [f"book_snap_{i}" for i in range(40)]
    book_raw = df.select(book_cols).to_numpy().astype(np.float32)

    # Target: return_5 (last occurrence if duplicated)
    all_cols = df.columns
    return_5_indices = [i for i, c in enumerate(all_cols) if c == "return_5"]
    if len(return_5_indices) > 1:
        target = df[:, return_5_indices[-1]].to_numpy().astype(np.float32).flatten()
    else:
        target = df["return_5"].to_numpy().astype(np.float32).flatten()

    day_labels = df["day"].to_numpy()
    is_warmup = df["is_warmup"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

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
    timestamps = timestamps[mask]

    # R3-corrected normalization
    price_idx = np.arange(0, 40, 2)
    size_idx = np.arange(1, 40, 2)

    # Channel 0: divide by TICK_SIZE to get tick offsets
    book_raw[:, price_idx] /= TICK_SIZE

    # Channel 1: log1p + z-score PER DAY
    # Each day is z-scored independently -- no leakage across days
    book_raw[:, size_idx] = np.log1p(np.abs(book_raw[:, size_idx]))
    for d in np.unique(day_labels):
        dm = day_labels == d
        for si in size_idx:
            col = book_raw[dm, si]
            mu, sigma = col.mean(), col.std()
            if sigma > 1e-8:
                book_raw[dm, si] = (col - mu) / sigma
            else:
                book_raw[dm, si] = 0.0

    # NaN/inf cleanup
    nan_count = int(np.isnan(book_raw).sum() + np.isinf(book_raw).sum())
    target_nan = int(np.isnan(target).sum() + np.isinf(target).sum())
    book_raw = np.nan_to_num(book_raw, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"    Samples after warmup filter: {len(book_raw)}, NaN replaced: book={nan_count}, target={target_nan}")

    return book_raw, target, day_labels, days, timestamps, nan_count, target_nan


def compute_bar_stats(day_labels, days, timestamps):
    """Compute bar count and duration statistics per day."""
    bars_per_day = {}
    durations_all = []

    for d in days:
        dm = day_labels == d
        day_ts = timestamps[dm]
        bars_per_day[str(d)] = int(dm.sum())

        if len(day_ts) > 1:
            ts_sorted = np.sort(day_ts.astype(np.int64))
            diffs = np.diff(ts_sorted)
            # Detect if nanoseconds
            if diffs.mean() > 1e6:
                diffs = diffs / 1e9
            durations_all.extend(diffs.tolist())

    bars_arr = np.array(list(bars_per_day.values()))
    dur_arr = np.array(durations_all) if durations_all else np.array([0.0])

    return {
        "bars_per_day": bars_per_day,
        "bars_per_day_mean": float(bars_arr.mean()),
        "bars_per_day_std": float(bars_arr.std()),
        "bars_per_day_cv": float(bars_arr.std() / bars_arr.mean()) if bars_arr.mean() > 0 else 0.0,
        "bars_per_day_min": int(bars_arr.min()),
        "bars_per_day_max": int(bars_arr.max()),
        "total_bars": int(bars_arr.sum()),
        "n_days": len(days),
        "duration_mean_s": float(dur_arr.mean()),
        "duration_median_s": float(np.median(dur_arr)),
        "duration_p10_s": float(np.percentile(dur_arr, 10)),
        "duration_p90_s": float(np.percentile(dur_arr, 90)),
    }


# ============================================================================
# CV Folds (5-fold expanding window on day boundaries)
# ============================================================================
def get_cv_folds(day_labels, days):
    """Identical fold structure to R3/9D/9E.

    Fold 1: train_pool = days 1-4,  test = days 5-7
    Fold 2: train_pool = days 1-7,  test = days 8-10
    Fold 3: train_pool = days 1-10, test = days 11-13
    Fold 4: train_pool = days 1-13, test = days 14-16
    Fold 5: train_pool = days 1-16, test = days 17-19

    80/20 val split from train_pool happens inside train_cnn().
    """
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


# ============================================================================
# CNN Training with proper validation (80/20 split, NO test leakage)
# ============================================================================
def make_loader(X, y, batch_size, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(X_t, y_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_cnn(train_X, train_y, seed):
    """Train CNN with proper validation: last 20% of training data as val."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = R3CNN(ch=59)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR_MIN
    )
    criterion = nn.MSELoss()

    # 80/20 chronological split
    n = len(train_X)
    n_val = max(1, int(n * 0.2))
    act_train_X, act_train_y = train_X[:-n_val], train_y[:-n_val]
    val_X, val_y = train_X[-n_val:], train_y[-n_val:]

    train_loader = make_loader(act_train_X, act_train_y, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_X, val_y, BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    lr_start = LR
    lr_end = LR
    epochs_trained = 0

    for epoch in range(MAX_EPOCHS):
        epochs_trained = epoch + 1

        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if torch.isnan(loss):
                return model, epochs_trained, True, lr_start, LR  # NaN abort
            loss.backward()
            optimizer.step()
        scheduler.step()

        lr_end = optimizer.param_groups[0]["lr"]
        if epoch == 0:
            lr_start = lr_end

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, epochs_trained, False, lr_start, lr_end


def eval_r2(model, X, y):
    """Compute R-squared on a dataset."""
    model.eval()
    loader = make_loader(X, y, BATCH_SIZE, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb).numpy())
    preds = np.concatenate(preds)

    # Check for NaN
    if np.isnan(preds).any():
        return float("nan")
    return float(r2_score(y, preds))


# ============================================================================
# Sanity Checks
# ============================================================================
def run_sanity_checks(book_data, day_labels, days):
    """Run all sanity checks from the spec."""
    checks = {}
    price_idx = np.arange(0, 40, 2)
    size_idx = np.arange(1, 40, 2)

    # Channel 0: price tick-quantized after TICK_SIZE division
    ch0 = book_data[:, price_idx]
    frac = np.abs(ch0 * 2 - np.round(ch0 * 2))
    checks["price_tick_quantized_fraction"] = float((frac < 0.01).mean())
    checks["price_range"] = [float(ch0.min()), float(ch0.max())]

    # CNN param count
    m = R3CNN(ch=59)
    pc = sum(p.numel() for p in m.parameters())
    checks["param_count"] = pc
    checks["param_count_ok"] = abs(pc - 12128) / 12128 < 0.05
    del m

    # NaN check
    checks["has_nan"] = bool(np.isnan(book_data).any())

    # Channel 1: per-day z-score verification
    per_day_means = []
    per_day_stds = []
    for d in days:
        dm = day_labels == d
        ds = book_data[dm][:, size_idx]
        per_day_means.append(float(ds.mean()))
        per_day_stds.append(float(ds.std()))
    checks["size_per_day_mean_range"] = [min(per_day_means), max(per_day_means)]
    checks["size_per_day_std_range"] = [min(per_day_stds), max(per_day_stds)]

    return checks


# ============================================================================
# Bar Export
# ============================================================================
def export_bars(threshold, output_csv):
    """Export tick bars using bar_feature_export binary."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        BAR_EXPORT_BIN,
        "--bar-type", "tick",
        "--bar-param", str(threshold),
        "--output", str(output_csv),
    ]
    print(f"  Exporting tick_{threshold}: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  EXPORT TIMEOUT (600s) for tick_{threshold}")
        return False

    export_time = time.time() - t0
    if result.returncode != 0:
        print(f"  EXPORT FAILED (rc={result.returncode}): {result.stderr[:500]}")
        return False

    print(f"  Export OK ({export_time:.1f}s): {output_csv}")
    return True


# ============================================================================
# Main Experiment
# ============================================================================
def run_experiment():
    wall_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "calibration").mkdir(exist_ok=True)

    print("=" * 70)
    print("R3b GENUINE TICK BARS -- FULL PROTOCOL")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    abort_triggered = False
    abort_reason = None
    all_sanity = {}
    mve_train_r2 = None
    mve_label = None

    # ==================================================================
    # PHASE 0: EXPORT + CALIBRATION
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 0: EXPORT + CALIBRATION")
    print("=" * 70)

    calibration = {}

    for thresh in THRESHOLDS:
        label = f"tick_{thresh}"
        csv_path = RESULTS_DIR / label / f"{label}.csv"

        # Export if not already done
        if not csv_path.exists():
            ok = export_bars(thresh, csv_path)
            if not ok:
                calibration[label] = {"export_failed": True}
                continue
        else:
            print(f"  {label}: CSV exists, skipping export")

        # Load and compute calibration statistics
        try:
            book_data, target, day_labels, days, timestamps, nan_cnt, tgt_nan = load_csv(csv_path)
            stats = compute_bar_stats(day_labels, days, timestamps)
            stats["nan_count_book"] = nan_cnt
            stats["nan_count_target"] = tgt_nan
            calibration[label] = stats

            # Save per-threshold stats
            thresh_dir = RESULTS_DIR / label
            with open(thresh_dir / "bar_statistics.json", "w") as f:
                json.dump(stats, f, indent=2, cls=NumpyEncoder)

            print(f"    {label}: {stats['total_bars']} bars, "
                  f"{stats['bars_per_day_mean']:.1f}/day "
                  f"(cv={stats['bars_per_day_cv']:.4f}), "
                  f"median_dur={stats['duration_median_s']:.1f}s, "
                  f"p10={stats['duration_p10_s']:.1f}s, "
                  f"p90={stats['duration_p90_s']:.1f}s")

            del book_data, target, day_labels, days, timestamps
        except Exception as e:
            print(f"  ERROR loading {label}: {e}")
            calibration[label] = {"load_error": str(e)}

    # Save full calibration
    with open(RESULTS_DIR / "calibration" / "threshold_sweep.json", "w") as f:
        json.dump(calibration, f, indent=2, cls=NumpyEncoder)

    # --- Gate A: bars_per_day_cv > 0 at ALL thresholds ---
    print(f"\n--- Gate A: bars_per_day_cv > 0 ---")
    gate_a_fail = False
    for label in [f"tick_{t}" for t in THRESHOLDS]:
        if label not in calibration:
            continue
        stats = calibration[label]
        if "export_failed" in stats or "load_error" in stats:
            continue
        cv = stats.get("bars_per_day_cv", 0)
        p10 = stats.get("duration_p10_s", 0)
        p90 = stats.get("duration_p90_s", 0)
        if cv == 0:
            print(f"  FATAL: {label} bars_per_day_cv = 0 (not genuine event bars)")
            gate_a_fail = True
        if p10 == p90 and p10 > 0:
            print(f"  WARNING: {label} p10 == p90 = {p10:.1f}s (bar durations identical)")

    if gate_a_fail:
        abort_triggered = True
        abort_reason = "bars_per_day_cv = 0 at one or more thresholds -- tick-bar-fix did not work"
        print(f"\n  ABORT: {abort_reason}")
        write_final_outputs(calibration, {}, all_sanity, abort_triggered, abort_reason,
                            wall_start, None, None, None, None, None, None, None, 0, None)
        return

    print("  Gate A PASSED: all thresholds have cv > 0")

    # --- Gate B: >= 3 viable thresholds ---
    print(f"\n--- Gate B: >= 3 viable thresholds (bars_per_day >= {VIABLE_MIN_BARS_PER_DAY}) ---")
    viable = {}
    non_viable = {}
    for label in [f"tick_{t}" for t in THRESHOLDS]:
        if label not in calibration:
            continue
        stats = calibration[label]
        if "export_failed" in stats or "load_error" in stats:
            continue
        if stats["bars_per_day_mean"] >= VIABLE_MIN_BARS_PER_DAY:
            viable[label] = stats
            print(f"  VIABLE: {label} ({stats['bars_per_day_mean']:.0f}/day)")
        else:
            non_viable[label] = stats
            print(f"  NON-VIABLE: {label} ({stats['bars_per_day_mean']:.0f}/day < {VIABLE_MIN_BARS_PER_DAY})")

    if len(viable) < 3:
        abort_triggered = True
        abort_reason = f"Only {len(viable)} viable thresholds (need >= 3)"
        print(f"\n  ABORT: {abort_reason}")
        write_final_outputs(calibration, {}, all_sanity, abort_triggered, abort_reason,
                            wall_start, None, None, None, None, None, None, None, 0, None)
        return

    print(f"  Gate B PASSED: {len(viable)} viable thresholds")

    # --- Threshold Selection (up to 4, spanning viable range in log-space) ---
    viable_labels = sorted(viable.keys(), key=lambda l: int(l.split("_")[1]))
    if len(viable_labels) <= MAX_SELECTED_THRESHOLDS:
        selected = viable_labels
    else:
        # Pick evenly spaced in log-space
        indices = np.linspace(0, len(viable_labels) - 1, MAX_SELECTED_THRESHOLDS).astype(int)
        selected = list(dict.fromkeys(viable_labels[i] for i in indices))  # deduplicate preserving order

    print(f"\n  Selected thresholds for CNN training: {selected}")

    # Data adequacy warnings
    for label in selected:
        bpd = calibration[label]["bars_per_day_mean"]
        fold1_est = bpd * 3  # ~3 effective train days in fold 1 after 80/20 split
        if fold1_est < 2000:
            print(f"  WARNING: {label} fold 1 estimated train bars = {fold1_est:.0f} < 2000 (data-starved)")

    # ==================================================================
    # PHASE 1: MVE (fold 5 on highest-data threshold)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: MVE (fold 5 on highest-data threshold)")
    print("=" * 70)

    mve_label = max(selected, key=lambda l: calibration[l]["bars_per_day_mean"])
    mve_csv = RESULTS_DIR / mve_label / f"{mve_label}.csv"
    print(f"  MVE threshold: {mve_label} ({calibration[mve_label]['bars_per_day_mean']:.0f} bars/day)")

    book_data, target, day_labels, days, timestamps, _, _ = load_csv(mve_csv)

    if len(days) != 19:
        abort_triggered = True
        abort_reason = f"Expected 19 days, got {len(days)} for {mve_label}"
        print(f"  ABORT: {abort_reason}")
        write_final_outputs(calibration, {}, all_sanity, abort_triggered, abort_reason,
                            wall_start, selected, None, None, None, None, None, None, 0, None)
        del book_data, target, day_labels, days, timestamps
        return

    # Sanity checks on MVE data
    mve_sanity = run_sanity_checks(book_data, day_labels, days)
    all_sanity[mve_label] = mve_sanity
    print(f"  Sanity: param_count={mve_sanity['param_count']}, "
          f"tick_quant={mve_sanity['price_tick_quantized_fraction']:.4f}, "
          f"has_nan={mve_sanity['has_nan']}")

    if not mve_sanity["param_count_ok"]:
        abort_triggered = True
        abort_reason = f"CNN param count {mve_sanity['param_count']} deviates >5% from 12128"
        print(f"  ABORT: {abort_reason}")
        write_final_outputs(calibration, {}, all_sanity, abort_triggered, abort_reason,
                            wall_start, selected, None, None, None, None, None, None, 0, None)
        del book_data, target, day_labels, days, timestamps
        return

    if mve_sanity["has_nan"]:
        abort_triggered = True
        abort_reason = "NaN in normalized book data"
        print(f"  ABORT: {abort_reason}")
        write_final_outputs(calibration, {}, all_sanity, abort_triggered, abort_reason,
                            wall_start, selected, None, None, None, None, None, None, 0, None)
        del book_data, target, day_labels, days, timestamps
        return

    # Train fold 5 only
    splits = get_cv_folds(day_labels, days)
    fold_idx = 4  # 0-based = fold 5
    tr_mask, te_mask = splits[fold_idx]
    tr_X = book_data[tr_mask].reshape(-1, 20, 2)
    tr_y = target[tr_mask]
    te_X = book_data[te_mask].reshape(-1, 20, 2)
    te_y = target[te_mask]

    print(f"  Fold 5: {len(tr_X)} train, {len(te_X)} test")

    seed = SEED_BASE + fold_idx
    mve_start = time.time()
    model, epochs, nan_abort, lr_s, lr_e = train_cnn(tr_X, tr_y, seed)
    mve_time = time.time() - mve_start

    if nan_abort:
        abort_triggered = True
        abort_reason = "NaN during MVE training"
        print(f"  ABORT: {abort_reason}")
        write_final_outputs(calibration, {}, all_sanity, abort_triggered, abort_reason,
                            wall_start, selected, None, None, None, None, None, None, 0, None)
        del book_data, target, day_labels, days, timestamps, model
        return

    mve_train_r2 = eval_r2(model, tr_X, tr_y)
    mve_test_r2 = eval_r2(model, te_X, te_y)
    print(f"  MVE fold 5: train_r2={mve_train_r2:.6f}, test_r2={mve_test_r2:.6f}, "
          f"epochs={epochs}, time={mve_time:.1f}s, LR={lr_s:.6f}->{lr_e:.6f}")

    # Verify LR decay
    if lr_e > lr_s * 0.5:
        print(f"  WARNING: LR did not decay significantly ({lr_s:.6f} -> {lr_e:.6f})")

    # Gate C/D: train R2 >= 0.05
    if mve_train_r2 < 0.05:
        abort_triggered = True
        abort_reason = f"MVE train R2 = {mve_train_r2:.6f} < 0.05 (normalization or export issue)"
        print(f"  ABORT: {abort_reason}")
        write_final_outputs(calibration, {}, all_sanity, abort_triggered, abort_reason,
                            wall_start, selected, None, None, None, None, None, None, 0, mve_train_r2)
        del book_data, target, day_labels, days, timestamps, model
        return

    print(f"  Gate C/D PASSED: train R2 = {mve_train_r2:.6f} >= 0.05")

    del book_data, target, day_labels, days, timestamps, model, tr_X, tr_y, te_X, te_y

    # ==================================================================
    # PHASE 2: FULL SWEEP (all selected thresholds x 5 folds)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FULL SWEEP")
    print("=" * 70)

    all_results = {}
    total_training_runs = 0
    wall_exceeded = False

    for label in selected:
        csv_path = RESULTS_DIR / label / f"{label}.csv"
        thresh_dir = RESULTS_DIR / label
        thresh_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- {label} ---")
        book_data, target, day_labels, days, timestamps, nan_cnt, tgt_nan = load_csv(csv_path)

        if len(days) != 19:
            print(f"  SKIP: {len(days)} days != 19")
            continue

        # Sanity checks
        sanity_thresh = run_sanity_checks(book_data, day_labels, days)
        all_sanity[label] = sanity_thresh

        # Write normalization verification
        verify_lines = [
            f"Normalization Verification -- {label}",
            "=" * 50,
            f"Price offsets (/ TICK_SIZE={TICK_SIZE}):",
            f"  Range: [{sanity_thresh['price_range'][0]:.1f}, {sanity_thresh['price_range'][1]:.1f}]",
            f"  Tick-quantized fraction: {sanity_thresh['price_tick_quantized_fraction']:.4f}",
            "",
            f"Size (log1p + per-day z-score):",
            f"  Per-day mean range: [{sanity_thresh['size_per_day_mean_range'][0]:.4f}, {sanity_thresh['size_per_day_mean_range'][1]:.4f}]",
            f"  Per-day std range: [{sanity_thresh['size_per_day_std_range'][0]:.4f}, {sanity_thresh['size_per_day_std_range'][1]:.4f}]",
            "",
            f"NaN/inf: book={nan_cnt}, target={tgt_nan}",
            f"Has NaN after cleanup: {sanity_thresh['has_nan']}",
            f"Param count: {sanity_thresh['param_count']} (ok: {sanity_thresh['param_count_ok']})",
        ]
        with open(thresh_dir / "normalization_verification.txt", "w") as f:
            f.write("\n".join(verify_lines))

        splits = get_cv_folds(day_labels, days)
        fold_results = []

        for fold_idx in range(5):
            fold_start = time.time()
            tr_mask, te_mask = splits[fold_idx]
            tr_X = book_data[tr_mask].reshape(-1, 20, 2)
            te_X = book_data[te_mask].reshape(-1, 20, 2)
            tr_y = target[tr_mask]
            te_y = target[te_mask]

            seed = SEED_BASE + fold_idx
            model, epochs, nan_abort_fold, lr_s, lr_e = train_cnn(tr_X, tr_y, seed)
            fold_time = time.time() - fold_start
            total_training_runs += 1

            if nan_abort_fold:
                print(f"  Fold {fold_idx+1}: NaN abort (epochs={epochs})")
                fold_results.append({
                    "fold": fold_idx + 1,
                    "train_r2": None,
                    "test_r2": None,
                    "epochs_trained": epochs,
                    "nan_abort": True,
                    "time_s": float(fold_time),
                    "train_size": int(len(tr_X)),
                    "test_size": int(len(te_X)),
                })
                del model
                continue

            train_r2 = eval_r2(model, tr_X, tr_y)
            test_r2 = eval_r2(model, te_X, te_y)

            print(f"  Fold {fold_idx+1}: train_r2={train_r2:.6f}, test_r2={test_r2:.6f}, "
                  f"ep={epochs}, {fold_time:.1f}s, train_n={len(tr_X)}, test_n={len(te_X)}")

            fold_results.append({
                "fold": fold_idx + 1,
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "epochs_trained": int(epochs),
                "lr_start": float(lr_s),
                "lr_end": float(lr_e),
                "nan_abort": False,
                "time_s": float(fold_time),
                "train_size": int(len(tr_X)),
                "test_size": int(len(te_X)),
            })

            del model

            # Per-run time check (5 min = 300s)
            if fold_time > 300:
                print(f"  WARNING: fold {fold_idx+1} took {fold_time:.1f}s > 300s")

            # Wall clock check
            if time.time() - wall_start > WALL_CLOCK_BUDGET:
                print(f"  WALL CLOCK EXCEEDED {WALL_CLOCK_BUDGET}s -- stopping")
                wall_exceeded = True
                break

        # Save fold results
        with open(thresh_dir / "fold_results.json", "w") as f:
            json.dump(fold_results, f, indent=2, cls=NumpyEncoder)

        # Compute threshold summary
        valid_test = [r["test_r2"] for r in fold_results if r["test_r2"] is not None]
        valid_train = [r["train_r2"] for r in fold_results if r["train_r2"] is not None]

        summary = {
            "threshold": label,
            "threshold_value": int(label.split("_")[1]),
            "n_folds_valid": len(valid_test),
            "mean_test_r2": float(np.mean(valid_test)) if valid_test else None,
            "std_test_r2": float(np.std(valid_test)) if valid_test else None,
            "mean_train_r2": float(np.mean(valid_train)) if valid_train else None,
            "per_fold_test_r2": valid_test,
            "per_fold_train_r2": valid_train,
            "fold3_test_r2": next((r["test_r2"] for r in fold_results if r["fold"] == 3 and r["test_r2"] is not None), None),
            "fold5_test_r2": next((r["test_r2"] for r in fold_results if r["fold"] == 5 and r["test_r2"] is not None), None),
            "bar_stats": calibration.get(label, {}),
            "sanity_checks": sanity_thresh,
        }
        all_results[label] = summary

        if valid_test:
            print(f"\n  SUMMARY {label}: mean_test_r2={summary['mean_test_r2']:.6f} "
                  f"(std={summary['std_test_r2']:.6f}), fold3={summary['fold3_test_r2']}")

        del book_data, target, day_labels, days, timestamps

        if wall_exceeded:
            abort_reason = "Wall clock budget exceeded during sweep"
            break

    # Check all-below-0.02 abort criterion
    all_mean_r2 = [all_results[l]["mean_test_r2"] for l in all_results
                   if all_results[l]["mean_test_r2"] is not None]
    if all_mean_r2 and all(r2 < 0.02 for r2 in all_mean_r2):
        print(f"\n  ALL thresholds mean test R2 < 0.02 (max={max(all_mean_r2):.6f})")
        abort_reason = "All thresholds produce mean test R2 < 0.02"
        abort_triggered = True

    # ==================================================================
    # PHASE 3: ANALYSIS
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: ANALYSIS")
    print("=" * 70)

    # --- sweep_summary.json ---
    with open(RESULTS_DIR / "sweep_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)

    # --- Peak threshold ---
    peak_label = None
    peak_r2 = -999.0
    for label in all_results:
        r2 = all_results[label].get("mean_test_r2")
        if r2 is not None and r2 > peak_r2:
            peak_r2 = r2
            peak_label = label
    peak_delta = peak_r2 - BASELINE_R2 if peak_r2 > -999 else None

    print(f"  Peak: {peak_label} (R2={peak_r2:.6f}, delta={peak_delta:+.6f})" if peak_label else "  No valid results")

    # --- r2_vs_barsize_curve.csv ---
    curve_rows = []
    for label in selected:
        if label not in all_results or all_results[label]["mean_test_r2"] is None:
            continue
        res = all_results[label]
        cal = calibration.get(label, {})
        curve_rows.append({
            "threshold": label,
            "threshold_value": res["threshold_value"],
            "mean_duration_sec": cal.get("duration_median_s", 0),
            "mean_test_r2": res["mean_test_r2"],
            "std_test_r2": res["std_test_r2"],
            "fold3_test_r2": res["fold3_test_r2"],
            "mean_train_r2": res["mean_train_r2"],
            "bars_per_day_mean": cal.get("bars_per_day_mean", 0),
            "bars_per_day_cv": cal.get("bars_per_day_cv", 0),
        })

    if curve_rows:
        with open(RESULTS_DIR / "r2_vs_barsize_curve.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=curve_rows[0].keys())
            writer.writeheader()
            writer.writerows(curve_rows)
        print(f"  R2 vs bar-size curve: {len(curve_rows)} data points")

    # --- Curve shape ---
    r2_sequence = [all_results[l]["mean_test_r2"] for l in selected
                   if l in all_results and all_results[l]["mean_test_r2"] is not None]
    if len(r2_sequence) >= 3:
        diffs = np.diff(r2_sequence)
        if np.all(diffs > 0):
            curve_shape = "Monotonic up"
        elif np.all(diffs < 0):
            curve_shape = "Monotonic down"
        elif len(diffs) >= 2 and diffs[0] > 0 and diffs[-1] < 0:
            curve_shape = "Inverted-U"
        else:
            curve_shape = "Non-monotonic"
    elif len(r2_sequence) == 2:
        curve_shape = "Only 2 points"
    else:
        curve_shape = "Insufficient data"

    print(f"  Curve shape: {curve_shape}")

    # --- Data volume confound (correlation bars_per_day vs mean R2) ---
    if len(curve_rows) >= 3:
        bpd = np.array([r["bars_per_day_mean"] for r in curve_rows])
        r2s = np.array([r["mean_test_r2"] for r in curve_rows])
        if bpd.std() > 0 and r2s.std() > 0:
            corr = float(np.corrcoef(bpd, r2s)[0, 1])
        else:
            corr = 0.0
    else:
        corr = None

    print(f"  Data volume correlation: {corr:.3f}" if corr is not None else "  Data volume correlation: N/A")

    # --- Decision framework ---
    if peak_r2 >= 0.107:
        # Check if broad pattern
        better_count = sum(1 for r in curve_rows if r["mean_test_r2"] >= 0.107)
        if better_count >= 2:
            decision = "BETTER"
            decision_detail = f"Peak R2={peak_r2:.4f} >= 0.107 at {peak_label}. Broad pattern: {better_count} thresholds in BETTER range."
        else:
            decision = "BETTER"
            decision_detail = (f"Peak R2={peak_r2:.4f} >= 0.107 at {peak_label}. "
                               f"Single-threshold peak -- multiple-testing concern noted.")
    elif all_mean_r2 and all(r2 < 0.071 for r2 in all_mean_r2):
        decision = "WORSE"
        decision_detail = f"All viable R2 < 0.071 (peak={peak_r2:.4f}). Time_5s definitively superior."
    elif corr is not None and abs(corr) > 0.8:
        decision = "INCONCLUSIVE"
        decision_detail = f"R2 correlates strongly with bars_per_day (r={corr:.3f}). Cannot separate bar-type from data-volume effect."
    elif peak_r2 >= 0.071:
        decision = "COMPARABLE"
        decision_detail = f"Peak R2={peak_r2:.4f} in [0.071, 0.107]. Time_5s is as good or better."
    else:
        decision = "WORSE"
        decision_detail = f"Peak R2={peak_r2:.4f} < 0.071."

    print(f"  Decision: {decision} -- {decision_detail}")

    # --- Fold 3 diagnostic ---
    fold3_improvements = {}
    for label in all_results:
        f3 = all_results[label].get("fold3_test_r2")
        if f3 is not None:
            fold3_improvements[label] = float(f3 - (-0.049))

    any_fold3_positive = any(all_results[l].get("fold3_test_r2", -1) > 0
                             for l in all_results if all_results[l].get("fold3_test_r2") is not None)
    if any_fold3_positive:
        print(f"  NOTABLE: At least one threshold has positive fold 3 R2 (time_5s baseline = -0.049)")

    # --- comparison_table.md ---
    table_lines = [
        "# Comparison Table: Tick Bars vs Time_5s Baseline",
        "",
        "| Threshold | Bars/Day | CV | Duration (med) | Mean R2 | Std | Fold 3 R2 | Delta vs 0.089 | Verdict |",
        "|-----------|----------|------|----------------|---------|------|-----------|----------------|---------|",
        f"| time_5s (baseline) | 4,630 | 0.000 | 5.0s | 0.0890 | 0.074 | -0.049 | -- | BASELINE |",
    ]
    for label in selected:
        if label not in all_results or all_results[label]["mean_test_r2"] is None:
            continue
        res = all_results[label]
        cal = calibration.get(label, {})
        delta = res["mean_test_r2"] - BASELINE_R2
        f3 = res["fold3_test_r2"]
        f3_str = f"{f3:.4f}" if f3 is not None else "N/A"

        if res["mean_test_r2"] >= 0.107:
            v = "BETTER"
        elif res["mean_test_r2"] >= 0.071:
            v = "COMPARABLE"
        else:
            v = "WORSE"

        table_lines.append(
            f"| {label} | {cal.get('bars_per_day_mean', 0):,.0f} | "
            f"{cal.get('bars_per_day_cv', 0):.3f} | "
            f"{cal.get('duration_median_s', 0):.1f}s | "
            f"{res['mean_test_r2']:.4f} | {res['std_test_r2']:.4f} | "
            f"{f3_str} | {delta:+.4f} | {v} |"
        )

    with open(RESULTS_DIR / "comparison_table.md", "w") as f:
        f.write("\n".join(table_lines))

    # --- analysis.md ---
    analysis_lines = [
        "# R3b Genuine Tick Bars -- Analysis",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Spec:** .kit/experiments/r3b-genuine-tick-bars.md",
        f"**Decision:** {decision}",
        f"**Detail:** {decision_detail}",
        "",
        "## Calibration (8 thresholds)",
        "",
        "| Threshold | Total Bars | Bars/Day | CV | Median Dur | p10 | p90 | Viable |",
        "|-----------|-----------|----------|------|------------|-----|-----|--------|",
    ]
    for thresh in THRESHOLDS:
        label = f"tick_{thresh}"
        if label in calibration and "bars_per_day_mean" in calibration[label]:
            cal = calibration[label]
            v = "YES" if cal["bars_per_day_mean"] >= VIABLE_MIN_BARS_PER_DAY else "NO"
            analysis_lines.append(
                f"| {label} | {cal['total_bars']:,} | {cal['bars_per_day_mean']:.0f} | "
                f"{cal['bars_per_day_cv']:.4f} | {cal['duration_median_s']:.1f}s | "
                f"{cal['duration_p10_s']:.1f}s | {cal['duration_p90_s']:.1f}s | {v} |"
            )
        else:
            analysis_lines.append(f"| {label} | -- | -- | -- | -- | -- | -- | FAILED |")

    analysis_lines.extend([
        "",
        f"**Selected thresholds:** {', '.join(selected)}",
        f"**Non-viable:** {', '.join(non_viable.keys()) if non_viable else 'none'}",
        "",
        "## MVE Gate",
        "",
        f"- Threshold: {mve_label}",
        f"- Fold 5 train R2: {mve_train_r2:.6f}",
        f"- Gate: {'PASSED' if mve_train_r2 >= 0.05 else 'FAILED'} (threshold: 0.05)",
        "",
        "## CNN Results (5-fold expanding window)",
        "",
    ])

    for label in selected:
        if label not in all_results or all_results[label]["mean_test_r2"] is None:
            continue
        res = all_results[label]
        cal = calibration.get(label, {})
        analysis_lines.extend([
            f"### {label} ({cal.get('bars_per_day_mean', 0):.0f} bars/day)",
            f"- Mean test R2: {res['mean_test_r2']:.6f} (std: {res['std_test_r2']:.6f})",
            f"- Mean train R2: {res['mean_train_r2']:.6f}",
            f"- Per-fold test R2: {[f'{r:.4f}' for r in res['per_fold_test_r2']]}",
            f"- Per-fold train R2: {[f'{r:.4f}' for r in res['per_fold_train_r2']]}",
            f"- Fold 3 test R2: {res['fold3_test_r2']}",
            f"- Fold 5 test R2: {res['fold5_test_r2']}",
            f"- Delta vs baseline: {res['mean_test_r2'] - BASELINE_R2:+.6f}",
            "",
        ])

    analysis_lines.extend([
        "## Decision Framework",
        "",
        f"- **Peak threshold:** {peak_label}",
        f"- **Peak mean test R2:** {peak_r2:.6f}",
    ])
    if peak_delta is not None:
        analysis_lines.append(f"- **Delta vs baseline (0.089):** {peak_delta:+.6f}")
    analysis_lines.extend([
        f"- **Curve shape:** {curve_shape}",
    ])
    if corr is not None:
        analysis_lines.append(f"- **Data volume correlation (r):** {corr:.3f}")
    analysis_lines.extend([
        "",
        f"### Verdict: **{decision}**",
        "",
        f"{decision_detail}",
        "",
    ])

    # Fold 3 diagnostic
    analysis_lines.extend([
        "## Fold 3 Diagnostic",
        "",
        "Fold 3 (Oct 2022) produces R2=-0.049 on time_5s baseline.",
        "",
    ])
    for label in all_results:
        f3 = all_results[label].get("fold3_test_r2")
        if f3 is not None:
            delta_f3 = f3 - (-0.049)
            flag = " **POSITIVE**" if f3 > 0 else ""
            analysis_lines.append(
                f"- {label}: fold 3 R2={f3:.4f} (delta vs baseline: {delta_f3:+.4f}){flag}"
            )
    analysis_lines.append("")

    # Data volume confound
    analysis_lines.extend([
        "## Data Volume Confound",
        "",
    ])
    if corr is not None:
        analysis_lines.append(f"Correlation between bars_per_day and mean R2: r = {corr:.3f}")
        if abs(corr) > 0.8:
            analysis_lines.append("**Strong correlation -- cannot separate bar-type effect from data-volume effect.**")
        elif abs(corr) > 0.5:
            analysis_lines.append("Moderate correlation -- data volume may partially confound results.")
        else:
            analysis_lines.append("Weak correlation -- bar-type effect is separable from data volume.")
    else:
        analysis_lines.append("Insufficient data points for correlation analysis.")

    # Find threshold closest to 4630 bars/day (time_5s equivalent)
    closest_label = None
    closest_dist = float("inf")
    for label in all_results:
        cal = calibration.get(label, {})
        bpd = cal.get("bars_per_day_mean", 0)
        dist = abs(bpd - BASELINE_BARS_PER_DAY)
        if dist < closest_dist:
            closest_dist = dist
            closest_label = label
    if closest_label:
        closest_bpd = calibration[closest_label].get("bars_per_day_mean", 0)
        closest_r2 = all_results[closest_label].get("mean_test_r2")
        analysis_lines.extend([
            "",
            f"Threshold closest to time_5s bar rate (4,630/day): {closest_label} ({closest_bpd:.0f}/day)",
            f"R2 at this threshold: {closest_r2:.4f}" if closest_r2 else "R2: N/A",
        ])

    analysis_lines.append("")

    with open(RESULTS_DIR / "analysis.md", "w") as f:
        f.write("\n".join(analysis_lines))

    # --- Write final metrics ---
    write_final_outputs(
        calibration, all_results, all_sanity, abort_triggered, abort_reason,
        wall_start, selected, peak_label, peak_r2, peak_delta,
        curve_shape, corr, decision, decision_detail,
        fold3_improvements, total_training_runs, mve_train_r2
    )

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Decision: {decision}")
    print(f"Peak: {peak_label} R2={peak_r2:.6f} (delta={peak_delta:+.6f})")
    print(f"Wall clock: {time.time() - wall_start:.1f}s")
    print(f"Total training runs: {total_training_runs}")
    print(f"{'='*70}")


def write_final_outputs(calibration, all_results, sanity, abort_triggered, abort_reason,
                        wall_start, selected, peak_label, peak_r2, peak_delta,
                        curve_shape, corr, decision, decision_detail,
                        fold3_improvements=None, total_runs=0, mve_train_r2=None):
    """Write metrics.json and config.json."""
    wall_clock = time.time() - wall_start

    # Build treatment section
    treatment = {}
    per_seed = []
    if all_results:
        for label in all_results:
            res = all_results[label]
            treatment[label] = {
                "mean_test_r2": res.get("mean_test_r2"),
                "std_test_r2": res.get("std_test_r2"),
                "mean_train_r2": res.get("mean_train_r2"),
                "per_fold_test_r2": res.get("per_fold_test_r2"),
                "per_fold_train_r2": res.get("per_fold_train_r2"),
                "fold3_test_r2": res.get("fold3_test_r2"),
                "fold5_test_r2": res.get("fold5_test_r2"),
                "bars_per_day_mean": res.get("bar_stats", {}).get("bars_per_day_mean"),
                "bars_per_day_cv": res.get("bar_stats", {}).get("bars_per_day_cv"),
            }
            # Per-seed entries
            if res.get("per_fold_test_r2"):
                for i, r2 in enumerate(res["per_fold_test_r2"]):
                    per_seed.append({
                        "threshold": label,
                        "seed": SEED_BASE + i,
                        "fold": i + 1,
                        "test_r2": r2,
                        "train_r2": res["per_fold_train_r2"][i] if i < len(res.get("per_fold_train_r2", [])) else None,
                    })

    # Calibration summary
    cal_summary = {}
    for label in calibration:
        cal = calibration[label]
        if "export_failed" not in cal and "load_error" not in cal:
            cal_summary[label] = {
                "bars_per_day_mean": cal.get("bars_per_day_mean"),
                "bars_per_day_std": cal.get("bars_per_day_std"),
                "bars_per_day_cv": cal.get("bars_per_day_cv"),
                "total_bars": cal.get("total_bars"),
                "duration_median_s": cal.get("duration_median_s"),
                "duration_p10_s": cal.get("duration_p10_s"),
                "duration_p90_s": cal.get("duration_p90_s"),
            }

    # R2 by threshold
    r2_by_threshold = {}
    if all_results:
        r2_by_threshold = {l: all_results[l].get("mean_test_r2") for l in all_results}

    metrics = {
        "experiment": "r3b-genuine-tick-bars",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "source": "9E (hybrid-model-corrected)",
            "bar_type": "time_5s",
            "mean_test_r2": BASELINE_R2,
            "per_fold_test_r2": BASELINE_PER_FOLD,
            "bars_per_day": BASELINE_BARS_PER_DAY,
        },
        "treatment": treatment,
        "per_seed": per_seed,
        "calibration": cal_summary,
        "primary_metrics": {
            "peak_tick_r2": peak_r2 if peak_r2 and peak_r2 > -999 else None,
            "peak_tick_threshold": peak_label,
            "peak_delta": peak_delta,
        },
        "secondary_metrics": {
            "r2_by_threshold": r2_by_threshold,
            "per_fold_r2": {l: all_results[l].get("per_fold_test_r2") for l in all_results} if all_results else {},
            "per_fold_train_r2": {l: all_results[l].get("per_fold_train_r2") for l in all_results} if all_results else {},
            "fold3_r2": {l: all_results[l].get("fold3_test_r2") for l in all_results} if all_results else {},
            "r2_std_by_threshold": {l: all_results[l].get("std_test_r2") for l in all_results} if all_results else {},
            "curve_shape": curve_shape,
            "bars_per_day": {l: cal_summary[l].get("bars_per_day_mean") for l in cal_summary} if cal_summary else {},
            "bar_duration": {l: {"median": cal_summary[l].get("duration_median_s"),
                                 "p10": cal_summary[l].get("duration_p10_s"),
                                 "p90": cal_summary[l].get("duration_p90_s")}
                            for l in cal_summary} if cal_summary else {},
            "total_bar_count": {l: cal_summary[l].get("total_bars") for l in cal_summary} if cal_summary else {},
            "data_volume_correlation": corr,
            "fold3_improvements": fold3_improvements or {},
            "selected_thresholds": selected or [],
        },
        "sanity_checks": sanity if isinstance(sanity, dict) else {},
        "mve": {
            "threshold": selected[0] if selected else None,
            "fold5_train_r2": mve_train_r2,
        } if mve_train_r2 is not None else {},
        "decision": {
            "outcome": decision,
            "detail": decision_detail,
        } if decision else {},
        "resource_usage": {
            "gpu_hours": 0,
            "wall_clock_seconds": float(wall_clock),
            "total_training_runs": total_runs,
            "total_runs": total_runs + len([l for l in calibration if "export_failed" not in calibration[l]]),
        },
        "abort_triggered": abort_triggered,
        "abort_reason": abort_reason,
        "notes": "",
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    # config.json
    config = {
        "experiment": "r3b-genuine-tick-bars",
        "spec": ".kit/experiments/r3b-genuine-tick-bars.md",
        "thresholds_calibrated": THRESHOLDS,
        "thresholds_selected": selected or [],
        "seed_base": SEED_BASE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "lr_min": LR_MIN,
        "weight_decay": WEIGHT_DECAY,
        "tick_size": TICK_SIZE,
        "viable_min_bars_per_day": VIABLE_MIN_BARS_PER_DAY,
        "device": DEVICE,
        "cnn_params": 12128,
        "bar_export_binary": BAR_EXPORT_BIN,
        "data_source": "DATA/GLBX-20260207-L953CAPU5B/",
        "baseline_r2": BASELINE_R2,
        "baseline_per_fold": BASELINE_PER_FOLD,
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  metrics.json and config.json written to {RESULTS_DIR}")
    print(f"  Wall clock: {wall_clock:.1f}s ({wall_clock/60:.1f}min)")


if __name__ == "__main__":
    run_experiment()
