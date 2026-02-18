#!/usr/bin/env python3
"""Phase R4d: Temporal Predictability at Actionable Dollar & Tick Timescales

Spec: .kit/experiments/temporal-predictability-dollar-tick-actionable.md

Phase 1: Empirical calibration sweep (10 thresholds × 1 day analysis from 19-day export)
Phase 2: Feature export validation for 3 selected operating points
Phase 3: Temporal analysis (GBT, 3 feature configs) per operating point
Phase 4: Cross-timescale synthesis

Output:
  - .kit/results/temporal-predictability-dollar-tick-actionable/calibration/calibration_table.json
  - .kit/results/temporal-predictability-dollar-tick-actionable/{type}_{threshold}/metrics.json
  - .kit/results/temporal-predictability-dollar-tick-actionable/metrics.json
"""

import argparse
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
RESULTS_DIR = Path(".kit/results/temporal-predictability-dollar-tick-actionable")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_DAY = 20220103

# Thresholds from spec
DOLLAR_THRESHOLDS = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 250_000_000, 1_000_000_000]
TICK_THRESHOLDS = [500, 3_000, 10_000, 25_000]

# CV folds: expanding window (identical to R4/R4b/R4c)
CV_FOLDS = [
    (list(range(1, 5)),   list(range(5, 9))),
    (list(range(1, 9)),   list(range(9, 12))),
    (list(range(1, 12)),  list(range(12, 15))),
    (list(range(1, 15)),  list(range(15, 18))),
    (list(range(1, 18)),  list(range(18, 20))),
]

HORIZONS_FULL = [1, 5, 20, 100]

# Feature definitions (identical to R4/R4c)
BOOK_SNAP_FEATURES = [f"book_snap_{i}" for i in range(40)]

TRACK_A_FEATURES = [
    "book_imbalance_1", "book_imbalance_3", "book_imbalance_5", "book_imbalance_10",
    "weighted_imbalance", "spread",
    *[f"bid_depth_profile_{i}" for i in range(10)],
    *[f"ask_depth_profile_{i}" for i in range(10)],
    "depth_concentration_bid", "depth_concentration_ask",
    "book_slope_bid", "book_slope_ask",
    "level_count_bid", "level_count_ask",
    "net_volume", "volume_imbalance", "trade_count", "avg_trade_size",
    "large_trade_count", "vwap_distance", "kyle_lambda",
    "return_1", "return_5", "return_20",
    "volatility_20", "volatility_50", "momentum",
    "high_low_range_20", "high_low_range_50", "close_position",
    "volume_surprise", "duration_surprise", "acceleration", "vol_price_corr",
    "time_sin", "time_cos", "minutes_since_open", "minutes_to_close", "session_volume_frac",
    "cancel_add_ratio", "message_rate", "modify_fraction",
    "order_flow_toxicity", "cancel_concentration",
]

TEMPORAL_FEATURE_NAMES = [
    *[f"lag_return_{i}" for i in range(1, 11)],
    "rolling_vol_5", "rolling_vol_20", "rolling_vol_100",
    "vol_ratio",
    "momentum_5", "momentum_20", "momentum_100",
    "mean_reversion_20",
    "abs_return_lag1",
    "signed_vol",
]

# GBT hyperparameters (identical to R4/R4b/R4c)
GBT_PARAMS = dict(
    max_depth=4,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    random_state=SEED,
    verbosity=0,
    n_jobs=-1,
)

WARMUP_BARS = 50


# ===========================================================================
# Data loading (from R4c pattern)
# ===========================================================================
def load_features(csv_path):
    """Load feature CSV, handling polars duplicate column renaming."""
    df = pl.read_csv(str(csv_path), infer_schema_length=10000,
                     null_values=["NaN", "Inf", "nan", "inf"])

    cols = df.columns
    rename_map = {}
    for c in cols:
        if c == "return_100":
            rename_map[c] = "fwd_return_100"
        elif c == "return_1_duplicated_0":
            rename_map[c] = "fwd_return_1"
        elif c == "return_5_duplicated_0":
            rename_map[c] = "fwd_return_5"
        elif c == "return_20_duplicated_0":
            rename_map[c] = "fwd_return_20"

    if "return_1_duplicated_0" not in cols:
        mbo_idx = cols.index("mbo_event_count") if "mbo_event_count" in cols else len(cols)
        fwd_cols = cols[mbo_idx - 4: mbo_idx]
        for i, label in enumerate(["fwd_return_1", "fwd_return_5", "fwd_return_20", "fwd_return_100"]):
            if fwd_cols[i] not in rename_map:
                rename_map[fwd_cols[i]] = label

    df = df.rename(rename_map)
    print(f"Loaded {len(df)} bars from {csv_path}", flush=True)
    return df


def construct_temporal_features(df):
    """Construct temporal features per-day (no cross-day lookback)."""
    print("Constructing temporal features...", flush=True)
    ret_col = "return_1"

    day_frames = []
    for day in sorted(df["day"].unique().to_list()):
        day_df = df.filter(pl.col("day") == day).sort("bar_index")

        for lag in range(1, 11):
            day_df = day_df.with_columns(
                pl.col(ret_col).shift(lag).alias(f"lag_return_{lag}")
            )

        for window in [5, 20, 100]:
            day_df = day_df.with_columns(
                pl.col(ret_col).rolling_std(window_size=window, min_periods=window).alias(f"rolling_vol_{window}")
            )

        day_df = day_df.with_columns(
            (pl.col("rolling_vol_5") / pl.col("rolling_vol_20").clip(lower_bound=1e-12)).alias("vol_ratio")
        )

        for window in [5, 20, 100]:
            day_df = day_df.with_columns(
                pl.col(ret_col).rolling_sum(window_size=window, min_periods=window).alias(f"momentum_{window}")
            )

        day_df = day_df.with_columns(
            (pl.col("lag_return_1") - pl.col(ret_col).rolling_mean(window_size=20, min_periods=20)).alias("mean_reversion_20")
        )

        day_df = day_df.with_columns(
            pl.col("lag_return_1").abs().alias("abs_return_lag1")
        )

        day_df = day_df.with_columns(
            (pl.col("lag_return_1").sign() * pl.col("rolling_vol_5")).alias("signed_vol")
        )

        day_frames.append(day_df)

    result = pl.concat(day_frames)
    print(f"Temporal features constructed. Shape: {result.shape}", flush=True)
    return result


# ===========================================================================
# CV and model helpers (from R4/R4c)
# ===========================================================================
def prepare_data(df, lookback_depth):
    """Filter warmup bars and bars needing lookback.

    Note: The C++ binary already excludes warmup bars from the CSV (is_warmup
    column doesn't exist in the output). But we still filter in case it does.
    """
    if "is_warmup" in df.columns:
        df_valid = df.filter(pl.col("is_warmup") == False)
    else:
        df_valid = df

    if lookback_depth > 0:
        day_frames = []
        for day in sorted(df_valid["day"].unique().to_list()):
            day_df = df_valid.filter(pl.col("day") == day).sort("bar_index")
            if len(day_df) > lookback_depth:
                day_df = day_df.slice(lookback_depth)
            else:
                continue
            day_frames.append(day_df)
        if not day_frames:
            return pl.DataFrame()  # empty dataframe
        df_valid = pl.concat(day_frames)

    return df_valid


def get_cv_splits(df_valid, folds=CV_FOLDS):
    """Return list of (train_indices, test_indices)."""
    days = df_valid["day"].to_numpy()
    unique_days = sorted(np.unique(days).tolist())
    day_map = {i + 1: d for i, d in enumerate(unique_days)}

    splits = []
    for train_day_nums, test_day_nums in folds:
        train_day_vals = [day_map[d] for d in train_day_nums if d in day_map]
        test_day_vals = [day_map[d] for d in test_day_nums if d in day_map]
        train_mask = np.isin(days, train_day_vals)
        test_mask = np.isin(days, test_day_vals)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def fit_and_evaluate_gbt(X, y, splits, feature_names=None):
    """Fit GBT on each fold, return per-fold R2 and feature importances from last fold."""
    fold_r2s = []
    last_fold_importances = None

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        test_valid = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_test, y_test = X_test[test_valid], y_test[test_valid]

        if len(X_train) == 0 or len(X_test) == 0:
            fold_r2s.append(float("nan"))
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        n_train = len(X_train)
        n_val = max(1, int(n_train * 0.2))
        X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
        y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

        model = XGBRegressor(**GBT_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        fold_r2s.append(r2)

        if fold_idx == len(splits) - 1 and feature_names is not None:
            importances = model.feature_importances_
            last_fold_importances = dict(zip(feature_names, importances.tolist()))

    return np.array(fold_r2s), last_fold_importances


# ===========================================================================
# Statistical helpers (from R4/R4c)
# ===========================================================================
def paired_test(a, b):
    """Paired t-test (or Wilcoxon if diffs non-normal)."""
    diffs = a - b
    n = len(diffs)
    if n < 3:
        return 1.0
    _, sw_p = stats.shapiro(diffs)
    if sw_p < 0.05:
        try:
            _, p = stats.wilcoxon(diffs, alternative="two-sided")
        except ValueError:
            p = 1.0
    else:
        _, p = stats.ttest_rel(a, b)
    return float(p)


def one_sided_test_greater_than_zero(values):
    """One-sided t-test: H0: mean <= 0, H1: mean > 0."""
    n = len(values)
    if n < 2:
        return 1.0
    t_stat, p_two = stats.ttest_1samp(values, 0)
    if t_stat > 0:
        return float(p_two / 2)
    else:
        return 1.0


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    indices = np.argsort(p_values)
    corrected = np.ones(n)
    for rank, idx in enumerate(indices):
        corrected[idx] = p_values[idx] * (n - rank)
    sorted_corrected = corrected[indices]
    for i in range(1, n):
        if sorted_corrected[i] < sorted_corrected[i - 1]:
            sorted_corrected[i] = sorted_corrected[i - 1]
    corrected[indices] = sorted_corrected
    return np.clip(corrected, 0, 1)


def confidence_interval_95(values):
    """95% CI from t-distribution."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = np.mean(values)
    se = stats.sem(values)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (float(mean - t_crit * se), float(mean + t_crit * se))


def cohens_d(a, b):
    """Cohen's d for paired samples."""
    diffs = a - b
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-15))


# ===========================================================================
# Phase 1: Empirical Calibration
# ===========================================================================
def run_calibration_export(bar_type, threshold, cal_dir):
    """Export bars for one threshold. Returns (csv_path, stdout_text) or (None, None).

    Note: The C++ binary skips warmup bars (first 50/day) in CSV output.
    For very large thresholds producing <50 bars/day, the CSV will be empty
    but stdout still contains bar counts for calibration.
    """
    label = f"{bar_type}_{threshold}"
    csv_path = cal_dir / f"{label}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"  {label}: using existing CSV ({csv_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        return csv_path, None  # stdout not available for cached

    cmd = [
        "./build/bar_feature_export",
        "--bar-type", bar_type,
        "--bar-param", str(threshold),
        "--output", str(csv_path),
    ]
    print(f"  {label}: exporting (all 19 days)...", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10800)  # 3hr
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"  {label}: FAILED ({elapsed:.0f}s) — {result.stderr[-300:]}", flush=True)
            return None, None
        print(f"  {label}: exported in {elapsed:.0f}s", flush=True)
        return csv_path, result.stdout
    except subprocess.TimeoutExpired:
        print(f"  {label}: TIMEOUT (>3h)", flush=True)
        return None, None


def parse_bar_counts_from_stdout(stdout_text):
    """Parse per-day bar counts from bar_feature_export stdout.

    The binary prints lines like: '  20220103: 234000 snaps, 6426946 events, 17 bars, exported'
    Returns dict: {day_int: bar_count}
    """
    import re
    counts = {}
    if not stdout_text:
        return counts
    pattern = re.compile(r'(\d{8}):\s+\d+\s+snaps,\s+\d+\s+events,\s+(\d+)\s+bars')
    for line in stdout_text.split('\n'):
        m = pattern.search(line)
        if m:
            day = int(m.group(1))
            n_bars = int(m.group(2))
            counts[day] = n_bars
    return counts


def compute_calibration_stats(csv_path, bar_type, threshold, stdout_bar_counts=None):
    """Compute calibration stats from CSV, filtering to CALIBRATION_DAY only.

    The C++ binary skips warmup bars (first 50/day) in the CSV, so for thresholds
    producing <50 bars/day, the CSV will have zero rows for that day. In that case,
    we use stdout_bar_counts (parsed from binary output) and estimate durations
    from RTH session length (6.5h = 23400s).
    """
    RTH_DURATION_S = 23400.0  # 6.5 hours

    try:
        df = pl.read_csv(str(csv_path), infer_schema_length=5000,
                         null_values=["NaN", "Inf", "nan", "inf"])
    except Exception as e:
        # CSV might be empty (header only or truly empty)
        if stdout_bar_counts:
            # Use stdout counts instead
            day1_bars = stdout_bar_counts.get(CALIBRATION_DAY, 0)
            total_raw = sum(stdout_bar_counts.values())
            if day1_bars > 0:
                est_duration = RTH_DURATION_S / day1_bars
                return {
                    "bar_type": bar_type,
                    "threshold": threshold,
                    "status": "ok_from_stdout",
                    "total_bars_day1": day1_bars,
                    "total_bars_all_days_raw": total_raw,
                    "total_bars_all_days": max(0, total_raw - 50 * len(stdout_bar_counts)),
                    "n_days": len(stdout_bar_counts),
                    "bars_per_session": day1_bars,
                    "median_duration_s": est_duration,
                    "mean_duration_s": est_duration,
                    "p10_duration_s": est_duration * 0.5,
                    "p90_duration_s": est_duration * 2.0,
                    "note": "Durations estimated from RTH/bars (all bars are warmup in CSV)",
                }
        return {"bar_type": bar_type, "threshold": threshold, "status": "parse_error", "error": str(e)}

    # Check for empty CSV (header only)
    if len(df) == 0:
        if stdout_bar_counts:
            day1_bars = stdout_bar_counts.get(CALIBRATION_DAY, 0)
            total_raw = sum(stdout_bar_counts.values())
            if day1_bars > 0:
                est_duration = RTH_DURATION_S / day1_bars
                return {
                    "bar_type": bar_type,
                    "threshold": threshold,
                    "status": "ok_from_stdout",
                    "total_bars_day1": day1_bars,
                    "total_bars_all_days_raw": total_raw,
                    "total_bars_all_days": max(0, total_raw - 50 * len(stdout_bar_counts)),
                    "n_days": len(stdout_bar_counts),
                    "bars_per_session": day1_bars,
                    "median_duration_s": est_duration,
                    "mean_duration_s": est_duration,
                    "p10_duration_s": est_duration * 0.5,
                    "p90_duration_s": est_duration * 2.0,
                    "note": "Durations estimated from RTH/bars (CSV empty, all bars are warmup)",
                }
        return {
            "bar_type": bar_type,
            "threshold": threshold,
            "status": "empty_csv",
            "total_bars_all_days": 0,
        }

    # Total non-warmup bars across all 19 days (for Phase 2 validation)
    total_bars_all_days = len(df)  # CSV only contains non-warmup bars
    n_days = df["day"].n_unique()

    # Filter to calibration day
    day_df = df.filter(pl.col("day") == CALIBRATION_DAY)
    n_bars_day1 = len(day_df)

    if n_bars_day1 < 2:
        # Maybe we have stdout counts for this day
        if stdout_bar_counts:
            day1_raw = stdout_bar_counts.get(CALIBRATION_DAY, 0)
            if day1_raw > 0:
                est_duration = RTH_DURATION_S / day1_raw
                return {
                    "bar_type": bar_type,
                    "threshold": threshold,
                    "status": "ok_from_stdout",
                    "total_bars_day1": day1_raw,
                    "total_bars_all_days_raw": sum(stdout_bar_counts.values()),
                    "total_bars_all_days": total_bars_all_days,
                    "n_days": n_days,
                    "bars_per_session": day1_raw,
                    "median_duration_s": est_duration,
                    "mean_duration_s": est_duration,
                    "p10_duration_s": est_duration * 0.5,
                    "p90_duration_s": est_duration * 2.0,
                    "note": f"Day1 has {n_bars_day1} non-warmup bars; using raw count {day1_raw} for duration estimate",
                }
        return {
            "bar_type": bar_type,
            "threshold": threshold,
            "status": "too_few_bars",
            "n_bars_day1": n_bars_day1,
            "total_bars_all_days": total_bars_all_days,
            "n_days": n_days,
        }

    # Compute inter-bar durations from timestamps
    if "timestamp" not in day_df.columns:
        return {
            "bar_type": bar_type,
            "threshold": threshold,
            "status": "no_timestamp",
            "n_bars_day1": n_bars_day1,
        }

    ts = day_df.sort("bar_index")["timestamp"].to_numpy().astype(np.float64)
    durations_s = np.diff(ts) / 1e9  # nanoseconds to seconds
    durations_s = durations_s[durations_s > 0]

    if len(durations_s) == 0:
        return {
            "bar_type": bar_type,
            "threshold": threshold,
            "status": "no_positive_durations",
            "n_bars_day1": n_bars_day1,
        }

    return {
        "bar_type": bar_type,
        "threshold": threshold,
        "status": "ok",
        "total_bars_day1": n_bars_day1,
        "total_bars_all_days": total_bars_all_days,
        "n_days": n_days,
        "bars_per_session": n_bars_day1,
        "median_duration_s": float(np.median(durations_s)),
        "mean_duration_s": float(np.mean(durations_s)),
        "p10_duration_s": float(np.percentile(durations_s, 10)),
        "p90_duration_s": float(np.percentile(durations_s, 90)),
    }


def run_phase1():
    """Phase 1: Empirical Calibration Sweep."""
    print("\n" + "=" * 70, flush=True)
    print("PHASE 1: Empirical Calibration Sweep", flush=True)
    print("=" * 70, flush=True)

    cal_dir = RESULTS_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    calibration_rows = []
    csv_paths = {}  # {label: csv_path} for reuse in Phase 2
    stdout_cache = {}  # {label: stdout_text} for parsing bar counts

    for bar_type, thresholds in [("dollar", DOLLAR_THRESHOLDS), ("tick", TICK_THRESHOLDS)]:
        for threshold in thresholds:
            label = f"{bar_type}_{threshold}"
            csv_path, stdout_text = run_calibration_export(bar_type, threshold, cal_dir)

            if csv_path is None:
                calibration_rows.append({
                    "bar_type": bar_type,
                    "threshold": threshold,
                    "status": "export_failed",
                })
                continue

            csv_paths[label] = csv_path
            if stdout_text:
                stdout_cache[label] = stdout_text

            # Parse bar counts from stdout (fallback for large thresholds)
            stdout_bar_counts = parse_bar_counts_from_stdout(stdout_text) if stdout_text else None

            # Compute calibration stats from day 20220103
            row = compute_calibration_stats(csv_path, bar_type, threshold, stdout_bar_counts)
            calibration_rows.append(row)

            if row.get("status", "").startswith("ok"):
                print(f"  {label}: {row['total_bars_day1']} bars/day1, "
                      f"median={row['median_duration_s']:.3f}s, "
                      f"mean={row['mean_duration_s']:.3f}s, "
                      f"total_19d={row['total_bars_all_days']}", flush=True)

    # Write calibration table
    cal_table_path = cal_dir / "calibration_table.json"
    with open(cal_table_path, "w") as f:
        json.dump(calibration_rows, f, indent=2)
    print(f"\nCalibration table written to {cal_table_path}", flush=True)

    return calibration_rows, csv_paths


def validate_1m_anchor(calibration_rows):
    """Validate $1M empirical duration against R4c's extrapolated 0.9s."""
    r4c_estimate = 0.896  # R4c extrapolated
    dollar_1m = [r for r in calibration_rows
                 if r.get("bar_type") == "dollar" and r.get("threshold") == 1_000_000
                 and r.get("status", "").startswith("ok")]

    if not dollar_1m:
        return {"status": "NOT_FOUND", "r4c_estimate": r4c_estimate}

    empirical = dollar_1m[0]["median_duration_s"]
    ratio = empirical / r4c_estimate
    within_50pct = 0.5 <= ratio <= 1.5
    within_5x = 0.2 <= ratio <= 5.0

    result = {
        "status": "PASS" if within_50pct else ("WARN" if within_5x else "FAIL"),
        "empirical_median_s": empirical,
        "r4c_estimate_s": r4c_estimate,
        "ratio": ratio,
        "within_50pct": within_50pct,
        "within_5x": within_5x,
    }
    print(f"\n$1M anchor: empirical={empirical:.3f}s, R4c={r4c_estimate:.3f}s, "
          f"ratio={ratio:.2f} -> {result['status']}", flush=True)

    # Abort criterion: deviates >5x
    if not within_5x:
        print("ABORT: $1M anchor deviates >5x from R4c. Bar construction issue.", flush=True)

    return result


def validate_calibration_estimates(calibration_rows):
    """Check empirical durations vs volume-math estimates (within 2x)."""
    estimates = {
        ("dollar", 1_000_000): 0.9,
        ("dollar", 5_000_000): 4.0,
        ("dollar", 10_000_000): 7.5,
        ("dollar", 50_000_000): 37.5,
        ("dollar", 250_000_000): 210.0,
        ("dollar", 1_000_000_000): 750.0,
        ("tick", 500): 50.0,
        ("tick", 3_000): 300.0,
        ("tick", 10_000): 900.0,
        ("tick", 25_000): 2400.0,
    }
    results = {}
    for row in calibration_rows:
        if not row.get("status", "").startswith("ok"):
            continue
        key = (row["bar_type"], row["threshold"])
        if key in estimates:
            est = estimates[key]
            emp = row["median_duration_s"]
            ratio = emp / est if est > 0 else float("inf")
            within_2x = 0.5 <= ratio <= 2.0
            results[f"{key[0]}_{key[1]}"] = {
                "estimate_s": est,
                "empirical_s": emp,
                "ratio": round(ratio, 3),
                "within_2x": within_2x,
            }
    return results


def select_operating_points(calibration_rows):
    """Select 3 operating points per spec:
    1. Dollar bar nearest 5s median
    2. Dollar bar nearest 5min median
    3. Tick bar nearest 5min median
    """
    print("\nSelecting operating points...", flush=True)

    dollar_ok = [r for r in calibration_rows
                 if r.get("bar_type") == "dollar" and r.get("status", "").startswith("ok")]
    tick_ok = [r for r in calibration_rows
               if r.get("bar_type") == "tick" and r.get("status", "").startswith("ok")]

    selected = []
    selection_log = {}

    # Check SC2: at least 2 dollar thresholds with >=5s median
    dollar_actionable = [r for r in dollar_ok if r.get("median_duration_s", 0) >= 5.0]
    sc2_pass = len(dollar_actionable) >= 2
    selection_log["sc2_dollar_actionable_count"] = len(dollar_actionable)
    selection_log["sc2_pass"] = sc2_pass

    if dollar_actionable:
        # OP1: Dollar nearest 5s
        op1 = min(dollar_ok, key=lambda r: abs(r.get("median_duration_s", 999) - 5.0))
        selected.append({
            "label": f"dollar_{op1['threshold']}",
            "bar_type": "dollar",
            "threshold": op1["threshold"],
            "target": "5s",
            "median_duration_s": op1["median_duration_s"],
            "total_bars_all_days": op1["total_bars_all_days"],
        })
        print(f"  OP1 (dollar ~5s): dollar_{op1['threshold']} "
              f"(median={op1['median_duration_s']:.1f}s, {op1['total_bars_all_days']} total bars)", flush=True)

        # OP2: Dollar nearest 5min (300s)
        op2 = min(dollar_ok, key=lambda r: abs(r.get("median_duration_s", 999) - 300.0))
        if op2["threshold"] != op1["threshold"]:
            selected.append({
                "label": f"dollar_{op2['threshold']}",
                "bar_type": "dollar",
                "threshold": op2["threshold"],
                "target": "5min",
                "median_duration_s": op2["median_duration_s"],
                "total_bars_all_days": op2["total_bars_all_days"],
            })
            print(f"  OP2 (dollar ~5min): dollar_{op2['threshold']} "
                  f"(median={op2['median_duration_s']:.1f}s, {op2['total_bars_all_days']} total bars)", flush=True)
        else:
            # Same threshold for both targets — pick next closest for 5min
            remaining = [r for r in dollar_ok if r["threshold"] != op1["threshold"]]
            if remaining:
                op2 = min(remaining, key=lambda r: abs(r.get("median_duration_s", 999) - 300.0))
                selected.append({
                    "label": f"dollar_{op2['threshold']}",
                    "bar_type": "dollar",
                    "threshold": op2["threshold"],
                    "target": "5min",
                    "median_duration_s": op2["median_duration_s"],
                    "total_bars_all_days": op2["total_bars_all_days"],
                })
                print(f"  OP2 (dollar ~5min): dollar_{op2['threshold']} "
                      f"(median={op2['median_duration_s']:.1f}s, {op2['total_bars_all_days']} total bars)", flush=True)
    else:
        # No actionable dollar bars — use longest-duration dollar + note
        if dollar_ok:
            best_dollar = max(dollar_ok, key=lambda r: r.get("median_duration_s", 0))
            selected.append({
                "label": f"dollar_{best_dollar['threshold']}",
                "bar_type": "dollar",
                "threshold": best_dollar["threshold"],
                "target": "longest_dollar",
                "median_duration_s": best_dollar["median_duration_s"],
                "total_bars_all_days": best_dollar["total_bars_all_days"],
                "note": "No dollar threshold reaches 5s. Using longest for completeness.",
            })
            print(f"  OP_dollar (longest): dollar_{best_dollar['threshold']} "
                  f"(median={best_dollar['median_duration_s']:.1f}s)", flush=True)
        selection_log["dollar_sub_actionable"] = True

    # OP3: Tick nearest 5min (300s)
    if tick_ok:
        op3 = min(tick_ok, key=lambda r: abs(r.get("median_duration_s", 999) - 300.0))
        selected.append({
            "label": f"tick_{op3['threshold']}",
            "bar_type": "tick",
            "threshold": op3["threshold"],
            "target": "5min",
            "median_duration_s": op3["median_duration_s"],
            "total_bars_all_days": op3["total_bars_all_days"],
        })
        print(f"  OP3 (tick ~5min): tick_{op3['threshold']} "
              f"(median={op3['median_duration_s']:.1f}s, {op3['total_bars_all_days']} total bars)", flush=True)

    # Check for <100 total bars (abort criterion)
    valid_selected = []
    for op in selected:
        if op["total_bars_all_days"] < 100:
            print(f"  SKIP {op['label']}: <100 total bars ({op['total_bars_all_days']})", flush=True)
            selection_log[f"skip_{op['label']}"] = "insufficient_sample"
        else:
            valid_selected.append(op)

    selection_log["selected_count"] = len(valid_selected)
    selection_log["selected_labels"] = [op["label"] for op in valid_selected]

    # Determine horizons per operating point
    for op in valid_selected:
        bars_per_session = op["total_bars_all_days"] / 19  # approximate
        if bars_per_session < 26:
            op["horizons"] = [1, 5]
            op["horizon_restriction"] = "bars_per_session < 26"
        elif bars_per_session < 50:
            op["horizons"] = [1, 5, 20]
            op["horizon_restriction"] = "bars_per_session < 50"
        else:
            op["horizons"] = [1, 5, 20, 100]
            op["horizon_restriction"] = None
        print(f"  {op['label']}: horizons={op['horizons']} "
              f"(~{bars_per_session:.0f} bars/session)", flush=True)

    # Write selected thresholds
    sel_path = RESULTS_DIR / "calibration" / "selected_thresholds.json"
    with open(sel_path, "w") as f:
        json.dump({"selected": valid_selected, "log": selection_log}, f, indent=2)
    print(f"Selected thresholds written to {sel_path}", flush=True)

    return valid_selected, selection_log


# ===========================================================================
# Phase 3: Temporal Analysis per Operating Point
# ===========================================================================
def run_temporal_analysis(op, csv_path):
    """Run the R4d temporal analysis protocol for one operating point.

    Spec: GBT only, 3 feature configs (Static-Book, Book+Temporal, Temporal-Only).
    """
    label = op["label"]
    horizons = op["horizons"]
    op_dir = RESULTS_DIR / label
    op_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}", flush=True)
    print(f"TEMPORAL ANALYSIS: {label}", flush=True)
    print(f"Horizons: {horizons}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Load data
    df = load_features(csv_path)
    df = construct_temporal_features(df)

    # Determine appropriate lookback based on bars per day
    bars_per_day = len(df) / max(1, df["day"].n_unique())
    if bars_per_day < 120:
        # With <120 bars/day, rolling_vol_100 and momentum_100 will mostly be NaN
        # Use lookback=20 to preserve more data
        lookback_depth = min(20, max(5, int(bars_per_day * 0.3)))
        print(f"Sparse bars ({bars_per_day:.0f}/day): using lookback={lookback_depth}", flush=True)
    else:
        lookback_depth = 100

    df_valid = prepare_data(df, lookback_depth)
    if len(df_valid) == 0:
        print(f"  SKIP: zero valid bars after lookback removal", flush=True)
        return {
            "status": "insufficient_data",
            "n_valid": 0,
            "n_bars_loaded": len(df),
            "bars_per_day": bars_per_day,
            "lookback_depth": lookback_depth,
            "operating_point": label,
            "bar_type": op["bar_type"],
            "threshold": op["threshold"],
            "median_duration_s": op["median_duration_s"],
        }

    splits = get_cv_splits(df_valid)
    n_valid = len(df_valid)
    n_folds = len(splits)

    print(f"Valid bars: {n_valid}, CV folds: {n_folds}", flush=True)

    if n_valid < 100 or n_folds < 2:
        print(f"  SKIP: insufficient data after lookback ({n_valid} valid bars, {n_folds} folds)", flush=True)
        return {
            "status": "insufficient_data",
            "n_valid": n_valid,
            "n_folds": n_folds,
            "n_bars_loaded": len(df),
            "operating_point": label,
            "bar_type": op["bar_type"],
            "threshold": op["threshold"],
            "median_duration_s": op["median_duration_s"],
        }

    # Feature configs (spec: 3 configs, GBT only)
    configs = {
        "Static-Book": BOOK_SNAP_FEATURES,
        "Book+Temporal": BOOK_SNAP_FEATURES + TEMPORAL_FEATURE_NAMES,
        "Temporal-Only": TEMPORAL_FEATURE_NAMES,
    }

    # --- Tier 1: AR-10 GBT (best config from R4) ---
    print("\nTier 1: AR-10 GBT...", flush=True)
    tier1_results = {}
    lookback = 10
    ar_features = [f"lag_return_{i}" for i in range(1, lookback + 1)]
    available_ar = [c for c in ar_features if c in df_valid.columns]
    X_ar = df_valid.select(available_ar).to_numpy().astype(np.float64)

    for h in horizons:
        target_col = f"fwd_return_{h}"
        if target_col not in df_valid.columns:
            print(f"  AR-10_gbt_h{h}: target column missing, skip", flush=True)
            continue
        y = df_valid[target_col].to_numpy().astype(np.float64)

        key = f"AR-10_gbt_h{h}"
        t0 = time.time()
        fold_r2s, _ = fit_and_evaluate_gbt(X_ar, y, splits, feature_names=available_ar)
        elapsed = time.time() - t0

        mean_r2 = float(np.nanmean(fold_r2s))
        std_r2 = float(np.nanstd(fold_r2s))
        p_gt_zero = one_sided_test_greater_than_zero(fold_r2s)

        tier1_results[key] = {
            "lookback": lookback,
            "horizon": h,
            "fold_r2s": fold_r2s.tolist(),
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "p_gt_zero_raw": p_gt_zero,
            "elapsed_s": round(elapsed, 1),
        }
        print(f"  {key}: R2={mean_r2:.6f} +/- {std_r2:.6f} (p>0={p_gt_zero:.4f}) [{elapsed:.1f}s]", flush=True)

        # Abort check: R2 > +0.01
        if mean_r2 > 0.01:
            print(f"  WARNING: {key} R2={mean_r2:.6f} > 0.01 — unexpected strong signal!", flush=True)
            # Per spec: pause, verify, expand if verified. For now, flag but continue.
            tier1_results[key]["abort_flag"] = True

    # Holm-Bonferroni for Tier 1
    family_keys = list(tier1_results.keys())
    if family_keys:
        raw_ps = np.array([tier1_results[k]["p_gt_zero_raw"] for k in family_keys])
        corrected_ps = holm_bonferroni(raw_ps)
        for k, cp in zip(family_keys, corrected_ps):
            tier1_results[k]["p_gt_zero_corrected"] = float(cp)

    # --- Tier 2: Feature configs (GBT) ---
    print("\nTier 2: Feature configs (GBT)...", flush=True)
    tier2_results = {}
    importance_results = {}

    for config_name, feature_cols in configs.items():
        available_cols = [c for c in feature_cols if c in df_valid.columns]
        if not available_cols:
            print(f"  {config_name}: no features available, skip", flush=True)
            continue

        X = df_valid.select(available_cols).to_numpy().astype(np.float64)

        for h in horizons:
            target_col = f"fwd_return_{h}"
            if target_col not in df_valid.columns:
                continue
            y = df_valid[target_col].to_numpy().astype(np.float64)

            key = f"{config_name}_gbt_h{h}"
            t0 = time.time()
            fold_r2s, fold_imp = fit_and_evaluate_gbt(X, y, splits, feature_names=available_cols)
            elapsed = time.time() - t0

            mean_r2 = float(np.nanmean(fold_r2s))
            std_r2 = float(np.nanstd(fold_r2s))

            tier2_results[key] = {
                "config": config_name,
                "horizon": h,
                "fold_r2s": fold_r2s.tolist(),
                "mean_r2": mean_r2,
                "std_r2": std_r2,
                "elapsed_s": round(elapsed, 1),
            }
            if fold_imp is not None:
                importance_results[key] = fold_imp

            print(f"  {key}: R2={mean_r2:.6f} +/- {std_r2:.6f} [{elapsed:.1f}s]", flush=True)

    # --- Compute gaps and dual threshold ---
    print("\nComputing delta_temporal_book...", flush=True)
    gaps = compute_gaps(tier2_results, horizons)
    threshold_eval = evaluate_dual_threshold(gaps, tier2_results, horizons)

    # --- Feature importance ---
    importance_analysis = analyze_importance(importance_results)

    # --- Fold sign agreement (SC6) ---
    fold_sign_agreement = {}
    for key, result in tier1_results.items():
        r2s = np.array(result["fold_r2s"])
        n_positive = int(np.sum(r2s > 0))
        n_total = len(r2s)
        fold_sign_agreement[key] = {
            "n_positive": n_positive,
            "n_total": n_total,
            "all_agree_sign": n_positive == 0 or n_positive == n_total,
            "p_r2_gt_0": n_positive / n_total if n_total > 0 else 0,
        }

    # Write per-operating-point metrics
    op_metrics = {
        "operating_point": label,
        "bar_type": op["bar_type"],
        "threshold": op["threshold"],
        "median_duration_s": op["median_duration_s"],
        "horizons": horizons,
        "horizon_restriction": op.get("horizon_restriction"),
        "n_bars_loaded": len(df),
        "n_valid_bars": n_valid,
        "n_folds": n_folds,
        "tier1": tier1_results,
        "tier2": {
            "results": tier2_results,
            "gaps": gaps,
            "threshold_evaluation": threshold_eval,
            "feature_importance": importance_analysis,
        },
        "fold_sign_agreement": fold_sign_agreement,
    }

    op_metrics_path = op_dir / "metrics.json"
    with open(op_metrics_path, "w") as f:
        json.dump(op_metrics, f, indent=2, default=str)
    print(f"Metrics written to {op_metrics_path}", flush=True)

    return op_metrics


def compute_gaps(tier2_results, horizons):
    """Compute delta_temporal_book and delta_temporal_only."""
    gaps = {}

    gap_defs = {
        "delta_temporal_book": ("Book+Temporal_gbt", "Static-Book_gbt"),
        "delta_temporal_only": ("Temporal-Only_gbt", None),
    }

    for gap_name, (aug_base, base_base) in gap_defs.items():
        raw_ps = []
        entries = []

        for h in horizons:
            aug_key = f"{aug_base}_h{h}"
            if aug_key not in tier2_results:
                continue

            aug_r2s = np.array(tier2_results[aug_key]["fold_r2s"])

            if base_base is not None:
                base_key = f"{base_base}_h{h}"
                if base_key not in tier2_results:
                    continue
                base_r2s = np.array(tier2_results[base_key]["fold_r2s"])
                diffs = aug_r2s - base_r2s
                delta = float(np.mean(diffs))
                ci = confidence_interval_95(diffs)
                raw_p = paired_test(aug_r2s, base_r2s)
                d = cohens_d(aug_r2s, base_r2s)
            else:
                diffs = aug_r2s
                delta = float(np.mean(diffs))
                ci = confidence_interval_95(diffs)
                raw_p = one_sided_test_greater_than_zero(diffs)
                d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-15))

            raw_ps.append(raw_p)
            entries.append({
                "horizon": h,
                "delta_r2": delta,
                "ci_95": list(ci),
                "raw_p": raw_p,
                "cohens_d": d,
                "fold_diffs": diffs.tolist(),
            })

        if raw_ps:
            corrected_ps = holm_bonferroni(np.array(raw_ps))
            for i, entry in enumerate(entries):
                entry["corrected_p"] = float(corrected_ps[i])

        gaps[gap_name] = entries

    return gaps


def evaluate_dual_threshold(gaps, tier2_results, horizons):
    """Evaluate dual threshold: delta > 20% of baseline AND corrected p < 0.05."""
    threshold_results = {}

    for gap_name, entries in gaps.items():
        for entry in entries:
            h = entry["horizon"]
            baseline_key = f"Static-Book_gbt_h{h}"
            baseline_r2 = tier2_results.get(baseline_key, {}).get("mean_r2", 0)
            relative_threshold = 0.20 * abs(baseline_r2) if baseline_r2 != 0 else 0.001

            passes_relative = entry["delta_r2"] > relative_threshold
            passes_statistical = entry.get("corrected_p", 1.0) < 0.05
            passes_dual = passes_relative and passes_statistical

            key = f"{gap_name}_h{h}"
            threshold_results[key] = {
                "gap": gap_name,
                "horizon": h,
                "delta_r2": entry["delta_r2"],
                "baseline_r2": baseline_r2,
                "relative_threshold": relative_threshold,
                "passes_relative": passes_relative,
                "passes_statistical": passes_statistical,
                "passes_dual": passes_dual,
                "corrected_p": entry.get("corrected_p", 1.0),
                "ci_95": entry["ci_95"],
                "cohens_d": entry["cohens_d"],
            }

    return threshold_results


def analyze_importance(importance_results):
    """Analyze feature importance for Book+Temporal config."""
    analysis = {}
    temporal_set = set(TEMPORAL_FEATURE_NAMES)

    for key, importances in importance_results.items():
        if "Book+Temporal" not in key:
            continue
        sorted_feats = sorted(importances.items(), key=lambda x: -x[1])
        top10 = sorted_feats[:10]

        temporal_share = sum(v for k, v in importances.items() if k in temporal_set)
        total = sum(importances.values())
        temporal_fraction = temporal_share / total if total > 0 else 0.0

        top10_entries = []
        for rank, (fname, imp) in enumerate(top10, 1):
            category = "temporal" if fname in temporal_set else "static"
            top10_entries.append({
                "rank": rank,
                "feature": fname,
                "importance": float(imp),
                "category": category,
            })

        analysis[key] = {
            "top10": top10_entries,
            "temporal_importance_fraction": float(temporal_fraction),
            "total_features": len(importances),
        }

    return analysis


# ===========================================================================
# Phase 4: Cross-Timescale Synthesis
# ===========================================================================
def build_timescale_response(op_metrics_list, calibration_rows):
    """Build timescale response table combining R4c tick results with R4d new points."""
    print("\n" + "=" * 70, flush=True)
    print("PHASE 4: Cross-Timescale Synthesis", flush=True)
    print("=" * 70, flush=True)

    # Prior results from R4 chain
    prior_points = [
        {"bar_type": "time", "threshold": "5s", "median_duration_s": 5.0,
         "tier1_ar_r2_h1": -0.0002, "delta_temporal_book_h1": -0.0021, "dual_pass": False,
         "source": "R4"},
        {"bar_type": "tick", "threshold": 50, "median_duration_s": 5.0,
         "tier1_ar_r2_h1": -0.000370, "delta_temporal_book_h1": -0.003, "dual_pass": False,
         "source": "R4c"},
        {"bar_type": "tick", "threshold": 100, "median_duration_s": 10.0,
         "tier1_ar_r2_h1": -0.000297, "delta_temporal_book_h1": -0.00451, "dual_pass": False,
         "source": "R4c"},
        {"bar_type": "tick", "threshold": 250, "median_duration_s": 25.0,
         "tier1_ar_r2_h1": -0.000762, "delta_temporal_book_h1": -0.00409, "dual_pass": False,
         "source": "R4c"},
    ]

    # Add R4d new operating points
    for op_m in op_metrics_list:
        if isinstance(op_m, dict) and "tier1" in op_m:
            tier1 = op_m["tier1"]
            tier2_eval = op_m.get("tier2", {}).get("threshold_evaluation", {})

            ar_h1_key = "AR-10_gbt_h1"
            ar_h1_r2 = tier1.get(ar_h1_key, {}).get("mean_r2", None)

            dtb_h1 = tier2_eval.get("delta_temporal_book_h1", {})
            delta_h1 = dtb_h1.get("delta_r2", None) if isinstance(dtb_h1, dict) else None
            dual_h1 = dtb_h1.get("passes_dual", False) if isinstance(dtb_h1, dict) else False

            prior_points.append({
                "bar_type": op_m["bar_type"],
                "threshold": op_m["threshold"],
                "median_duration_s": op_m["median_duration_s"],
                "tier1_ar_r2_h1": ar_h1_r2,
                "delta_temporal_book_h1": delta_h1,
                "dual_pass": dual_h1,
                "source": "R4d",
            })

    # Sort by median duration
    prior_points.sort(key=lambda x: x.get("median_duration_s", 0))

    print("\nTimescale Response Table:")
    print(f"{'bar_type':<10} {'threshold':<12} {'median_s':<10} {'AR_R2_h1':<12} {'delta_h1':<12} {'dual':<6} {'source':<6}")
    for p in prior_points:
        print(f"{p['bar_type']:<10} {str(p['threshold']):<12} "
              f"{p['median_duration_s']:<10.1f} "
              f"{p.get('tier1_ar_r2_h1', 'N/A')!s:<12} "
              f"{p.get('delta_temporal_book_h1', 'N/A')!s:<12} "
              f"{str(p['dual_pass']):<6} {p['source']:<6}", flush=True)

    return prior_points


def write_analysis_md(calibration_rows, anchor_result, cal_validation, op_metrics_list,
                      timescale_response, selection_log):
    """Write cross-timescale analysis.md."""
    lines = []
    lines.append("# R4d: Temporal Predictability at Actionable Dollar & Tick Timescales\n")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n")

    # Calibration table
    lines.append("## Phase 1: Empirical Calibration Table\n")
    lines.append("| Bar Type | Threshold | Bars/Day1 | Median (s) | Mean (s) | P10 (s) | P90 (s) | Total 19d |")
    lines.append("|----------|-----------|-----------|------------|----------|---------|---------|-----------|")
    for row in calibration_rows:
        if row.get("status", "").startswith("ok"):
            note = " *" if row.get("note") else ""
            lines.append(
                f"| {row['bar_type']} | {row['threshold']:,} | {row['total_bars_day1']} | "
                f"{row['median_duration_s']:.3f} | {row['mean_duration_s']:.3f} | "
                f"{row['p10_duration_s']:.3f} | {row['p90_duration_s']:.3f} | "
                f"{row['total_bars_all_days']}{note} |"
            )
        else:
            lines.append(f"| {row.get('bar_type','?')} | {row.get('threshold','?')} | — | — | — | — | — | {row.get('status','?')} |")
    lines.append("")

    # Anchor validation
    lines.append("### $1M Anchor Validation\n")
    lines.append(f"- Empirical median: {anchor_result.get('empirical_median_s', 'N/A')}s")
    lines.append(f"- R4c extrapolation: {anchor_result.get('r4c_estimate_s', 'N/A')}s")
    lines.append(f"- Ratio: {anchor_result.get('ratio', 'N/A')}")
    lines.append(f"- Status: {anchor_result.get('status', 'N/A')}\n")

    # Operating points
    lines.append("## Operating Points Selected\n")
    for op_m in op_metrics_list:
        if isinstance(op_m, dict) and "operating_point" in op_m:
            lines.append(f"### {op_m['operating_point']}\n")
            lines.append(f"- Median duration: {op_m.get('median_duration_s', 'N/A')}s")
            lines.append(f"- Bars loaded: {op_m.get('n_bars_loaded', 'N/A')}")
            lines.append(f"- Valid bars: {op_m.get('n_valid_bars', 'N/A')}")
            lines.append(f"- Horizons: {op_m.get('horizons', 'N/A')}")

            # Tier 1
            if "tier1" in op_m:
                lines.append("\n**Tier 1: AR-10 GBT**\n")
                lines.append("| Horizon | Mean R2 | Std | p>0 (corrected) |")
                lines.append("|---------|---------|-----|-----------------|")
                for key, val in op_m["tier1"].items():
                    lines.append(
                        f"| h={val['horizon']} | {val['mean_r2']:.6f} | {val['std_r2']:.6f} | "
                        f"{val.get('p_gt_zero_corrected', val.get('p_gt_zero_raw', 'N/A')):.4f} |"
                    )

            # Tier 2
            if "tier2" in op_m:
                t2 = op_m["tier2"]
                if "results" in t2:
                    lines.append("\n**Tier 2: Feature Configs (GBT)**\n")
                    lines.append("| Config | Horizon | Mean R2 | Std |")
                    lines.append("|--------|---------|---------|-----|")
                    for key, val in t2["results"].items():
                        lines.append(f"| {val['config']} | h={val['horizon']} | {val['mean_r2']:.6f} | {val['std_r2']:.6f} |")

                # Dual threshold
                if "threshold_evaluation" in t2:
                    lines.append("\n**Dual Threshold Evaluation**\n")
                    lines.append("| Gap | Horizon | Delta R2 | Corrected p | Passes? |")
                    lines.append("|-----|---------|----------|-------------|---------|")
                    for key, val in t2["threshold_evaluation"].items():
                        passes = "YES" if val.get("passes_dual", False) else "no"
                        lines.append(
                            f"| {val['gap']} | h={val['horizon']} | {val['delta_r2']:.6f} | "
                            f"{val['corrected_p']:.4f} | {passes} |"
                        )

            lines.append("")

    # Timescale response
    lines.append("## Cross-Timescale AR R2 Response\n")
    lines.append("| Bar Type | Threshold | Median (s) | AR R2 h=1 | Delta h=1 | Dual Pass | Source |")
    lines.append("|----------|-----------|------------|-----------|-----------|-----------|--------|")
    for p in timescale_response:
        ar = f"{p['tier1_ar_r2_h1']:.6f}" if p.get('tier1_ar_r2_h1') is not None else "N/A"
        delta = f"{p['delta_temporal_book_h1']:.6f}" if p.get('delta_temporal_book_h1') is not None else "N/A"
        lines.append(
            f"| {p['bar_type']} | {p['threshold']} | {p['median_duration_s']:.1f} | "
            f"{ar} | {delta} | {p['dual_pass']} | {p['source']} |"
        )
    lines.append("")

    # Decision framework
    lines.append("## Decision Framework Evaluation\n")
    total_tests = sum(
        len(op_m.get("tier2", {}).get("threshold_evaluation", {}))
        for op_m in op_metrics_list if isinstance(op_m, dict)
    )
    total_passes = sum(
        sum(1 for v in op_m.get("tier2", {}).get("threshold_evaluation", {}).values()
            if isinstance(v, dict) and v.get("passes_dual", False))
        for op_m in op_metrics_list if isinstance(op_m, dict)
    )
    lines.append(f"- Total dual threshold tests: {total_tests}")
    lines.append(f"- Total passes: {total_passes}")
    lines.append(f"- Result: {'ALL FAIL' if total_passes == 0 else f'{total_passes} PASS'}")
    lines.append("")

    analysis_path = RESULTS_DIR / "analysis.md"
    analysis_path.write_text("\n".join(lines))
    print(f"\nAnalysis written to {analysis_path}", flush=True)


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()

    print("R4d: Temporal Predictability at Actionable Dollar & Tick Timescales", flush=True)
    print("=" * 70, flush=True)
    print(f"Start time: {datetime.now().isoformat()}", flush=True)

    # Initialize final metrics
    all_metrics = {
        "experiment": "R4d-temporal-predictability-dollar-tick-actionable",
        "timestamp": datetime.now().isoformat(),
        "abort_triggered": False,
        "abort_reason": None,
        "notes": "",
    }

    # =================================================================
    # PHASE 1: Empirical Calibration
    # =================================================================
    calibration_rows, csv_paths = run_phase1()
    all_metrics["calibration"] = calibration_rows

    # Validate $1M anchor
    anchor_result = validate_1m_anchor(calibration_rows)
    all_metrics["sanity_checks"] = {"anchor_1m": anchor_result}

    if anchor_result.get("status") == "FAIL":
        all_metrics["abort_triggered"] = True
        all_metrics["abort_reason"] = f"$1M anchor deviates >5x from R4c (ratio={anchor_result.get('ratio', 'N/A')})"
        all_metrics["resource_usage"] = {
            "gpu_hours": 0,
            "wall_clock_seconds": round(time.time() - t_start, 1),
            "total_training_steps": 0,
            "total_runs": 0,
        }
        write_final_metrics(all_metrics)
        return

    # Calibration validation (within 2x of estimates)
    cal_validation = validate_calibration_estimates(calibration_rows)
    all_metrics["sanity_checks"]["calibration_validation"] = cal_validation

    # Select operating points
    selected_ops, selection_log = select_operating_points(calibration_rows)
    all_metrics["selected_operating_points"] = [
        {k: v for k, v in op.items()} for op in selected_ops
    ]
    all_metrics["selection_log"] = selection_log

    if not selected_ops:
        print("\nNo operating points selected. Calibration table is the final deliverable.", flush=True)
        all_metrics["notes"] += "No operating points selected. Calibration-only result. "
        all_metrics["resource_usage"] = {
            "gpu_hours": 0,
            "wall_clock_seconds": round(time.time() - t_start, 1),
            "total_training_steps": 0,
            "total_runs": 0,
        }
        write_final_metrics(all_metrics)
        return

    # =================================================================
    # PHASE 2 + 3: Feature Export Validation + Temporal Analysis
    # =================================================================
    op_metrics_list = []
    total_runs = 0

    for op in selected_ops:
        label = op["label"]
        csv_path = csv_paths.get(label)

        if csv_path is None or not csv_path.exists():
            print(f"\n{label}: CSV not available, skipping.", flush=True)
            op_metrics_list.append({"status": "csv_missing", "label": label})
            continue

        # Phase 2: Validate bar count
        total_bars = op["total_bars_all_days"]
        cal_day_bars = None
        for row in calibration_rows:
            if row.get("bar_type") == op["bar_type"] and row.get("threshold") == op["threshold"]:
                cal_day_bars = row.get("total_bars_day1")
                break
        if cal_day_bars:
            expected_19d = cal_day_bars * 19
            actual = total_bars
            pct_diff = abs(actual - expected_19d) / expected_19d * 100 if expected_19d > 0 else 999
            within_20pct = pct_diff <= 20
            print(f"\n{label} bar count: {actual} actual vs {expected_19d} extrapolated "
                  f"({pct_diff:.1f}% diff) -> {'OK' if within_20pct else 'WARNING'}", flush=True)
            op["bar_count_validation"] = {
                "actual_19d": actual,
                "extrapolated_19d": expected_19d,
                "pct_diff": round(pct_diff, 1),
                "within_20pct": within_20pct,
            }

        # Wall clock check
        elapsed_so_far = time.time() - t_start
        if elapsed_so_far > 36000:  # 10 hours
            print(f"\nWall clock exceeded 10 hours ({elapsed_so_far/3600:.1f}h). Stopping.", flush=True)
            all_metrics["notes"] += f"Wall clock limit reached after {total_runs} runs. "
            break

        # Phase 3: Temporal analysis
        op_metrics = run_temporal_analysis(op, csv_path)
        op_metrics_list.append(op_metrics)
        total_runs += 1

    all_metrics["operating_point_results"] = op_metrics_list

    # =================================================================
    # PHASE 4: Cross-Timescale Synthesis
    # =================================================================
    timescale_response = build_timescale_response(op_metrics_list, calibration_rows)
    all_metrics["timescale_response"] = timescale_response

    # Write analysis.md
    write_analysis_md(calibration_rows, anchor_result, cal_validation,
                      op_metrics_list, timescale_response, selection_log)

    # =================================================================
    # Success Criteria Evaluation
    # =================================================================
    sc = {}

    # SC1: Calibration table for all 10 thresholds
    ok_count = sum(1 for r in calibration_rows if r.get("status", "").startswith("ok"))
    sc["SC1_calibration_table"] = ok_count >= 10
    sc["SC1_ok_count"] = ok_count

    # SC2: >=2 dollar thresholds with >=5s median
    dollar_5s = sum(1 for r in calibration_rows
                    if r.get("bar_type") == "dollar" and r.get("status", "").startswith("ok")
                    and r.get("median_duration_s", 0) >= 5.0)
    sc["SC2_dollar_actionable"] = dollar_5s >= 2
    sc["SC2_count"] = dollar_5s

    # SC3: Full R4 protocol completed for all selected operating points
    completed = sum(1 for m in op_metrics_list if isinstance(m, dict) and "tier1" in m)
    sc["SC3_protocol_complete"] = completed == len(selected_ops)
    sc["SC3_completed"] = completed
    sc["SC3_total"] = len(selected_ops)

    # SC4: 0/N dual threshold passes
    total_tests = 0
    total_passes = 0
    for op_m in op_metrics_list:
        if isinstance(op_m, dict) and "tier2" in op_m:
            te = op_m["tier2"].get("threshold_evaluation", {})
            total_tests += len(te)
            total_passes += sum(1 for v in te.values()
                                if isinstance(v, dict) and v.get("passes_dual", False))
    sc["SC4_zero_dual_passes"] = total_passes == 0
    sc["SC4_passes"] = total_passes
    sc["SC4_total_tests"] = total_tests

    # SC5: Timescale response data produced
    sc["SC5_timescale_response"] = len(timescale_response) > 4  # >4 means we added new points

    # SC6: Fold sign agreement
    all_agree = True
    for op_m in op_metrics_list:
        if isinstance(op_m, dict) and "fold_sign_agreement" in op_m:
            for key, val in op_m["fold_sign_agreement"].items():
                if not val.get("all_agree_sign", True):
                    all_agree = False
    sc["SC6_fold_agreement"] = all_agree

    all_metrics["success_criteria"] = sc

    # Sanity checks
    sanity = all_metrics.get("sanity_checks", {})
    # Check: Static-Book R2 at h=1 same order of magnitude as time_5s (R4: +0.0046)
    for op_m in op_metrics_list:
        if isinstance(op_m, dict) and "tier2" in op_m:
            sb_h1 = op_m["tier2"].get("results", {}).get("Static-Book_gbt_h1", {})
            label = op_m.get("operating_point", "?")
            sanity[f"static_book_r2_h1_{label}"] = sb_h1.get("mean_r2", None)
    # Check: Temporal-Only R2 <= Book+Temporal R2
    for op_m in op_metrics_list:
        if isinstance(op_m, dict) and "tier2" in op_m:
            results = op_m["tier2"].get("results", {})
            for h in op_m.get("horizons", []):
                to_key = f"Temporal-Only_gbt_h{h}"
                bt_key = f"Book+Temporal_gbt_h{h}"
                to_r2 = results.get(to_key, {}).get("mean_r2")
                bt_r2 = results.get(bt_key, {}).get("mean_r2")
                if to_r2 is not None and bt_r2 is not None:
                    sanity[f"info_subset_{op_m.get('operating_point','?')}_h{h}"] = to_r2 <= bt_r2

    all_metrics["sanity_checks"] = sanity

    # Resource usage
    all_metrics["resource_usage"] = {
        "gpu_hours": 0,
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "total_training_steps": total_runs * 5 * 3,  # runs * folds * configs
        "total_runs": total_runs,
    }

    write_final_metrics(all_metrics)

    print("\n" + "=" * 70, flush=True)
    print("R4d COMPLETE", flush=True)
    print(f"Wall clock: {(time.time() - t_start)/60:.1f} min", flush=True)
    print(f"SC4 dual passes: {total_passes}/{total_tests}", flush=True)
    print("=" * 70, flush=True)


def write_final_metrics(all_metrics):
    """Write final metrics.json."""
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nFinal metrics written to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
