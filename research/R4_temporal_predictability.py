#!/usr/bin/env python3
"""Phase R4: Temporal Predictability — Analysis Script

Spec: .kit/experiments/temporal-predictability.md

Tests whether MES 5-second bar returns contain exploitable autoregressive
structure beyond the current-bar feature set.

Tier 1: Pure return autoregression (Linear, Ridge, GBT) at multiple lookback depths.
Tier 2: Temporal feature augmentation of static feature sets with GBT.

Input:
  - .kit/results/info-decomposition/features.csv (reuse R2 export)

Output:
  - .kit/results/temporal-predictability/metrics.json
  - .kit/results/temporal-predictability/analysis.md
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_INPUT_CSV = ".kit/results/info-decomposition/features.csv"
DEFAULT_OUTPUT_DIR = ".kit/results/temporal-predictability"
DEFAULT_BAR_LABEL = "time_5s"

# These globals are set in main() from CLI args
INPUT_CSV = Path(DEFAULT_INPUT_CSV)
RESULTS_DIR = Path(DEFAULT_OUTPUT_DIR)
BAR_LABEL = DEFAULT_BAR_LABEL
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 5, 20, 100]
LOOKBACKS = [10, 50, 100]
SEED = 42

# CV folds: expanding window (1-indexed day numbers)
CV_FOLDS = [
    (list(range(1, 5)),   list(range(5, 9))),    # fold 1: train 1-4, test 5-8
    (list(range(1, 9)),   list(range(9, 12))),    # fold 2: train 1-8, test 9-11
    (list(range(1, 12)),  list(range(12, 15))),   # fold 3: train 1-11, test 12-14
    (list(range(1, 15)),  list(range(15, 18))),   # fold 4: train 1-14, test 15-17
    (list(range(1, 18)),  list(range(18, 20))),   # fold 5: train 1-17, test 18-19
]

# Track A: 62 hand-crafted features (same as R2)
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

BOOK_SNAP_FEATURES = [f"book_snap_{i}" for i in range(40)]

TEMPORAL_FEATURE_NAMES = [
    *[f"lag_return_{i}" for i in range(1, 11)],
    "rolling_vol_5", "rolling_vol_20", "rolling_vol_100",
    "vol_ratio",
    "momentum_5", "momentum_20", "momentum_100",
    "mean_reversion_20",
    "abs_return_lag1",
    "signed_vol",
]


# ===========================================================================
# Data loading
# ===========================================================================
def load_features():
    """Load feature CSV, handling polars duplicate column renaming."""
    df = pl.read_csv(str(INPUT_CSV), infer_schema_length=10000,
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
    print(f"Loaded {len(df)} bars from {INPUT_CSV}", flush=True)
    print(f"Days: {sorted(df['day'].unique().to_list())}", flush=True)
    return df


def construct_temporal_features(df):
    """Construct temporal features per-day (no cross-day lookback).

    Returns df with additional columns for all temporal features.
    """
    print("Constructing temporal features...", flush=True)

    # We need return_1 (the within-bar return) for lagged returns
    # This is the Track A return_1 (not fwd_return_1)
    ret_col = "return_1"

    # Process per day to avoid cross-day lookback
    day_frames = []
    for day in sorted(df["day"].unique().to_list()):
        day_df = df.filter(pl.col("day") == day).sort("bar_index")

        # Lagged returns
        for lag in range(1, 11):
            day_df = day_df.with_columns(
                pl.col(ret_col).shift(lag).alias(f"lag_return_{lag}")
            )

        # Rolling volatility (std of returns over window)
        for window in [5, 20, 100]:
            day_df = day_df.with_columns(
                pl.col(ret_col).rolling_std(window_size=window, min_periods=window).alias(f"rolling_vol_{window}")
            )

        # Vol ratio
        day_df = day_df.with_columns(
            (pl.col("rolling_vol_5") / pl.col("rolling_vol_20").clip(lower_bound=1e-12)).alias("vol_ratio")
        )

        # Momentum (sum of returns over window)
        for window in [5, 20, 100]:
            day_df = day_df.with_columns(
                pl.col(ret_col).rolling_sum(window_size=window, min_periods=window).alias(f"momentum_{window}")
            )

        # Mean reversion: return_{t-1} - mean(return_{t-20..t-1})
        day_df = day_df.with_columns(
            (pl.col("lag_return_1") - pl.col(ret_col).rolling_mean(window_size=20, min_periods=20)).alias("mean_reversion_20")
        )

        # Abs return lag 1
        day_df = day_df.with_columns(
            pl.col("lag_return_1").abs().alias("abs_return_lag1")
        )

        # Signed vol
        day_df = day_df.with_columns(
            (pl.col("lag_return_1").sign() * pl.col("rolling_vol_5")).alias("signed_vol")
        )

        day_frames.append(day_df)

    result = pl.concat(day_frames)
    print(f"Temporal features constructed. Shape: {result.shape}", flush=True)
    return result


def prepare_data(df, lookback_depth):
    """Filter warmup bars and bars needing lookback, return non-warmup df.

    For Tier 2 configs that use temporal features, lookback_depth=100 is used
    (the max window for rolling_vol_100 and momentum_100).
    """
    # Remove warmup bars
    df_valid = df.filter(pl.col("is_warmup") == False)

    # For each day, also remove the first `lookback_depth` bars
    # (they have null temporal features due to insufficient history)
    if lookback_depth > 0:
        day_frames = []
        for day in sorted(df_valid["day"].unique().to_list()):
            day_df = df_valid.filter(pl.col("day") == day).sort("bar_index")
            if len(day_df) > lookback_depth:
                day_df = day_df.slice(lookback_depth)
            else:
                continue
            day_frames.append(day_df)
        df_valid = pl.concat(day_frames)

    return df_valid


# ===========================================================================
# CV helpers
# ===========================================================================
def get_cv_splits(df_valid, folds=CV_FOLDS):
    """Return list of (train_indices, test_indices) for numpy arrays.

    CV folds use 1-indexed day numbers (1-19). Map to actual YYYYMMDD day values.
    """
    days = df_valid["day"].to_numpy()
    unique_days = sorted(np.unique(days).tolist())
    # Map 1-indexed position to actual day value
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


def fit_and_evaluate(X, y, splits, model_type, feature_names=None):
    """Fit model on each fold, return per-fold R² and feature importances from last fold.

    model_type: 'linear', 'ridge', or 'gbt'
    """
    fold_r2s = []
    last_fold_importances = None

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Remove rows with NaN/Inf
        train_valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        test_valid = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
        X_train, y_train = X_train[train_valid], y_train[train_valid]
        X_test, y_test = X_test[test_valid], y_test[test_valid]

        if len(X_train) == 0 or len(X_test) == 0:
            fold_r2s.append(float("nan"))
            continue

        # Z-score standardize using training stats
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if model_type == "linear":
            model = LinearRegression()
            model.fit(X_train, y_train)
        elif model_type == "ridge":
            # Nested 3-fold CV for alpha
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=3)
            model.fit(X_train, y_train)
        elif model_type == "gbt":
            # Split last 20% of training as validation for early stopping
            n_train = len(X_train)
            n_val = max(1, int(n_train * 0.2))
            X_tr, X_val = X_train[:-n_val], X_train[-n_val:]
            y_tr, y_val = y_train[:-n_val], y_train[-n_val:]

            model = XGBRegressor(
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
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        fold_r2s.append(r2)

        # Extract feature importances from last fold for GBT
        if fold_idx == len(splits) - 1 and model_type == "gbt" and feature_names is not None:
            importances = model.feature_importances_
            last_fold_importances = dict(zip(feature_names, importances.tolist()))

    return np.array(fold_r2s), last_fold_importances


# ===========================================================================
# Statistical tests
# ===========================================================================
def paired_test(a, b):
    """Paired t-test (or Wilcoxon if diffs non-normal). Returns raw p-value."""
    diffs = a - b
    n = len(diffs)
    if n < 3:
        return 1.0
    # Shapiro-Wilk on differences
    if n >= 3:
        _, sw_p = stats.shapiro(diffs)
    else:
        sw_p = 1.0

    if sw_p < 0.05:
        # Non-normal: Wilcoxon signed-rank
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
    # One-sided: only significant if t > 0
    if t_stat > 0:
        return float(p_two / 2)
    else:
        return 1.0


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction. Returns corrected p-values in original order."""
    n = len(p_values)
    indices = np.argsort(p_values)
    corrected = np.ones(n)
    for rank, idx in enumerate(indices):
        corrected[idx] = p_values[idx] * (n - rank)
    # Enforce monotonicity
    sorted_corrected = corrected[indices]
    for i in range(1, n):
        if sorted_corrected[i] < sorted_corrected[i - 1]:
            sorted_corrected[i] = sorted_corrected[i - 1]
    corrected[indices] = sorted_corrected
    return np.clip(corrected, 0, 1)


def confidence_interval_95(values):
    """95% CI from t-distribution with df=n-1."""
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
# Tier 1: Pure Return Autoregression
# ===========================================================================
def run_tier1(df_full):
    """Run Tier 1: AR models on lagged returns."""
    print("\n" + "=" * 70, flush=True)
    print("TIER 1: Pure Return Autoregression", flush=True)
    print("=" * 70, flush=True)

    results = {}  # key: (lookback, model, horizon) -> dict

    for lookback in LOOKBACKS:
        # Prepare data with appropriate lookback exclusion
        df_valid = prepare_data(df_full, lookback)
        splits = get_cv_splits(df_valid)

        print(f"\nLookback={lookback}: {len(df_valid)} bars, {len(splits)} folds", flush=True)

        # Build AR feature matrix
        ar_feature_names = [f"lag_return_{i}" for i in range(1, lookback + 1)]

        # For lookback > 10, we need to construct additional lagged features
        if lookback > 10:
            # We already have lag_return_1..10 from construct_temporal_features
            # Need to construct lag_return_11..lookback per day
            day_frames = []
            for day in sorted(df_valid["day"].unique().to_list()):
                day_df = df_valid.filter(pl.col("day") == day).sort("bar_index")
                for lag in range(11, lookback + 1):
                    day_df = day_df.with_columns(
                        pl.col("return_1").shift(lag).alias(f"lag_return_{lag}")
                    )
                day_frames.append(day_df)
            df_lookback = pl.concat(day_frames)
        else:
            df_lookback = df_valid

        # Extract numpy arrays
        X_cols = [f"lag_return_{i}" for i in range(1, lookback + 1)]
        X = df_lookback.select(X_cols).to_numpy().astype(np.float64)

        splits_lb = get_cv_splits(df_lookback)

        for horizon in HORIZONS:
            target_col = f"fwd_return_{horizon}"
            y = df_lookback[target_col].to_numpy().astype(np.float64)

            for model_type in ["linear", "ridge", "gbt"]:
                key = f"AR-{lookback}_{model_type}_h{horizon}"
                fold_r2s, _ = fit_and_evaluate(X, y, splits_lb, model_type, feature_names=X_cols)

                mean_r2 = float(np.nanmean(fold_r2s))
                std_r2 = float(np.nanstd(fold_r2s))
                p_gt_zero = one_sided_test_greater_than_zero(fold_r2s)

                results[key] = {
                    "lookback": lookback,
                    "model": model_type,
                    "horizon": horizon,
                    "fold_r2s": fold_r2s.tolist(),
                    "mean_r2": mean_r2,
                    "std_r2": std_r2,
                    "p_gt_zero_raw": p_gt_zero,
                }

                print(f"  {key}: R²={mean_r2:.6f} ± {std_r2:.6f} (p>{0}={p_gt_zero:.4f})", flush=True)

    # Holm-Bonferroni correction within each model family (12 tests per model)
    for model_type in ["linear", "ridge", "gbt"]:
        family_keys = [k for k in results if results[k]["model"] == model_type]
        raw_ps = np.array([results[k]["p_gt_zero_raw"] for k in family_keys])
        corrected_ps = holm_bonferroni(raw_ps)
        for k, cp in zip(family_keys, corrected_ps):
            results[k]["p_gt_zero_corrected"] = float(cp)

    return results


# ===========================================================================
# Tier 2: Temporal Feature Augmentation
# ===========================================================================
def run_tier2(df_full):
    """Run Tier 2: Static features + temporal feature augmentation."""
    print("\n" + "=" * 70, flush=True)
    print("TIER 2: Temporal Feature Augmentation", flush=True)
    print("=" * 70, flush=True)

    # Use lookback=100 for all Tier 2 (max temporal window)
    df_valid = prepare_data(df_full, 100)
    splits = get_cv_splits(df_valid)
    print(f"Tier 2: {len(df_valid)} bars, {len(splits)} folds", flush=True)

    # Define configs
    configs = {
        "Static-Book": BOOK_SNAP_FEATURES,
        "Static-HC": TRACK_A_FEATURES,
        "Book+Temporal": BOOK_SNAP_FEATURES + TEMPORAL_FEATURE_NAMES,
        "HC+Temporal": TRACK_A_FEATURES + TEMPORAL_FEATURE_NAMES,
        "Temporal-Only": TEMPORAL_FEATURE_NAMES,
    }

    # Configs that also get linear model: Static-Book, Book+Temporal
    linear_configs = ["Static-Book", "Book+Temporal"]

    results = {}
    importance_results = {}

    for config_name, feature_cols in configs.items():
        # Check that all feature columns exist
        available_cols = [c for c in feature_cols if c in df_valid.columns]
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            print(f"  WARNING: {config_name} missing columns: {missing}", flush=True)
        feature_cols = available_cols

        X = df_valid.select(feature_cols).to_numpy().astype(np.float64)

        for horizon in HORIZONS:
            target_col = f"fwd_return_{horizon}"
            y = df_valid[target_col].to_numpy().astype(np.float64)

            # GBT
            key = f"{config_name}_gbt_h{horizon}"
            fold_r2s, fold_importances = fit_and_evaluate(
                X, y, splits, "gbt", feature_names=feature_cols
            )
            mean_r2 = float(np.nanmean(fold_r2s))
            std_r2 = float(np.nanstd(fold_r2s))

            results[key] = {
                "config": config_name,
                "model": "gbt",
                "horizon": horizon,
                "fold_r2s": fold_r2s.tolist(),
                "mean_r2": mean_r2,
                "std_r2": std_r2,
            }
            if fold_importances is not None:
                importance_results[key] = fold_importances

            print(f"  {key}: R²={mean_r2:.6f} ± {std_r2:.6f}", flush=True)

            # Linear for select configs
            if config_name in linear_configs:
                key_lin = f"{config_name}_linear_h{horizon}"
                fold_r2s_lin, _ = fit_and_evaluate(X, y, splits, "linear")
                results[key_lin] = {
                    "config": config_name,
                    "model": "linear",
                    "horizon": horizon,
                    "fold_r2s": fold_r2s_lin.tolist(),
                    "mean_r2": float(np.nanmean(fold_r2s_lin)),
                    "std_r2": float(np.nanstd(fold_r2s_lin)),
                }
                print(f"  {key_lin}: R²={results[key_lin]['mean_r2']:.6f} ± {results[key_lin]['std_r2']:.6f}", flush=True)

    return results, importance_results


# ===========================================================================
# Tier 2 statistical analysis
# ===========================================================================
def compute_tier2_gaps(tier2_results):
    """Compute information gaps with paired tests and Holm-Bonferroni."""
    gaps = {}

    gap_definitions = {
        "delta_temporal_book": ("Book+Temporal_gbt", "Static-Book_gbt"),
        "delta_temporal_hc": ("HC+Temporal_gbt", "Static-HC_gbt"),
        "delta_temporal_only": ("Temporal-Only_gbt", None),  # vs. zero
        "delta_static_comparison": ("Static-Book_gbt", "Static-HC_gbt"),
    }

    for gap_name, (aug_key_base, base_key_base) in gap_definitions.items():
        raw_ps = []
        horizon_entries = []

        for horizon in HORIZONS:
            aug_key = f"{aug_key_base}_h{horizon}"
            aug_r2s = np.array(tier2_results[aug_key]["fold_r2s"])

            if base_key_base is not None:
                base_key = f"{base_key_base}_h{horizon}"
                base_r2s = np.array(tier2_results[base_key]["fold_r2s"])
                diffs = aug_r2s - base_r2s
                delta = float(np.mean(diffs))
                ci = confidence_interval_95(diffs)
                raw_p = paired_test(aug_r2s, base_r2s)
                d = cohens_d(aug_r2s, base_r2s)
            else:
                # vs. zero
                diffs = aug_r2s
                delta = float(np.mean(diffs))
                ci = confidence_interval_95(diffs)
                raw_p = one_sided_test_greater_than_zero(diffs)
                d = float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-15))

            raw_ps.append(raw_p)
            horizon_entries.append({
                "horizon": horizon,
                "delta_r2": delta,
                "ci_95": list(ci),
                "raw_p": raw_p,
                "cohens_d": d,
                "fold_diffs": diffs.tolist(),
            })

        # Holm-Bonferroni within this gap family (4 horizons)
        corrected_ps = holm_bonferroni(np.array(raw_ps))
        for i, entry in enumerate(horizon_entries):
            entry["corrected_p"] = float(corrected_ps[i])

        gaps[gap_name] = horizon_entries

    return gaps


def evaluate_threshold(gaps, tier2_results):
    """Evaluate decision rules.

    Threshold: delta > 20% of baseline R² AND corrected p < 0.05.
    Baseline = Static-Book GBT R².
    """
    threshold_results = {}
    for gap_name, entries in gaps.items():
        for entry in entries:
            h = entry["horizon"]
            baseline_key = f"Static-Book_gbt_h{h}"
            baseline_r2 = tier2_results[baseline_key]["mean_r2"]
            relative_threshold = 0.20 * abs(baseline_r2) if baseline_r2 != 0 else 0.001

            passes_relative = entry["delta_r2"] > relative_threshold
            passes_statistical = entry["corrected_p"] < 0.05
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
                "corrected_p": entry["corrected_p"],
                "ci_95": entry["ci_95"],
                "cohens_d": entry["cohens_d"],
            }

    return threshold_results


# ===========================================================================
# Feature importance analysis
# ===========================================================================
def analyze_feature_importance(importance_results):
    """Analyze feature importance for Book+Temporal and HC+Temporal configs."""
    analysis = {}
    for key, importances in importance_results.items():
        if "Book+Temporal" not in key and "HC+Temporal" not in key:
            continue

        # Sort by importance
        sorted_feats = sorted(importances.items(), key=lambda x: -x[1])
        top10 = sorted_feats[:10]

        # Categorize: temporal vs. static
        temporal_set = set(TEMPORAL_FEATURE_NAMES)
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
# Tier 1 pairwise comparisons (GBT vs Linear)
# ===========================================================================
def tier1_pairwise(tier1_results):
    """Pairwise tests for Tier 1: GBT vs. Linear within each lookback/horizon."""
    pairwise = {}
    for lookback in LOOKBACKS:
        for horizon in HORIZONS:
            gbt_key = f"AR-{lookback}_gbt_h{horizon}"
            lin_key = f"AR-{lookback}_linear_h{horizon}"
            gbt_r2s = np.array(tier1_results[gbt_key]["fold_r2s"])
            lin_r2s = np.array(tier1_results[lin_key]["fold_r2s"])
            p = paired_test(gbt_r2s, lin_r2s)
            d = cohens_d(gbt_r2s, lin_r2s)
            key = f"GBT_vs_Linear_AR-{lookback}_h{horizon}"
            pairwise[key] = {
                "lookback": lookback,
                "horizon": horizon,
                "gbt_mean_r2": float(np.nanmean(gbt_r2s)),
                "linear_mean_r2": float(np.nanmean(lin_r2s)),
                "delta": float(np.nanmean(gbt_r2s) - np.nanmean(lin_r2s)),
                "raw_p": p,
                "cohens_d": d,
            }

    # Holm-Bonferroni across all 12 comparisons
    keys = list(pairwise.keys())
    raw_ps = np.array([pairwise[k]["raw_p"] for k in keys])
    corrected = holm_bonferroni(raw_ps)
    for k, cp in zip(keys, corrected):
        pairwise[k]["corrected_p"] = float(cp)

    return pairwise


# ===========================================================================
# Decision rule evaluation
# ===========================================================================
def evaluate_decision_rules(tier1_results, tier2_gaps, threshold_results):
    """Evaluate the four decision rules from the spec."""
    rules = {}

    # RULE 1: Pure return AR — any AR GBT R² > 0 at h > 1?
    rule1_passes = []
    for lookback in LOOKBACKS:
        for horizon in [5, 20, 100]:
            key = f"AR-{lookback}_gbt_h{horizon}"
            r = tier1_results[key]
            if r["mean_r2"] > 0 and r.get("p_gt_zero_corrected", 1.0) < 0.05:
                rule1_passes.append(key)
    rules["rule1_ar_structure"] = {
        "passes": len(rule1_passes) > 0,
        "passing_configs": rule1_passes,
        "interpretation": "Genuine temporal structure exists" if rule1_passes else "Returns are martingale at horizons > 5s",
    }

    # RULE 2: Temporal augmentation passes dual threshold?
    rule2_passes = []
    for key, tr in threshold_results.items():
        if tr["gap"] == "delta_temporal_book" and tr["passes_dual"]:
            rule2_passes.append(key)
    rules["rule2_temporal_augmentation"] = {
        "passes": len(rule2_passes) > 0,
        "passing_horizons": rule2_passes,
        "interpretation": (
            "Temporal encoder justified" if rule2_passes
            else "Current-bar features sufficient. Drop temporal encoder/SSM."
        ),
    }

    # RULE 3: Temporal-only signal?
    rule3_passes = []
    for key, tr in threshold_results.items():
        if tr["gap"] == "delta_temporal_only" and tr["corrected_p"] < 0.05 and tr["delta_r2"] > 0:
            rule3_passes.append(key)
    rules["rule3_temporal_only"] = {
        "passes": len(rule3_passes) > 0,
        "passing_horizons": rule3_passes,
        "interpretation": (
            "Temporal features have standalone predictive power" if rule3_passes
            else "No temporal signal. Martingale confirmed."
        ),
    }

    # RULE 4: Reconciliation with R2
    r2_delta_temporal_negative = True  # R2 found Δ_temporal = -0.006
    if rules["rule2_temporal_augmentation"]["passes"]:
        rules["rule4_reconciliation"] = {
            "outcome": "R4_temporal_R2_negative",
            "interpretation": "Low-dim temporal features succeed where high-dim book concat failed. SSM should operate on derived features.",
        }
    else:
        rules["rule4_reconciliation"] = {
            "outcome": "converging_evidence",
            "interpretation": "R4 confirms R2: temporal encoder adds no value for MES at 5s bars. Strong recommendation to drop SSM.",
        }

    return rules


# ===========================================================================
# Analysis markdown generation
# ===========================================================================
def generate_analysis(tier1_results, tier2_results, tier2_gaps, threshold_results,
                      importance_analysis, tier1_pairwise_results, decision_rules):
    """Generate analysis.md."""
    lines = []
    lines.append(f"# R4: Temporal Predictability — Analysis [{BAR_LABEL}]\n")
    lines.append(f"**Date:** 2026-02-17  \n**Bar type:** {BAR_LABEL}\n")

    # Table 1: Tier 1
    lines.append("## Table 1: Tier 1 — Pure Return AR\n")
    lines.append("| Lookback | Model | return_1 | return_5 | return_20 | return_100 |")
    lines.append("|----------|-------|----------|----------|-----------|------------|")
    for lookback in LOOKBACKS:
        for model in ["linear", "ridge", "gbt"]:
            row = f"| AR-{lookback} | {model} |"
            for h in HORIZONS:
                key = f"AR-{lookback}_{model}_h{h}"
                r = tier1_results[key]
                cell = f" {r['mean_r2']:.6f} ± {r['std_r2']:.6f}"
                cp = r.get("p_gt_zero_corrected", 1.0)
                if r["mean_r2"] > 0 and cp < 0.05:
                    cell = f" **{r['mean_r2']:.6f} ± {r['std_r2']:.6f}**"
                row += cell + " |"
            lines.append(row)
    lines.append("")

    # Table 2: Tier 2
    lines.append("## Table 2: Tier 2 — Temporal Feature Augmentation (GBT)\n")
    lines.append("| Config | return_1 | return_5 | return_20 | return_100 |")
    lines.append("|--------|----------|----------|-----------|------------|")
    for config in ["Static-Book", "Static-HC", "Book+Temporal", "HC+Temporal", "Temporal-Only"]:
        row = f"| {config} |"
        for h in HORIZONS:
            key = f"{config}_gbt_h{h}"
            r = tier2_results[key]
            row += f" {r['mean_r2']:.6f} ± {r['std_r2']:.6f} |"
        lines.append(row)
    lines.append("")

    # Table 3: Information Gaps
    lines.append("## Table 3: Information Gaps (GBT, Tier 2)\n")
    lines.append("| Gap | Horizon | Δ_R² | 95% CI | Raw p | Corrected p | Passes? |")
    lines.append("|-----|---------|------|--------|-------|-------------|---------|")
    for gap_name in ["delta_temporal_book", "delta_temporal_hc", "delta_temporal_only", "delta_static_comparison"]:
        for entry in tier2_gaps[gap_name]:
            h = entry["horizon"]
            key = f"{gap_name}_h{h}"
            tr = threshold_results.get(key, {})
            passes = "YES" if tr.get("passes_dual", False) else "no"
            ci = entry["ci_95"]
            lines.append(
                f"| {gap_name} | {h} | {entry['delta_r2']:.6f} | "
                f"[{ci[0]:.6f}, {ci[1]:.6f}] | {entry['raw_p']:.4f} | "
                f"{entry['corrected_p']:.4f} | {passes} |"
            )
    lines.append("")

    # Table 4: Feature Importance
    lines.append("## Table 4: Feature Importance (GBT, Fold 5)\n")
    for key, analysis in importance_analysis.items():
        lines.append(f"### {key}\n")
        lines.append("| Rank | Feature | Importance | Category |")
        lines.append("|------|---------|------------|----------|")
        for entry in analysis["top10"]:
            lines.append(f"| {entry['rank']} | {entry['feature']} | {entry['importance']:.6f} | {entry['category']} |")
        lines.append(f"\nTemporal feature share: {analysis['temporal_importance_fraction']*100:.1f}% of total importance\n")
    lines.append("")

    # Tier 1 pairwise: GBT vs Linear
    lines.append("## Tier 1 Pairwise: GBT vs. Linear\n")
    lines.append("| Lookback | Horizon | GBT R² | Linear R² | Δ | Corrected p | Cohen's d |")
    lines.append("|----------|---------|--------|-----------|---|-------------|-----------|")
    for lookback in LOOKBACKS:
        for h in HORIZONS:
            key = f"GBT_vs_Linear_AR-{lookback}_h{h}"
            r = tier1_pairwise_results[key]
            lines.append(
                f"| AR-{lookback} | {h} | {r['gbt_mean_r2']:.6f} | {r['linear_mean_r2']:.6f} | "
                f"{r['delta']:.6f} | {r['corrected_p']:.4f} | {r['cohens_d']:.3f} |"
            )
    lines.append("")

    # Decision Rules
    lines.append("## Decision Rules\n")
    for rule_name, rule in decision_rules.items():
        lines.append(f"### {rule_name}")
        lines.append(f"- **Passes**: {rule.get('passes', 'N/A')}")
        lines.append(f"- **Interpretation**: {rule.get('interpretation', rule.get('outcome', 'N/A'))}")
        if "passing_configs" in rule:
            lines.append(f"- **Passing configs**: {rule['passing_configs']}")
        if "passing_horizons" in rule:
            lines.append(f"- **Passing horizons**: {rule['passing_horizons']}")
        lines.append("")

    # Summary Finding
    lines.append("## Summary Finding\n")
    rule2 = decision_rules["rule2_temporal_augmentation"]
    rule3 = decision_rules["rule3_temporal_only"]
    if rule2["passes"]:
        lines.append("**TEMPORAL SIGNAL**: Temporal features pass the dual threshold. SSM/temporal encoder justified.")
    elif rule3["passes"]:
        lines.append("**MARGINAL SIGNAL**: Positive R² in temporal-only but augmentation fails threshold. Temporal structure exists but is redundant with static features.")
    else:
        lines.append("**NO TEMPORAL SIGNAL**: Returns are a martingale difference sequence at the 5s scale. Drop SSM. Converges with R2 Δ_temporal finding.")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================
def main():
    global INPUT_CSV, RESULTS_DIR, BAR_LABEL

    parser = argparse.ArgumentParser(description="R4: Temporal Predictability")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="Input feature CSV path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for results")
    parser.add_argument("--bar-label", default=DEFAULT_BAR_LABEL, help="Bar type label for reporting")
    args = parser.parse_args()

    INPUT_CSV = Path(args.input_csv)
    RESULTS_DIR = Path(args.output_dir)
    BAR_LABEL = args.bar_label
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"R4: Temporal Predictability [{BAR_LABEL}]", flush=True)
    print("=" * 70, flush=True)

    # Load and prepare data
    df = load_features()
    df = construct_temporal_features(df)

    # Tier 1
    tier1_results = run_tier1(df)
    tier1_pw = tier1_pairwise(tier1_results)

    # Tier 2
    tier2_results, importance_raw = run_tier2(df)
    tier2_gaps = compute_tier2_gaps(tier2_results)
    threshold_results = evaluate_threshold(tier2_gaps, tier2_results)
    importance_analysis = analyze_feature_importance(importance_raw)

    # Decision rules
    decision_rules = evaluate_decision_rules(tier1_results, tier2_gaps, threshold_results)

    # Generate analysis
    analysis_md = generate_analysis(
        tier1_results, tier2_results, tier2_gaps, threshold_results,
        importance_analysis, tier1_pw, decision_rules,
    )

    # Write analysis.md
    analysis_path = RESULTS_DIR / "analysis.md"
    analysis_path.write_text(analysis_md)
    print(f"\nAnalysis written to {analysis_path}", flush=True)

    # Build metrics.json
    metrics = {
        "experiment": f"R4-temporal-predictability-{BAR_LABEL}",
        "date": "2026-02-17",
        "bar_label": BAR_LABEL,
        "data": {
            "input": str(INPUT_CSV),
            "total_bars_loaded": len(df),
        },
        "tier1": {
            "results": {k: {kk: vv for kk, vv in v.items()} for k, v in tier1_results.items()},
            "pairwise_gbt_vs_linear": tier1_pw,
        },
        "tier2": {
            "results": {k: {kk: vv for kk, vv in v.items()} for k, v in tier2_results.items()},
            "information_gaps": {k: v for k, v in tier2_gaps.items()},
            "threshold_evaluation": threshold_results,
            "feature_importance": importance_analysis,
        },
        "decision_rules": decision_rules,
        "summary": {
            "finding": (
                "TEMPORAL SIGNAL" if decision_rules["rule2_temporal_augmentation"]["passes"]
                else ("MARGINAL SIGNAL" if decision_rules["rule3_temporal_only"]["passes"]
                      else "NO TEMPORAL SIGNAL")
            ),
            "recommendation": decision_rules["rule2_temporal_augmentation"]["interpretation"],
            "reconciliation_with_R2": decision_rules["rule4_reconciliation"]["interpretation"],
        },
    }

    metrics_path = RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"Metrics written to {metrics_path}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"R4 COMPLETE: {metrics['summary']['finding']}", flush=True)
    print(f"Recommendation: {metrics['summary']['recommendation']}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
