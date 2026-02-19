#!/usr/bin/env python3
"""R4d Phase 3: Temporal analysis for a single operating point.

Usage:
  python research/R4d_phase3_single.py \
    --input-csv .kit/results/.../features.csv \
    --output-dir .kit/results/.../dollar_10M \
    --bar-label dollar_10M \
    --bar-type dollar \
    --threshold 10000000 \
    --median-duration 13.9 \
    --horizons 1 5 20 100
"""

import argparse
import json
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

CV_FOLDS = [
    (list(range(1, 5)),   list(range(5, 9))),
    (list(range(1, 9)),   list(range(9, 12))),
    (list(range(1, 12)),  list(range(12, 15))),
    (list(range(1, 15)),  list(range(15, 18))),
    (list(range(1, 18)),  list(range(18, 20))),
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


def load_features(csv_path):
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


def prepare_data(df, lookback_depth):
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
            return pl.DataFrame()
        df_valid = pl.concat(day_frames)
    return df_valid


def get_cv_splits(df_valid, folds=CV_FOLDS):
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


def paired_test(a, b):
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
    n = len(values)
    if n < 2:
        return 1.0
    t_stat, p_two = stats.ttest_1samp(values, 0)
    if t_stat > 0:
        return float(p_two / 2)
    else:
        return 1.0


def holm_bonferroni(p_values):
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
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    mean = np.mean(values)
    se = stats.sem(values)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (float(mean - t_crit * se), float(mean + t_crit * se))


def cohens_d(a, b):
    diffs = a - b
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-15))


def compute_gaps(tier2_results, horizons):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bar-label", required=True)
    parser.add_argument("--bar-type", required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--median-duration", type=float, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20, 100])
    args = parser.parse_args()

    t_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = args.bar_label
    horizons = args.horizons

    print(f"R4d Phase 3: {label}", flush=True)
    print(f"Horizons: {horizons}", flush=True)

    # Load and prepare
    df = load_features(args.input_csv)
    df = construct_temporal_features(df)

    bars_per_day = len(df) / max(1, df["day"].n_unique())
    if bars_per_day < 120:
        lookback_depth = min(20, max(5, int(bars_per_day * 0.3)))
        print(f"Sparse bars ({bars_per_day:.0f}/day): lookback={lookback_depth}", flush=True)
    else:
        lookback_depth = 100

    df_valid = prepare_data(df, lookback_depth)
    if len(df_valid) == 0:
        print("SKIP: zero valid bars", flush=True)
        result = {"status": "insufficient_data", "n_valid": 0, "n_bars_loaded": len(df)}
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    splits = get_cv_splits(df_valid)
    n_valid = len(df_valid)
    n_folds = len(splits)
    print(f"Valid bars: {n_valid}, CV folds: {n_folds}", flush=True)

    if n_valid < 100 or n_folds < 2:
        print(f"SKIP: insufficient data ({n_valid} valid, {n_folds} folds)", flush=True)
        result = {"status": "insufficient_data", "n_valid": n_valid, "n_folds": n_folds}
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        return

    configs = {
        "Static-Book": BOOK_SNAP_FEATURES,
        "Book+Temporal": BOOK_SNAP_FEATURES + TEMPORAL_FEATURE_NAMES,
        "Temporal-Only": TEMPORAL_FEATURE_NAMES,
    }

    # Tier 1: AR-10 GBT
    print("\nTier 1: AR-10 GBT...", flush=True)
    tier1_results = {}
    lookback = 10
    ar_features = [f"lag_return_{i}" for i in range(1, lookback + 1)]
    available_ar = [c for c in ar_features if c in df_valid.columns]
    X_ar = df_valid.select(available_ar).to_numpy().astype(np.float64)

    for h in horizons:
        target_col = f"fwd_return_{h}"
        if target_col not in df_valid.columns:
            print(f"  AR-10_gbt_h{h}: target missing, skip", flush=True)
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
            "lookback": lookback, "horizon": h,
            "fold_r2s": fold_r2s.tolist(),
            "mean_r2": mean_r2, "std_r2": std_r2,
            "p_gt_zero_raw": p_gt_zero, "elapsed_s": round(elapsed, 1),
        }
        print(f"  {key}: R2={mean_r2:.6f} +/- {std_r2:.6f} [{elapsed:.1f}s]", flush=True)
        if mean_r2 > 0.01:
            print(f"  WARNING: R2 > 0.01!", flush=True)
            tier1_results[key]["abort_flag"] = True

    # Holm-Bonferroni for Tier 1
    family_keys = list(tier1_results.keys())
    if family_keys:
        raw_ps = np.array([tier1_results[k]["p_gt_zero_raw"] for k in family_keys])
        corrected_ps = holm_bonferroni(raw_ps)
        for k, cp in zip(family_keys, corrected_ps):
            tier1_results[k]["p_gt_zero_corrected"] = float(cp)

    # Tier 2: Feature configs
    print("\nTier 2: Feature configs...", flush=True)
    tier2_results = {}
    importance_results = {}

    for config_name, feature_cols in configs.items():
        available_cols = [c for c in feature_cols if c in df_valid.columns]
        if not available_cols:
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
                "config": config_name, "horizon": h,
                "fold_r2s": fold_r2s.tolist(),
                "mean_r2": mean_r2, "std_r2": std_r2,
                "elapsed_s": round(elapsed, 1),
            }
            if fold_imp is not None:
                importance_results[key] = fold_imp
            print(f"  {key}: R2={mean_r2:.6f} +/- {std_r2:.6f} [{elapsed:.1f}s]", flush=True)

    # Gaps and dual threshold
    gaps = compute_gaps(tier2_results, horizons)
    threshold_eval = evaluate_dual_threshold(gaps, tier2_results, horizons)
    importance_analysis = analyze_importance(importance_results)

    # Fold sign agreement
    fold_sign_agreement = {}
    for key, result in tier1_results.items():
        r2s = np.array(result["fold_r2s"])
        n_positive = int(np.sum(r2s > 0))
        n_total = len(r2s)
        fold_sign_agreement[key] = {
            "n_positive": n_positive, "n_total": n_total,
            "all_agree_sign": n_positive == 0 or n_positive == n_total,
            "p_r2_gt_0": n_positive / n_total if n_total > 0 else 0,
        }

    op_metrics = {
        "operating_point": label,
        "bar_type": args.bar_type,
        "threshold": args.threshold,
        "median_duration_s": args.median_duration,
        "horizons": horizons,
        "horizon_restriction": None if len(horizons) == 4 else f"restricted to {horizons}",
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

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(op_metrics, f, indent=2, default=str)
    elapsed_total = time.time() - t_start
    print(f"\nMetrics written to {metrics_path} ({elapsed_total:.1f}s)", flush=True)

    # Summary
    total_dual_tests = len(threshold_eval)
    dual_passes = sum(1 for v in threshold_eval.values() if v.get("passes_dual", False))
    print(f"Dual threshold: {dual_passes}/{total_dual_tests} passes", flush=True)


if __name__ == "__main__":
    main()
