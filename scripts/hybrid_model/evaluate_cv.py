"""Full 5-fold expanding window cross-validation pipeline."""

import json
import os
import sys

import numpy as np
import torch

from . import config
from .data_loader import (
    load_csv, split_fold, extract_book_arrays,
    normalize_book_sizes, extract_non_spatial, normalize_features,
)
from .cnn_encoder import CNNWithHead
from .train_cnn import train_cnn_regression, train_cnn_classifier, set_seed
from .extract_embeddings import extract_embeddings
from .train_xgboost import train_xgboost, predict_xgboost
from .metrics import r_squared, classification_metrics, compute_pnl


def run_fold(df, fold_idx, horizon="h5"):
    """Run full pipeline for a single fold.

    Args:
        df: full dataframe
        fold_idx: 0-based fold index
        horizon: "h1" or "h5"

    Returns:
        dict with all per-fold metrics + predictions
    """
    train_df, test_df = split_fold(df, fold_idx)
    N_train = len(train_df)
    N_test = len(test_df)

    # Combine for normalized array extraction
    combined_df = train_df._append(test_df, ignore_index=True)
    train_mask = np.zeros(len(combined_df), dtype=bool)
    train_mask[:N_train] = True

    # Book arrays
    book_all = extract_book_arrays(combined_df)
    book_all = normalize_book_sizes(book_all, train_mask)
    train_book = book_all[:N_train]
    test_book = book_all[N_train:]

    # Forward return targets
    fwd_col = config.FWD_RETURN_H1 if horizon == "h1" else config.FWD_RETURN_H5
    train_targets = train_df[fwd_col].values.astype(np.float32)
    test_targets = test_df[fwd_col].values.astype(np.float32)

    # Use 10% of training data as validation for early stopping
    val_size = max(1, int(N_train * 0.1))
    cnn_train_book = train_book[:-val_size]
    cnn_val_book = train_book[-val_size:]
    cnn_train_targets = train_targets[:-val_size]
    cnn_val_targets = train_targets[-val_size:]

    # Stage 1: CNN encoder training
    model, history = train_cnn_regression(
        cnn_train_book, cnn_train_targets,
        cnn_val_book, cnn_val_targets,
    )

    # CNN R-squared on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(test_book, dtype=torch.float32)).numpy()
    cnn_r2 = r_squared(test_targets, test_preds)

    # Stage 2: Extract embeddings
    train_emb = extract_embeddings(model, train_book)
    test_emb = extract_embeddings(model, test_book)

    # Non-spatial features
    ns_all = extract_non_spatial(combined_df)
    ns_all, ns_mean, ns_std = normalize_features(ns_all, train_mask)
    train_ns = ns_all[:N_train]
    test_ns = ns_all[N_train:]

    # Combined features for XGBoost
    train_X = np.concatenate([train_emb, train_ns], axis=1)
    test_X = np.concatenate([test_emb, test_ns], axis=1)

    # TB labels
    train_labels = train_df[config.TB_LABEL_COL].values
    test_labels = test_df[config.TB_LABEL_COL].values

    # Stage 2: XGBoost
    xgb_model = train_xgboost(train_X, train_labels)
    test_preds_xgb = predict_xgboost(xgb_model, test_X)

    # Classification metrics
    cls_metrics = classification_metrics(test_labels, test_preds_xgb)

    # PnL metrics under each cost scenario
    pnl_results = {}
    for scenario_name in config.COST_SCENARIOS:
        cost = config.round_trip_cost(scenario_name)
        pnl = compute_pnl(
            predictions=test_preds_xgb,
            tb_labels=test_labels,
            tb_exit_types=test_df[config.TB_EXIT_TYPE_COL].values,
            tb_bars_held=test_df[config.TB_BARS_HELD_COL].values,
            entry_mids=None,
            bars_df=test_df,
            cost_per_trade=cost,
        )
        pnl_results[scenario_name] = {
            "expectancy": pnl["expectancy"],
            "profit_factor": pnl["profit_factor"],
            "trade_count": pnl["trade_count"],
            "total_pnl": pnl["total_pnl"],
            "sharpe": pnl["sharpe"],
        }

    return {
        "fold": fold_idx + 1,
        "horizon": horizon,
        "n_train": N_train,
        "n_test": N_test,
        "cnn_r2": float(cnn_r2),
        "cnn_epochs": len(history["train_loss"]),
        "xgb_accuracy": cls_metrics["accuracy"],
        "xgb_f1_macro": cls_metrics["f1_macro"],
        "pnl": pnl_results,
        "test_predictions": test_preds_xgb.tolist(),
        "test_labels": test_labels.tolist(),
        "model_state": model,
        "xgb_model": xgb_model,
    }


def run_ablation_gbt_only(df):
    """GBT-only baseline: XGBoost on all 62 Track A features (no CNN)."""
    results = []
    for fold_idx in range(len(config.FOLDS)):
        train_df, test_df = split_fold(df, fold_idx)
        N_train = len(train_df)

        combined_df = train_df._append(test_df, ignore_index=True)
        train_mask = np.zeros(len(combined_df), dtype=bool)
        train_mask[:N_train] = True

        # All 62 Track A features
        features_all = combined_df[config.TRACK_A_FEATURES].values.astype(np.float32)
        features_all, _, _ = normalize_features(features_all, train_mask)
        train_X = features_all[:N_train]
        test_X = features_all[N_train:]

        train_labels = train_df[config.TB_LABEL_COL].values
        test_labels = test_df[config.TB_LABEL_COL].values

        xgb_model = train_xgboost(train_X, train_labels)
        preds = predict_xgboost(xgb_model, test_X)

        cls = classification_metrics(test_labels, preds)

        cost = config.round_trip_cost("base")
        pnl = compute_pnl(
            predictions=preds,
            tb_labels=test_labels,
            tb_exit_types=test_df[config.TB_EXIT_TYPE_COL].values,
            tb_bars_held=test_df[config.TB_BARS_HELD_COL].values,
            entry_mids=None, bars_df=test_df,
            cost_per_trade=cost,
        )

        results.append({
            "fold": fold_idx + 1,
            "accuracy": cls["accuracy"],
            "f1_macro": cls["f1_macro"],
            "expectancy": pnl["expectancy"],
            "profit_factor": pnl["profit_factor"],
            "trade_count": pnl["trade_count"],
        })

    return {
        "folds": results,
        "mean_accuracy": float(np.mean([r["accuracy"] for r in results])),
        "mean_expectancy": float(np.mean([r["expectancy"] for r in results])),
        "mean_profit_factor": float(np.mean([r["profit_factor"] for r in results])),
    }


def run_ablation_cnn_only(df):
    """CNN-only baseline: CNN classifier directly on TB labels (no XGBoost)."""
    results = []
    for fold_idx in range(len(config.FOLDS)):
        train_df, test_df = split_fold(df, fold_idx)
        N_train = len(train_df)

        combined_df = train_df._append(test_df, ignore_index=True)
        train_mask = np.zeros(len(combined_df), dtype=bool)
        train_mask[:N_train] = True

        book_all = extract_book_arrays(combined_df)
        book_all = normalize_book_sizes(book_all, train_mask)
        train_book = book_all[:N_train]
        test_book = book_all[N_train:]

        train_labels = train_df[config.TB_LABEL_COL].values
        test_labels = test_df[config.TB_LABEL_COL].values

        model = train_cnn_classifier(train_book, train_labels, test_book, test_labels)

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(test_book, dtype=torch.float32))
            preds_mapped = logits.argmax(dim=1).numpy()

        # Map back: {0: -1, 1: 0, 2: +1}
        inv_map = {0: -1, 1: 0, 2: 1}
        preds = np.array([inv_map[int(p)] for p in preds_mapped])

        cls = classification_metrics(test_labels, preds)

        cost = config.round_trip_cost("base")
        pnl = compute_pnl(
            predictions=preds,
            tb_labels=test_labels,
            tb_exit_types=test_df[config.TB_EXIT_TYPE_COL].values,
            tb_bars_held=test_df[config.TB_BARS_HELD_COL].values,
            entry_mids=None, bars_df=test_df,
            cost_per_trade=cost,
        )

        results.append({
            "fold": fold_idx + 1,
            "accuracy": cls["accuracy"],
            "f1_macro": cls["f1_macro"],
            "expectancy": pnl["expectancy"],
            "profit_factor": pnl["profit_factor"],
            "trade_count": pnl["trade_count"],
        })

    return {
        "folds": results,
        "mean_accuracy": float(np.mean([r["accuracy"] for r in results])),
        "mean_expectancy": float(np.mean([r["expectancy"] for r in results])),
        "mean_profit_factor": float(np.mean([r["profit_factor"] for r in results])),
    }


def run_full_cv(csv_path=None):
    """Run the full 5-fold CV pipeline with ablations.

    Returns:
        aggregate results dict
    """
    set_seed()
    print("Loading data...", flush=True)
    df = load_csv(csv_path)
    print(f"  Loaded {len(df)} bars from {df['day'].nunique()} days", flush=True)

    # Label distribution check
    label_counts = df[config.TB_LABEL_COL].value_counts()
    total = len(df)
    print(f"  TB label distribution: {dict(label_counts)}", flush=True)
    for lbl, cnt in label_counts.items():
        frac = cnt / total
        if frac > 0.60:
            print(f"  WARNING: class {lbl} is {frac:.1%} of total (>60%)", flush=True)

    # Run folds for h5 (primary) and h1 (report-only)
    fold_results_h5 = []
    fold_results_h1 = []

    for fold_idx in range(len(config.FOLDS)):
        print(f"\n=== Fold {fold_idx + 1}/{len(config.FOLDS)} ===", flush=True)

        # h5 (primary horizon)
        print(f"  Training CNN (h=5)...", flush=True)
        res_h5 = run_fold(df, fold_idx, horizon="h5")
        print(f"  CNN R² (h=5): {res_h5['cnn_r2']:.6f}", flush=True)
        print(f"  XGB accuracy: {res_h5['xgb_accuracy']:.4f}", flush=True)
        print(f"  XGB F1 macro: {res_h5['xgb_f1_macro']:.4f}", flush=True)
        print(f"  Base expectancy: ${res_h5['pnl']['base']['expectancy']:.2f}/trade", flush=True)
        fold_results_h5.append(res_h5)

        # h1 (report-only)
        print(f"  Training CNN (h=1)...", flush=True)
        res_h1 = run_fold(df, fold_idx, horizon="h1")
        print(f"  CNN R² (h=1): {res_h1['cnn_r2']:.6f}", flush=True)
        fold_results_h1.append(res_h1)

        # Save per-fold results
        fold_dir = config.RESULTS_DIR / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        _save_json(fold_dir / "cnn_h5_metrics.json", {
            "r2": res_h5["cnn_r2"], "epochs": res_h5["cnn_epochs"],
        })
        _save_json(fold_dir / "cnn_h1_metrics.json", {
            "r2": res_h1["cnn_r2"], "epochs": res_h1["cnn_epochs"],
        })
        _save_json(fold_dir / "xgb_metrics.json", {
            "accuracy": res_h5["xgb_accuracy"],
            "f1_macro": res_h5["xgb_f1_macro"],
            "pnl": res_h5["pnl"],
        })

        # Save CNN weights
        torch.save(res_h5["model_state"].state_dict(), fold_dir / "cnn_encoder_h5.pt")

        # Save XGBoost model
        res_h5["xgb_model"].save_model(str(fold_dir / "xgb_model.json"))

    # Aggregate metrics
    h5_r2s = [r["cnn_r2"] for r in fold_results_h5]
    h1_r2s = [r["cnn_r2"] for r in fold_results_h1]
    accuracies = [r["xgb_accuracy"] for r in fold_results_h5]
    f1s = [r["xgb_f1_macro"] for r in fold_results_h5]

    # Pool all test predictions for aggregate PnL
    all_preds = np.concatenate([np.array(r["test_predictions"]) for r in fold_results_h5])
    all_labels = np.concatenate([np.array(r["test_labels"]) for r in fold_results_h5])

    aggregate = {
        "mean_cnn_r2_h5": float(np.mean(h5_r2s)),
        "std_cnn_r2_h5": float(np.std(h5_r2s)),
        "mean_cnn_r2_h1": float(np.mean(h1_r2s)),
        "std_cnn_r2_h1": float(np.std(h1_r2s)),
        "negative_fold_count_h5": int(sum(1 for r in h5_r2s if r < 0)),
        "mean_xgb_accuracy": float(np.mean(accuracies)),
        "std_xgb_accuracy": float(np.std(accuracies)),
        "mean_xgb_f1_macro": float(np.mean(f1s)),
        "per_fold_h5": [{
            "fold": r["fold"], "cnn_r2": r["cnn_r2"],
            "xgb_accuracy": r["xgb_accuracy"], "xgb_f1_macro": r["xgb_f1_macro"],
            "pnl": r["pnl"],
        } for r in fold_results_h5],
        "per_fold_h1": [{"fold": r["fold"], "cnn_r2": r["cnn_r2"]} for r in fold_results_h1],
    }

    # Cost sensitivity
    cost_sensitivity = {}
    for scenario_name in config.COST_SCENARIOS:
        scenario_expectations = [r["pnl"][scenario_name]["expectancy"] for r in fold_results_h5]
        scenario_pfs = [r["pnl"][scenario_name]["profit_factor"] for r in fold_results_h5]
        cost_sensitivity[scenario_name] = {
            "mean_expectancy": float(np.mean(scenario_expectations)),
            "mean_profit_factor": float(np.mean(scenario_pfs)),
            "rt_cost": config.round_trip_cost(scenario_name),
        }
    aggregate["cost_sensitivity"] = cost_sensitivity

    print("\n=== Ablation: GBT-only baseline ===", flush=True)
    gbt_only = run_ablation_gbt_only(df)
    print(f"  Mean accuracy: {gbt_only['mean_accuracy']:.4f}", flush=True)
    print(f"  Mean expectancy: ${gbt_only['mean_expectancy']:.2f}/trade", flush=True)

    print("\n=== Ablation: CNN-only baseline ===", flush=True)
    cnn_only = run_ablation_cnn_only(df)
    print(f"  Mean accuracy: {cnn_only['mean_accuracy']:.4f}", flush=True)
    print(f"  Mean expectancy: ${cnn_only['mean_expectancy']:.2f}/trade", flush=True)

    # Save results
    _save_json(config.RESULTS_DIR / "aggregate_metrics.json", aggregate)
    _save_json(config.RESULTS_DIR / "ablation_gbt_only.json", gbt_only)
    _save_json(config.RESULTS_DIR / "ablation_cnn_only.json", cnn_only)
    _save_json(config.RESULTS_DIR / "cost_sensitivity.json", cost_sensitivity)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  CNN R² (h=5): {aggregate['mean_cnn_r2_h5']:.6f} ± {aggregate['std_cnn_r2_h5']:.6f}", flush=True)
    print(f"  CNN R² (h=1): {aggregate['mean_cnn_r2_h1']:.6f} ± {aggregate['std_cnn_r2_h1']:.6f}", flush=True)
    print(f"  Negative R² folds (h=5): {aggregate['negative_fold_count_h5']}", flush=True)
    print(f"  XGB accuracy: {aggregate['mean_xgb_accuracy']:.4f} ± {aggregate['std_xgb_accuracy']:.4f}", flush=True)
    print(f"  Base expectancy: ${cost_sensitivity['base']['mean_expectancy']:.2f}/trade", flush=True)
    print(f"  Base PF: {cost_sensitivity['base']['mean_profit_factor']:.2f}", flush=True)
    print(f"  GBT-only accuracy: {gbt_only['mean_accuracy']:.4f}", flush=True)
    print(f"  CNN-only accuracy: {cnn_only['mean_accuracy']:.4f}", flush=True)

    return aggregate


def _save_json(path, data):
    """Save data as JSON, handling numpy types."""
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_cv(csv_path)
