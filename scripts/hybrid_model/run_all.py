"""End-to-end: export -> train -> evaluate.

Usage:
    python -m scripts.hybrid_model.run_all [--csv PATH] [--skip-export]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from . import config
from .evaluate_cv import run_full_cv


def main():
    parser = argparse.ArgumentParser(description="CNN+GBT Hybrid model pipeline")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to pre-exported CSV (skip C++ export)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip C++ data export step")
    args = parser.parse_args()

    csv_path = args.csv or str(config.DATA_CSV)

    if not args.skip_export and not args.csv:
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print("Running C++ data export...", flush=True)
            csv_file.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run([
                str(config.PROJECT_ROOT / "build" / "bar_feature_export"),
                "--bar-type", "time",
                "--bar-param", "5",
                "--output", csv_path,
            ], check=True)
            print("Export complete.", flush=True)
        else:
            print(f"CSV already exists: {csv_path}", flush=True)

    results = run_full_cv(csv_path)

    # Write analysis document
    _write_analysis(results)

    return results


def _write_analysis(results):
    """Write analysis.md summarizing results."""
    out_path = config.RESULTS_DIR / "analysis.md"

    lines = ["# CNN+GBT Hybrid Model — Analysis\n"]
    lines.append(f"## CNN Regression R²\n")
    lines.append(f"- h=5 mean: {results['mean_cnn_r2_h5']:.6f} ± {results['std_cnn_r2_h5']:.6f}")
    lines.append(f"- h=1 mean: {results['mean_cnn_r2_h1']:.6f} ± {results['std_cnn_r2_h1']:.6f}")
    lines.append(f"- Negative R² folds (h=5): {results['negative_fold_count_h5']}\n")

    lines.append(f"## XGBoost Classification\n")
    lines.append(f"- Mean accuracy: {results['mean_xgb_accuracy']:.4f} ± {results['std_xgb_accuracy']:.4f}")
    lines.append(f"- Mean F1 macro: {results['mean_xgb_f1_macro']:.4f}\n")

    lines.append("## Cost Sensitivity\n")
    lines.append("| Scenario | RT Cost | Expectancy | PF |")
    lines.append("|----------|---------|------------|-----|")
    for name, data in results.get("cost_sensitivity", {}).items():
        lines.append(f"| {name} | ${data['rt_cost']:.2f} | ${data['mean_expectancy']:.2f} | {data['mean_profit_factor']:.2f} |")
    lines.append("")

    lines.append("## Per-Fold Results (h=5)\n")
    lines.append("| Fold | CNN R² | XGB Acc | Base Exp | Base PF |")
    lines.append("|------|--------|---------|----------|---------|")
    for f in results.get("per_fold_h5", []):
        base = f.get("pnl", {}).get("base", {})
        lines.append(f"| {f['fold']} | {f['cnn_r2']:.6f} | {f['xgb_accuracy']:.4f} | ${base.get('expectancy', 0):.2f} | {base.get('profit_factor', 0):.2f} |")
    lines.append("")

    with open(out_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"Analysis written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
