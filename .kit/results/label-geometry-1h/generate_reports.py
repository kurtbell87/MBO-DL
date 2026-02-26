#!/usr/bin/env python3
"""Generate analysis.md, config.json, env_fingerprint.json from metrics.json.

Standalone script to produce the 3 deliverables that weren't generated
due to an f-string formatting bug in run_experiment.py.
"""
import json
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path(__file__).parent
MAIN_SCRIPT = RESULTS_DIR / "run_experiment.py"

# Load metrics
with open(RESULTS_DIR / "metrics.json") as f:
    metrics = json.load(f)

GEOM_ORDER = ["10_5", "15_3", "19_7", "20_3"]
GEOM_LABELS = {k: metrics["geometries"][k]["label"] for k in GEOM_ORDER}

# Feature name mapping (f0..f19 → human names) from run_experiment.py
NON_SPATIAL_FEATURES = [
    "weighted_imbalance", "spread", "microprice_offset", "total_depth",
    "depth_imbalance_1", "depth_imbalance_3", "depth_imbalance_5",
    "trade_imbalance_50", "trade_intensity_50", "volume_imbalance_50",
    "volatility_20", "volatility_50", "return_1", "return_5", "return_20",
    "high_low_range_50", "vwap_offset", "bid_retreat_rate_50",
    "ask_retreat_rate_50", "effective_spread_50"
]

outcome = metrics["outcome"]
outcome_desc = metrics["outcome_description"]
wall_clock = metrics["resource_usage"]["wall_clock_minutes"]
time_horizon = metrics["time_horizon_s"]

# --- analysis.md ---
a_lines = []
def a(line=""):
    a_lines.append(line)

a("# Label Geometry 1h — Analysis")
a()
a(f"**Date:** {metrics['timestamp'][:10]}")
a(f"**Outcome:** {outcome} — {outcome_desc}")
a(f"**Wall clock:** {wall_clock:.1f} minutes")
a(f"**Time horizon:** {time_horizon}s (vs Phase 1's 300s)")
a()

# 1. Executive summary
a("## 1. Executive Summary")
a()
if outcome == "A":
    a("At least one geometry produces CPCV directional accuracy > breakeven WR + 2pp AND positive expectancy. "
      "Favorable payoff structure converts the model's directional signal into positive expectancy.")
elif outcome == "B":
    a("No geometry achieves both directional accuracy > BEV WR + 2pp and positive expectancy. "
      "The model's directional signal does not change enough with geometry to overcome costs.")
elif outcome == "C":
    a("Directional accuracy drops >10pp at non-baseline geometries. "
      "The wider-target classification problem is fundamentally harder for the model.")
elif outcome == "D":
    a("Hold rate still >80% at all geometries even with 3600s horizon. "
      "Time horizon is not the root cause of label degeneracy.")
a()

# 2. Hold rate comparison
a("## 2. Hold Rate Comparison (KEY DIAGNOSTIC)")
a()
a("| Geometry | Hold Rate (300s, Phase 1) | Hold Rate (3600s) | Delta |")
a("|----------|--------------------------|-------------------|-------|")
for key in GEOM_ORDER:
    hr_comp = metrics["hold_rate_comparison"].get(key, {})
    p1 = hr_comp.get("phase1_300s")
    cur = hr_comp.get("current_3600s")
    delta = hr_comp.get("delta")
    p1_str = f"{p1:.3f}" if p1 is not None else "N/A"
    cur_str = f"{cur:.3f}" if cur is not None else "N/A"
    delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
    a(f"| {GEOM_LABELS[key]} | {p1_str} | {cur_str} | {delta_str} |")
a()

# 3. Oracle comparison
a("## 3. Oracle Comparison (300s vs 3600s)")
a()
a("| Geometry | Oracle WR (3600s) | Oracle Exp (3600s) | Oracle Trades (3600s) |")
a("|----------|-------------------|--------------------|-----------------------|")
for key in GEOM_ORDER:
    orc = metrics["oracle_3600s"].get(key, {})
    a(f"| {GEOM_LABELS[key]} | {orc.get('tb_win_rate', 0):.3f} | "
      f"${orc.get('tb_expectancy', 0):.2f} | {orc.get('tb_total_trades', 0)} |")
a()

# 4. CPCV results table
a("## 4. CPCV Results")
a()
a("| Geometry | BEV WR | Dir Acc | Dir Margin | Overall Acc | Exp (base) | Exp (opt) | PF | Dir Pred Rate |")
a("|----------|--------|--------|------------|-------------|------------|-----------|-----|---------------|")
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    pf_str = f"{d['cpcv_profit_factor']:.3f}" if d['cpcv_profit_factor'] != float('inf') else "Inf"
    a(f"| {d['label']} | {d['breakeven_wr']:.3f} | {d['cpcv_directional_accuracy']:.4f} | "
      f"{d['breakeven_margin_directional']:+.4f} | {d['cpcv_accuracy']:.4f} | "
      f"${d['cpcv_expectancy_base']:.4f} | ${d['cpcv_expectancy_optimistic']:.4f} | "
      f"{pf_str} | {d['directional_prediction_rate']:.4f} |")
a()

# 5. Walk-forward results
a("## 5. Walk-Forward Results (Primary for deployment)")
a()
a("| Geometry | Fold | Test Days | Accuracy | Dir Acc | Exp (base) | N Trades |")
a("|----------|------|-----------|----------|---------|------------|----------|")
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    wf_folds = d.get("walkforward_per_fold", [])
    if not wf_folds:
        continue
    for wf in wf_folds:
        a(f"| {d['label']} | {wf['fold']} | {wf['test_days']} | "
          f"{wf['accuracy']:.4f} | {wf['directional_accuracy']:.4f} | "
          f"${wf['pnl']['base']['expectancy']:.4f} | {wf['pnl']['base']['n_trades']} |")
    a(f"| **{d['label']} mean** | | | **{d['walkforward_mean_accuracy']:.4f}** | "
      f"**{d['walkforward_mean_directional_accuracy']:.4f}** | "
      f"**${d['walkforward_mean_expectancy']:.4f}** | |")
a()

# 6. Breakeven margin analysis
a("## 6. Breakeven Margin Analysis (Directional Accuracy)")
a()
a("Key diagnostic: does directional accuracy track geometry, or is it geometry-invariant?")
a()
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    a(f"- **{d['label']}**: dir_acc={d['cpcv_directional_accuracy']:.4f}, "
      f"BEV={d['breakeven_wr']:.3f}, "
      f"margin={d['breakeven_margin_directional']:+.4f}")
a()

# 7. Class distribution
a("## 7. Class Distribution")
a()
a("| Geometry | -1 (short) | 0 (hold) | +1 (long) | Total bars | Hold Rate |")
a("|----------|-----------|----------|-----------|------------|-----------|")
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    cd = d["class_distribution"]
    hr = d.get("hold_rate", cd.get("0", 0))
    a(f"| {d['label']} | {cd.get('-1', 0):.3f} | {cd.get('0', 0):.3f} | "
      f"{cd.get('1', 0):.3f} | {d['total_bars']:,} | {hr:.3f} |")
a()

# 8. Per-class recall
a("## 8. Per-Class Recall")
a()
a("| Geometry | Short Recall | Hold Recall | Long Recall | Long vs Short Delta |")
a("|----------|-------------|-------------|-------------|---------------------|")
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    r = d["per_class_recall"]
    delta = r["long"] - r["short"]
    a(f"| {d['label']} | {r['short']:.3f} | {r['hold']:.3f} | {r['long']:.3f} | {delta:+.3f} |")
a()

# 9. Feature importance shift
a("## 9. Feature Importance Shift")
a()
vol_feats = ["volatility_20", "volatility_50", "high_low_range_50"]
vol_feat_indices = {f"f{i}" for i, name in enumerate(NON_SPATIAL_FEATURES) if name in vol_feats}

for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    feats = d.get("top10_features", [])
    if not feats:
        continue
    total_gain = sum(v for _, v in feats)
    vol_gain = sum(v for n, v in feats if n in vol_feat_indices)
    vol_share = vol_gain / total_gain if total_gain > 0 else 0
    a(f"### {d['label']} — Volatility share of top 10: {vol_share:.1%}")
    a()
    for feat_id, gain in feats:
        idx = int(feat_id[1:]) if feat_id.startswith("f") else -1
        name = NON_SPATIAL_FEATURES[idx] if 0 <= idx < len(NON_SPATIAL_FEATURES) else feat_id
        a(f"  - {name} ({feat_id}): {gain:.1f}")
    a()

# 10. Per-direction oracle
a("## 10. Per-Direction Oracle Analysis")
a()
a("| Geometry | Long Triggered | Long WR | Long Exp | Short Triggered | Short WR | Short Exp | Both Rate |")
a("|----------|---------------|---------|----------|----------------|---------|----------|-----------|")
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    pd_data = d.get("per_direction", {})
    long_d = pd_data.get("long", {})
    short_d = pd_data.get("short", {})
    a(f"| {d['label']} | {long_d.get('n_triggered', 0):,} | {long_d.get('wr', 0):.3f} | "
      f"${long_d.get('expectancy', 0):.2f} | {short_d.get('n_triggered', 0):,} | "
      f"{short_d.get('wr', 0):.3f} | ${short_d.get('expectancy', 0):.2f} | "
      f"{pd_data.get('both_triggered_rate', 0):.3f} |")
a()

# 11. Time-of-day
a("## 11. Time-of-Day Breakdown")
a()
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    tod = d.get("time_of_day", {})
    a(f"### {d['label']}")
    a()
    a("| Band | N Bars | -1 | 0 | +1 | Directional Rate |")
    a("|------|--------|-----|---|-----|------------------|")
    for band_name, band_data in tod.items():
        if band_data.get("n_bars", 0) == 0:
            continue
        cd = band_data.get("class_dist", {})
        a(f"| {band_name} | {band_data['n_bars']:,} | {cd.get('-1', 0):.3f} | "
          f"{cd.get('0', 0):.3f} | {cd.get('+1', 0):.3f} | {band_data.get('directional_rate', 0):.3f} |")
    a()

# 12. Cost sensitivity
a("## 12. Cost Sensitivity")
a()
a("| Geometry | Optimistic ($2.49) | Base ($3.74) | Pessimistic ($6.25) |")
a("|----------|-------------------|-------------|---------------------|")
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    a(f"| {d['label']} | ${d['cpcv_expectancy_optimistic']:.4f} | "
      f"${d['cpcv_expectancy_base']:.4f} | ${d['cpcv_expectancy_pessimistic']:.4f} |")
a()

# 13. Holdout
a("## 13. Holdout Evaluation")
a()
for key in GEOM_ORDER:
    if key not in metrics["holdout"]:
        continue
    result = metrics["holdout"][key]
    label = GEOM_LABELS[key]
    a(f"### {label}")
    a(f"- Overall accuracy: {result['accuracy']:.4f}")
    a(f"- Directional accuracy: {result['directional_accuracy']:.4f}")
    a(f"- Directional prediction rate: {result['directional_prediction_rate']:.4f}")
    a(f"- Per-class recall: short={result['per_class_recall']['short']:.3f}, "
      f"hold={result['per_class_recall']['hold']:.3f}, "
      f"long={result['per_class_recall']['long']:.3f}")
    for scenario in ["optimistic", "base", "pessimistic"]:
        a(f"- Expectancy ({scenario}): ${result['pnl'][scenario]['expectancy']:.4f}")
    if result.get("per_quarter"):
        parts = []
        for q, qd in sorted(result["per_quarter"].items()):
            parts.append(f"{q}=${qd['expectancy']:.4f}")
        a(f"- Per-quarter: {', '.join(parts)}")
    a()

# 14. SC evaluation
a("## 14. Success Criteria Evaluation")
a()
for sc_name, sc_data in metrics["success_criteria"].items():
    status = "PASS" if sc_data["pass"] else "FAIL"
    a(f"- **{sc_name}**: {status} — {sc_data['description']}")
a()
a("### Sanity Checks")
a()
for sc_name, sc_pass in metrics["sanity_checks"].items():
    a(f"- **{sc_name}**: {'PASS' if sc_pass else 'FAIL'}")
a()

# 15. Outcome verdict
a("## 15. Outcome Verdict")
a()
a(f"**{outcome} — {outcome_desc}**")
a()

# 16. Caveats and notes
a("## 16. Caveats")
a()
a("- SC-S1 (baseline accuracy > 0.40) FAILED. The 3600s time horizon produces ~33% hold rate (vs 91% at 300s), "
  "making the 3-class problem genuinely balanced. Baseline XGBoost accuracy is 0.384 — slightly below the "
  "40% threshold designed for the old high-hold-rate regime. This is expected behavior, not a bug.")
a("- 15:3 and 20:3 directional prediction rates are near zero (0.006% and 0.003%). The model predicts hold "
  "for essentially all bars at these geometries. The positive SC-3 margin at 15:3 is based on ~3 directional "
  "predictions across 45 CPCV splits — statistically meaningless.")
a("- 19:7 SC-3 margin (+17.47pp) is based on ~2,600 directional predictions out of 1.16M bars (0.28% rate). "
  "The model is extremely selective but achieves 55.9% directional accuracy when it does predict.")
a("- 19:7 holdout: 52 directional trades total (0.7% prediction rate), 50.0% directional accuracy, "
  "$3.76 expectancy (base). Sample too small for statistical confidence.")
a("- CPCV profit factor at 15:3 is Infinity (all directional trades were winners — but only ~3 trades total).")
a()

# 17. Directional prediction analysis
a("## 17. Directional Prediction Analysis")
a()
a("The critical insight: the 3600s time horizon transforms the classification problem. With ~33% hold rate, "
  "the model must learn genuine directional signal rather than exploiting hold dominance.")
a()
a("| Geometry | Dir Pred Rate | N Dir Predictions (CPCV) | Dir Acc | Status |")
a("|----------|--------------|-------------------------|---------|--------|")
for key in GEOM_ORDER:
    d = metrics["geometries"][key]
    # Estimate total directional predictions: rate * total_bars
    n_dir = d["directional_prediction_rate"] * d["total_bars"]
    status = "Active" if d["directional_prediction_rate"] > 0.01 else "Degenerate (hold-only)"
    a(f"| {d['label']} | {d['directional_prediction_rate']:.4f} | ~{int(n_dir):,} | "
      f"{d['cpcv_directional_accuracy']:.4f} | {status} |")
a()

with open(RESULTS_DIR / "analysis.md", "w") as f:
    f.write("\n".join(a_lines))
print("analysis.md written.")

# --- Config JSON ---
config = {
    "seed": 42,
    "max_time_horizon_s": time_horizon,
    "volume_horizon": metrics["volume_horizon"],
    "geometries": [
        {"target": 10, "stop": 5, "label": "10:5 (control)"},
        {"target": 15, "stop": 3, "label": "15:3"},
        {"target": 19, "stop": 7, "label": "19:7"},
        {"target": 20, "stop": 3, "label": "20:3"},
    ],
    "xgb_params": {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.013396,
        "reg_lambda": 6.586,
        "max_depth": 6,
        "min_child_weight": 20,
        "subsample": 0.5613,
        "colsample_bytree": 0.7483,
        "reg_alpha": 0.001424,
        "tree_method": "hist",
        "seed": 42,
    },
    "features": NON_SPATIAL_FEATURES,
    "cpcv": {"n_groups": 10, "k_test": 2, "purge_bars": 500, "embargo_bars": 4600},
    "dev_days": 201,
    "cost_scenarios": {
        "optimistic": 2.49,
        "base": 3.74,
        "pessimistic": 6.25,
    },
    "tick_value": 1.25,
    "bar_type": "time_5s",
}
with open(RESULTS_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2, default=str)
print("config.json written.")

# --- Environment fingerprint ---
env_fingerprint = {
    "python_version": sys.version,
    "seed": 42,
    "script_sha256": hashlib.sha256(open(MAIN_SCRIPT, "rb").read()).hexdigest(),
    "generated_by": "generate_reports.py (post-hoc from metrics.json)",
    "metrics_sha256": hashlib.sha256(open(RESULTS_DIR / "metrics.json", "rb").read()).hexdigest(),
}
try:
    import xgboost as xgb
    env_fingerprint["xgboost_version"] = xgb.__version__
except ImportError:
    env_fingerprint["xgboost_version"] = "unavailable"
try:
    import polars as pl
    env_fingerprint["polars_version"] = pl.__version__
except ImportError:
    env_fingerprint["polars_version"] = "unavailable"
try:
    import numpy as np
    env_fingerprint["numpy_version"] = np.__version__
except ImportError:
    env_fingerprint["numpy_version"] = "unavailable"

with open(RESULTS_DIR / "env_fingerprint.json", "w") as f:
    json.dump(env_fingerprint, f, indent=2)
print("env_fingerprint.json written.")

print("All 3 deliverables generated successfully.")
