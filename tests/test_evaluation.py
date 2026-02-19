"""
test_evaluation.py -- TDD RED phase tests for evaluation pipeline (spec tests 22-26)
Spec: .kit/docs/hybrid-model.md §PnL Computation, §Transaction Cost Sensitivity,
      §Evaluation Metrics, §Ablation Comparisons

Tests:
  22. PnL computation: known prediction + known price path -> correct PnL
  23. Cost scenarios: 3 cost levels produce decreasing expectancy
  24. Full pipeline integration: 2-fold subset, output JSON schema, metrics, no NaN
  25. Ablation GBT-only: produces valid metrics
  26. Ablation CNN-only: produces valid metrics
"""

import sys
import os
import json
import tempfile

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.hybrid_model.metrics import (
    compute_pnl,
    compute_expectancy,
    compute_profit_factor,
    compute_sharpe,
    compute_accuracy,
    compute_f1_macro,
    compute_metrics_suite,
)
from scripts.hybrid_model.config import COST_SCENARIOS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def known_trade_scenario():
    """A hand-computed PnL scenario.

    Prediction: +1 (long)
    Entry mid_price: 4500.00
    TB outcome: target hit, exit at 4502.50 (10 ticks up)
    Gross PnL: (4502.50 - 4500.00) * $5.00/point = $12.50
    Base cost: $3.74 per round-trip
    Net PnL: $12.50 - $3.74 = $8.76
    """
    return {
        "prediction": 1,
        "entry_price": 4500.00,
        "exit_price": 4502.50,
        "direction": 1,
        "tb_label": 1,
        "tb_exit_type": "target",
        "point_value": 5.00,
        "base_cost_rt": 3.74,
        "expected_gross_pnl": 12.50,
        "expected_net_pnl": 8.76,
    }


@pytest.fixture
def multiple_trades():
    """A set of trades with known PnL outcomes."""
    return pd.DataFrame({
        "prediction": [1, -1, 0, 1, -1, 1],
        "entry_price": [4500.0, 4500.0, 4500.0, 4500.0, 4500.0, 4500.0],
        "exit_price": [4502.5, 4498.75, 4500.0, 4498.75, 4501.25, 4501.25],
        "direction": [1, -1, 0, 1, -1, 1],
        "tb_label": [1, 1, 0, -1, -1, 1],
    })


@pytest.fixture
def synthetic_fold_results():
    """Synthetic per-fold evaluation results for integration test."""
    np.random.seed(42)
    n = 200
    predictions = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.3, 0.4])
    actuals = np.random.choice([-1, 0, 1], size=n, p=[0.3, 0.3, 0.4])
    entry_prices = np.full(n, 4500.0)
    # Random exit prices within reasonable range
    exit_prices = 4500.0 + np.random.randn(n) * 2.5

    return pd.DataFrame({
        "prediction": predictions,
        "actual": actuals,
        "entry_price": entry_prices,
        "exit_price": exit_prices,
        "tb_label": actuals,
        "tb_exit_type": np.random.choice(["target", "stop", "expiry"], size=n),
    })


# ===========================================================================
# Test 22: PnL computation — known prediction + price path -> correct PnL
# ===========================================================================

class TestPnLComputation:
    def test_long_target_hit_pnl(self, known_trade_scenario):
        """Hand-computed: long entry at 4500, exit at 4502.50, gross = $12.50."""
        s = known_trade_scenario
        gross_pnl = compute_pnl(
            direction=s["direction"],
            entry_price=s["entry_price"],
            exit_price=s["exit_price"],
            point_value=s["point_value"],
            cost_rt=0.0,
        )
        assert abs(gross_pnl - s["expected_gross_pnl"]) < 0.01, \
            f"Gross PnL {gross_pnl:.2f} != expected {s['expected_gross_pnl']:.2f}"

    def test_long_target_hit_net_pnl(self, known_trade_scenario):
        """Net PnL after base cost deduction."""
        s = known_trade_scenario
        net_pnl = compute_pnl(
            direction=s["direction"],
            entry_price=s["entry_price"],
            exit_price=s["exit_price"],
            point_value=s["point_value"],
            cost_rt=s["base_cost_rt"],
        )
        assert abs(net_pnl - s["expected_net_pnl"]) < 0.01, \
            f"Net PnL {net_pnl:.2f} != expected {s['expected_net_pnl']:.2f}"

    def test_short_stop_hit_pnl(self):
        """Short prediction, stop hit (price goes up → loss for short)."""
        # Short entry at 4500, exit at 4501.25 (5 ticks up = stop for short)
        # Gross PnL for short: (-1) * (4501.25 - 4500.0) * 5.0 = -6.25
        pnl = compute_pnl(
            direction=-1,
            entry_price=4500.0,
            exit_price=4501.25,
            point_value=5.0,
            cost_rt=0.0,
        )
        assert abs(pnl - (-6.25)) < 0.01, f"Short loss PnL {pnl:.2f} != -6.25"

    def test_hold_prediction_zero_pnl(self):
        """Prediction = 0 (hold) should produce PnL = 0 regardless of prices."""
        pnl = compute_pnl(
            direction=0,
            entry_price=4500.0,
            exit_price=4510.0,
            point_value=5.0,
            cost_rt=3.74,
        )
        assert pnl == 0.0, f"Hold prediction should have PnL=0, got {pnl:.2f}"

    def test_pnl_formula_direction_times_delta(self):
        """PnL = direction * (exit - entry) * point_value - cost."""
        # Long, +2 points: 1 * 2 * 5 - 3 = 7.0
        pnl = compute_pnl(1, 4500.0, 4502.0, 5.0, 3.0)
        assert abs(pnl - 7.0) < 0.01

        # Short, -2 points (profitable for short): -1 * -2 * 5 - 3 = 7.0
        pnl = compute_pnl(-1, 4500.0, 4498.0, 5.0, 3.0)
        assert abs(pnl - 7.0) < 0.01


# ===========================================================================
# Test 23: Cost scenarios — 3 cost levels produce decreasing expectancy
# ===========================================================================

class TestCostScenarios:
    def test_three_cost_scenarios_defined(self):
        """Spec defines 3 cost scenarios: optimistic, base, pessimistic."""
        assert len(COST_SCENARIOS) == 3, \
            f"Expected 3 cost scenarios, got {len(COST_SCENARIOS)}"

    def test_cost_scenarios_have_required_fields(self):
        """Each scenario must have name, commission, spread, slippage, total_rt_cost."""
        required_fields = {"name", "commission_per_side", "spread_ticks",
                           "slippage_ticks", "total_rt_cost"}
        for scenario in COST_SCENARIOS:
            missing = required_fields - set(scenario.keys())
            assert missing == set(), \
                f"Scenario '{scenario.get('name', '?')}' missing fields: {missing}"

    def test_optimistic_less_than_base_less_than_pessimistic(self):
        """Cost levels: optimistic < base < pessimistic."""
        costs = [s["total_rt_cost"] for s in COST_SCENARIOS]
        # Sort scenarios by name to ensure order
        scenario_map = {s["name"]: s["total_rt_cost"] for s in COST_SCENARIOS}

        assert scenario_map["optimistic"] < scenario_map["base"], \
            "Optimistic cost should be less than base cost"
        assert scenario_map["base"] < scenario_map["pessimistic"], \
            "Base cost should be less than pessimistic cost"

    def test_cost_values_match_spec(self):
        """Verify exact cost values from spec."""
        scenario_map = {s["name"]: s for s in COST_SCENARIOS}

        opt = scenario_map["optimistic"]
        assert abs(opt["total_rt_cost"] - 2.49) < 0.01, \
            f"Optimistic total RT cost should be $2.49, got {opt['total_rt_cost']}"

        base = scenario_map["base"]
        assert abs(base["total_rt_cost"] - 3.74) < 0.01, \
            f"Base total RT cost should be $3.74, got {base['total_rt_cost']}"

        pess = scenario_map["pessimistic"]
        assert abs(pess["total_rt_cost"] - 6.25) < 0.01, \
            f"Pessimistic total RT cost should be $6.25, got {pess['total_rt_cost']}"

    def test_decreasing_expectancy_with_increasing_costs(self, multiple_trades):
        """Higher costs must produce lower (or equal) expectancy."""
        costs_ordered = sorted(COST_SCENARIOS, key=lambda s: s["total_rt_cost"])

        expectancies = []
        for scenario in costs_ordered:
            pnls = []
            for _, trade in multiple_trades.iterrows():
                pnl = compute_pnl(
                    direction=int(trade["prediction"]),
                    entry_price=trade["entry_price"],
                    exit_price=trade["exit_price"],
                    point_value=5.0,
                    cost_rt=scenario["total_rt_cost"],
                )
                pnls.append(pnl)
            expectancy = compute_expectancy(pnls)
            expectancies.append(expectancy)

        # Each subsequent expectancy should be <= the previous
        for i in range(1, len(expectancies)):
            assert expectancies[i] <= expectancies[i - 1] + 0.01, \
                f"Expectancy with higher cost ({costs_ordered[i]['name']}: " \
                f"${expectancies[i]:.2f}) should be <= lower cost " \
                f"({costs_ordered[i-1]['name']}: ${expectancies[i-1]:.2f})"


# ===========================================================================
# Test 24: Full pipeline integration — output JSON schema, metrics, no NaN
# ===========================================================================

class TestFullPipelineIntegration:
    def test_metrics_suite_returns_all_required_fields(self, synthetic_fold_results):
        """compute_metrics_suite should return all per-fold metrics from spec."""
        results = synthetic_fold_results
        metrics = compute_metrics_suite(
            predictions=results["prediction"].values,
            actuals=results["actual"].values,
            entry_prices=results["entry_price"].values,
            exit_prices=results["exit_price"].values,
            point_value=5.0,
            cost_rt=3.74,
        )

        required_keys = {
            "xgb_accuracy", "xgb_f1_macro",
            "expectancy", "profit_factor",
            "trade_count", "sharpe",
        }
        missing = required_keys - set(metrics.keys())
        assert missing == set(), \
            f"Metrics suite missing keys: {missing}"

    def test_no_nan_in_metrics(self, synthetic_fold_results):
        """No metric value should be NaN."""
        results = synthetic_fold_results
        metrics = compute_metrics_suite(
            predictions=results["prediction"].values,
            actuals=results["actual"].values,
            entry_prices=results["entry_price"].values,
            exit_prices=results["exit_price"].values,
            point_value=5.0,
            cost_rt=3.74,
        )

        for key, value in metrics.items():
            if isinstance(value, float):
                assert not np.isnan(value), \
                    f"Metric '{key}' is NaN"
                assert not np.isinf(value), \
                    f"Metric '{key}' is Inf"

    def test_metrics_json_serializable(self, synthetic_fold_results):
        """All metrics must be JSON-serializable."""
        results = synthetic_fold_results
        metrics = compute_metrics_suite(
            predictions=results["prediction"].values,
            actuals=results["actual"].values,
            entry_prices=results["entry_price"].values,
            exit_prices=results["exit_price"].values,
            point_value=5.0,
            cost_rt=3.74,
        )

        # Should not raise
        json_str = json.dumps(metrics)
        assert len(json_str) > 0

    def test_accuracy_in_valid_range(self, synthetic_fold_results):
        """Accuracy must be in [0, 1]."""
        results = synthetic_fold_results
        acc = compute_accuracy(results["prediction"].values, results["actual"].values)
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} not in [0, 1]"

    def test_trade_count_excludes_holds(self, synthetic_fold_results):
        """Trade count should only count non-HOLD predictions."""
        results = synthetic_fold_results
        predictions = results["prediction"].values
        expected_trades = np.sum(predictions != 0)

        metrics = compute_metrics_suite(
            predictions=results["prediction"].values,
            actuals=results["actual"].values,
            entry_prices=results["entry_price"].values,
            exit_prices=results["exit_price"].values,
            point_value=5.0,
            cost_rt=3.74,
        )

        assert metrics["trade_count"] == expected_trades, \
            f"Trade count {metrics['trade_count']} != expected {expected_trades}"

    def test_profit_factor_non_negative(self, synthetic_fold_results):
        """Profit factor must be >= 0 (or inf if no losses)."""
        results = synthetic_fold_results
        metrics = compute_metrics_suite(
            predictions=results["prediction"].values,
            actuals=results["actual"].values,
            entry_prices=results["entry_price"].values,
            exit_prices=results["exit_price"].values,
            point_value=5.0,
            cost_rt=3.74,
        )
        assert metrics["profit_factor"] >= 0.0, \
            f"Profit factor {metrics['profit_factor']} should be >= 0"


# ===========================================================================
# Test 25: Ablation GBT-only — runs and produces valid metrics
# ===========================================================================

class TestAblationGBTOnly:
    def test_gbt_only_produces_valid_metrics(self):
        """GBT-only baseline (XGBoost on all 62 features, no CNN) runs."""
        np.random.seed(42)
        n = 300
        X_train = np.random.randn(n, 62)  # all 62 Track A features
        y_train = np.random.choice([-1, 0, 1], size=n)
        X_test = np.random.randn(100, 62)
        y_test = np.random.choice([-1, 0, 1], size=100)

        from scripts.hybrid_model.train_xgboost import train_xgboost_classifier, predict_xgboost

        model = train_xgboost_classifier(X_train, y_train, seed=42)
        predictions = predict_xgboost(model, X_test)

        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({-1, 0, 1})

    def test_gbt_only_metrics_all_present(self):
        """GBT-only baseline should produce all required metrics."""
        np.random.seed(42)
        n = 300
        predictions = np.random.choice([-1, 0, 1], size=n)
        actuals = np.random.choice([-1, 0, 1], size=n)
        entry_prices = np.full(n, 4500.0)
        exit_prices = 4500.0 + np.random.randn(n) * 2.5

        metrics = compute_metrics_suite(
            predictions=predictions,
            actuals=actuals,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            point_value=5.0,
            cost_rt=3.74,
        )

        required = {"xgb_accuracy", "xgb_f1_macro", "expectancy",
                     "profit_factor", "trade_count", "sharpe"}
        assert required.issubset(set(metrics.keys()))

    def test_gbt_only_62_features_accepted(self):
        """GBT-only baseline must accept 62 input features (no CNN dims)."""
        np.random.seed(42)
        X = np.random.randn(200, 62)
        y = np.random.choice([-1, 0, 1], size=200)

        from scripts.hybrid_model.train_xgboost import train_xgboost_classifier
        model = train_xgboost_classifier(X, y, seed=42)
        assert model is not None


# ===========================================================================
# Test 26: Ablation CNN-only — runs and produces valid metrics
# ===========================================================================

class TestAblationCNNOnly:
    def test_cnn_only_classification_runs(self):
        """CNN-only baseline (CNN with classification head, no XGBoost)."""
        import torch
        from scripts.hybrid_model.cnn_encoder import CNNEncoder

        torch.manual_seed(42)
        encoder = CNNEncoder()
        # Classification head: Linear(16, 3) for 3-class TB labels
        head = torch.nn.Linear(16, 3)

        X = torch.randn(100, 2, 20)
        embedding = encoder(X)
        logits = head(embedding)

        assert logits.shape == (100, 3), \
            f"CNN-only classification output shape should be (100, 3), got {logits.shape}"

    def test_cnn_only_produces_valid_predictions(self):
        """CNN-only predictions should map to {-1, 0, +1}."""
        import torch
        from scripts.hybrid_model.cnn_encoder import CNNEncoder

        torch.manual_seed(42)
        encoder = CNNEncoder()
        head = torch.nn.Linear(16, 3)

        X = torch.randn(50, 2, 20)
        with torch.no_grad():
            embedding = encoder(X)
            logits = head(embedding)
            # Map softmax class indices {0, 1, 2} to labels {-1, 0, +1}
            class_indices = logits.argmax(dim=1).numpy()
            label_map = {0: -1, 1: 0, 2: 1}
            predictions = np.array([label_map[c] for c in class_indices])

        assert set(predictions).issubset({-1, 0, 1})
        assert len(predictions) == 50

    def test_cnn_only_metrics_computable(self):
        """CNN-only baseline metrics should be computable."""
        np.random.seed(42)
        n = 200
        predictions = np.random.choice([-1, 0, 1], size=n)
        actuals = np.random.choice([-1, 0, 1], size=n)
        entry_prices = np.full(n, 4500.0)
        exit_prices = 4500.0 + np.random.randn(n) * 2.5

        metrics = compute_metrics_suite(
            predictions=predictions,
            actuals=actuals,
            entry_prices=entry_prices,
            exit_prices=exit_prices,
            point_value=5.0,
            cost_rt=3.74,
        )

        assert "xgb_accuracy" in metrics
        assert "expectancy" in metrics
        assert "profit_factor" in metrics


# ===========================================================================
# Additional: Metric function unit tests
# ===========================================================================

class TestMetricFunctions:
    def test_compute_expectancy_simple(self):
        """Expectancy = mean of PnL list."""
        pnls = [10.0, -5.0, 8.0, -3.0, 0.0]
        exp = compute_expectancy(pnls)
        assert abs(exp - 2.0) < 0.01, f"Expectancy {exp:.2f} != 2.0"

    def test_compute_expectancy_empty_list(self):
        """Empty PnL list should return 0.0."""
        exp = compute_expectancy([])
        assert exp == 0.0

    def test_compute_profit_factor_simple(self):
        """PF = gross_profit / gross_loss."""
        # Profits: 10 + 8 = 18. Losses: |(-5)| + |(-3)| = 8. PF = 18/8 = 2.25
        pnls = [10.0, -5.0, 8.0, -3.0]
        pf = compute_profit_factor(pnls)
        assert abs(pf - 2.25) < 0.01, f"Profit factor {pf:.2f} != 2.25"

    def test_compute_profit_factor_no_losses(self):
        """All winning trades → PF should be inf or a very large number."""
        pnls = [10.0, 5.0, 8.0]
        pf = compute_profit_factor(pnls)
        assert pf > 100.0 or pf == float("inf"), \
            f"PF with no losses should be very large, got {pf}"

    def test_compute_sharpe_positive(self):
        """Positive returns should have positive Sharpe."""
        daily_pnls = [10.0, 5.0, 8.0, 3.0, 7.0]
        sharpe = compute_sharpe(daily_pnls)
        assert sharpe > 0, f"Sharpe {sharpe:.4f} should be > 0 for positive returns"

    def test_compute_f1_macro(self):
        """F1 macro for perfect predictions should be 1.0."""
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = np.array([-1, 0, 1, -1, 0, 1])
        f1 = compute_f1_macro(y_pred, y_true)
        assert abs(f1 - 1.0) < 0.01, f"Perfect predictions should give F1=1.0, got {f1}"

    def test_compute_accuracy_perfect(self):
        """Perfect predictions should give accuracy 1.0."""
        y_true = np.array([-1, 0, 1, -1, 0, 1])
        y_pred = np.array([-1, 0, 1, -1, 0, 1])
        acc = compute_accuracy(y_pred, y_true)
        assert abs(acc - 1.0) < 0.01
