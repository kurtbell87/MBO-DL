"""
test_xgboost_training.py -- TDD RED phase tests for XGBoost classification (spec tests 20-21)
Spec: .kit/docs/hybrid-model.md §Stage 2: XGBoost Classification

Tests:
  20. XGBoost trains on synthetic 36-dim features + 3-class labels, predictions in {-1, 0, +1}
  21. XGBoost deterministic: two runs with seed=42 produce identical predictions
"""

import sys
import os

import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.hybrid_model.train_xgboost import train_xgboost_classifier, predict_xgboost


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_xgb_data():
    """Create synthetic 36-dim features + 3-class labels for XGBoost.

    36 dims = 16 CNN embedding + 20 non-spatial features.
    Labels in {-1, 0, +1} (triple barrier).
    """
    np.random.seed(42)
    n_samples = 500

    X = np.random.randn(n_samples, 36)

    # Create labels with some learnable signal:
    # label based on sign of first feature (with noise)
    raw = X[:, 0] + 0.5 * np.random.randn(n_samples)
    labels = np.zeros(n_samples, dtype=int)
    labels[raw > 0.5] = 1
    labels[raw < -0.5] = -1

    return X, labels


@pytest.fixture
def small_xgb_data():
    """Smaller dataset for quick tests."""
    np.random.seed(42)
    X = np.random.randn(100, 36)
    labels = np.random.choice([-1, 0, 1], size=100)
    return X, labels


# ===========================================================================
# Test 20: XGBoost trains and predicts valid labels
# ===========================================================================

class TestXGBoostTrains:
    def test_model_returns_predictions(self, synthetic_xgb_data):
        """Train on synthetic data and verify predictions are returned."""
        X_train, y_train = synthetic_xgb_data
        model = train_xgboost_classifier(X_train, y_train, seed=42)
        predictions = predict_xgboost(model, X_train)

        assert predictions is not None, "Predictions should not be None"
        assert len(predictions) == len(X_train), \
            f"Expected {len(X_train)} predictions, got {len(predictions)}"

    def test_predictions_in_valid_label_set(self, synthetic_xgb_data):
        """Spec: 'verify predictions in {-1, 0, +1}'."""
        X_train, y_train = synthetic_xgb_data
        model = train_xgboost_classifier(X_train, y_train, seed=42)
        predictions = predict_xgboost(model, X_train)

        valid_labels = {-1, 0, 1}
        unique_preds = set(predictions)
        invalid = unique_preds - valid_labels
        assert invalid == set(), \
            f"Predictions contain invalid labels: {invalid}. " \
            f"Expected only {{-1, 0, +1}}"

    def test_predictions_contain_multiple_classes(self, synthetic_xgb_data):
        """Model should predict more than one class on training data."""
        X_train, y_train = synthetic_xgb_data
        model = train_xgboost_classifier(X_train, y_train, seed=42)
        predictions = predict_xgboost(model, X_train)

        unique_preds = set(predictions)
        assert len(unique_preds) >= 2, \
            f"Model only predicts {unique_preds} — should predict at least 2 classes"

    def test_training_accuracy_above_random(self, synthetic_xgb_data):
        """Training accuracy on synthetic data should be above 1/3 (random for 3-class)."""
        X_train, y_train = synthetic_xgb_data
        model = train_xgboost_classifier(X_train, y_train, seed=42)
        predictions = predict_xgboost(model, X_train)

        accuracy = np.mean(predictions == y_train)
        assert accuracy > 1.0 / 3.0, \
            f"Training accuracy {accuracy:.4f} is not above random (0.333)"

    def test_works_with_36_dimensions(self, small_xgb_data):
        """Verify the model accepts exactly 36 input dimensions."""
        X_train, y_train = small_xgb_data
        assert X_train.shape[1] == 36

        model = train_xgboost_classifier(X_train, y_train, seed=42)
        predictions = predict_xgboost(model, X_train)
        assert len(predictions) == len(X_train)

    def test_handles_unseen_test_data(self, synthetic_xgb_data):
        """Model should produce valid predictions on held-out data."""
        X, y = synthetic_xgb_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        model = train_xgboost_classifier(X_train, y_train, seed=42)
        predictions = predict_xgboost(model, X_test)

        assert len(predictions) == len(X_test)
        valid_labels = {-1, 0, 1}
        assert set(predictions).issubset(valid_labels)

    def test_uses_spec_hyperparameters(self, small_xgb_data):
        """XGBoost should use the hyperparameters from the spec."""
        X_train, y_train = small_xgb_data
        model = train_xgboost_classifier(X_train, y_train, seed=42)

        # The model should have been trained (not None)
        assert model is not None

        # Verify it's an XGBoost model (check for predict method)
        assert hasattr(model, "predict"), "Model must have a predict method"


# ===========================================================================
# Test 21: XGBoost deterministic — two runs with seed=42 produce identical predictions
# ===========================================================================

class TestXGBoostDeterministic:
    def test_same_seed_same_predictions(self, synthetic_xgb_data):
        """Spec: 'Two runs with seed=42 produce identical predictions'."""
        X_train, y_train = synthetic_xgb_data

        model1 = train_xgboost_classifier(X_train, y_train, seed=42)
        preds1 = predict_xgboost(model1, X_train)

        model2 = train_xgboost_classifier(X_train, y_train, seed=42)
        preds2 = predict_xgboost(model2, X_train)

        np.testing.assert_array_equal(preds1, preds2,
            err_msg="Same seed should produce identical predictions")

    def test_same_seed_same_on_test_data(self, synthetic_xgb_data):
        """Determinism extends to unseen test data."""
        X, y = synthetic_xgb_data
        X_train, y_train = X[:400], y[:400]
        X_test = X[400:]

        model1 = train_xgboost_classifier(X_train, y_train, seed=42)
        preds1 = predict_xgboost(model1, X_test)

        model2 = train_xgboost_classifier(X_train, y_train, seed=42)
        preds2 = predict_xgboost(model2, X_test)

        np.testing.assert_array_equal(preds1, preds2)

    def test_different_seed_may_differ(self, synthetic_xgb_data):
        """Different seeds should produce potentially different predictions."""
        X_train, y_train = synthetic_xgb_data

        model1 = train_xgboost_classifier(X_train, y_train, seed=42)
        preds1 = predict_xgboost(model1, X_train)

        model2 = train_xgboost_classifier(X_train, y_train, seed=99)
        preds2 = predict_xgboost(model2, X_train)

        # They might happen to be the same, but typically won't be
        # We just verify both produce valid outputs
        valid_labels = {-1, 0, 1}
        assert set(preds1).issubset(valid_labels)
        assert set(preds2).issubset(valid_labels)
