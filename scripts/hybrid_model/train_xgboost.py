"""XGBoost classifier training for triple barrier labels."""

import numpy as np
import xgboost as xgb


def train_xgboost_classifier(X_train, y_train, seed=42):
    """Train XGBoost classifier on features + TB labels.

    Labels are {-1, 0, +1}, mapped to {0, 1, 2} for XGBoost multi:softmax.

    Args:
        X_train: (n_samples, n_features) numpy array
        y_train: (n_samples,) numpy array with values in {-1, 0, 1}
        seed: random seed for reproducibility

    Returns:
        Trained XGBoost Booster model
    """
    # Map labels: {-1, 0, +1} -> {0, 1, 2}
    y_mapped = y_train + 1

    dtrain = xgb.DMatrix(X_train, label=y_mapped)

    params = {
        "objective": "multi:softmax",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": seed,
        "verbosity": 0,
    }

    model = xgb.train(params, dtrain, num_boost_round=500)
    return model


def predict_xgboost(model, X):
    """Predict labels using trained XGBoost model.

    Returns labels in {-1, 0, +1}.
    """
    dtest = xgb.DMatrix(X)
    # Predictions are class indices {0, 1, 2}
    preds = model.predict(dtest)
    # Map back: {0, 1, 2} -> {-1, 0, +1}
    return (preds - 1).astype(int)
