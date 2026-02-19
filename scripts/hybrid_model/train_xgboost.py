"""XGBoost classifier for triple barrier labels.

Hyperparameters from spec:
  max_depth=6, lr=0.05, n_estimators=500, subsample=0.8,
  colsample_bytree=0.8, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0
"""

import numpy as np
import xgboost as xgb


def train_xgboost_classifier(X, y, seed=42):
    """Train XGBoost multi-class classifier on triple barrier labels.

    Args:
        X: (N, D) feature array
        y: (N,) labels in {-1, 0, +1}
        seed: random seed

    Returns:
        Trained XGBClassifier
    """
    # Map labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
    y_mapped = y + 1  # {-1,0,1} -> {0,1,2}

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X, y_mapped)
    return model


def predict_xgboost(model, X):
    """Predict triple barrier labels.

    Returns:
        (N,) array of labels in {-1, 0, +1}
    """
    preds_mapped = model.predict(X)  # {0, 1, 2}
    return preds_mapped.astype(int) - 1  # back to {-1, 0, 1}


def predict_xgboost_proba(model, X):
    """Predict class probabilities.

    Returns:
        (N, 3) array of probabilities for classes {-1, 0, +1}
    """
    return model.predict_proba(X)
