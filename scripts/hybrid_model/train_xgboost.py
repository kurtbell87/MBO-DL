"""Stage 2: XGBoost classification on CNN embeddings + non-spatial features."""

import numpy as np
import xgboost as xgb

from . import config


def _map_labels_to_xgb(labels):
    """Map {-1, 0, +1} -> {0, 1, 2} for XGBoost multi:softmax."""
    label_map = {-1: 0, 0: 1, 1: 2}
    return np.array([label_map[int(l)] for l in labels])


def _map_labels_from_xgb(preds):
    """Map {0, 1, 2} -> {-1, 0, +1} back to triple barrier labels."""
    inv_map = {0: -1, 1: 0, 2: 1}
    return np.array([inv_map[int(p)] for p in preds])


def train_xgboost(train_X, train_labels, val_X=None, val_labels=None, params=None):
    """Train XGBoost classifier on combined features.

    Args:
        train_X: (N_train, D) features (embeddings + non-spatial)
        train_labels: (N_train,) TB labels in {-1, 0, +1}
        val_X: optional validation features
        val_labels: optional validation labels
        params: override XGB_PARAMS

    Returns:
        model: trained xgb.XGBClassifier
    """
    xgb_params = dict(config.XGB_PARAMS)
    if params:
        xgb_params.update(params)

    n_estimators = xgb_params.pop("n_estimators", 500)
    seed = xgb_params.pop("seed", 42)

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        use_label_encoder=False,
        **xgb_params,
    )

    train_y = _map_labels_to_xgb(train_labels)

    fit_kwargs = {}
    if val_X is not None and val_labels is not None:
        val_y = _map_labels_to_xgb(val_labels)
        fit_kwargs["eval_set"] = [(val_X, val_y)]
        fit_kwargs["verbose"] = False

    model.fit(train_X, train_y, **fit_kwargs)
    return model


def predict_xgboost(model, X):
    """Predict TB labels using trained XGBoost model.

    Returns:
        (N,) array of labels in {-1, 0, +1}
    """
    raw_preds = model.predict(X)
    return _map_labels_from_xgb(raw_preds)
