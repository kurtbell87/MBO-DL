"""Data loading, feature selection, normalization, and fold splitting."""

import numpy as np
import pandas as pd

from .config import NON_SPATIAL_FEATURES, SELECTED_DAYS


def select_non_spatial_features(df):
    """Select the 20 non-spatial features from a DataFrame."""
    return df[NON_SPATIAL_FEATURES].copy()


def normalize_features(df):
    """Z-score normalize features.

    Returns:
        (normalized_df, means, stds) where means and stds are arrays of length n_features.
    """
    means = df.mean().values
    stds = df.std().values
    stds = np.where(stds < 1e-8, 1.0, stds)
    normalized = (df - means) / stds
    normalized = normalized.fillna(0.0)
    return normalized, means, stds


def create_expanding_window_folds(df):
    """Create 5-fold expanding window CV splits.

    Uses the days present in df, sorted chronologically.
    Fold k: train on first N_k days, test on next M_k days.

    Structure (19 days):
      Fold 1: train days 1-4, test days 5-7
      Fold 2: train days 1-7, test days 8-10
      Fold 3: train days 1-10, test days 11-13
      Fold 4: train days 1-13, test days 14-16
      Fold 5: train days 1-16, test days 17-19
    """
    unique_days = sorted(df["day"].unique())

    # Define fold boundaries using day indices
    fold_splits = [
        (0, 4, 4, 7),    # Fold 1: train [0:4), test [4:7)
        (0, 7, 7, 10),   # Fold 2: train [0:7), test [7:10)
        (0, 10, 10, 13),  # Fold 3: train [0:10), test [10:13)
        (0, 13, 13, 16),  # Fold 4: train [0:13), test [13:16)
        (0, 16, 16, 19),  # Fold 5: train [0:16), test [16:19)
    ]

    folds = []
    for train_start, train_end, test_start, test_end in fold_splits:
        train_end = min(train_end, len(unique_days))
        test_end = min(test_end, len(unique_days))

        train_days = set(unique_days[train_start:train_end])
        test_days = set(unique_days[test_start:test_end])

        fold = {
            "train": df[df["day"].isin(train_days)].copy(),
            "test": df[df["day"].isin(test_days)].copy(),
        }
        folds.append(fold)

    return folds


def load_and_prepare_data(csv_path):
    """Load exported CSV and prepare for training."""
    df = pd.read_csv(csv_path)
    return df
