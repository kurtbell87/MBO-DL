"""Load exported CSV, normalize features, split into folds."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config


def load_csv(path=None):
    """Load the exported CSV and fix duplicate column names.

    The C++ exporter produces duplicate column names for return_1/5/20
    (Track A features) and forward returns. We resolve this by positional
    renaming of the forward return columns.
    """
    csv_path = path or config.DATA_CSV
    df = pd.read_csv(csv_path)

    # The CSV has 62 Track A features, then book_snap, msg_summary,
    # then 4 forward returns named return_1,return_5,return_20,return_100.
    # Pandas auto-deduplicates to return_1.1, return_5.1, return_20.1, return_100.
    # Rename them to fwd_return_*.
    rename_map = {}
    if "return_1.1" in df.columns:
        rename_map["return_1.1"] = "fwd_return_1"
    if "return_5.1" in df.columns:
        rename_map["return_5.1"] = "fwd_return_5"
    if "return_20.1" in df.columns:
        rename_map["return_20.1"] = "fwd_return_20"
    # return_100 is unique (no Track A duplicate)
    if "return_100" in df.columns and "return_100.1" not in df.columns:
        rename_map["return_100"] = "fwd_return_100"
    elif "return_100.1" in df.columns:
        rename_map["return_100.1"] = "fwd_return_100"

    df.rename(columns=rename_map, inplace=True)
    return df


def split_fold(df, fold_idx):
    """Split dataframe into train/test sets for a given fold index (0-based)."""
    fold = config.FOLDS[fold_idx]
    train_mask = df["day"].isin(fold["train"])
    test_mask = df["day"].isin(fold["test"])
    return df[train_mask].copy(), df[test_mask].copy()


def extract_book_arrays(df):
    """Extract (N, 2, 20) book arrays from dataframe.

    Channel 0: price offsets from mid (20 levels)
    Channel 1: sizes (20 levels)

    CSV layout: book_snap_{2i} = price_offset, book_snap_{2i+1} = size
    """
    book_data = df[config.BOOK_SNAP_COLS].values  # (N, 40)
    N = book_data.shape[0]
    prices = book_data[:, 0::2]  # (N, 20) — even indices
    sizes = book_data[:, 1::2]   # (N, 20) — odd indices
    return np.stack([prices, sizes], axis=1)  # (N, 2, 20)


def normalize_book_sizes(book_arr, train_mask=None):
    """Z-score normalize book sizes (channel 1) per day or using training stats.

    Args:
        book_arr: (N, 2, 20) array
        train_mask: if provided, compute stats only from train rows

    Returns:
        normalized copy of book_arr
    """
    result = book_arr.copy()
    sizes = result[:, 1, :]  # (N, 20)

    if train_mask is not None:
        train_sizes = sizes[train_mask]
    else:
        train_sizes = sizes

    mean = train_sizes.mean()
    std = train_sizes.std()
    result[:, 1, :] = (sizes - mean) / (std + 1e-8)
    return result


def extract_non_spatial(df):
    """Extract the 20 non-spatial features as (N, 20) array."""
    return df[config.NON_SPATIAL_FEATURES].values.astype(np.float32)


def normalize_features(features, train_mask=None):
    """Z-score normalize features using training set statistics only.

    Args:
        features: (N, D) array
        train_mask: boolean mask for training rows

    Returns:
        normalized (N, D) array
    """
    result = features.copy()
    if train_mask is not None:
        train_data = features[train_mask]
    else:
        train_data = features

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    result = (result - mean) / (std + 1e-8)
    # Handle NaN (rare, only first few bars excluded by warmup)
    result = np.nan_to_num(result, nan=0.0)
    return result, mean, std


class BookDataset(Dataset):
    """PyTorch dataset for CNN encoder training."""

    def __init__(self, book_arr, targets):
        """
        Args:
            book_arr: (N, 2, 20) float array
            targets: (N,) float array (forward returns)
        """
        self.book = torch.tensor(book_arr, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.book[idx], self.targets[idx]


class HybridDataset(Dataset):
    """PyTorch-compatible dataset for combined embeddings + features."""

    def __init__(self, embeddings, features, labels):
        """
        Args:
            embeddings: (N, 16) CNN embeddings
            features: (N, 20) non-spatial features
            labels: (N,) triple barrier labels
        """
        self.X = np.concatenate([embeddings, features], axis=1)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def get_X(self):
        return self.X

    def get_labels(self):
        return self.labels
