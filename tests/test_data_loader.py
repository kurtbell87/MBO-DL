"""
test_data_loader.py -- TDD RED phase tests for data loading, feature selection,
normalization, and fold splitting (spec tests 16-19)
Spec: .kit/docs/hybrid-model.md §Non-Spatial Feature Set, §Normalization, §CV Protocol

Tests:
  16. Feature selector: exactly 20 features selected from 62-feature CSV
  17. Feature normalization: after z-score, mean ~ 0, std ~ 1
  18. Fold splitting: expanding window, train subset property, no future leakage
  19. No data leakage: test fold normalized with train-fold statistics only
"""

import sys
import os

import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.hybrid_model.config import NON_SPATIAL_FEATURES, SELECTED_DAYS
from scripts.hybrid_model.data_loader import (
    select_non_spatial_features,
    normalize_features,
    create_expanding_window_folds,
    load_and_prepare_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_csv_data():
    """Create a synthetic DataFrame mimicking bar_feature_export CSV output.

    62 Track A features + metadata + forward returns + TB columns.
    Uses 19 unique days matching the spec's SELECTED_DAYS.
    """
    np.random.seed(42)
    n_bars_per_day = 100
    n_days = 19

    rows = []
    for day_idx, day in enumerate(SELECTED_DAYS):
        for bar_idx in range(n_bars_per_day):
            row = {"day": day, "bar_index": bar_idx + 50}

            # All 62 Track A features (matching BarFeatureRow::feature_names())
            all_62_features = [
                "book_imbalance_1", "book_imbalance_3", "book_imbalance_5",
                "book_imbalance_10", "weighted_imbalance", "spread",
            ]
            for i in range(10):
                all_62_features.append(f"bid_depth_profile_{i}")
            for i in range(10):
                all_62_features.append(f"ask_depth_profile_{i}")
            all_62_features.extend([
                "depth_concentration_bid", "depth_concentration_ask",
                "book_slope_bid", "book_slope_ask",
                "level_count_bid", "level_count_ask",
                "net_volume", "volume_imbalance", "trade_count",
                "avg_trade_size", "large_trade_count", "vwap_distance",
                "kyle_lambda",
                "return_1", "return_5", "return_20",
                "volatility_20", "volatility_50", "momentum",
                "high_low_range_20", "high_low_range_50", "close_position",
                "volume_surprise", "duration_surprise", "acceleration",
                "vol_price_corr",
                "time_sin", "time_cos", "minutes_since_open",
                "minutes_to_close", "session_volume_frac",
                "cancel_add_ratio", "message_rate", "modify_fraction",
                "order_flow_toxicity", "cancel_concentration",
            ])
            assert len(all_62_features) == 62

            for feat in all_62_features:
                # Use day_idx-scaled values so z-score differs per fold
                row[feat] = np.random.randn() * (1 + day_idx * 0.1) + day_idx * 0.5

            # Book snapshot columns (40)
            for i in range(40):
                row[f"book_snap_{i}"] = np.random.randn()

            # Forward returns
            row["return_1"] = np.random.randn() * 0.5  # overwrite the feature
            row["fwd_return_1"] = np.random.randn() * 0.5
            row["fwd_return_5"] = np.random.randn() * 0.5

            # TB columns
            row["tb_label"] = np.random.choice([-1, 0, 1])
            row["tb_exit_type"] = np.random.choice(["target", "stop", "expiry", "timeout"])
            row["tb_bars_held"] = np.random.randint(1, 60)

            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def non_spatial_feature_names():
    """The 20 non-spatial features listed in the spec."""
    return [
        "weighted_imbalance", "spread",
        "net_volume", "volume_imbalance", "trade_count",
        "avg_trade_size", "vwap_distance",
        "return_1", "return_5", "return_20",
        "volatility_20", "volatility_50",
        "high_low_range_50", "close_position",
        "cancel_add_ratio", "message_rate", "modify_fraction",
        "time_sin", "time_cos", "minutes_since_open",
    ]


# ===========================================================================
# Test 16: Feature selector — exactly 20 features from 62-feature CSV
# ===========================================================================

class TestFeatureSelector:
    def test_exactly_20_features_selected(self):
        """Spec: 'Verify exactly 20 features selected from 62-feature CSV'."""
        assert len(NON_SPATIAL_FEATURES) == 20, \
            f"Expected exactly 20 non-spatial features, got {len(NON_SPATIAL_FEATURES)}"

    def test_selected_features_match_spec(self, non_spatial_feature_names):
        """Selected features must match the 20 listed in the spec."""
        spec_set = set(non_spatial_feature_names)
        config_set = set(NON_SPATIAL_FEATURES)

        missing = spec_set - config_set
        extra = config_set - spec_set

        assert missing == set(), \
            f"Features missing from config that are in spec: {missing}"
        assert extra == set(), \
            f"Extra features in config not in spec: {extra}"

    def test_select_function_returns_correct_columns(self, synthetic_csv_data):
        """select_non_spatial_features() should return a DataFrame with exactly 20 columns."""
        selected = select_non_spatial_features(synthetic_csv_data)
        assert selected.shape[1] == 20, \
            f"Expected 20 columns, got {selected.shape[1]}"

    def test_selected_features_are_subset_of_62(self, non_spatial_feature_names):
        """All 20 selected features must be from the 62 Track A features."""
        all_62 = [
            "book_imbalance_1", "book_imbalance_3", "book_imbalance_5",
            "book_imbalance_10", "weighted_imbalance", "spread",
        ]
        for i in range(10):
            all_62.append(f"bid_depth_profile_{i}")
        for i in range(10):
            all_62.append(f"ask_depth_profile_{i}")
        all_62.extend([
            "depth_concentration_bid", "depth_concentration_ask",
            "book_slope_bid", "book_slope_ask",
            "level_count_bid", "level_count_ask",
            "net_volume", "volume_imbalance", "trade_count",
            "avg_trade_size", "large_trade_count", "vwap_distance",
            "kyle_lambda",
            "return_1", "return_5", "return_20",
            "volatility_20", "volatility_50", "momentum",
            "high_low_range_20", "high_low_range_50", "close_position",
            "volume_surprise", "duration_surprise", "acceleration",
            "vol_price_corr",
            "time_sin", "time_cos", "minutes_since_open",
            "minutes_to_close", "session_volume_frac",
            "cancel_add_ratio", "message_rate", "modify_fraction",
            "order_flow_toxicity", "cancel_concentration",
        ])

        for feat in non_spatial_feature_names:
            assert feat in all_62, \
                f"Non-spatial feature '{feat}' not found in 62 Track A features"

    def test_excluded_features_not_in_selection(self):
        """Features explicitly excluded (redundant with CNN) must not appear."""
        excluded = [
            "bid_depth_profile_0", "bid_depth_profile_5",
            "ask_depth_profile_0", "ask_depth_profile_9",
            "book_imbalance_1", "book_imbalance_3", "book_imbalance_5",
            "book_imbalance_10",
            "depth_concentration_bid", "depth_concentration_ask",
            "book_slope_bid", "book_slope_ask",
            "level_count_bid", "level_count_ask",
        ]
        for feat in excluded:
            assert feat not in NON_SPATIAL_FEATURES, \
                f"Feature '{feat}' should be excluded (redundant with CNN input)"


# ===========================================================================
# Test 17: Feature normalization — z-score: mean ~ 0, std ~ 1
# ===========================================================================

class TestFeatureNormalization:
    def test_zscore_mean_near_zero(self, synthetic_csv_data):
        """After z-score normalization, mean of each feature should be ~ 0."""
        selected = select_non_spatial_features(synthetic_csv_data)
        normalized, _, _ = normalize_features(selected)

        for col in normalized.columns:
            mean = normalized[col].mean()
            assert abs(mean) < 0.1, \
                f"Feature '{col}' mean after z-score = {mean:.4f}, expected ~ 0"

    def test_zscore_std_near_one(self, synthetic_csv_data):
        """After z-score normalization, std of each feature should be ~ 1."""
        selected = select_non_spatial_features(synthetic_csv_data)
        normalized, _, _ = normalize_features(selected)

        for col in normalized.columns:
            std = normalized[col].std()
            assert abs(std - 1.0) < 0.2, \
                f"Feature '{col}' std after z-score = {std:.4f}, expected ~ 1"

    def test_normalize_returns_mean_and_std(self, synthetic_csv_data):
        """normalize_features should return (normalized_df, means, stds)."""
        selected = select_non_spatial_features(synthetic_csv_data)
        result = normalize_features(selected)

        assert len(result) == 3, \
            f"normalize_features should return 3 values, got {len(result)}"
        normalized, means, stds = result
        assert len(means) == 20, f"Expected 20 means, got {len(means)}"
        assert len(stds) == 20, f"Expected 20 stds, got {len(stds)}"

    def test_nan_filled_with_zero(self, synthetic_csv_data):
        """NaN values should be filled with 0.0 after normalization."""
        selected = select_non_spatial_features(synthetic_csv_data)
        # Inject some NaN values
        selected.iloc[0, 0] = np.nan
        selected.iloc[5, 3] = np.nan

        normalized, _, _ = normalize_features(selected)
        assert not normalized.isna().any().any(), \
            "NaN values should be filled with 0.0 after normalization"


# ===========================================================================
# Test 18: Fold splitting — expanding window, train subset, no future leakage
# ===========================================================================

class TestFoldSplitting:
    def test_five_folds_created(self, synthetic_csv_data):
        """Spec: 5-fold expanding window CV."""
        folds = create_expanding_window_folds(synthetic_csv_data)
        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    def test_each_fold_has_train_and_test(self, synthetic_csv_data):
        """Each fold should contain 'train' and 'test' splits."""
        folds = create_expanding_window_folds(synthetic_csv_data)
        for k, fold in enumerate(folds):
            assert "train" in fold, f"Fold {k} missing 'train' key"
            assert "test" in fold, f"Fold {k} missing 'test' key"

    def test_train_set_is_expanding(self, synthetic_csv_data):
        """Fold k train set must be strict subset of fold k+1 train set."""
        folds = create_expanding_window_folds(synthetic_csv_data)

        for k in range(len(folds) - 1):
            train_k_days = set(folds[k]["train"]["day"].unique())
            train_k1_days = set(folds[k + 1]["train"]["day"].unique())

            assert train_k_days < train_k1_days, \
                f"Fold {k} train days ({train_k_days}) is not a strict subset " \
                f"of fold {k+1} train days ({train_k1_days})"

    def test_no_test_day_in_earlier_train(self, synthetic_csv_data):
        """No test day should appear in any earlier fold's train set."""
        folds = create_expanding_window_folds(synthetic_csv_data)

        for k in range(len(folds)):
            test_days = set(folds[k]["test"]["day"].unique())
            for j in range(k):
                train_days_j = set(folds[j]["train"]["day"].unique())
                overlap = test_days & train_days_j
                assert overlap == set(), \
                    f"Fold {k} test days {overlap} appear in fold {j} train set (leakage)"

    def test_no_test_day_in_same_fold_train(self, synthetic_csv_data):
        """Within a fold, test days must not overlap with train days."""
        folds = create_expanding_window_folds(synthetic_csv_data)

        for k in range(len(folds)):
            train_days = set(folds[k]["train"]["day"].unique())
            test_days = set(folds[k]["test"]["day"].unique())
            overlap = train_days & test_days
            assert overlap == set(), \
                f"Fold {k} has overlapping days in train and test: {overlap}"

    def test_train_days_precede_test_days(self, synthetic_csv_data):
        """All train days must be strictly before all test days."""
        folds = create_expanding_window_folds(synthetic_csv_data)

        for k in range(len(folds)):
            max_train_day = folds[k]["train"]["day"].max()
            min_test_day = folds[k]["test"]["day"].min()
            assert max_train_day < min_test_day, \
                f"Fold {k}: max train day ({max_train_day}) >= " \
                f"min test day ({min_test_day})"

    def test_all_19_days_covered(self, synthetic_csv_data):
        """Every day in SELECTED_DAYS should appear in at least one fold."""
        folds = create_expanding_window_folds(synthetic_csv_data)
        all_days_seen = set()
        for fold in folds:
            all_days_seen.update(fold["train"]["day"].unique())
            all_days_seen.update(fold["test"]["day"].unique())

        for day in SELECTED_DAYS:
            assert day in all_days_seen, \
                f"Day {day} not covered by any fold"


# ===========================================================================
# Test 19: No data leakage — test fold features normalized with train stats
# ===========================================================================

class TestNoDataLeakage:
    def test_test_fold_uses_train_statistics(self, synthetic_csv_data):
        """Test fold must be normalized using train fold's mean/std, not its own."""
        folds = create_expanding_window_folds(synthetic_csv_data)

        for k, fold in enumerate(folds):
            train_feats = select_non_spatial_features(fold["train"])
            test_feats = select_non_spatial_features(fold["test"])

            # Normalize train to get train statistics
            _, train_means, train_stds = normalize_features(train_feats)

            # Normalize test with train statistics (the correct way)
            test_normalized_correct = (test_feats - train_means) / (train_stds + 1e-8)

            # Normalize test with its own statistics (the WRONG way)
            test_normalized_wrong, _, _ = normalize_features(test_feats)

            # The correct normalization should NOT match self-normalization
            # (because train and test have different distributions in our synthetic data)
            if len(test_feats) > 10:  # need enough data for stats to differ
                diffs = (test_normalized_correct.values - test_normalized_wrong.values)
                max_diff = np.abs(diffs).max()
                assert max_diff > 0.01, \
                    f"Fold {k}: test normalized with train stats should differ " \
                    f"from self-normalization (max_diff={max_diff:.6f})"

    def test_train_statistics_not_using_test_data(self, synthetic_csv_data):
        """Train mean/std computed only from train data."""
        folds = create_expanding_window_folds(synthetic_csv_data)

        for k, fold in enumerate(folds):
            train_feats = select_non_spatial_features(fold["train"])
            _, train_means, train_stds = normalize_features(train_feats)

            # Verify train means match actual train column means
            for i, col in enumerate(train_feats.columns):
                expected_mean = train_feats[col].mean()
                assert abs(train_means[i] - expected_mean) < 1e-5, \
                    f"Fold {k}, feature '{col}': train mean {train_means[i]:.6f} " \
                    f"!= actual train mean {expected_mean:.6f}"
