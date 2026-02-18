"""
test_cnn_encoder.py -- TDD RED phase tests for CNN encoder (spec tests 11-15)
Spec: .kit/docs/hybrid-model.md §Architecture, §CNN Encoder Detail, §Two-Stage Training

Tests the CNN encoder module contract:
  11. Output shape: (B, 2, 20) -> (B, 16)
  12. No NaN in output for random input
  13. Gradient flow: all parameters receive non-zero gradients
  14. Regression loss decreases over 10 epochs on synthetic data
  15. Deterministic: two runs with seed=42 produce identical loss at epoch 5
"""

import sys
import os

import pytest
import torch
import numpy as np

# The CNN encoder module will live at scripts/hybrid_model/cnn_encoder.py
# Add the project root to sys.path so we can import it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.hybrid_model.cnn_encoder import CNNEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encoder():
    """Create a fresh CNNEncoder instance."""
    return CNNEncoder()


@pytest.fixture
def synthetic_batch():
    """Create a synthetic batch of shape (B=4, 2, 20)."""
    torch.manual_seed(42)
    return torch.randn(4, 2, 20)


@pytest.fixture
def synthetic_dataset():
    """Create a larger synthetic dataset for training tests.
    X: (200, 2, 20) book snapshots
    y: (200,) forward returns (regression targets)
    """
    torch.manual_seed(42)
    X = torch.randn(200, 2, 20)
    # Create a simple learnable signal: target = mean of first channel
    y = X[:, 0, :].mean(dim=1) + 0.1 * torch.randn(200)
    return X, y


# ===========================================================================
# Test 11: CNN output shape — Input (B=4, 2, 20) -> output (4, 16)
# ===========================================================================

class TestCNNOutputShape:
    def test_output_shape_batch_4(self, encoder, synthetic_batch):
        """Spec: Input (B=4, 2, 20) -> output (4, 16)."""
        output = encoder(synthetic_batch)
        assert output.shape == (4, 16), \
            f"Expected output shape (4, 16), got {output.shape}"

    def test_output_shape_batch_1(self, encoder):
        """Single sample input should work."""
        torch.manual_seed(42)
        x = torch.randn(1, 2, 20)
        output = encoder(x)
        assert output.shape == (1, 16), \
            f"Expected output shape (1, 16), got {output.shape}"

    def test_output_shape_batch_256(self, encoder):
        """Batch size 256 (training batch size from spec)."""
        torch.manual_seed(42)
        x = torch.randn(256, 2, 20)
        output = encoder(x)
        assert output.shape == (256, 16), \
            f"Expected output shape (256, 16), got {output.shape}"

    def test_input_channels_is_2(self, encoder):
        """CNN input must have exactly 2 channels (price_offset, size)."""
        torch.manual_seed(42)
        # Wrong number of channels should fail
        x_wrong = torch.randn(4, 3, 20)
        with pytest.raises(Exception):
            encoder(x_wrong)

    def test_input_spatial_dim_is_20(self, encoder):
        """CNN input spatial dimension must be 20 (book levels)."""
        torch.manual_seed(42)
        # Spatial dim 10 instead of 20 -- encoder may still run but
        # we test the contract with the correct dimension.
        x_correct = torch.randn(4, 2, 20)
        output = encoder(x_correct)
        assert output.shape[-1] == 16


# ===========================================================================
# Test 12: CNN forward no NaN — Random input produces finite output
# ===========================================================================

class TestCNNNoNaN:
    def test_output_is_finite(self, encoder, synthetic_batch):
        """Random input should produce finite (no NaN, no Inf) output."""
        output = encoder(synthetic_batch)
        assert torch.isfinite(output).all(), \
            f"Output contains non-finite values: {output}"

    def test_output_no_nan(self, encoder, synthetic_batch):
        """Explicit NaN check."""
        output = encoder(synthetic_batch)
        assert not torch.isnan(output).any(), \
            f"Output contains NaN values"

    def test_output_no_inf(self, encoder, synthetic_batch):
        """Explicit Inf check."""
        output = encoder(synthetic_batch)
        assert not torch.isinf(output).any(), \
            f"Output contains Inf values"

    def test_zero_input_produces_finite_output(self, encoder):
        """All-zero input should still produce finite output."""
        x = torch.zeros(4, 2, 20)
        output = encoder(x)
        assert torch.isfinite(output).all(), \
            f"Zero input produced non-finite output: {output}"

    def test_large_input_produces_finite_output(self, encoder):
        """Large but reasonable input should produce finite output."""
        torch.manual_seed(42)
        x = torch.randn(4, 2, 20) * 100  # large values
        output = encoder(x)
        assert torch.isfinite(output).all(), \
            f"Large input produced non-finite output"


# ===========================================================================
# Test 13: CNN gradient flow — All parameters receive non-zero gradients
# ===========================================================================

class TestCNNGradientFlow:
    def test_all_parameters_receive_gradients(self, encoder, synthetic_batch):
        """After backward pass, all parameters must have non-zero gradients."""
        output = encoder(synthetic_batch)
        loss = output.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, \
                f"Parameter '{name}' has no gradient"
            assert param.grad.abs().sum() > 0, \
                f"Parameter '{name}' has zero gradient"

    def test_regression_head_gradient_flow(self, encoder, synthetic_batch):
        """With a regression head (Linear(16, 1)), gradients flow back."""
        # The spec says Stage 1 uses a Linear(16, 1) head for fwd_return_h regression.
        head = torch.nn.Linear(16, 1)
        embedding = encoder(synthetic_batch)
        prediction = head(embedding)
        target = torch.randn(4, 1)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()

        for name, param in encoder.named_parameters():
            assert param.grad is not None, \
                f"Parameter '{name}' received no gradient through regression head"
            assert param.grad.abs().sum() > 0, \
                f"Parameter '{name}' has zero gradient through regression head"


# ===========================================================================
# Test 14: CNN regression loss decreases — 10 epochs on synthetic data
# ===========================================================================

class TestCNNLossDecreases:
    def test_mse_decreases_over_10_epochs(self, synthetic_dataset):
        """Spec: '10 epochs on synthetic data, MSE decreases'."""
        X, y = synthetic_dataset
        torch.manual_seed(42)

        encoder = CNNEncoder()
        head = torch.nn.Linear(16, 1)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=1e-3, weight_decay=1e-5
        )

        losses = []
        for epoch in range(10):
            encoder.train()
            head.train()
            optimizer.zero_grad()

            embedding = encoder(X)
            prediction = head(embedding).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at epoch 10 must be strictly less than loss at epoch 1
        assert losses[-1] < losses[0], \
            f"MSE did not decrease: epoch 1 loss={losses[0]:.6f}, " \
            f"epoch 10 loss={losses[-1]:.6f}"

    def test_loss_decreases_monotonically_in_early_epochs(self, synthetic_dataset):
        """Loss should generally decrease in early epochs (first 5)."""
        X, y = synthetic_dataset
        torch.manual_seed(42)

        encoder = CNNEncoder()
        head = torch.nn.Linear(16, 1)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=1e-3, weight_decay=1e-5
        )

        losses = []
        for epoch in range(5):
            encoder.train()
            head.train()
            optimizer.zero_grad()

            embedding = encoder(X)
            prediction = head(embedding).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # At least 3 out of 4 transitions should be decreasing
        decreasing_count = sum(1 for i in range(1, len(losses)) if losses[i] < losses[i-1])
        assert decreasing_count >= 3, \
            f"Expected at least 3/4 decreasing transitions in first 5 epochs, " \
            f"got {decreasing_count}. Losses: {losses}"


# ===========================================================================
# Test 15: CNN deterministic — Two runs with seed=42 produce identical loss
# ===========================================================================

class TestCNNDeterministic:
    def _train_and_get_loss_at_epoch(self, seed, n_epochs):
        """Train CNN encoder and return loss at each epoch."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        X = torch.randn(200, 2, 20)
        y = X[:, 0, :].mean(dim=1) + 0.1 * torch.randn(200)

        torch.manual_seed(seed)
        encoder = CNNEncoder()
        head = torch.nn.Linear(16, 1)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(head.parameters()),
            lr=1e-3, weight_decay=1e-5
        )

        losses = []
        for epoch in range(n_epochs):
            encoder.train()
            head.train()
            optimizer.zero_grad()

            embedding = encoder(X)
            prediction = head(embedding).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses

    def test_same_seed_same_loss_at_epoch_5(self):
        """Spec: 'Two runs with seed=42 produce identical loss at epoch 5'."""
        losses_run1 = self._train_and_get_loss_at_epoch(42, 5)
        losses_run2 = self._train_and_get_loss_at_epoch(42, 5)

        assert losses_run1[4] == pytest.approx(losses_run2[4], abs=1e-6), \
            f"Loss at epoch 5 differs between runs: " \
            f"{losses_run1[4]:.8f} vs {losses_run2[4]:.8f}"

    def test_all_epochs_match_with_same_seed(self):
        """All epoch losses should match between runs with same seed."""
        losses_run1 = self._train_and_get_loss_at_epoch(42, 10)
        losses_run2 = self._train_and_get_loss_at_epoch(42, 10)

        for epoch in range(10):
            assert losses_run1[epoch] == pytest.approx(losses_run2[epoch], abs=1e-6), \
                f"Loss at epoch {epoch+1} differs: " \
                f"{losses_run1[epoch]:.8f} vs {losses_run2[epoch]:.8f}"

    def test_different_seed_different_loss(self):
        """Different seeds should produce different losses."""
        losses_42 = self._train_and_get_loss_at_epoch(42, 5)
        losses_99 = self._train_and_get_loss_at_epoch(99, 5)

        # At least one epoch should differ
        any_different = any(
            abs(losses_42[i] - losses_99[i]) > 1e-4 for i in range(5)
        )
        assert any_different, \
            "Different seeds should produce different training trajectories"
