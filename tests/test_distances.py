"""Tests for distance metrics (moment matching)."""

import torch

from diffusionsde.distances import moment_matching_loss


class TestMomentMatchingLoss:
    """Tests for the pointwise moment-matching cost function."""

    def test_zero_for_identical_ensembles(self):
        """Loss is zero when generated and observed are the same."""
        data = torch.randn(50, 20)
        loss = moment_matching_loss(data, data.clone())
        assert loss.item() == 0.0

    def test_positive_for_different_ensembles(self):
        """Loss is positive when ensembles differ."""
        a = torch.randn(50, 20)
        b = torch.randn(50, 20) + 2.0
        loss = moment_matching_loss(a, b)
        assert loss.item() > 0.0

    def test_differentiable(self):
        """Loss supports backpropagation."""
        x = torch.randn(50, 20, requires_grad=True)
        obs = torch.randn(50, 20)
        loss = moment_matching_loss(x, obs)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_precomputed_stats(self):
        """Precomputed stats give the same result as computing from data."""
        gen = torch.randn(50, 20)
        obs = torch.randn(50, 20)
        loss_direct = moment_matching_loss(gen, obs)
        precomputed = (obs.mean(dim=0), obs.var(dim=0))
        loss_precomputed = moment_matching_loss(gen, obs, precomputed_obs_stats=precomputed)
        assert torch.isclose(loss_direct, loss_precomputed, atol=1e-6)

    def test_weights(self):
        """Mean and variance weights scale the respective terms."""
        gen = torch.randn(50, 20) + 1.0
        obs = torch.randn(50, 20)
        loss_default = moment_matching_loss(gen, obs)
        loss_mean_only = moment_matching_loss(gen, obs, variance_weight=0.0)
        loss_var_only = moment_matching_loss(gen, obs, mean_weight=0.0)
        # mean_only + var_only ≈ default (with default weights = 1)
        assert torch.isclose(loss_mean_only + loss_var_only, loss_default, atol=1e-6)
