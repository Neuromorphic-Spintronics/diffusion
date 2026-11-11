"""Tests for differentiable ensemble moment matching."""

import torch

from diffusionsde.distances import moment_matching_loss


def test_moment_matching_loss_has_gradients_and_weighted_terms():
    generated = torch.randn(32, 12, requires_grad=True)
    observed = torch.randn(32, 12)

    combined = moment_matching_loss(generated, observed)
    mean_only = moment_matching_loss(generated, observed, variance_weight=0.0)
    variance_only = moment_matching_loss(generated, observed, mean_weight=0.0)
    combined.backward()

    assert torch.isclose(mean_only + variance_only, combined.detach(), atol=1e-6)
    assert generated.grad is not None
    assert generated.grad.shape == generated.shape


def test_precomputed_observation_statistics_match_direct_loss():
    generated = torch.randn(32, 12)
    observed = torch.randn(24, 12)
    precomputed = (observed.mean(dim=0), observed.var(dim=0))

    direct = moment_matching_loss(generated, observed)
    cached = moment_matching_loss(generated, observed, precomputed_obs_stats=precomputed)

    assert torch.isclose(direct, cached, atol=1e-6)
