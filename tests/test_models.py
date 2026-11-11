"""Tests for causal information flow in the conditional U-Net."""

import pytest
import torch

from diffusionsde.models import ConditionalUNet


@pytest.mark.parametrize("causal", [True, False])
def test_causality_controls_future_leakage(causal):
    torch.manual_seed(0)
    input_dim = 33
    cutoff = input_dim // 2
    model = ConditionalUNet(
        input_dim=input_dim,
        hidden_dim=8,
        n_layers=2,
        n_input_channels=3,
        n_output_channels=2,
        causal=causal,
    )
    model.eval()

    x = torch.randn(1, input_dim)
    conditioning = torch.randn(1, 2, input_dim)
    changed_x = x.clone()
    changed_conditioning = conditioning.clone()
    changed_x[:, cutoff:] += torch.randn_like(changed_x[:, cutoff:]) * 5
    changed_conditioning[:, cutoff:] += torch.randn_like(changed_conditioning[:, cutoff:]) * 5
    timestep = torch.tensor([2])

    with torch.no_grad():
        original = model(x, timestep, conditioning)
        changed = model(changed_x, timestep, changed_conditioning)

    assert original.shape == (1, 2, input_dim)
    same_past = torch.allclose(original[..., :cutoff], changed[..., :cutoff], atol=1e-5)
    assert same_past is causal
