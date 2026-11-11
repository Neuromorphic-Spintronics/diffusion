"""Core checks for forward diffusion and training loss."""

import torch


def test_forward_diffusion_matches_closed_form(diffusion_model, sample_data):
    data, _ = sample_data
    timesteps = torch.zeros(data.shape[0], dtype=torch.long)
    noise = torch.randn_like(data)
    noisy, returned_noise = diffusion_model.forward_diffusion(data, timesteps, noise)
    expected = (
        diffusion_model.sqrt_alphas_cumprod[timesteps, None] * data
        + diffusion_model.sqrt_one_minus_alphas_cumprod[timesteps, None] * noise
    )
    torch.testing.assert_close(noisy, expected)
    torch.testing.assert_close(returned_noise, noise)


def test_compute_loss_backpropagates(diffusion_model, sample_data):
    data, conditioning = sample_data
    data = data.detach().requires_grad_(True)
    loss = diffusion_model.compute_loss(data, conditioning)
    assert torch.isfinite(loss)
    loss.backward()
    assert data.grad is not None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in diffusion_model.model.parameters()
    )
