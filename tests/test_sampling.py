"""Core behavior checks for reverse samplers."""

import pytest
import torch

from diffusionsde.sampling import DDIMSampler, DDPMSampler


@pytest.fixture(params=[DDPMSampler, DDIMSampler])
def sampler(request, config, unet):
    betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    if request.param is DDIMSampler:
        return DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.0)
    return DDPMSampler(config, unet, betas, alphas_cumprod)


def test_sample_shape_and_no_grad(sampler, config):
    result = sampler.sample(torch.randn(config.trajectory_length), n_samples=2, show_progress=False)
    assert result.shape == (2, config.trajectory_length)
    assert torch.isfinite(result).all()
    assert not result.requires_grad


def test_ddim_deterministic_and_stochastic(config, unet):
    betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    conditioning = torch.randn(config.trajectory_length)
    initial_noise = torch.randn(1, config.trajectory_length)
    deterministic = DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.0)
    torch.manual_seed(1)
    first = deterministic.sample(conditioning, initial_noise=initial_noise, show_progress=False)
    torch.manual_seed(2)
    second = deterministic.sample(conditioning, initial_noise=initial_noise, show_progress=False)
    torch.testing.assert_close(first, second)

    stochastic = DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.5)
    torch.manual_seed(1)
    first = stochastic.sample(conditioning, initial_noise=initial_noise, show_progress=False)
    torch.manual_seed(2)
    second = stochastic.sample(conditioning, initial_noise=initial_noise, show_progress=False)
    assert not torch.allclose(first, second)


def test_terminal_steps_do_not_add_noise(config, unet):
    betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    x = torch.randn(2, config.trajectory_length)
    conditioning = torch.randn_like(x)
    t0 = torch.zeros(2, dtype=torch.long)
    ddpm = DDPMSampler(config, unet, betas, alphas_cumprod)
    torch.manual_seed(1)
    a = ddpm.reverse_step(x, t0, conditioning, add_noise=True)
    torch.manual_seed(2)
    b = ddpm.reverse_step(x, t0, conditioning, add_noise=True)
    torch.testing.assert_close(a, b)

    ddim = DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.5)
    torch.manual_seed(1)
    a = ddim.reverse_step(x, config.n_timesteps - 1, -1, conditioning)
    torch.manual_seed(2)
    b = ddim.reverse_step(x, config.n_timesteps - 1, -1, conditioning)
    torch.testing.assert_close(a, b)


def test_gradient_flow_under_outer_no_grad_and_input_unchanged(config, unet):
    betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sampler = DDIMSampler(config, unet, betas, alphas_cumprod)
    conditioning = torch.randn(1, config.trajectory_length, requires_grad=True)
    with torch.no_grad():
        result = sampler.sample(
            conditioning,
            require_grad=True,
            show_progress=False,
            grad_steps=1,
            grad_steps_spread=True,
        )
    result.square().mean().backward()
    assert conditioning.grad is not None
    unchanged = torch.randn(config.trajectory_length)
    sampler.sample(unchanged, show_progress=False)
    assert unchanged.requires_grad is False
    assert unet.training is True


def test_restores_model_mode_after_sampling_error(config, unet):
    betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sampler = DDIMSampler(config, unet, betas, alphas_cumprod)
    unet.train()
    with pytest.raises(ValueError, match="batch size"):
        sampler.sample(
            torch.randn(2, config.trajectory_length),
            initial_noise=torch.randn(1, config.trajectory_length),
            show_progress=False,
        )
    assert unet.training is True
    unet.eval()
    sampler.sample(torch.randn(config.trajectory_length), show_progress=False)
    assert unet.training is False


def test_invalid_gradient_and_schedule_counts(config, unet):
    betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    with pytest.raises(ValueError, match="num_inference_steps"):
        DDIMSampler(config, unet, betas, alphas_cumprod, num_inference_steps=0)
    sampler = DDIMSampler(config, unet, betas, alphas_cumprod)
    with pytest.raises(ValueError, match="grad_steps"):
        sampler.sample(torch.randn(config.trajectory_length), grad_steps=0, show_progress=False)


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_inference_schedule_has_requested_bounded_length(config, unet, steps):
    betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sampler = DDIMSampler(config, unet, betas, alphas_cumprod, num_inference_steps=steps)
    assert len(sampler.timesteps) == min(steps, config.n_timesteps)
    assert torch.all(sampler.timesteps[1:] > sampler.timesteps[:-1])
