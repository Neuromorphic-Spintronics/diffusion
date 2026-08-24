"""Tests for sampling methods."""

import pytest
import torch

from diffusionsde.sampling import DDIMSampler, DDPMSampler, create_sampler
from tests.conftest import assert_finite_with_shape


class TestDDPMSampler:
    """Test suite for DDPM sampler."""

    @pytest.fixture
    def ddpm_sampler(self, config, unet):
        """Create a DDPM sampler for testing."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return DDPMSampler(config, unet, betas, alphas_cumprod)

    def test_reverse_step_shape(self, ddpm_sampler, config):
        """Test that reverse step produces correct shape with finite values."""
        batch_size = 4
        x_t = torch.randn(batch_size, config.trajectory_length)
        t = torch.full((batch_size,), 5, dtype=torch.long)
        conditioning = torch.randn(batch_size, config.trajectory_length)

        x_t_minus_1 = ddpm_sampler.reverse_step(x_t, t, conditioning, add_noise=True)

        assert_finite_with_shape(x_t_minus_1, x_t.shape)

    def test_reverse_step_no_noise_at_t0(self, ddpm_sampler, config):
        """Test that no noise is added at t=0."""
        x_t = torch.randn(2, config.trajectory_length)
        t = torch.full((2,), 0, dtype=torch.long)
        conditioning = torch.randn(2, config.trajectory_length)

        torch.manual_seed(1)
        result1 = ddpm_sampler.reverse_step(x_t, t, conditioning, add_noise=True)
        torch.manual_seed(2)
        result2 = ddpm_sampler.reverse_step(x_t, t, conditioning, add_noise=True)

        assert torch.allclose(result1, result2)

    def test_sample_shape(self, ddpm_sampler, config):
        """Test that sampling produces correct shape with finite values."""
        conditioning = torch.randn(config.trajectory_length)
        samples = ddpm_sampler.sample(conditioning, n_samples=3, require_grad=False)

        assert_finite_with_shape(samples, (3, config.trajectory_length))


class TestDDIMSampler:
    """Test suite for DDIM sampler."""

    @pytest.fixture
    def ddim_sampler(self, config, unet):
        """Create a DDIM sampler for testing."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.0)

    def test_reverse_step_shape(self, ddim_sampler, config):
        """Test that DDIM reverse step produces correct shape with finite values."""
        batch_size = 4
        x_t = torch.randn(batch_size, config.trajectory_length)
        conditioning = torch.randn(batch_size, config.trajectory_length)

        x_t_prev = ddim_sampler.reverse_step(x_t, t=5, t_prev=4, conditioning=conditioning)

        assert_finite_with_shape(x_t_prev, x_t.shape)

    def test_deterministic_sampling_eta_0(self, ddim_sampler, config):
        """Test that DDIM with eta=0 is deterministic."""
        conditioning = torch.randn(config.trajectory_length)

        torch.manual_seed(42)
        samples1 = ddim_sampler.sample(conditioning, n_samples=1, require_grad=False)
        torch.manual_seed(42)
        samples2 = ddim_sampler.sample(conditioning, n_samples=1, require_grad=False)

        assert torch.allclose(samples1, samples2)

    def test_stochastic_sampling_eta_positive(self, config, unet):
        """Test that DDIM with eta>0 adds stochasticity."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sampler = DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.5)

        conditioning = torch.randn(config.trajectory_length)

        torch.manual_seed(1)
        samples1 = sampler.sample(conditioning, n_samples=1, require_grad=False)
        torch.manual_seed(2)
        samples2 = sampler.sample(conditioning, n_samples=1, require_grad=False)

        assert not torch.allclose(samples1, samples2)

    def test_sample_shape(self, ddim_sampler, config):
        """Test that DDIM sampling produces correct shape."""
        conditioning = torch.randn(config.trajectory_length)
        samples = ddim_sampler.sample(conditioning, n_samples=3, require_grad=False)

        assert samples.shape == (3, config.trajectory_length)

    def test_reduced_steps(self, config, unet):
        """Test DDIM with fewer inference steps."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sampler = DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.0, num_inference_steps=5)

        assert sampler.num_inference_steps == 5

        conditioning = torch.randn(config.trajectory_length)
        samples = sampler.sample(conditioning, n_samples=1, require_grad=False)

        assert samples.shape == (1, config.trajectory_length)
        assert not torch.isnan(samples).any()

    @pytest.mark.parametrize("num_steps", [1, 2, 5, 32])
    def test_reduced_steps_schedule_matches_request(self, config, unet, num_steps):
        """Test that DDIM honours the requested inference-step count exactly."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sampler = DDIMSampler(
            config, unet, betas, alphas_cumprod, eta=0.0, num_inference_steps=num_steps
        )

        expected_steps = min(num_steps, config.n_timesteps)

        assert len(sampler.timesteps) == expected_steps
        assert torch.all(sampler.timesteps[1:] > sampler.timesteps[:-1])
        assert int(sampler.timesteps[-1].item()) == config.n_timesteps - 1
        if expected_steps > 1:
            assert int(sampler.timesteps[0].item()) == 0
        else:
            assert int(sampler.timesteps[0].item()) == config.n_timesteps - 1

    def test_grad_steps_preserves_gradients(self, config, unet):
        """Test that bounded-gradient DDIM sampling still backpropagates."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        sampler = DDIMSampler(config, unet, betas, alphas_cumprod, eta=0.0, num_inference_steps=5)

        conditioning = torch.randn(2, config.trajectory_length, requires_grad=True)
        initial_noise = torch.randn(2, config.trajectory_length)

        samples = sampler.sample(
            conditioning,
            n_samples=1,
            require_grad=True,
            initial_noise=initial_noise,
            show_progress=False,
            grad_steps=1,
        )
        loss = samples.square().mean()
        loss.backward()

        assert conditioning.grad is not None
        assert conditioning.grad.shape == conditioning.shape
        assert not torch.isnan(conditioning.grad).any()


class TestCreateSampler:
    """Test suite for sampler factory function."""

    def test_create_ddpm_sampler(self, config, unet):
        """Test creating DDPM sampler."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        sampler = create_sampler("ddpm", config, unet, betas, alphas_cumprod)

        assert isinstance(sampler, DDPMSampler)

    def test_create_ddim_sampler(self, config, unet):
        """Test creating DDIM sampler."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        sampler = create_sampler("ddim", config, unet, betas, alphas_cumprod)

        assert isinstance(sampler, DDIMSampler)

    def test_create_ddim_with_kwargs(self, config, unet):
        """Test creating DDIM sampler with custom kwargs."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        sampler = create_sampler(
            "ddim", config, unet, betas, alphas_cumprod, eta=0.5, num_inference_steps=5
        )

        assert isinstance(sampler, DDIMSampler)
        assert sampler.eta == 0.5
        assert sampler.num_inference_steps == 5

    def test_create_sampler_case_insensitive(self, config, unet):
        """Test that sampler type is case insensitive."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        sampler1 = create_sampler("DDPM", config, unet, betas, alphas_cumprod)
        sampler2 = create_sampler("DdIm", config, unet, betas, alphas_cumprod)

        assert isinstance(sampler1, DDPMSampler)
        assert isinstance(sampler2, DDIMSampler)

    def test_create_sampler_invalid_type(self, config, unet):
        """Test that invalid sampler type raises error."""
        betas = torch.linspace(config.beta_start, config.beta_end, config.n_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        with pytest.raises(ValueError, match="Unknown sampler type"):
            create_sampler("invalid", config, unet, betas, alphas_cumprod)
