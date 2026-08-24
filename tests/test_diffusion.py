"""Tests for diffusion model."""

import pytest
import torch

from tests.conftest import assert_finite


class TestNoiseSchedule:
    """Test suite for noise schedule generation."""

    def test_noise_schedule_shape(self, diffusion_model, config):
        """Test that noise schedule has correct shape."""
        betas = diffusion_model.betas
        assert len(betas) == config.n_timesteps

    def test_noise_schedule_bounds(self, diffusion_model, config):
        """Test that noise schedule values are bounded correctly."""
        betas = diffusion_model.betas

        assert betas[0] == pytest.approx(config.beta_start, rel=1e-5)
        assert betas[-1] == pytest.approx(config.beta_end, rel=1e-5)
        assert torch.all(betas > 0)
        assert torch.all(betas < 1)

    def test_noise_schedule_monotonic(self, diffusion_model):
        """Test that noise schedule is monotonically increasing."""
        betas = diffusion_model.betas
        assert torch.all(betas[1:] >= betas[:-1])

    def test_alphas_cumprod(self, diffusion_model):
        """Test that cumulative product of alphas is computed correctly."""
        alphas = diffusion_model.alphas
        alphas_cumprod = diffusion_model.alphas_cumprod

        expected_cumprod = torch.cumprod(alphas, dim=0)
        assert torch.allclose(alphas_cumprod, expected_cumprod)


class TestForwardDiffusion:
    """Test suite for forward diffusion process."""

    def test_forward_diffusion_shape(self, diffusion_model, sample_data, config):
        """Test forward diffusion output shapes and values are finite."""
        data, _ = sample_data

        t = torch.randint(0, config.n_timesteps, (config.batch_size,))
        x_t, noise = diffusion_model.forward_diffusion(data, t)

        assert x_t.shape == data.shape
        assert noise.shape == data.shape
        assert_finite(x_t)

    def test_forward_diffusion_noise_level_increases(self, diffusion_model, sample_data, config):
        """Test that noise increases with timestep."""
        data, _ = sample_data
        batch_size = config.batch_size

        t_early = torch.zeros(batch_size, dtype=torch.long)
        x_t_early, _ = diffusion_model.forward_diffusion(data, t_early)

        t_late = torch.full((batch_size,), config.n_timesteps - 1, dtype=torch.long)
        x_t_late, _ = diffusion_model.forward_diffusion(data, t_late)

        mse_early = torch.mean((x_t_early - data) ** 2)
        mse_late = torch.mean((x_t_late - data) ** 2)

        assert mse_early < mse_late

    def test_forward_diffusion_deterministic_with_noise(self, diffusion_model, sample_data, config):
        """Test that forward diffusion is deterministic when noise is provided."""
        data, _ = sample_data
        t = torch.randint(0, config.n_timesteps, (config.batch_size,))
        noise = torch.randn_like(data)

        x_t_1, _ = diffusion_model.forward_diffusion(data, t, noise=noise)
        x_t_2, _ = diffusion_model.forward_diffusion(data, t, noise=noise)

        assert torch.allclose(x_t_1, x_t_2)

    def test_forward_diffusion_t0_preserves_signal(self, diffusion_model, sample_data):
        """Test that at t=0, the signal is mostly preserved."""
        data, _ = sample_data
        batch_size = data.shape[0]

        t = torch.zeros(batch_size, dtype=torch.long)
        x_t, _ = diffusion_model.forward_diffusion(data, t)

        correlation = torch.corrcoef(torch.stack([data.flatten(), x_t.flatten()]))[0, 1]
        assert correlation > 0.9


class TestComputeLoss:
    """Test suite for loss computation."""

    def test_loss_non_negative_and_finite(self, diffusion_model, sample_data):
        """Test that loss is non-negative and finite."""
        data, conditioning = sample_data
        loss = diffusion_model.compute_loss(data, conditioning)
        assert loss.item() >= 0
        assert_finite(loss)

    def test_loss_differentiable(self, diffusion_model, sample_data):
        """Test that loss is differentiable."""
        data, conditioning = sample_data
        data.requires_grad_(True)

        loss = diffusion_model.compute_loss(data, conditioning)
        loss.backward()

        assert data.grad is not None


class TestSampleGeneration:
    """Test suite for sample generation."""

    def test_sample_shape_1d_conditioning(self, diffusion_model, config):
        """Test sample shape with 1D conditioning input."""
        conditioning = torch.randn(config.trajectory_length)
        samples = diffusion_model.sampler.sample(conditioning, n_samples=3, require_grad=False)

        assert samples.shape == (3, config.trajectory_length)
        assert not torch.isnan(samples).any()

    def test_sample_shape_2d_conditioning(self, diffusion_model, sample_data, config):
        """Test sample shape with 2D conditioning input."""
        _, conditioning = sample_data
        samples = diffusion_model.sampler.sample(conditioning, n_samples=2, require_grad=False)

        expected_batch = conditioning.shape[0] * 2
        assert samples.shape == (expected_batch, config.trajectory_length)

    def test_sample_require_grad(self, diffusion_model, config):
        """Test that gradients can be computed through sampling."""
        conditioning = torch.randn(config.trajectory_length, requires_grad=True)
        samples = diffusion_model.sampler.sample(conditioning, n_samples=1, require_grad=True)

        loss = samples.sum()
        loss.backward()

        assert conditioning.grad is not None

    def test_samples_vary_across_runs(self, diffusion_model, config):
        """Test that different random seeds produce different samples."""
        conditioning = torch.randn(config.trajectory_length)

        torch.manual_seed(1)
        samples1 = diffusion_model.sampler.sample(conditioning, n_samples=1, require_grad=False)

        torch.manual_seed(2)
        samples2 = diffusion_model.sampler.sample(conditioning, n_samples=1, require_grad=False)

        assert not torch.allclose(samples1, samples2)
