"""Pytest fixtures for DDPM tests."""

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from diffusionsde.config import DiffusionConfig
from diffusionsde.diffusion import DiffusionModel
from diffusionsde.models import ConditionalUNet


@pytest.fixture
def config():
    """Create a test configuration with small parameters for fast testing."""
    return DiffusionConfig(
        n_timesteps=10,
        trajectory_length=50,
        batch_size=4,
        learning_rate=1e-3,
        n_epochs=2,
        hidden_dim=32,
        n_layers=2,
        device="cpu",
    )


@pytest.fixture
def unet(config):
    """Create a ConditionalUNet for testing."""
    return ConditionalUNet(
        input_dim=config.trajectory_length,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        time_emb_dim=32,
    )


@pytest.fixture
def diffusion_model(config, unet):
    """Create a DiffusionModel for testing with default DDIM sampler."""
    return DiffusionModel(config, unet, sampler_type="ddim")


@pytest.fixture
def sample_data(config):
    """Create sample data for testing."""
    batch_size = config.batch_size
    trajectory_length = config.trajectory_length

    data = torch.randn(batch_size, trajectory_length)
    conditioning = torch.randn(batch_size, trajectory_length)

    return data, conditioning


@pytest.fixture
def sample_dataset(config):
    """Create a sample TensorDataset for training tests."""
    n_samples = 20
    data = torch.randn(n_samples, config.trajectory_length)
    conditioning = torch.randn(n_samples, config.trajectory_length)
    return TensorDataset(data, conditioning)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility in tests."""
    torch.manual_seed(42)
    np.random.seed(42)


def assert_finite(tensor):
    """Assert that a tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values"


def assert_finite_with_shape(tensor, expected_shape):
    """Assert that a tensor has the expected shape and contains no NaN or Inf."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    assert_finite(tensor)
