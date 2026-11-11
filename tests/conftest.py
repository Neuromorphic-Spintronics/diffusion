"""Pytest fixtures for DDPM tests."""

import matplotlib
import pytest
import torch
from torch.utils.data import TensorDataset

from diffusionsde.config import DiffusionConfig
from diffusionsde.diffusion import DiffusionModel
from diffusionsde.models import ConditionalUNet

matplotlib.use("Agg")
torch.set_num_threads(1)


@pytest.fixture
def config():
    """Create a test configuration with small parameters for fast testing."""
    return DiffusionConfig(
        n_timesteps=4,
        trajectory_length=16,
        batch_size=4,
        learning_rate=1e-3,
        n_epochs=2,
        hidden_dim=8,
        n_layers=2,
        device="cpu",
        checkpoint_interval=0,
    )


@pytest.fixture
def unet(config):
    """Create a ConditionalUNet for testing."""
    return ConditionalUNet(
        input_dim=config.trajectory_length,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        time_emb_dim=8,
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
    n_samples = 8
    data = torch.randn(n_samples, config.trajectory_length)
    conditioning = torch.randn(n_samples, config.trajectory_length)
    return TensorDataset(data, conditioning)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility in tests."""
    torch.manual_seed(42)
