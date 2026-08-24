"""Tests for DiffusionConfig."""

from diffusionsde.config import DiffusionConfig


class TestDiffusionConfig:
    """Test suite for DiffusionConfig dataclass."""

    def test_default_values(self):
        """Test that default configuration values are sensible."""
        config = DiffusionConfig()

        assert config.n_timesteps == 1000
        assert config.trajectory_length == 250
        assert config.beta_start == 0.0001
        assert config.beta_end == 0.02
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.n_epochs == 50
        assert config.hidden_dim == 128
        assert config.n_layers == 4

    def test_device_auto_detection(self):
        """Test that device is auto-detected when not specified."""
        config = DiffusionConfig()
        # Device should be a valid string
        assert config.device in ["cpu", "cuda", "mps"]
