"""Tests for high-level DDPM API."""

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from diffusionsde import DDPM, DiffusionConfig
from tests.conftest import assert_finite_with_shape


class TestDDPMInitialisation:
    """Test suite for DDPM initialisation."""

    def test_default_init(self):
        """Test DDPM initialisation with defaults."""
        ddpm = DDPM()

        assert ddpm.config is not None
        assert ddpm.sampler_type == "ddim"
        assert ddpm._model is None


class TestDDPMFit:
    """Test suite for DDPM fit method."""

    def test_fit_returns_metrics(self, config, sample_dataset):
        """Test that fit returns training metrics."""
        ddpm = DDPM(config=config)
        metrics = ddpm.fit(sample_dataset)

        assert "train_loss" in metrics
        assert len(metrics["train_loss"]) == config.n_epochs

    def test_fit_with_validation(self, config, sample_dataset):
        """Test fit with validation dataset."""
        val_dataset = TensorDataset(
            torch.randn(10, config.trajectory_length), torch.randn(10, config.trajectory_length)
        )

        ddpm = DDPM(config=config)
        metrics = ddpm.fit(sample_dataset, val_dataset=val_dataset)

        assert "val_loss" in metrics

    def test_fit_override_epochs(self, config, sample_dataset):
        """Test that n_epochs can be overridden in fit."""
        ddpm = DDPM(config=config)
        metrics = ddpm.fit(sample_dataset, n_epochs=1)

        assert len(metrics["train_loss"]) == 1


class TestDDPMGenerate:
    """Test suite for DDPM generate method."""

    def test_generate_requires_training(self, config):
        """Test that generate raises error before training."""
        ddpm = DDPM(config=config)

        conditioning = torch.randn(config.trajectory_length)
        with pytest.raises(RuntimeError, match="Model not trained"):
            ddpm.generate(conditioning)

    def test_generate_shape_and_values(self, config, sample_dataset):
        """Test that generate produces correct shape with finite values."""
        ddpm = DDPM(config=config)
        ddpm.fit(sample_dataset)

        conditioning = torch.randn(config.trajectory_length)
        samples = ddpm.generate(conditioning, n_samples=3)

        assert_finite_with_shape(samples, (3, config.trajectory_length))

    def test_generate_with_multi_channel_conditioning(self, config):
        """Test generation when the model is conditioned on two parameter channels."""
        multi_config = replace(config, n_conditioning_channels=2)
        n_samples = 12
        data = torch.randn(n_samples, multi_config.trajectory_length)
        conditioning = torch.randn(n_samples, 2, multi_config.trajectory_length)
        dataset = TensorDataset(data, conditioning)

        ddpm = DDPM(config=multi_config)
        ddpm.fit(dataset, n_epochs=1)

        conditioning_query = torch.randn(2, multi_config.trajectory_length)
        samples = ddpm.generate(conditioning_query, n_samples=3, show_progress=False)

        assert_finite_with_shape(samples, (3, multi_config.trajectory_length))


class TestDDPMSaveLoad:
    """Test suite for DDPM save/load methods."""

    def test_save_requires_training(self, config):
        """Test that save raises error before training."""
        ddpm = DDPM(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            with pytest.raises(RuntimeError, match="Model not trained"):
                ddpm.save(path)

    def test_save_load_roundtrip(self, config, sample_dataset):
        """Test that model can be saved and loaded."""
        ddpm = DDPM(config=config)
        ddpm.fit(sample_dataset)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pt"
            ddpm.save(path)

            ddpm_loaded = DDPM(config=config)
            ddpm_loaded.load(path)

            device = ddpm_loaded.config.device
            conditioning = torch.randn(config.trajectory_length, device=device)

            torch.manual_seed(42)
            samples = ddpm_loaded.generate(conditioning, n_samples=1)

            assert samples.shape == (1, config.trajectory_length)
            assert not torch.isnan(samples).any()

    def test_save_load_roundtrip_multi_channel(self, config):
        """Test checkpoint roundtrip for a two-channel conditioned model."""
        multi_config = replace(config, n_conditioning_channels=2)
        n_samples = 12
        data = torch.randn(n_samples, multi_config.trajectory_length)
        conditioning = torch.randn(n_samples, 2, multi_config.trajectory_length)
        dataset = TensorDataset(data, conditioning)

        ddpm = DDPM(config=multi_config)
        ddpm.fit(dataset, n_epochs=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model_multi.pt"
            ddpm.save(path)

            ddpm_loaded = DDPM(config=DiffusionConfig())
            ddpm_loaded.load(path)

            assert ddpm_loaded.config.n_conditioning_channels == 2

            device = ddpm_loaded.config.device
            conditioning_query = torch.randn(2, multi_config.trajectory_length, device=device)
            samples = ddpm_loaded.generate(conditioning_query, n_samples=1, show_progress=False)

            assert samples.shape == (1, multi_config.trajectory_length)
            assert not torch.isnan(samples).any()


class TestDDPMModelProperty:
    """Test suite for DDPM model property."""

    def test_model_property_before_training(self, config):
        """Test that model property raises error before training."""
        ddpm = DDPM(config=config)

        with pytest.raises(RuntimeError, match="Model not initialised"):
            _ = ddpm.model

    def test_model_property_after_training(self, config, sample_dataset):
        """Test that model property returns model after training."""
        ddpm = DDPM(config=config)
        ddpm.fit(sample_dataset)

        model = ddpm.model
        assert model is not None
