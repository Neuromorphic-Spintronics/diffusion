"""Tests for trainer."""

import math
import tempfile
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from diffusionsde.models import ConditionalUNet
from diffusionsde.trainer import DiffusionTrainer


class TestTrainerInitialisation:
    """Test suite for trainer initialisation."""

    def test_trainer_init(self, diffusion_model, sample_dataset, config):
        """Test trainer initialisation."""
        trainer = DiffusionTrainer(diffusion_model, sample_dataset, config=config)

        assert trainer.model == diffusion_model
        assert trainer.train_dataset == sample_dataset
        assert len(trainer.train_losses) == 0
        assert len(trainer.val_losses) == 0

    def test_trainer_with_validation_set(self, diffusion_model, sample_dataset, config):
        """Test trainer initialisation with validation dataset."""
        val_dataset = TensorDataset(
            torch.randn(10, config.trajectory_length), torch.randn(10, config.trajectory_length)
        )

        trainer = DiffusionTrainer(
            diffusion_model, sample_dataset, val_dataset=val_dataset, config=config
        )

        assert trainer.val_dataset == val_dataset
        assert trainer.val_loader is not None


class TestTraining:
    """Test suite for training loop."""

    def test_train_returns_metrics(self, diffusion_model, sample_dataset, config):
        """Test that training returns finite metrics."""
        trainer = DiffusionTrainer(diffusion_model, sample_dataset, config=config)
        metrics = trainer.train()

        assert "train_loss" in metrics
        assert len(metrics["train_loss"]) == config.n_epochs
        assert all(math.isfinite(loss) for loss in metrics["train_loss"])

    def test_train_with_validation(self, diffusion_model, sample_dataset, config):
        """Test training with validation set."""
        val_dataset = TensorDataset(
            torch.randn(10, config.trajectory_length), torch.randn(10, config.trajectory_length)
        )

        trainer = DiffusionTrainer(
            diffusion_model, sample_dataset, val_dataset=val_dataset, config=config
        )
        metrics = trainer.train()

        assert "val_loss" in metrics
        assert len(metrics["val_loss"]) == config.n_epochs

    def test_training_updates_model(self, diffusion_model, sample_dataset, config):
        """Test that training actually updates model parameters."""
        trainer = DiffusionTrainer(diffusion_model, sample_dataset, config=config)

        initial_params = {
            name: param.clone() for name, param in diffusion_model.model.named_parameters()
        }

        trainer.train()

        params_changed = False
        for name, param in diffusion_model.model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed, "Model parameters should change during training"


class TestCheckpointing:
    """Test suite for checkpoint save/load."""

    def test_save_checkpoint(self, diffusion_model, sample_dataset, config):
        """Test saving a checkpoint."""
        trainer = DiffusionTrainer(diffusion_model, sample_dataset, config=config)
        trainer.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            assert checkpoint_path.exists()

    def test_load_checkpoint(self, diffusion_model, sample_dataset, config, unet):
        """Test loading a checkpoint."""
        from diffusionsde.diffusion import DiffusionModel

        trainer = DiffusionTrainer(diffusion_model, sample_dataset, config=config)
        trainer.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            new_unet = ConditionalUNet(
                input_dim=config.trajectory_length,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                time_emb_dim=32,
            )
            new_model = DiffusionModel(config, new_unet)
            new_trainer = DiffusionTrainer(new_model, sample_dataset, config=config)

            new_trainer.load_checkpoint(checkpoint_path)

            assert len(new_trainer.train_losses) == len(trainer.train_losses)

    def test_checkpoint_contains_required_keys(self, diffusion_model, sample_dataset, config):
        """Test that checkpoint contains all required keys."""
        trainer = DiffusionTrainer(diffusion_model, sample_dataset, config=config)
        trainer.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            checkpoint = torch.load(checkpoint_path, weights_only=False)

            assert "model_state_dict" in checkpoint
            assert "optimiser_state_dict" in checkpoint
            assert "scheduler_state_dict" in checkpoint
            assert "train_losses" in checkpoint
            assert "config" in checkpoint
