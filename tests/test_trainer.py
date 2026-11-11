"""Focused trainer continuation and validation-mode tests."""

import pytest
import torch

from diffusionsde.diffusion import DiffusionModel
from diffusionsde.models import ConditionalUNet
from diffusionsde.trainer import DiffusionTrainer


def make_model(config):
    unet = ConditionalUNet(
        input_dim=config.trajectory_length,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        time_emb_dim=32,
    )
    return DiffusionModel(config, unet, sampler_type="ddim")


def test_checkpoint_resume_preserves_epoch_and_best_state(config, sample_dataset, tmp_path):
    config.n_epochs = 1
    trainer = DiffusionTrainer(make_model(config), sample_dataset, sample_dataset, config)
    trainer.train()
    checkpoint = tmp_path / "resume.pt"
    trainer.save_checkpoint(checkpoint)

    config.n_epochs = 2
    resumed = DiffusionTrainer(make_model(config), sample_dataset, sample_dataset, config)
    resumed.load_checkpoint(checkpoint)
    assert resumed.completed_epochs == 1
    assert resumed.best_state_dict is not None
    previous_weights = {
        name: value.clone() for name, value in resumed.model.model.state_dict().items()
    }
    resumed.train()

    assert resumed.completed_epochs == 2
    assert len(resumed.train_losses) == len(resumed.val_losses) == 2
    assert resumed.best_epoch is not None
    assert resumed.resume_state_dict is not None
    assert any(
        not torch.equal(previous_weights[name], value)
        for name, value in resumed.resume_state_dict.items()
    )
    for name, value in resumed.best_state_dict.items():
        torch.testing.assert_close(resumed.model.model.state_dict()[name], value)
    resumed.train()
    assert len(resumed.train_losses) == 2


def test_validation_restores_model_mode_on_success_and_error(config, sample_dataset, monkeypatch):
    trainer = DiffusionTrainer(make_model(config), sample_dataset, sample_dataset, config)
    trainer.model.model.eval()
    trainer._validate()
    assert not trainer.model.model.training

    def fail(*args, **kwargs):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(trainer.model, "compute_loss", fail)
    with pytest.raises(RuntimeError, match="validation failed"):
        trainer._validate()
    assert not trainer.model.model.training
