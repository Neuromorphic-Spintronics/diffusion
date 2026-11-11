"""Focused end-to-end tests for the public DDPM lifecycle."""

import pytest
import torch

from diffusionsde import DDPM


def test_fit_save_load_and_save_after_load(config, sample_dataset, tmp_path):
    ddpm = DDPM(config)
    metrics = ddpm.fit(sample_dataset, n_epochs=1)
    assert len(metrics["train_loss"]) == 1

    first = tmp_path / "first.pt"
    ddpm.save(first)

    loaded = DDPM(config)
    loaded.load(first)
    assert loaded._trainer is None
    second = tmp_path / "nested" / "second.pt"
    loaded.save(second)
    conditioning = torch.randn(config.trajectory_length)
    noise = torch.randn(1, config.trajectory_length)
    torch.testing.assert_close(
        ddpm.generate(conditioning, initial_noise=noise, show_progress=False),
        loaded.generate(conditioning, initial_noise=noise, show_progress=False),
    )
    loaded.freeze()
    assert not loaded.model.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.model.parameters())
    loaded.freeze(False)
    assert loaded.model.model.training
    assert all(parameter.requires_grad for parameter in loaded.model.model.parameters())

    # Loading into a previously fitted instance must discard its stale trainer.
    prior_model, prior_trainer = ddpm.model, ddpm._trainer
    with pytest.raises(FileNotFoundError):
        ddpm.load(tmp_path / "missing.pt")
    assert ddpm.model is prior_model
    assert ddpm._trainer is prior_trainer

    ddpm.load(second)
    assert ddpm._trainer is None
    third = tmp_path / "third.pt"
    ddpm.save(third)
    saved = torch.load(third, map_location="cpu", weights_only=False)
    for name, parameter in loaded.model.model.state_dict().items():
        torch.testing.assert_close(saved["model_state_dict"][name], parameter)


def test_failed_fit_does_not_publish_untrained_model(config, sample_dataset, monkeypatch):
    from diffusionsde.trainer import DiffusionTrainer

    def fail(self):
        raise RuntimeError("training failed")

    monkeypatch.setattr(DiffusionTrainer, "train", fail)
    ddpm = DDPM(config)
    with pytest.raises(RuntimeError, match="training failed"):
        ddpm.fit(sample_dataset)
    assert ddpm._model is None
    assert ddpm._trainer is None

    # A failed replacement must preserve the previous configuration as well.
    previous_config = ddpm.config
    with pytest.raises(RuntimeError, match="training failed"):
        ddpm.fit(sample_dataset, n_epochs=9)
    assert ddpm.config is previous_config
    assert ddpm.config.n_epochs == config.n_epochs
