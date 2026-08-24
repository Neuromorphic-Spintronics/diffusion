"""Unit tests for W&B logging utilities.

Tests all logging functionality without actually writing to W&B.
Focused on verifying graceful fallback and correct API usage.
"""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt

# Import private module variables for mocking
import diffusionsde.logging as logging_module
from diffusionsde.config import DiffusionConfig
from diffusionsde.logging import (
    _detect_project_name,
    _flatten_config,
    _generate_run_name,
    wandb_finish,
    wandb_log,
    wandb_log_artifact,
)


class TestWandbDisabled:
    """Test that all W&B functions work silently when disabled."""

    @patch.object(logging_module, "_WANDB_ENABLED", False)
    @patch.object(logging_module, "wandb", None)
    def test_all_functions_silent_when_disabled(self, tmp_path):
        """All W&B functions should work without errors when disabled."""
        from diffusionsde.logging import is_wandb_enabled, wandb_init, wandb_log_figure

        # Check disabled state
        assert is_wandb_enabled() is False

        # Init should return False
        config = DiffusionConfig(trajectory_length=250)
        assert wandb_init(config=config, run_prefix="test") is False

        # All logging functions should silently do nothing
        wandb_log({"loss": 0.5, "epoch": 1})
        wandb_log({"train/loss": 0.3}, step=10)

        # Figure logging
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        wandb_log_figure(fig, "test_plot")
        plt.close(fig)

        # Artifact logging
        test_file = tmp_path / "model.pt"
        test_file.write_text("test data")
        wandb_log_artifact(test_file, name="test-model")

        # Finish
        wandb_finish()


class TestHelperFunctions:
    """Test internal helper functions."""

    @patch("inspect.stack")
    def test_detect_project_name(self, mock_stack):
        """Should detect project name from examples/ directory."""
        # Mock being in examples/duffing/train.py
        mock_frame = MagicMock()
        mock_frame.filename = "/some/path/examples/duffing/train.py"
        mock_stack.return_value = [mock_frame]

        assert _detect_project_name() == "duffing"

    def test_generate_run_name(self):
        """Should generate run name with prefix and params."""
        name = _generate_run_name(prefix="train", params={"gamma": 0.5})

        assert name.startswith("train_")
        assert "gamma-0.5" in name or "gamma-0.50" in name
        # Should end with timestamp
        assert len(name.split("_")[-1]) == 15  # YYYYMMDD-HHMMSS

    def test_flatten_config(self):
        """Should flatten nested config structures."""
        # Simple dict
        assert _flatten_config({"lr": 1e-4, "epochs": 100}) == {"lr": 1e-4, "epochs": 100}

        # Nested dict
        nested = {"model": {"hidden_dim": 128}, "train": {"lr": 1e-4}}
        flat = _flatten_config(nested)
        assert flat == {"model/hidden_dim": 128, "train/lr": 1e-4}

        # Dataclass
        config = DiffusionConfig(trajectory_length=250, batch_size=32)
        flat = _flatten_config(config)
        assert flat["trajectory_length"] == 250
        assert flat["batch_size"] == 32


class TestWandbEnabled:
    """Test W&B functionality when enabled (using mocks)."""

    @patch.object(logging_module, "_WANDB_ENABLED", True)
    @patch.object(logging_module, "_WANDB_ENTITY", "test-entity")
    @patch.object(logging_module, "wandb")
    def test_wandb_init_calls_api(self, mock_wandb):
        """Should call wandb.init with correct parameters."""
        from diffusionsde.logging import wandb_init

        mock_wandb.run = None
        config = DiffusionConfig(trajectory_length=250, batch_size=32)

        result = wandb_init(config=config, run_name="test-run", tags=["test"])

        assert result is True
        mock_wandb.init.assert_called_once()
        call_kwargs = mock_wandb.init.call_args[1]
        assert call_kwargs["name"] == "test-run"
        assert call_kwargs["tags"] == ["test"]
        assert "trajectory_length" in call_kwargs["config"]

    @patch.object(logging_module, "_WANDB_ENABLED", True)
    @patch.object(logging_module, "wandb")
    def test_wandb_log_with_active_run(self, mock_wandb):
        """Should log metrics when run is active."""
        mock_wandb.run = MagicMock()

        wandb_log({"loss": 0.5}, step=10)

        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=10)

    @patch.object(logging_module, "_WANDB_ENABLED", True)
    @patch.object(logging_module, "wandb")
    def test_wandb_log_artifact(self, mock_wandb, tmp_path):
        """Should log artifacts when run is active."""
        mock_wandb.run = MagicMock()
        mock_artifact = MagicMock()
        mock_wandb.Artifact.return_value = mock_artifact

        test_file = tmp_path / "model.pt"
        test_file.write_text("test data")

        wandb_log_artifact(test_file, name="model", artifact_type="model")

        mock_wandb.Artifact.assert_called_once()
        mock_artifact.add_file.assert_called_once()
        mock_wandb.log_artifact.assert_called_once()
