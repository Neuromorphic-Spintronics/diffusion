"""DDPM: Denoising Diffusion Probabilistic Models.

A modular implementation of DDPMs for time series generation with conditioning.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from diffusionsde.config import DiffusionConfig
from diffusionsde.diffusion import DiffusionModel
from diffusionsde.distances import moment_matching_loss
from diffusionsde.evaluator import (
    StatisticsEvaluator,
    compute_higher_moments,
    compute_marginal_wasserstein,
    compute_psd,
    sample_acf,
)
from diffusionsde.inverse import InverseProblem, InverseProblemConfig, InverseResult
from diffusionsde.models import ConditionalUNet, compute_dilated_rf
from diffusionsde.plotting import (
    configure_matplotlib,
    plot_acf_comparison,
    plot_loss_curve,
    plot_marginal_distribution,
    plot_phase_space,
    plot_psd_comparison,
    plot_sample_trajectories,
    plot_trajectory_statistics,
)
from diffusionsde.trainer import DiffusionTrainer


class DDPM:
    """High-level API for training and using Denoising Diffusion Probabilistic Models.

    This class provides a simple interface for training DDPMs on time series data
    with conditioning information.

    Args:
        config: Diffusion configuration
    """

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        sampler_type: str = "ddim",
        sampler_kwargs: dict | None = None,
    ):
        """Initialise DDPM with configuration.

        Args:
            config: Diffusion configuration
            sampler_type: Type of sampler ('ddpm' or 'ddim')
            sampler_kwargs: Additional arguments for the sampler
        """
        self.config = config or DiffusionConfig()
        self.sampler_type = sampler_type
        self.sampler_kwargs = sampler_kwargs or {}
        self._model: DiffusionModel | None = None
        self._trainer: DiffusionTrainer | None = None

    def _build_unet(self, config: DiffusionConfig | None = None) -> ConditionalUNet:
        """Build the U-Net architecture from config."""
        config = config or self.config
        return ConditionalUNet(
            input_dim=config.trajectory_length,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_input_channels=config.n_output_channels + config.n_conditioning_channels,
            n_output_channels=config.n_output_channels,
            use_position_encoding=config.use_position_encoding,
            dilation_schedule=config.dilation_schedule,
            kernel_size=config.kernel_size,
            causal=getattr(config, "causal", False),
        )

    def fit(
        self,
        dataset: Any,
        val_dataset: Any | None = None,
        n_epochs: int | None = None,
    ):
        """Train the diffusion model on the provided dataset.

        Args:
            dataset: Training dataset that returns (data, conditioning) pairs
            val_dataset: Optional validation dataset
            n_epochs: Number of training epochs (overrides config if provided)

        Returns:
            Dictionary of training metrics
        """
        config = (
            replace(self.config, n_epochs=n_epochs)
            if n_epochs is not None
            else replace(self.config)
        )

        # Keep the current public model usable if initialisation or training fails.
        unet = self._build_unet(config)
        model = DiffusionModel(config, unet, self.sampler_type, self.sampler_kwargs)

        # Initialise trainer
        trainer = DiffusionTrainer(model, dataset, val_dataset=val_dataset, config=config)

        # Train
        print(f"Training DDPM on {self.config.device}...")
        print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
        metrics = trainer.train()
        self.config = config
        self._model = model
        self._trainer = trainer

        return metrics

    def generate(
        self,
        conditioning: torch.Tensor,
        n_samples: int = 1,
        require_grad: bool = False,
        initial_noise: torch.Tensor | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples given conditioning data.

        Args:
            conditioning: Conditioning data of shape [trajectory_length] or [batch_size, trajectory_length]
            n_samples: Number of samples to generate per conditioning
            require_grad: If True, enable gradient computation
            initial_noise: Optional fixed initial noise tensor for deterministic generation
            show_progress: If True, show tqdm progress bar. Disable for optimisation loops.
            **kwargs: Additional keyword arguments forwarded to the sampler (e.g. ``grad_steps=1``
                to use only the final denoising step for gradient computation).

        Returns:
            Generated samples of shape [n_samples, trajectory_length]
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        return self._model.sampler.sample(
            conditioning, n_samples, require_grad, initial_noise, show_progress, **kwargs
        )

    def save(self, path: str | Path):
        """Save the trained model to disk.

        Args:
            path: Path to save the model checkpoint
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        path = Path(path).absolute()
        if self._trainer is not None:
            self._trainer.save_checkpoint(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": self._model.model.state_dict(), "config": self.config},
                path,
            )

    def load(self, path: str | Path):
        """Load a trained model from disk.

        Args:
            path: Path to the model checkpoint
        """
        map_location = self.config.device

        checkpoint = torch.load(path, weights_only=False, map_location=map_location)

        # Build fully before publishing so failed loads leave current state intact.
        config = checkpoint["config"]
        config.device = map_location
        model = DiffusionModel(
            config, self._build_unet(config), self.sampler_type, self.sampler_kwargs
        )
        model.model.load_state_dict(checkpoint["model_state_dict"])
        self.config = config
        self._model = model
        self._trainer = None

        print(f"Model loaded from {path}")

    def freeze(self, frozen: bool = True) -> None:
        """Freeze model parameters for inference, or unfreeze them with ``False``."""
        model = self.model
        model.model.train(not frozen)
        for param in model.model.parameters():
            param.requires_grad_(not frozen)

    @property
    def model(self) -> DiffusionModel:
        """Get the underlying diffusion model."""
        if self._model is None:
            raise RuntimeError("Model not initialised. Call fit() first.")
        return self._model


__all__ = [
    "DDPM",
    "DiffusionConfig",
    "ConditionalUNet",
    "compute_dilated_rf",
    "StatisticsEvaluator",
    "sample_acf",
    "compute_psd",
    "compute_marginal_wasserstein",
    "compute_higher_moments",
    "configure_matplotlib",
    "plot_trajectory_statistics",
    "plot_sample_trajectories",
    "plot_acf_comparison",
    "plot_psd_comparison",
    "plot_marginal_distribution",
    "plot_loss_curve",
    "plot_phase_space",
    "moment_matching_loss",
    "InverseProblem",
    "InverseProblemConfig",
    "InverseResult",
]
