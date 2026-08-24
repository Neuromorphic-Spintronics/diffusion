"""DDPM: Denoising Diffusion Probabilistic Models.

A modular implementation of DDPMs for time series generation with conditioning.
"""

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
from diffusionsde.logging import (
    wandb_finish,
    wandb_init,
    wandb_log,
    wandb_log_artifact,
)
from diffusionsde.models import ConditionalUNet, compute_dilated_rf
from diffusionsde.ornstein_uhlenbeck import (
    RANDOM_SEED,
    TRAJECTORY_LENGTH,
    VALID_CONDITIONING_MODES,
    OUDataset,
    OUParams,
    build_conditioning_batch,
    canonicalise_conditioning_mode,
    conditioning_channels,
    conditioning_mode_label,
    conditioning_mode_suffix,
    generate_observed_trajectories,
    generate_training_data,
    integrate_ou,
    metadata_filename_for_mode,
    model_filename_for_mode,
    output_stem_for_mode,
    save_data,
    split_trajectories,
)
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

    def _build_unet(self) -> ConditionalUNet:
        """Build the U-Net architecture from config."""
        return ConditionalUNet(
            input_dim=self.config.trajectory_length,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            n_input_channels=self.config.n_output_channels + self.config.n_conditioning_channels,
            n_output_channels=self.config.n_output_channels,
            use_position_encoding=self.config.use_position_encoding,
            dilation_schedule=self.config.dilation_schedule,
            kernel_size=self.config.kernel_size,
            causal=getattr(self.config, "causal", False),
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
        if n_epochs is not None:
            self.config.n_epochs = n_epochs

        # Initialise model
        unet = self._build_unet()
        self._model = DiffusionModel(self.config, unet, self.sampler_type, self.sampler_kwargs)

        # Initialise trainer
        self._trainer = DiffusionTrainer(
            self._model, dataset, val_dataset=val_dataset, config=self.config
        )

        # Train
        print(f"Training DDPM on {self.config.device}...")
        print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")
        metrics = self._trainer.train()

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
        if self._trainer is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        self._trainer.save_checkpoint(Path(path))

    def load(self, path: str | Path):
        """Load a trained model from disk.

        Args:
            path: Path to the model checkpoint
        """
        # Use the default device detection from DiffusionConfig
        map_location = DiffusionConfig().device

        checkpoint = torch.load(path, weights_only=False, map_location=map_location)

        # Initialise model from checkpoint config
        self.config = checkpoint["config"]
        self.config.device = map_location

        unet = self._build_unet()
        self._model = DiffusionModel(self.config, unet, self.sampler_type, self.sampler_kwargs)
        self._model.model.load_state_dict(checkpoint["model_state_dict"])

        print(f"Model loaded from {path}")

    def freeze(self) -> None:
        """Put model in frozen eval mode (no gradient computation)."""
        model = self.model
        model.model.eval()
        for param in model.model.parameters():
            param.requires_grad_(False)

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
    "wandb_init",
    "wandb_log",
    "wandb_log_artifact",
    "wandb_finish",
    "moment_matching_loss",
    "InverseProblem",
    "InverseProblemConfig",
    "InverseResult",
    "OUParams",
    "OUDataset",
    "TRAJECTORY_LENGTH",
    "RANDOM_SEED",
    "VALID_CONDITIONING_MODES",
    "integrate_ou",
    "generate_training_data",
    "generate_observed_trajectories",
    "build_conditioning_batch",
    "canonicalise_conditioning_mode",
    "conditioning_channels",
    "conditioning_mode_label",
    "conditioning_mode_suffix",
    "metadata_filename_for_mode",
    "model_filename_for_mode",
    "output_stem_for_mode",
    "save_data",
    "split_trajectories",
]
