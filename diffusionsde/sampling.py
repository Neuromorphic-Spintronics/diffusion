"""Sampling methods for diffusion models.

This module implements different sampling strategies for denoising diffusion models:
- DDPM: Original sampling from Ho et al. (2020)
- DDIM: Deterministic sampling from Song et al. (2021) - faster inference
"""

from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from diffusionsde.config import DiffusionConfig
from diffusionsde.models import ConditionalUNet


class DiffusionSampler(ABC):
    """Abstract base class for diffusion samplers.

    Args:
        config: Diffusion configuration
        model: Conditional U-Net for denoising
        betas: Noise schedule
        alphas_cumprod: Cumulative product of alphas
    """

    def __init__(
        self,
        config: DiffusionConfig,
        model: ConditionalUNet,
        betas: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ):
        self.config = config
        self.model = model
        self.device = config.device
        self.betas = betas
        self.alphas_cumprod = alphas_cumprod
        self.alphas = 1.0 - betas

    @abstractmethod
    def sample(
        self,
        conditioning: torch.Tensor,
        n_samples: int = 1,
        require_grad: bool = False,
        initial_noise: torch.Tensor | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples given conditioning data.

        Args:
            conditioning: Conditioning data
            n_samples: Number of samples to generate
            require_grad: If True, enable gradient computation
            initial_noise: Optional fixed initial noise tensor
            show_progress: If True, show tqdm progress bar

        Returns:
            Generated samples
        """
        pass

    def _prepare_sampling(
        self,
        conditioning: torch.Tensor,
        n_samples: int,
        require_grad: bool,
        initial_noise: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare conditioning and initial noise for sampling.

        Handles model eval mode, conditioning expansion, device transfer,
        gradient setup, and noise initialisation.

        Args:
            conditioning: Raw conditioning data
            n_samples: Number of samples per conditioning
            require_grad: Whether to enable gradient computation
            initial_noise: Optional fixed initial noise

        Returns:
            Tuple of (prepared conditioning, initial noise x_t)
        """
        if not require_grad:
            self.model.eval()

        # Ensure conditioning has batch dimension
        if conditioning.dim() == 1:
            conditioning = conditioning.unsqueeze(0).repeat(n_samples, 1)
        elif conditioning.dim() == 2:
            if self.config.n_conditioning_channels > 1:
                conditioning = conditioning.unsqueeze(0).repeat(n_samples, 1, 1)
            else:
                conditioning = conditioning.repeat_interleave(n_samples, dim=0)
        elif conditioning.dim() == 3:
            conditioning = conditioning.repeat_interleave(n_samples, dim=0)

        conditioning = conditioning.to(self.device)
        if require_grad and not conditioning.requires_grad:
            conditioning.requires_grad_(True)

        if initial_noise is not None:
            x_t = initial_noise.to(self.device)
        else:
            batch_size = conditioning.shape[0]
            seq_len = conditioning.shape[-1]
            n_output_channels = getattr(self.config, "n_output_channels", 1)
            if n_output_channels > 1:
                x_t = torch.randn(batch_size, n_output_channels, seq_len, device=self.device)
            else:
                x_t = torch.randn(batch_size, seq_len, device=self.device)

        return conditioning, x_t


class DDPMSampler(DiffusionSampler):
    """DDPM sampling (Ho et al., 2020).

    Uses the full reverse diffusion process with stochastic sampling.
    Requires all T timesteps for generation.
    """

    def __init__(
        self,
        config: DiffusionConfig,
        model: ConditionalUNet,
        betas: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ):
        super().__init__(config, model, betas, alphas_cumprod)

        # Precompute posterior variance for efficiency
        alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def reverse_step(
        self, x_t: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor, add_noise: bool = True
    ) -> torch.Tensor:
        """Single DDPM reverse diffusion step.

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep
            conditioning: Conditioning data
            add_noise: Whether to add noise (False for t=0)

        Returns:
            x_{t-1}: Less noisy data
        """
        # Predict noise
        if add_noise or x_t.requires_grad or conditioning.requires_grad:
            predicted_noise = self.model(x_t, t, conditioning)
        else:
            with torch.no_grad():
                predicted_noise = self.model(x_t, t, conditioning)

        # Extract coefficients and broadcast across data and channel dimensions.
        coeff_shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
        alpha = self.alphas[t].reshape(coeff_shape)
        alpha_cumprod = self.alphas_cumprod[t].reshape(coeff_shape)
        beta = self.betas[t].reshape(coeff_shape)

        # Compute mean
        mean = (x_t - beta * predicted_noise / torch.sqrt(1.0 - alpha_cumprod)) / torch.sqrt(alpha)

        if add_noise and t[0] > 0:
            # Add noise except for the last step
            posterior_variance = self.posterior_variance[t].reshape(coeff_shape)
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(posterior_variance) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    def sample(
        self,
        conditioning: torch.Tensor,
        n_samples: int = 1,
        require_grad: bool = False,
        initial_noise: torch.Tensor | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples using DDPM sampling.

        Args:
            conditioning: Conditioning data of shape [trajectory_length] or [batch_size, trajectory_length]
            n_samples: Number of samples to generate per conditioning
            require_grad: If True, enable gradient computation
            initial_noise: Optional fixed initial noise tensor. If provided, must have shape
                          [batch_size * n_samples, trajectory_length]. Useful for consistent
                          gradients in optimisation.
            show_progress: If True, show tqdm progress bar. Disable for optimisation loops.

        Returns:
            Generated samples of shape [n_samples, trajectory_length]
        """
        conditioning, x_t = self._prepare_sampling(
            conditioning, n_samples, require_grad, initial_noise
        )

        # Reverse diffusion
        timestep_iterator = range(self.config.n_timesteps - 1, -1, -1)
        if show_progress:
            timestep_iterator = tqdm(timestep_iterator, desc="DDPM Sampling")

        for t_idx in timestep_iterator:
            t = torch.full((conditioning.shape[0],), t_idx, dtype=torch.long).to(self.device)
            x_t = self.reverse_step(x_t, t, conditioning, add_noise=(t_idx > 0))

        return x_t


class DDIMSampler(DiffusionSampler):
    """DDIM sampling (Song et al., 2021).

    Deterministic (or semi-deterministic) sampling that allows for faster inference
    by skipping timesteps. Can generate samples in fewer steps than DDPM.

    Args:
        config: Diffusion configuration
        model: Conditional U-Net for denoising
        betas: Noise schedule
        alphas_cumprod: Cumulative product of alphas
        eta: Controls stochasticity (0 = deterministic, 1 = stochastic like DDPM)
        num_inference_steps: Number of steps to use (can be less than training steps)
    """

    def __init__(
        self,
        config: DiffusionConfig,
        model: ConditionalUNet,
        betas: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        eta: float = 0.0,
        num_inference_steps: int | None = None,
    ):
        super().__init__(config, model, betas, alphas_cumprod)
        self.eta = eta

        # Use fewer steps for faster inference
        self.num_inference_steps = num_inference_steps or config.n_timesteps

        # Create timestep schedule (evenly spaced) while honouring the
        # requested step count exactly.  The schedule is stored in ascending
        # order and reversed at sampling time.
        if self.num_inference_steps < config.n_timesteps:
            self.timesteps = (
                torch.linspace(
                    config.n_timesteps - 1,
                    0,
                    steps=self.num_inference_steps,
                )
                .round()
                .long()
                .flip(0)
            )
            self.timesteps = torch.unique_consecutive(self.timesteps)
        else:
            self.timesteps = torch.arange(config.n_timesteps).long()

    def reverse_step(
        self, x_t: torch.Tensor, t: int, t_prev: int, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """Single DDIM reverse step.

        Uses the deterministic (or semi-deterministic) DDIM update rule:
        x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon + sigma_t * noise

        where pred_x0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)

        Args:
            x_t: Noisy data at timestep t
            t: Current timestep index
            t_prev: Previous timestep index (can skip timesteps)
            conditioning: Conditioning data

        Returns:
            x_{t_prev}: Less noisy data at previous timestep
        """
        # Predict noise
        t_tensor = torch.full((x_t.shape[0],), t, dtype=torch.long, device=self.device)

        if x_t.requires_grad or conditioning.requires_grad:
            predicted_noise = self.model(x_t, t_tensor, conditioning)
        else:
            with torch.no_grad():
                predicted_noise = self.model(x_t, t_tensor, conditioning)

        # Get alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0 from x_t and predicted noise
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

        # Compute variance (eta controls stochasticity)
        sigma_t = self.eta * torch.sqrt(
            (1.0 - alpha_t_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_t_prev)
        )

        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_t_prev - sigma_t**2) * predicted_noise

        # Compute x_{t-1}
        x_t_prev = torch.sqrt(alpha_t_prev) * pred_x0 + dir_xt

        # Add noise if eta > 0 and not at the last step
        if self.eta > 0 and t_prev >= 0:
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise

        return x_t_prev

    def sample(
        self,
        conditioning: torch.Tensor,
        n_samples: int = 1,
        require_grad: bool = False,
        initial_noise: torch.Tensor | None = None,
        show_progress: bool = True,
        grad_steps: int | None = None,
        grad_steps_spread: bool = False,
    ) -> torch.Tensor:
        """Generate samples using DDIM sampling.

        Args:
            conditioning: Conditioning data of shape [trajectory_length] or [batch_size, trajectory_length]
            n_samples: Number of samples to generate per conditioning
            require_grad: If True, enable gradient computation
            initial_noise: Optional fixed initial noise tensor. If provided, must have shape
                          [batch_size * n_samples, trajectory_length]. Useful for consistent
                          gradients in optimisation.
            show_progress: If True, show tqdm progress bar. Disable for optimisation loops.
            grad_steps: If set, restricts autograd to a subset of denoising steps.  By
                default (``grad_steps_spread=False``) these are the *final* ``grad_steps``
                steps (largest denoising effect, lowest noise).  If ``grad_steps_spread=True``
                the steps are chosen *linearly across the full chain*, capturing gradient
                signal at the start, middle, and end of denoising.
            grad_steps_spread: If ``True``, spread the ``grad_steps`` gradient-carrying
                steps uniformly across the chain instead of concentrating them at the end.

        Returns:
            Generated samples of shape [n_samples, trajectory_length]
        """
        conditioning, x_t = self._prepare_sampling(
            conditioning, n_samples, require_grad, initial_noise
        )

        if x_t.shape[0] != conditioning.shape[0]:
            raise ValueError(
                f"initial_noise batch size ({x_t.shape[0]}) must match "
                f"conditioning batch size ({conditioning.shape[0]})"
            )

        # Reverse diffusion with potentially fewer steps
        timesteps_reversed = self.timesteps.flip(0)
        n_total = len(timesteps_reversed)

        # Determine which step indices carry gradients
        if grad_steps is None:
            grad_indices: set[int] | None = None
        elif grad_steps_spread:
            # Linearly spaced across the full chain (inclusive of first and last)
            grad_indices = set(
                int(round(i))
                for i in torch.linspace(0, n_total - 1, min(grad_steps, n_total)).tolist()
            )
        else:
            # Final grad_steps steps only
            grad_indices = set(range(n_total - grad_steps, n_total))

        if show_progress:
            desc = f"DDIM Sampling ({self.num_inference_steps} steps)"
            timestep_iterator = enumerate(tqdm(timesteps_reversed, desc=desc))
        else:
            timestep_iterator = enumerate(timesteps_reversed)

        for i, t_idx in timestep_iterator:
            t = int(t_idx.item())
            # Get previous timestep (or -1 for the last step)
            t_prev = (
                int(timesteps_reversed[i + 1].item()) if i + 1 < len(timesteps_reversed) else -1
            )

            # Optionally restrict autograd graph to selected steps
            if grad_indices is not None and i not in grad_indices:
                with torch.no_grad():
                    x_t = self.reverse_step(x_t.detach(), t, t_prev, conditioning.detach())
            else:
                x_t = self.reverse_step(x_t, t, t_prev, conditioning)

        return x_t


def create_sampler(
    sampler_type: str,
    config: DiffusionConfig,
    model: ConditionalUNet,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    **kwargs,
) -> DiffusionSampler:
    """Factory function to create a sampler.

    Args:
        sampler_type: Type of sampler ('ddpm' or 'ddim')
        config: Diffusion configuration
        model: Conditional U-Net
        betas: Noise schedule
        alphas_cumprod: Cumulative product of alphas
        **kwargs: Additional arguments for specific samplers
            - For DDIM: eta (float), num_inference_steps (int)

    Returns:
        Sampler instance
    """
    sampler_type = sampler_type.lower()

    if sampler_type == "ddpm":
        return DDPMSampler(config, model, betas, alphas_cumprod)
    elif sampler_type == "ddim":
        return DDIMSampler(
            config,
            model,
            betas,
            alphas_cumprod,
            eta=kwargs.get("eta", 0.0),
            num_inference_steps=kwargs.get("num_inference_steps", None),
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}. Choose 'ddpm' or 'ddim'.")
