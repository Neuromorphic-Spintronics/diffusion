"""Core diffusion model implementation."""

import torch
import torch.nn.functional as F

from diffusionsde.config import DiffusionConfig
from diffusionsde.models import ConditionalUNet
from diffusionsde.sampling import DiffusionSampler, create_sampler


class DiffusionModel:
    r"""DDPM (Denoising Diffusion Probabilistic Model) implementation.

    See Ho, et al., (2020).

    This class implements the forward diffusion process $q(x_t|x_0)$ and
    the learned reverse process $p_\theta(x_{t-1}|x_t, c)$.

    The forward process adds Gaussian noise:
        $q(x_t|x_{t-1}) = N(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$

    The reverse process removes noise conditioned on conditioning data:
        $p_\theta(x_{t-1}|x_t, c) = N(x_{t-1}; \mu_\theta(x_t, t, c), \sigma_t^2 I)$

    Args:
        config: Diffusion configuration
        model: Conditional U-Net for denoising
        sampler_type: Type of sampler to use ('ddpm' or 'ddim')
        sampler_kwargs: Additional arguments for the sampler
    """

    def __init__(
        self,
        config: DiffusionConfig,
        model: ConditionalUNet,
        sampler_type: str = "ddim",
        sampler_kwargs: dict | None = None,
    ) -> None:
        r"""Initialise diffusion model with noise schedule."""
        self.config = config
        self.model = model.to(config.device)
        self.device = config.device

        # Create linear noise schedule $\beta_t$
        self.betas = torch.linspace(
            self.config.beta_start, self.config.beta_end, self.config.n_timesteps
        ).to(self.device)

        # Precompute values for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Create sampler
        sampler_kwargs = sampler_kwargs or {}
        self.sampler: DiffusionSampler = create_sampler(
            sampler_type, config, self.model, self.betas, self.alphas_cumprod, **sampler_kwargs
        )

    def forward_diffusion(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Forward diffusion process $q(x_t|x_0)$.

        Directly samples $x_t$ from $x_0$ using the formula:
            $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon$
        where $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

        Args:
            x_0: Clean data of shape [batch_size, trajectory_length]
            t: Timesteps of shape [batch_size]
            noise: Optional pre-generated noise

        Returns:
            Tuple of (x_t, noise) where x_t is noisy data
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for the given timesteps
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting across data and channel dimensions.
        shape = (x_0.shape[0],) + (1,) * (x_0.dim() - 1)
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.reshape(shape)
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.reshape(shape)

        # Sample x_t
        x_t = sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def compute_loss(self, x_0: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        r"""Compute training loss for diffusion model.

        The loss is the MSE between predicted and true noise:
            $L = \mathbb{E}_{t,x_0,\varepsilon} \left[||\varepsilon - \varepsilon_\theta(x_t, t, c)||^2\right]$

        Args:
            x_0: Clean data
            conditioning: Conditioning data

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.config.n_timesteps, (batch_size,)).to(self.device)

        # Forward diffusion
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_diffusion(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t, conditioning)

        # Loss
        if getattr(self.config, "loss_type", "mse") == "huber":
            delta = getattr(self.config, "huber_delta", 1.0)
            loss = F.smooth_l1_loss(predicted_noise, noise, beta=delta)
        else:
            loss = F.mse_loss(predicted_noise, noise)

        return loss
