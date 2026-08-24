"""Configuration for diffusion models."""

from dataclasses import dataclass

import torch


@dataclass
class DiffusionConfig:
    r"""Configuration for the diffusion model.

    Attributes:
        n_timesteps: Number of DENOISING timesteps T (algorithmic time for diffusion process)
        trajectory_length: Length of PHYSICAL trajectory (number of time points in data)
        beta_start: Starting value of noise schedule $\beta_0$
        beta_end: Ending value of noise schedule $\beta_T$
        batch_size: Training batch size
        learning_rate: Optimiser learning rate
        n_epochs: Number of training epochs
        hidden_dim: Hidden dimension for U-Net
        n_layers: Number of layers in U-Net
        device: Device for computation
    """

    n_timesteps: int = 1000  # Denoising algorithm timesteps
    trajectory_length: int = 250  # Physical trajectory length
    beta_start: float = 0.0001  # Noise schedule: initial value
    beta_end: float = 0.02  # Noise schedule: final value
    batch_size: int = 32
    learning_rate: float = 1e-4
    n_epochs: int = 50
    hidden_dim: int = 128  # Neural network architecture
    n_layers: int = 4  # Neural network architecture
    n_conditioning_channels: int = (
        1  # Number of conditioning channels (1 for scalar, >1 for multi-channel)
    )
    n_output_channels: int = 1  # Number of output channels (1 for scalar, >1 for multi-channel)
    use_position_encoding: bool = False  # Enable physical position encoding in U-Net
    dilation_schedule: list[int] | None = None  # Per-level dilation factors (None = no dilation)
    kernel_size: int = 3  # Convolution kernel size (must be odd)
    causal: bool = False  # Use causal convolutions (outputs depend only on the past)
    loss_type: str = "mse"  # "mse" or "huber"
    huber_delta: float = 1.0  # $\delta$ for the Huber/smooth-L1 threshold
    checkpoint_interval: int = 10  # Epoch interval; zero disables periodic checkpoints
    checkpoint_dir: str | None = None  # Optional directory for periodic checkpoints
    device: str = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
