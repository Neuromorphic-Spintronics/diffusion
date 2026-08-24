"""Neural network architectures for diffusion models."""

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep.

    Maps diffusion timestep t to a high-dimensional embedding using sinusoidal functions
    similar to transformer positional encodings (Vaswani, et al., (2017))

    Args:
        dim: Dimension of the embedding
        max_period: Maximum period for sinusoidal functions
    """

    dim: int
    max_period: float

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestep t.

        Args:
            t: Timestep tensor of shape [batch_size]

        Returns:
            Encoded timestep of shape [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        # Create frequency bands
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]

        # Apply sin and cos
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb


class CausalGroupNorm(nn.Module):
    r"""Group normalisation that uses only past timesteps.

    Standard group normalisation computes statistics over the full sequence,
    which lets future values influence earlier outputs.  This module instead
    computes cumulative statistics over channels in each group and timesteps
    up to the current position, preserving strict causality.

    Args:
        num_groups: Number of channel groups.
        num_channels: Number of input channels.
        eps: Small constant added to the variance for numerical stability.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.channels_per_group = num_channels // num_groups
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, seq_len = x.shape
        channels_per_group = self.channels_per_group
        grouped = x.reshape(batch_size, self.num_groups, channels_per_group, seq_len)

        cumulative_sum = grouped.cumsum(dim=-1)
        cumulative_sum_sq = (grouped * grouped).cumsum(dim=-1)

        counts = channels_per_group * torch.arange(
            1, seq_len + 1, device=x.device, dtype=x.dtype
        ).view(1, 1, 1, seq_len)
        group_sum = cumulative_sum.sum(dim=2, keepdim=True)
        group_sum_sq = cumulative_sum_sq.sum(dim=2, keepdim=True)

        mean = group_sum / counts
        variance = (group_sum_sq / counts - mean * mean).clamp_min(0.0)
        normalised = (grouped - mean) / torch.sqrt(variance + self.eps)

        output = normalised.reshape(batch_size, num_channels, seq_len)
        return output * self.weight.view(1, num_channels, 1) + self.bias.view(1, num_channels, 1)


def _make_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    dilation: int = 1,
    causal: bool = False,
) -> nn.Conv1d:
    """Build a length-preserving 1D convolution with optional causality."""
    if causal:
        return nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
    pad = ((kernel_size - 1) // 2) * dilation
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=pad,
        padding_mode="replicate",
        dilation=dilation,
    )


def _apply_conv(conv: nn.Conv1d, x: torch.Tensor, causal: bool) -> torch.Tensor:
    """Apply a convolution, zero-padding only on the left for causal mode."""
    if not causal:
        return conv(x)
    pad = int(conv.dilation[0]) * (int(conv.kernel_size[0]) - 1)
    return conv(F.pad(x, (pad, 0)))


class ResidualBlock(nn.Module):
    r"""Residual block with time conditioning and configurable dilation.

    Uses ``padding_mode="replicate"`` directly on the convolutions so that
    length preservation is correct for any odd kernel size and dilation $d$.

    See Ho, et al., (2020), or He, et al., (2016).

    Args:
        in_channels: Input channel count $C_{\mathrm{in}}$.
        out_channels: Output channel count $C_{\mathrm{out}}$.
        time_emb_dim: Dimension of the diffusion-timestep embedding $D_t$.
        dilation: Dilation factor $d$ (default $1$ = no dilation).
        kernel_size: Convolution kernel size $k$ (must be odd).
        causal: If True, use causal convolutions and causal group
            normalisation so outputs depend only on the past.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dilation: int = 1,
        kernel_size: int = 3,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.conv1 = _make_conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, causal=causal
        )
        self.conv2 = _make_conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, causal=causal
        )
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        norm_type = CausalGroupNorm if causal else nn.GroupNorm
        self.norm1 = norm_type(8, out_channels)
        self.norm2 = norm_type(8, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = _apply_conv(self.conv1, x, self.causal)
        h = self.norm1(h)
        h = self.act(h)

        time_emb = self.time_mlp(time_emb)[:, :, None]
        h = h + time_emb

        h = _apply_conv(self.conv2, h, self.causal)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    r"""Conditional U-Net for denoising time series signals.

    See Ronneberger, et al., (2015). As implemented in Ho, et al., (2020).

    This network takes noisy data $x_t$, diffusion timestep $t$, and conditioning data,
    and predicts the noise $\varepsilon$ to be removed.

    The architecture follows: $x_t, t, c \to \varepsilon_\theta(x_t, t, c)$

    When ``dilation_schedule`` is provided, a per-level dilation is applied to the
    encoder, middle, and decoder residual-block convolutions.  Dilation grows the
    theoretical receptive field at zero parameter cost.  The init, downsample,
    upsample and final convolutions keep $d = 1$.

    Args:
        input_dim: Dimension of input time series.
        hidden_dim: Hidden dimension for layers.
        n_layers: Number of layers in encoder/decoder.
        time_emb_dim: Dimension of timestep embedding.
        n_input_channels: Number of input channels (e.g., 2 for signal + conditioning).
        n_output_channels: Number of output channels (1 for scalar, >1 for joint multi-channel).
        use_position_encoding: If True, inject physical-time positional encoding.
        dilation_schedule: Per-level dilation factors, length ``n_layers`` (default: no dilation).
        kernel_size: Convolution kernel size $k$ (must be odd).
        causal: If True, use causal convolutions throughout the U-Net so that
            each output position depends only on input positions in the past.
    """

    input_dim: int
    hidden_dim: int
    n_layers: int
    channels: list[int]
    dilation_schedule: list[int]
    causal: bool
    physical_position_encoding: torch.Tensor

    def __init__(
        self,
        input_dim: int = 250,
        hidden_dim: int = 128,
        n_layers: int = 4,
        time_emb_dim: int = 128,
        n_input_channels: int = 2,
        n_output_channels: int = 1,
        use_position_encoding: bool = False,
        dilation_schedule: list[int] | None = None,
        kernel_size: int = 3,
        causal: bool = False,
    ) -> None:
        super().__init__()

        if dilation_schedule is None:
            dilation_schedule = [1] * n_layers
        if len(dilation_schedule) != n_layers:
            raise ValueError(
                f"dilation_schedule length {len(dilation_schedule)} must match n_layers {n_layers}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dilation_schedule = list(dilation_schedule)
        self.kernel_size = kernel_size
        self.causal = causal
        self.n_output_channels = n_output_channels

        k = kernel_size

        # Time embedding
        self.time_encoder = SinusoidalPositionalEncoding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Inject physical-time information so the network can distinguish
        # absolute trajectory positions away from the boundaries.
        # Only enabled for models trained with this feature (e.g. DWO).
        self.use_position_encoding = use_position_encoding
        if use_position_encoding:
            self.register_buffer(
                "physical_position_encoding",
                self._build_physical_position_encoding(input_dim, hidden_dim),
                persistent=False,
            )

        # Initial projection: concatenate $x_t$ and conditioning.
        self.init_conv = _make_conv1d(n_input_channels, hidden_dim, k, causal=causal)

        # Store channel configuration
        self.channels = [hidden_dim * (2**i) for i in range(n_layers)]

        # Encoder (downsampling)
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i in range(n_layers):
            in_ch = self.channels[i - 1] if i > 0 else hidden_dim
            out_ch = self.channels[i]
            d = self.dilation_schedule[i]

            self.encoder_blocks.append(
                ResidualBlock(in_ch, out_ch, time_emb_dim, dilation=d, kernel_size=k, causal=causal)
            )

            if i < n_layers - 1:
                self.downsample.append(_make_conv1d(out_ch, out_ch, k, stride=2, causal=causal))

        # Middle block (deepest dilation).
        mid_ch = self.channels[-1]
        mid_d = self.dilation_schedule[-1]
        self.middle_block = ResidualBlock(
            mid_ch, mid_ch, time_emb_dim, dilation=mid_d, kernel_size=k, causal=causal
        )

        # Decoder (upsampling) --- mirror of encoder.
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(n_layers - 1, -1, -1):
            if i == n_layers - 1:
                in_ch = self.channels[i]
            else:
                in_ch = self.channels[i] * 2

            out_ch = self.channels[i - 1] if i > 0 else hidden_dim
            d = self.dilation_schedule[i]

            self.decoder_blocks.append(
                ResidualBlock(in_ch, out_ch, time_emb_dim, dilation=d, kernel_size=k, causal=causal)
            )

            if i > 0:
                if causal:
                    upsample_layer = nn.Upsample(scale_factor=2, mode="nearest")
                else:
                    upsample_layer = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
                self.upsample.append(
                    nn.Sequential(
                        upsample_layer,
                        _make_conv1d(out_ch, out_ch, k, causal=causal),
                    )
                )

        # Final projection to the requested number of output channels.
        self.final_conv = _make_conv1d(hidden_dim, n_output_channels, k, causal=causal)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward pass for noise prediction.

        Args:
            x_t: Noisy data $[B, T]$.
            t: Diffusion timestep $[B]$.
            conditioning: Conditioning data $[B, T]$ (or $[B, C, T]$).

        Returns:
            Predicted noise $\varepsilon_\theta(x_t, t, c)$, shape $[B, T]$ for
            single-channel output, or $[B, C, T]$ for multi-channel output.
        """
        input_was_2d = x_t.dim() == 2
        if input_was_2d:
            x_t = x_t.unsqueeze(1)  # $[B, 1, T]$
        if conditioning.dim() == 2:
            conditioning = conditioning.unsqueeze(1)  # $[B, 1, T]$
        x = torch.cat([x_t, conditioning], dim=1)  # $[B, 1 + C, T]$

        time_emb = self.time_encoder(t)
        time_emb = self.time_mlp(time_emb)

        x = _apply_conv(self.init_conv, x, self.causal)
        if self.use_position_encoding:
            x = x + self.physical_position_encoding.to(dtype=x.dtype)

        skip_connections: list[torch.Tensor] = []
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x, time_emb)
            if i < len(self.downsample):
                skip_connections.append(x)
                x = _apply_conv(cast(nn.Conv1d, self.downsample[i]), x, self.causal)

        x = self.middle_block(x, time_emb)

        skip_idx = len(skip_connections) - 1
        for i in range(len(self.decoder_blocks)):
            if i > 0 and skip_idx >= 0:
                skip = skip_connections[skip_idx]
                if x.shape[2] != skip.shape[2]:
                    x = nn.functional.interpolate(
                        x,
                        size=skip.shape[2],
                        mode="nearest" if self.causal else "linear",
                        align_corners=None if self.causal else True,
                    )
                x = torch.cat([x, skip], dim=1)
                skip_idx -= 1

            x = self.decoder_blocks[i](x, time_emb)

            if i < len(self.upsample):
                upsample_block = cast(nn.Sequential, self.upsample[i])
                upsample_layer = upsample_block[0]
                conv_layer = upsample_block[1]
                x = upsample_layer(x)
                x = _apply_conv(cast(nn.Conv1d, conv_layer), x, self.causal)

        x = _apply_conv(self.final_conv, x, self.causal)
        if self.n_output_channels == 1 and input_was_2d:
            x = x.squeeze(1)  # $[B, T]$

        return x

    @staticmethod
    def _build_physical_position_encoding(input_dim: int, hidden_dim: int) -> torch.Tensor:
        """Build a fixed encoding for physical trajectory positions."""
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        positions = torch.arange(input_dim, dtype=torch.float32).unsqueeze(1)
        encoding = torch.zeros(input_dim, hidden_dim, dtype=torch.float32)

        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / max(hidden_dim, 1))
        )
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])

        # Reserve one channel for a monotonic physical-time coordinate.
        encoding[:, 0] = torch.linspace(-1.0, 1.0, input_dim, dtype=torch.float32)

        return encoding.transpose(0, 1).unsqueeze(0)


def compute_dilated_rf(
    n_layers: int,
    kernel_size: int = 3,
    dilation_schedule: list[int] | None = None,
) -> tuple[int, int]:
    r"""Theoretical receptive field of :class:`ConditionalUNet` with dilation.

    Returns ``(bottleneck_rf, full_unet_rf)`` in timesteps.

    .. math::

        \mathrm{RF} = 1 + \sum_{\ell \in \mathcal{P}} (k_\ell - 1)\, d_\ell\, J_\ell,
        \qquad J_\ell = \prod_{\ell' \prec \ell} s_{\ell'}

    where $k_\ell$, $d_\ell$, $s_\ell$ are the kernel, dilation and stride of
    layer $\ell$ and $J_\ell$ is the cumulative stride ("jump") of all preceding
    layers.  Dilation is applied to the residual-block convs only; the init,
    downsample, upsample and final convs keep $d = 1$.
    """
    if dilation_schedule is None:
        dilation_schedule = [1] * n_layers
    k = kernel_size
    rf = 1
    jump = 1

    # init_conv: $k$, stride 1, dilation 1.
    rf += (k - 1) * 1 * jump

    # Encoder: residual block ($2$ convs at dilation $d_i$) then downsample.
    for i in range(n_layers):
        d = dilation_schedule[i]
        rf += 2 * (k - 1) * d * jump
        if i < n_layers - 1:
            rf += (k - 1) * 1 * jump  # downsample conv (stride $2$, dilation $1$)
            jump *= 2

    # Middle block: $2$ convs at deepest dilation.
    d = dilation_schedule[-1]
    rf += 2 * (k - 1) * d * jump
    bottleneck_rf = rf

    # Decoder: residual block then upsample (halves $J$, then conv).
    for i in range(n_layers - 1, -1, -1):
        d = dilation_schedule[i]
        rf += 2 * (k - 1) * d * jump
        if i > 0:
            jump //= 2  # bilinear upsample: halves jump, no RF contribution
            rf += (k - 1) * 1 * jump  # upsample conv (stride $1$, dilation $1$)

    # final_conv: $k$, stride 1, dilation 1.
    rf += (k - 1) * 1 * jump
    return bottleneck_rf, rf
