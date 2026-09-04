# diffusionsde

Modular implementation of denoising diffusion probabilistic models (DDPMs)
for time series generation, with a conditional 1D U-Net denoiser and DDIM
sampling.

## Features

- Conditional DDPM over multivariate time series: `[batch, channels, time]`
- 1D residual U-Net denoiser with sinusoidal timestep embeddings, dilated
  convolutions and optional conditioning channels
- DDIM and DDPM samplers
- Training loop with validation, checkpointing and Weights & Biases logging
- Evaluation metrics for distributional comparison of generated trajectories
- Inverse problem solvers conditioned on partial observations

## Installation

```bash
uv add diffusionsde
# or
pip install diffusionsde
```

Requires Python >= 3.13.

## Quickstart

```python
import torch
from torch.utils.data import TensorDataset

from diffusionsde import DDPM, DiffusionConfig

config = DiffusionConfig(
    trajectory_length=50,
    n_output_channels=2,
    n_conditioning_channels=1,
    hidden_dim=64,
    n_layers=3,
    n_epochs=10,
)

ddpm = DDPM(config)

n_samples = 256
data = torch.randn(n_samples, 2, 50)  # [batch, channels, time]
conditioning = torch.randn(n_samples, 1, 50)  # [batch, condition channels, time]
dataset = TensorDataset(data, conditioning)

ddpm.fit(dataset)

samples = ddpm.generate(conditioning[:8], n_samples=8)
```

Example notebooks (Ornstein-Uhlenbeck, Duffing oscillator) live in
`notebooks/` and are self-contained.

## Development

```bash
uv sync

uv run pytest                # full test suite
uv run pytest -m "not slow"  # fast subset
uv run ruff check .
uv run ruff format .
uv run ty check
```

## Licence

MIT
