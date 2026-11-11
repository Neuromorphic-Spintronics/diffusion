# diffusionsde

Modular implementation of denoising diffusion probabilistic models (DDPMs)
for time series generation, with a conditional 1D U-Net denoiser and DDIM
sampling.

## Features

- Conditional DDPM over multivariate time series: `[batch, channels, time]`
- 1D residual U-Net denoiser with sinusoidal timestep embeddings, dilated
  convolutions and optional conditioning channels
- DDIM and DDPM samplers
- Training loop with validation and checkpointing
- Evaluation metrics for distributional comparison of generated trajectories
- Inverse problem solvers conditioned on partial observations

## Installation

Install from GitHub when the package is not available on PyPI:

```bash
uv add "diffusionsde @ git+https://github.com/Neuromorphic-Spintronics/diffusion.git"
# or
python -m pip install git+https://github.com/Neuromorphic-Spintronics/diffusion.git
```

Then import it using the package name:

```python
from diffusionsde import DDPM, DiffusionConfig
```

When a PyPI release is available, install it with:

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
uv run ruff check .
uv run ruff format .
uv run ty check
```

Tests use small CPU models and a non-interactive plotting backend. CI runs the
same full suite on Python 3.13 and 3.14.

`DDPM.fit()` starts a fresh model. To continue an interrupted training run, create
a `DiffusionTrainer`, call `load_checkpoint()`, then `train()`; `config.n_epochs`
is the total target epoch count. Checkpoints retain continuation weights and
the best validation snapshot separately. `DDPM.load()` loads a model for use
and saving; it does not attach a trainer. `freeze(False)` makes frozen model
parameters trainable again.

## Licence

MIT
