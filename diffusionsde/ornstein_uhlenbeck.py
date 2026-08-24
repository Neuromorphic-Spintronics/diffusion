r"""Simulation and conditioning utilities for the Ornstein-Uhlenbeck process.

The process uses the stochastic differential equation

.. math:: dx = -\theta x\,dt + \sqrt{2D}\,dW,

with :math:`\theta > 0` and :math:`D \geq 0`.  The numerical integrator is
Euler-Maruyama.  The module contains the small dataset adapter used by the OU
diffusion examples, so it can also be used without the example scripts.
"""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

TRAJECTORY_LENGTH = 256
RANDOM_SEED = 42
VALID_CONDITIONING_MODES = ("r", "theta_D")


def _as_finite_float(value: Any, name: str) -> float:
    """Convert a real scalar to float and reject non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass
class OUParams:
    """Parameters of a scalar Ornstein-Uhlenbeck process."""

    theta: float = 1.0
    D: float = 0.5

    def __post_init__(self) -> None:
        self.theta = _as_finite_float(self.theta, "theta")
        self.D = _as_finite_float(self.D, "D")
        if self.theta <= 0:
            raise ValueError("theta must be greater than zero")
        if self.D < 0:
            raise ValueError("D must be greater than or equal to zero")


def canonicalise_conditioning_mode(mode: str) -> str:
    """Return the canonical name for an accepted conditioning-mode alias."""
    if not isinstance(mode, str):
        raise TypeError(f"conditioning mode must be a string, got {type(mode).__name__}")
    aliases = {
        "r": "r",
        "ratio": "r",
        "theta_D": "theta_D",
        "theta_d": "theta_D",
        "theta-d": "theta_D",
        "thetad": "theta_D",
    }
    try:
        return aliases[mode.strip()]
    except KeyError as exc:
        valid_modes = ", ".join(VALID_CONDITIONING_MODES)
        raise ValueError(
            f"Unknown conditioning mode {mode!r}. Expected one of: {valid_modes}."
        ) from exc


def conditioning_channels(conditioning_mode: str) -> int:
    """Return the number of conditioning channels for a mode."""
    return 2 if canonicalise_conditioning_mode(conditioning_mode) == "theta_D" else 1


def conditioning_mode_label(conditioning_mode: str) -> str:
    """Return a human-readable label for a conditioning mode."""
    return (
        r"theta, D"
        if canonicalise_conditioning_mode(conditioning_mode) == "theta_D"
        else r"r = D/theta"
    )


def conditioning_mode_suffix(conditioning_mode: str) -> str:
    """Return the filename suffix used for a conditioning mode."""
    return "" if canonicalise_conditioning_mode(conditioning_mode) == "r" else "_theta_D"


def metadata_filename_for_mode(conditioning_mode: str) -> str:
    """Return the metadata filename for a conditioning mode."""
    return f"metadata{conditioning_mode_suffix(conditioning_mode)}.pt"


def model_filename_for_mode(conditioning_mode: str) -> str:
    """Return the model filename for a conditioning mode."""
    return f"ou{conditioning_mode_suffix(conditioning_mode)}_model.pt"


def output_stem_for_mode(base_stem: str, conditioning_mode: str) -> str:
    """Return a mode-specific output stem while keeping legacy ``r`` names."""
    return f"{base_stem}{conditioning_mode_suffix(conditioning_mode)}"


def _validate_time_grid(t_span: tuple[float, float], n_steps: int) -> tuple[float, float]:
    """Validate an integration interval and return its float endpoints."""
    if not isinstance(n_steps, Integral) or isinstance(n_steps, bool):
        raise TypeError("n_steps must be an integer")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise ValueError("t_span must contain exactly two values")
    t0 = _as_finite_float(t_span[0], "t_span[0]")
    t1 = _as_finite_float(t_span[1], "t_span[1]")
    if t1 <= t0:
        raise ValueError("t_span must be strictly increasing")
    return t0, t1


def integrate_ou(
    params: OUParams,
    t_span: tuple[float, float],
    n_steps: int,
    x0: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an OU process with Euler-Maruyama.

    A local random generator is used when a seed is provided.  This gives
    reproducible trajectories without changing NumPy's global random state.
    """
    if not isinstance(params, OUParams):
        raise TypeError(f"params must be an OUParams instance, got {type(params).__name__}")
    t0, t1 = _validate_time_grid(t_span, n_steps)
    x0_value = _as_finite_float(x0, "x0")
    t = np.linspace(t0, t1, int(n_steps))
    return t, _integrate_ou_on_grid(params, t, x0_value, seed)


def _integrate_ou_on_grid(
    params: OUParams,
    time_array: np.ndarray,
    x0: float,
    seed: int | None,
) -> np.ndarray:
    """Integrate on a validated, possibly non-uniform time grid."""
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be a valid NumPy random seed") from exc
    x = np.empty(time_array.size, dtype=np.float64)
    x[0] = x0
    for i in range(1, time_array.size):
        dt = time_array[i] - time_array[i - 1]
        decay = 1.0 - params.theta * dt
        noise_scale = np.sqrt(2.0 * params.D * dt)
        x[i] = decay * x[i - 1] + noise_scale * rng.standard_normal()
    return x


def _validate_range(values: tuple[float, float], name: str) -> tuple[float, float]:
    """Validate a finite inclusive parameter range."""
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower = _as_finite_float(values[0], f"{name}[0]")
    upper = _as_finite_float(values[1], f"{name}[1]")
    if upper < lower:
        raise ValueError(f"{name} lower bound must not exceed upper bound")
    return lower, upper


def generate_training_data(
    n_trajectories: int,
    t_span: tuple[float, float],
    n_steps: int,
    theta_range: tuple[float, float] = (0.5, 3.0),
    D_range: tuple[float, float] = (0.1, 1.5),
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate trajectories and their randomly sampled OU parameters."""
    if not isinstance(n_trajectories, Integral) or isinstance(n_trajectories, bool):
        raise TypeError("n_trajectories must be an integer")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be greater than zero")
    t0, t1 = _validate_time_grid(t_span, n_steps)
    theta_low, theta_high = _validate_range(theta_range, "theta_range")
    D_low, D_high = _validate_range(D_range, "D_range")
    if theta_low <= 0:
        raise ValueError("theta_range must contain only values greater than zero")
    if D_low < 0:
        raise ValueError("D_range must contain only non-negative values")
    try:
        rng = np.random.default_rng(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be a valid NumPy random seed") from exc
    theta_values = rng.uniform(theta_low, theta_high, int(n_trajectories))
    D_values = rng.uniform(D_low, D_high, int(n_trajectories))
    time_array = np.linspace(t0, t1, int(n_steps))
    trajectories = np.empty((int(n_trajectories), int(n_steps)), dtype=np.float64)
    for i, (theta, D) in enumerate(
        tqdm(zip(theta_values, D_values, strict=True), total=int(n_trajectories), desc="Generating")
    ):
        _, trajectories[i] = integrate_ou(
            OUParams(theta, D), (t0, t1), int(n_steps), seed=int(seed) + i + 1
        )
    return time_array, trajectories, theta_values, D_values


def _as_time_array(time_array: np.ndarray) -> np.ndarray:
    """Convert and validate an observation time array."""
    result = np.asarray(time_array, dtype=np.float64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError("time_array must be one-dimensional with at least two values")
    if not np.all(np.isfinite(result)) or not np.all(np.diff(result) > 0):
        raise ValueError("time_array must be finite and strictly increasing")
    return result


def generate_observed_trajectories(
    theta: float,
    D: float,
    n_trajectories: int,
    time_array: np.ndarray,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate observations with fixed OU parameters."""
    if not isinstance(n_trajectories, Integral) or isinstance(n_trajectories, bool):
        raise TypeError("n_trajectories must be an integer")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be greater than zero")
    params = OUParams(theta, D)
    times = _as_time_array(time_array)
    trajectories = np.empty((int(n_trajectories), times.size), dtype=np.float64)
    for i in range(int(n_trajectories)):
        trajectories[i] = _integrate_ou_on_grid(params, times, 0.0, int(seed) + i + 1)
    return trajectories


class OUDataset:
    """Dataset for OU trajectories with ratio or joint-parameter conditioning."""

    def __init__(
        self,
        trajectories: Any,
        time_array: Any,
        theta_values: Any,
        D_values: Any,
        normalise: bool = True,
        conditioning_mode: str = "r",
    ) -> None:
        mode = canonicalise_conditioning_mode(conditioning_mode)
        x_array = np.asarray(trajectories, dtype=np.float32)
        times = _as_time_array(time_array).astype(np.float32)
        theta_array = np.asarray(theta_values, dtype=np.float32).reshape(-1)
        D_array = np.asarray(D_values, dtype=np.float32).reshape(-1)
        if x_array.ndim != 2:
            raise ValueError("trajectories must have shape (n_trajectories, n_steps)")
        if x_array.shape[1] != times.size:
            raise ValueError("trajectory length must match time_array length")
        if x_array.shape[0] == 0:
            raise ValueError("trajectories must contain at least one trajectory")
        if theta_array.size != x_array.shape[0] or D_array.size != x_array.shape[0]:
            raise ValueError("parameter arrays must have one value per trajectory")
        if not np.all(np.isfinite(x_array)):
            raise ValueError("trajectories must contain only finite values")
        if not np.all(np.isfinite(theta_array)) or np.any(theta_array <= 0):
            raise ValueError("theta_values must contain finite values greater than zero")
        if not np.all(np.isfinite(D_array)) or np.any(D_array < 0):
            raise ValueError("D_values must contain finite non-negative values")

        self.time_array = times
        self.trajectory_length = int(times.size)
        self.normalise = bool(normalise)
        self.conditioning_mode = mode
        self.x_data = torch.as_tensor(x_array, dtype=torch.float32).clone()
        self.theta_values = torch.as_tensor(theta_array, dtype=torch.float32).clone()
        self.D_values = torch.as_tensor(D_array, dtype=torch.float32).clone()
        self.r_values = self.D_values / self.theta_values

        self.r_min = self.r_values.min()
        self.r_max = self.r_values.max()
        self.r_norm = self._normalise_to_unit_interval(self.r_values, self.r_min, self.r_max)
        self.theta_min = self.theta_values.min()
        self.theta_max = self.theta_values.max()
        self.D_min = self.D_values.min()
        self.D_max = self.D_values.max()
        self.theta_norm = self._normalise_to_unit_interval(
            self.theta_values, self.theta_min, self.theta_max
        )
        self.D_norm = self._normalise_to_unit_interval(self.D_values, self.D_min, self.D_max)

        self.x_min = self.x_data.min()
        self.x_max = self.x_data.max()
        if self.normalise:
            self.x_data = self._normalise_to_unit_interval(self.x_data, self.x_min, self.x_max)

    @staticmethod
    def _normalise_to_unit_interval(
        values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """Map values to [-1, 1], returning zero for a degenerate range."""
        if bool(torch.isclose(upper, lower)):
            return torch.zeros_like(values)
        return 2 * (values - lower) / (upper - lower) - 1

    def normalise_theta_val(self, theta: float) -> torch.Tensor:
        """Map a physical theta value to the dataset's [-1, 1] range."""
        return self._normalise_to_unit_interval(
            torch.tensor(_as_finite_float(theta, "theta"), dtype=torch.float32),
            self.theta_min,
            self.theta_max,
        )

    def normalise_D_val(self, D: float) -> torch.Tensor:
        """Map a physical D value to the dataset's [-1, 1] range."""
        return self._normalise_to_unit_interval(
            torch.tensor(_as_finite_float(D, "D"), dtype=torch.float32), self.D_min, self.D_max
        )

    def condition_trajectory(self, theta: float, D: float) -> torch.Tensor:
        """Build conditioning for one physical parameter pair."""
        return self.get_conditioning(theta=theta, D=D)

    def get_conditioning(self, theta: float, D: float) -> torch.Tensor:
        """Build a conditioning tensor in the configured mode."""
        params = OUParams(theta, D)
        if self.conditioning_mode == "theta_D":
            conditioning = torch.zeros(2, self.trajectory_length, dtype=torch.float32)
            conditioning[0].fill_(self.normalise_theta_val(params.theta))
            conditioning[1].fill_(self.normalise_D_val(params.D))
            return conditioning
        r = params.D / params.theta
        conditioning = torch.zeros(self.trajectory_length, dtype=torch.float32)
        conditioning[0] = self._normalise_to_unit_interval(
            torch.tensor(r, dtype=torch.float32), self.r_min, self.r_max
        )
        return conditioning

    def __len__(self) -> int:
        return int(self.x_data.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(idx, Integral) or isinstance(idx, bool):
            raise TypeError(f"OUDataset indices must be integers, got {type(idx).__name__}")
        if self.conditioning_mode == "theta_D":
            conditioning = torch.stack(
                [
                    torch.full((self.trajectory_length,), self.theta_norm[idx].item()),
                    torch.full((self.trajectory_length,), self.D_norm[idx].item()),
                ]
            )
        else:
            conditioning = torch.zeros(self.trajectory_length, dtype=torch.float32)
            conditioning[0] = self.r_norm[idx]
        return self.x_data[idx], conditioning

    def normalise_r(self, r: float) -> float:
        """Map a physical ratio to the dataset's [-1, 1] range."""
        value = self._normalise_to_unit_interval(
            torch.tensor(_as_finite_float(r, "r"), dtype=torch.float32), self.r_min, self.r_max
        )
        return float(value)


def split_trajectories(
    n_trajectories: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reproducible random train, validation, and test index splits."""
    if not isinstance(n_trajectories, Integral) or isinstance(n_trajectories, bool):
        raise TypeError("n_trajectories must be an integer")
    if n_trajectories < 0:
        raise ValueError("n_trajectories must not be negative")
    train = _as_finite_float(train_ratio, "train_ratio")
    validation = _as_finite_float(val_ratio, "val_ratio")
    if train < 0 or validation < 0 or train + validation > 1:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to at most one")
    permutation = np.random.default_rng(random_seed).permutation(int(n_trajectories))
    n_train = int(train * int(n_trajectories))
    n_val = int(validation * int(n_trajectories))
    return (
        permutation[:n_train],
        permutation[n_train : n_train + n_val],
        permutation[n_train + n_val :],
    )


def save_data(
    filepath: Path | str,
    trajectories: Any,
    time_array: Any,
    theta_values: Any,
    D_values: Any,
) -> None:
    """Save OU trajectories and parameter arrays in a Torch checkpoint."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "trajectories": trajectories,
            "time_array": time_array,
            "theta_values": theta_values,
            "D_values": D_values,
        },
        path,
    )


def _parameter_array(values: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Convert a batch parameter input to a one-dimensional NumPy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def build_conditioning_batch(
    dataset: OUDataset,
    theta_values: np.ndarray | torch.Tensor,
    D_values: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Build a batch of conditioning tensors from physical OU parameters."""
    if not isinstance(dataset, OUDataset):
        raise TypeError(f"dataset must be an OUDataset, got {type(dataset).__name__}")
    theta_array = _parameter_array(theta_values, "theta_values")
    D_array = _parameter_array(D_values, "D_values")
    if theta_array.size != D_array.size:
        raise ValueError("theta_values and D_values must have the same length")
    if theta_array.size == 0:
        raise ValueError("parameter arrays must contain at least one value")
    conditioning = torch.stack(
        [
            dataset.condition_trajectory(float(theta), float(D))
            for theta, D in zip(theta_array, D_array, strict=True)
        ]
    )
    if device is not None:
        conditioning = conditioning.to(device)
    return conditioning


__all__ = [
    "TRAJECTORY_LENGTH",
    "RANDOM_SEED",
    "VALID_CONDITIONING_MODES",
    "OUParams",
    "OUDataset",
    "build_conditioning_batch",
    "canonicalise_conditioning_mode",
    "conditioning_channels",
    "conditioning_mode_label",
    "conditioning_mode_suffix",
    "generate_observed_trajectories",
    "generate_training_data",
    "integrate_ou",
    "metadata_filename_for_mode",
    "model_filename_for_mode",
    "output_stem_for_mode",
    "save_data",
    "split_trajectories",
]
