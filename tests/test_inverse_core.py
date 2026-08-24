"""High-value tests for the core inverse solver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from diffusionsde.inverse import InverseProblem, InverseProblemConfig


class _ToyDDPM:
    """Small differentiable model that returns its first condition channel."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(device="cpu")

    def generate(
        self,
        conditioning: torch.Tensor,
        n_samples: int = 1,
        require_grad: bool = False,
        initial_noise: torch.Tensor | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del n_samples, require_grad, initial_noise, show_progress, kwargs
        signal = conditioning[:, 0] if conditioning.dim() == 3 else conditioning
        return signal.unsqueeze(1)


def _conditioning_from_parameter(
    parameter: torch.Tensor,
    n_obs: int,
    trajectory_length: int,
    device: str,
    run_idx: int,
) -> torch.Tensor:
    del run_idx
    value = parameter.reshape(-1)
    assert value.numel() == 1
    return value.to(device).reshape(1, 1, 1).expand(n_obs, 1, trajectory_length)


@pytest.mark.parametrize("parameterisation", ["physical", "sigmoid_bounded"])
def test_inverse_problem_recovers_physical_parameter(parameterisation: str) -> None:
    target = 40.0
    observed = torch.full((3, 5), target)
    problem = InverseProblem(
        cast(Any, _ToyDDPM()),
        observed,
        _conditioning_from_parameter,
        parameter_range=(0.0, 100.0),
        config=InverseProblemConfig(
            n_iterations=120,
            lr=0.05,
            grad_steps=1,
            parameterisation=parameterisation,
            lr_floor_ratio=0.1,
        ),
    )

    result = problem.solve(
        n_runs=1,
        parameter_inits=np.array([0.0], dtype=np.float32),
        return_trajectories=True,
    )

    assert result.trajectories is not None
    assert result.parameter_inits[0] == pytest.approx(0.0)
    assert result.estimates[0] == pytest.approx(target, abs=2.0)
    assert result.estimates[0] > 1.0


def test_inverse_problem_tail_slicing_supports_per_run_observations() -> None:
    observed = torch.tensor(
        [
            [[90.0, 90.0, 10.0, 10.0, 10.0], [85.0, 85.0, 10.0, 10.0, 10.0]],
            [[20.0, 20.0, 70.0, 70.0, 70.0], [15.0, 15.0, 70.0, 70.0, 70.0]],
        ],
        dtype=torch.float32,
    )
    problem = InverseProblem(
        cast(Any, _ToyDDPM()),
        observed,
        _conditioning_from_parameter,
        parameter_range=(0.0, 100.0),
        config=InverseProblemConfig(
            n_iterations=120,
            lr=0.05,
            grad_steps=1,
            tail_start=2,
            parameterisation="sigmoid_bounded",
            lr_floor_ratio=0.1,
        ),
    )

    result = problem.solve(n_runs=2, parameter_inits=np.array([0.0, 100.0], dtype=np.float32))

    np.testing.assert_allclose(result.estimates, np.array([10.0, 70.0]), atol=2.0)
