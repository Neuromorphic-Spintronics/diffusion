"""Core correctness checks for the inverse solver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from diffusionsde.inverse import InverseProblem, InverseProblemConfig


class _ToyDDPM:
    def __init__(self) -> None:
        self.config = SimpleNamespace(device="cpu")

    def generate(self, conditioning: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        signal = conditioning[:, 0] if conditioning.dim() == 3 else conditioning
        return signal.unsqueeze(1)


def _conditioning(parameter, n_obs, trajectory_length, device, run_idx):
    del run_idx
    return parameter.reshape(1, 1, 1).to(device).expand(n_obs, 1, trajectory_length)


def _problem(use_best_iterate: bool) -> InverseProblem:
    return InverseProblem(
        cast(Any, _ToyDDPM()),
        torch.tensor([[40.0] * 5, [70.0] * 5]),
        _conditioning,
        (0.0, 100.0),
        InverseProblemConfig(
            n_iterations=4,
            lr=0.05,
            grad_steps=1,
            use_best_iterate=use_best_iterate,
            parameterisation="physical",
        ),
    )


@pytest.mark.parametrize("use_best", [True, False])
def test_reported_loss_matches_reported_iterate(use_best: bool) -> None:
    problem = _problem(use_best)
    result = problem.solve(
        n_runs=2,
        parameter_inits=np.array([10.0, 90.0], dtype=np.float32),
        return_trajectories=True,
        runs_per_chunk=1,
    )
    assert result.trajectories is not None
    assert result.loss_trajectories is not None
    for run, (trajectory, losses) in enumerate(
        zip(result.trajectories, result.loss_trajectories, strict=True)
    ):
        if use_best:
            best_index = int(np.argmin(losses))
            assert result.estimates[run] == pytest.approx(trajectory[best_index])
            assert result.losses[run] == pytest.approx(losses[best_index])
        else:
            assert result.estimates[run] == pytest.approx(trajectory[-1])
            assert result.losses[run] == pytest.approx(losses[-1])


def test_chunking_does_not_change_estimates_or_losses() -> None:
    inits = np.array([10.0, 90.0], dtype=np.float32)
    one = _problem(False).solve(n_runs=2, parameter_inits=inits, runs_per_chunk=1)
    two = _problem(False).solve(n_runs=2, parameter_inits=inits, runs_per_chunk=2)
    np.testing.assert_allclose(one.estimates, two.estimates)
    np.testing.assert_allclose(one.losses, two.losses)


@pytest.mark.parametrize("parameterisation", ["physical", "sigmoid_bounded"])
def test_inverse_recovers_parameter_with_per_run_tail(parameterisation: str) -> None:
    observed = torch.stack([torch.full((3, 5), value) for value in (40.0, 70.0)])
    problem = InverseProblem(
        cast(Any, _ToyDDPM()),
        observed,
        _conditioning,
        (0.0, 100.0),
        InverseProblemConfig(
            n_iterations=100,
            lr=0.05,
            grad_steps=1,
            tail_start=2,
            parameterisation=parameterisation,
        ),
    )
    result = problem.solve(
        n_runs=2,
        parameter_inits=np.array([10.0, 90.0], dtype=np.float32),
    )
    np.testing.assert_allclose(result.estimates, np.array([40.0, 70.0]), atol=2.0)


def test_rejects_undefined_iteration_and_run_counts() -> None:
    problem = _problem(False)
    problem.config.n_iterations = 0
    with pytest.raises(ValueError, match="n_iterations"):
        problem.solve()
    problem.config.n_iterations = 1
    with pytest.raises(ValueError, match="n_runs"):
        problem.solve(n_runs=0)
