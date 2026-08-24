"""Tests for the public Ornstein-Uhlenbeck utilities."""

import numpy as np
import pytest
import torch

from diffusionsde import OUDataset, OUParams, integrate_ou
from diffusionsde.ornstein_uhlenbeck import (
    build_conditioning_batch,
    canonicalise_conditioning_mode,
    generate_observed_trajectories,
    generate_training_data,
    split_trajectories,
)


def make_dataset(mode: str = "r", *, degenerate: bool = False) -> OUDataset:
    trajectories = np.array([[0.0, 0.1, 0.2, 0.3], [0.0, -0.1, -0.2, -0.3]], dtype=np.float32)
    theta = np.array([1.0, 1.0] if degenerate else [0.5, 1.5], dtype=np.float32)
    diffusion = np.array([0.2, 0.2] if degenerate else [0.2, 1.2], dtype=np.float32)
    return OUDataset(
        trajectories, np.linspace(0.0, 1.0, 4), theta, diffusion, conditioning_mode=mode
    )


def test_integration_is_reproducible_and_has_expected_grid() -> None:
    times, first = integrate_ou(OUParams(theta=1.0, D=0.5), (0.0, 1.0), 11, seed=7)
    _, second = integrate_ou(OUParams(theta=1.0, D=0.5), (0.0, 1.0), 11, seed=7)

    assert np.array_equal(times, np.linspace(0.0, 1.0, 11))
    assert np.array_equal(first, second)
    assert first[0] == 0.0


def test_integration_deterministic_when_diffusion_is_zero() -> None:
    _, values = integrate_ou(OUParams(theta=2.0, D=0.0), (0.0, 1.0), 3, x0=1.0, seed=1)

    np.testing.assert_allclose(values, [1.0, 0.0, 0.0])


def test_integration_matches_finite_time_euler_variance() -> None:
    """The ensemble law follows the variance of the implemented Euler scheme."""
    theta = 1.2
    diffusion = 0.7
    n_steps = 181
    dt = 1.0 / (n_steps - 1)
    decay = 1.0 - theta * dt
    expected_variance = (
        2.0 * diffusion * dt * (1.0 - decay ** (2 * (n_steps - 1))) / (1.0 - decay**2)
    )
    trajectories = np.stack(
        [
            integrate_ou(OUParams(theta, diffusion), (0.0, 1.0), n_steps, seed=seed)[1]
            for seed in range(3000)
        ]
    )

    assert abs(float(trajectories[:, -1].mean())) < 0.04
    assert float(trajectories[:, -1].var()) == pytest.approx(expected_variance, rel=0.06)


def test_training_and_observed_generation_are_batched_and_reproducible() -> None:
    result = generate_training_data(4, (0.0, 1.0), 8, seed=12)
    repeated = generate_training_data(4, (0.0, 1.0), 8, seed=12)
    assert result[1].shape == (4, 8)
    assert result[2].shape == (4,)
    assert result[3].shape == (4,)
    for left, right in zip(result, repeated, strict=True):
        np.testing.assert_array_equal(left, right)

    observations = generate_observed_trajectories(1.0, 0.5, 3, np.linspace(0, 1, 8), seed=4)
    assert observations.shape == (3, 8)

    non_uniform_times = np.array([0.0, 0.1, 0.3, 1.0])
    non_uniform = generate_observed_trajectories(1.0, 0.5, 1, non_uniform_times, seed=4)
    uniform = generate_observed_trajectories(1.0, 0.5, 1, np.linspace(0, 1, 4), seed=4)
    assert not np.array_equal(non_uniform, uniform)


def test_dataset_conditioning_shapes_and_constant_ranges() -> None:
    ratio = make_dataset("ratio")
    sample, conditioning = ratio[0]
    assert sample.shape == (4,)
    assert conditioning.shape == (4,)
    assert conditioning[0] == ratio.r_norm[0]
    assert torch.count_nonzero(conditioning[1:]) == 0

    joint = make_dataset("theta_D")
    _, joint_conditioning = joint[1]
    assert joint_conditioning.shape == (2, 4)
    assert torch.all(joint_conditioning[0] == joint_conditioning[0, 0])
    assert torch.all(joint_conditioning[1] == joint_conditioning[1, 0])

    degenerate = make_dataset("r", degenerate=True)
    assert torch.all(degenerate.r_norm == 0)
    assert degenerate.normalise_r(0.2) == 0.0
    _, degenerate_conditioning = degenerate[0]
    assert torch.all(degenerate_conditioning == 0)


def test_conditioning_batch_accepts_torch_parameters_and_device() -> None:
    dataset = make_dataset("theta_D")
    batch = build_conditioning_batch(
        dataset,
        torch.tensor([0.5, 1.0]),
        torch.tensor([0.2, 0.7]),
        device="cpu",
    )
    assert batch.shape == (2, 2, 4)
    assert batch.device.type == "cpu"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: OUParams(theta=0.0), "theta"),
        (lambda: OUParams(D=-1.0), "D"),
        (lambda: integrate_ou(OUParams(), (0.0, 1.0), 1), "n_steps"),
        (lambda: integrate_ou(OUParams(), (1.0, 0.0), 3), "increasing"),
        (lambda: OUDataset(np.zeros((2, 3)), [0, 1, 2], [1], [0.1, 0.2]), "parameter"),
    ],
)
def test_invalid_inputs_raise(call, message: str) -> None:  # noqa: ANN001
    with pytest.raises((TypeError, ValueError), match=message):
        call()


def test_mode_aliases_and_split_reproducibility() -> None:
    assert canonicalise_conditioning_mode("theta_d") == "theta_D"
    first = split_trajectories(20, random_seed=3)
    second = split_trajectories(20, random_seed=3)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert sorted(np.concatenate(first).tolist()) == list(range(20))


def test_public_api_imports() -> None:
    import diffusionsde

    assert diffusionsde.OUParams is OUParams
    assert diffusionsde.OUDataset is OUDataset
