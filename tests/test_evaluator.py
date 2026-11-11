"""Analytical checks for trajectory evaluation metrics."""

import math

import numpy as np
import pytest
import torch

from diffusionsde.evaluator import (
    StatisticsEvaluator,
    compute_higher_moments,
    compute_marginal_wasserstein,
    compute_psd,
    sample_acf,
)


def test_ou_autocorrelation_matches_exponential_decay():
    rng = np.random.default_rng(0)
    theta, dt, n_steps = 1.0, 0.01, 20_000
    trajectory = np.zeros(n_steps)
    noise = rng.standard_normal(n_steps)
    for i in range(1, n_steps):
        trajectory[i] = trajectory[i - 1] * (1 - theta * dt) + math.sqrt(2 * dt) * noise[i]

    max_lag = 50
    acf = sample_acf(
        torch.tensor(trajectory[n_steps // 2 :][None, :], dtype=torch.float32), max_lag
    )
    analytic = np.exp(-theta * np.arange(max_lag + 1) * dt)

    assert np.abs(acf.squeeze(0).numpy() - analytic).mean() < 0.05


def test_psd_recovers_sinusoid_frequency():
    dt, frequency = 1 / 1000, 50.0
    time = np.arange(512) * dt
    signal = np.sin(2 * np.pi * frequency * time)
    data = torch.tensor(np.tile(signal, (8, 1)), dtype=torch.float32)

    frequencies, psd = compute_psd(data, dt)

    assert abs(float(frequencies[np.argmax(psd)]) - frequency) < 5.0


def test_wasserstein_recovers_translation():
    torch.manual_seed(0)
    real = torch.randn(2000, 1)
    shift = 2.0

    assert compute_marginal_wasserstein(real, real + shift) == pytest.approx(shift, abs=1e-6)


def test_exponential_marginal_has_positive_skewness():
    torch.manual_seed(3)
    data = torch.distributions.Exponential(1.0).sample((300, 256))

    assert compute_higher_moments(data)["skewness"] > 1.0


def test_extended_metrics_agree_for_identical_data():
    torch.manual_seed(7)
    data = torch.randn(40, 128)
    metrics = StatisticsEvaluator(data, data.clone()).compute_extended_metrics(dt=0.01, max_lag=16)

    for key in ("mu_mse", "sigma_mse", "mu_mae", "sigma_mae", "acf_mse", "psd_log_mae"):
        assert metrics[key] == 0.0
    assert metrics["mu_correlation"] == pytest.approx(1.0)
    assert metrics["sigma_correlation"] == pytest.approx(1.0)
    assert metrics["wasserstein_1"] == 0.0
