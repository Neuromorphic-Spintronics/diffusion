"""Tests for the evaluation utilities."""

from __future__ import annotations

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


class TestStatisticsEvaluator:
    """Test suite for StatisticsEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator with sample data."""
        real_data = torch.randn(100, 50)
        generated_data = torch.randn(100, 50)
        return StatisticsEvaluator(real_data, generated_data)

    @pytest.fixture
    def identical_evaluator(self):
        """Create an evaluator where real and generated data are identical."""
        data = torch.randn(100, 50)
        return StatisticsEvaluator(data, data.clone())

    def test_compute_statistics_shape(self, evaluator):
        """Test that computed statistics have correct shape."""
        mu, sigma = evaluator.compute_statistics(evaluator.real_data)
        assert mu.shape == (50,)
        assert sigma.shape == (50,)

    def test_compute_statistics_values(self):
        """Test that statistics are computed correctly."""
        n_runs = 1000
        n_timesteps = 10
        data = torch.randn(n_runs, n_timesteps)
        evaluator = StatisticsEvaluator(data, data)
        mu, sigma = evaluator.compute_statistics(data)
        assert torch.allclose(mu, torch.zeros(n_timesteps), atol=0.1)
        assert torch.allclose(sigma, torch.ones(n_timesteps), atol=0.1)

    def test_compute_metrics_keys(self, evaluator):
        """Test that compute_metrics returns expected keys."""
        metrics = evaluator.compute_metrics()
        expected_keys = [
            "mu_mse",
            "sigma_mse",
            "mu_mae",
            "sigma_mae",
            "mu_correlation",
            "sigma_correlation",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_compute_metrics_non_negative_errors(self, evaluator):
        """Test that error metrics are non-negative and finite."""
        metrics = evaluator.compute_metrics()
        assert metrics["mu_mse"] >= 0
        assert metrics["sigma_mse"] >= 0
        assert metrics["mu_mae"] >= 0
        assert metrics["sigma_mae"] >= 0
        for key, value in metrics.items():
            assert math.isfinite(value), f"{key} is not finite: {value}"

    def test_compute_metrics_identical_data(self, identical_evaluator):
        """Test metrics when real and generated data are identical."""
        metrics = identical_evaluator.compute_metrics()
        assert metrics["mu_mse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["sigma_mse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mu_mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["sigma_mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mu_correlation"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["sigma_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_different_distributions(self):
        """Test evaluator with clearly different distributions."""
        n_runs = 100
        n_timesteps = 50
        t = torch.linspace(0, 2 * 3.14159, n_timesteps)
        real_data = torch.sin(t).unsqueeze(0).repeat(n_runs, 1)
        real_data = real_data + 0.1 * torch.randn_like(real_data)
        generated_data = torch.randn(n_runs, n_timesteps)
        evaluator = StatisticsEvaluator(real_data, generated_data)
        metrics = evaluator.compute_metrics()
        assert metrics["mu_mse"] > 0.1


# ---------------------------------------------------------------------------
# sample_acf
# ---------------------------------------------------------------------------


class TestSampleAcf:
    def test_lag0_is_one(self):
        data = torch.randn(50, 128)
        acf = sample_acf(data, max_lag=10)
        assert acf.shape == (50, 11)
        assert torch.allclose(acf[:, 0], torch.ones(50), atol=1e-6)

    def test_shape(self):
        data = torch.randn(20, 64)
        acf = sample_acf(data, max_lag=16)
        assert acf.shape == (20, 17)

    def test_ou_exponential_decay(self):
        """ACF of a long OU trajectory should be close to exp(-theta * tau)."""
        rng = np.random.default_rng(0)
        theta = 1.0
        dt = 0.01
        n_steps = 20_000
        x = np.zeros(n_steps)
        noise = rng.standard_normal(n_steps)
        for i in range(1, n_steps):
            x[i] = x[i - 1] - theta * x[i - 1] * dt + math.sqrt(2 * dt) * noise[i]
        x_tensor = torch.tensor(x[n_steps // 2 :][None, :], dtype=torch.float32)
        max_lag = 50
        acf = sample_acf(x_tensor, max_lag).squeeze(0).numpy()
        lag_times = np.arange(max_lag + 1) * dt
        analytic = np.exp(-theta * lag_times)
        mae = float(np.abs(acf - analytic).mean())
        assert mae < 0.05, f"ACF MAE from analytic = {mae:.4f}"

    def test_white_noise_acf(self):
        """White noise should have ACF ~= 0 for lag > 0."""
        torch.manual_seed(42)
        data = torch.randn(200, 512)
        acf = sample_acf(data, max_lag=10)
        mean_nonzero_lag = acf[:, 1:].abs().mean().item()
        assert mean_nonzero_lag < 0.05, f"white-noise non-zero lag ACF = {mean_nonzero_lag:.4f}"


# ---------------------------------------------------------------------------
# compute_psd
# ---------------------------------------------------------------------------


class TestComputePsd:
    def test_shape(self):
        data = torch.randn(30, 256)
        freqs, psd = compute_psd(data, dt=0.01)
        assert freqs.shape == psd.shape
        assert freqs.ndim == 1

    def test_dc_at_zero(self):
        data = torch.randn(30, 256)
        freqs, _ = compute_psd(data, dt=0.01)
        assert freqs[0] == pytest.approx(0.0)

    def test_nyquist_limit(self):
        dt = 0.01
        data = torch.randn(30, 256)
        freqs, _ = compute_psd(data, dt=dt)
        assert freqs[-1] <= 0.5 / dt + 1e-9

    def test_dominant_frequency(self):
        """PSD of a sinusoid should peak near the signal frequency."""
        dt = 1.0 / 1000.0
        t = np.arange(512) * dt
        f0 = 50.0
        signal = np.sin(2 * np.pi * f0 * t)
        data = torch.tensor(signal[None, :].repeat(20, axis=0), dtype=torch.float32)
        freqs, psd = compute_psd(data, dt=dt)
        peak_freq = float(freqs[np.argmax(psd)])
        assert abs(peak_freq - f0) < 5.0, f"peak freq {peak_freq:.1f} Hz, expected {f0:.1f} Hz"


# ---------------------------------------------------------------------------
# compute_marginal_wasserstein
# ---------------------------------------------------------------------------


class TestComputeMarginalWasserstein:
    def test_identical_is_zero(self):
        data = torch.randn(100, 128)
        w1 = compute_marginal_wasserstein(data, data)
        assert abs(w1) < 1e-3

    def test_shifted_distribution(self):
        """W1 of N(0,1) vs N(mu,1) should be approximately mu."""
        torch.manual_seed(0)
        n, t = 2000, 1
        real = torch.randn(n, t)
        shift = 2.0
        gen = real + shift
        w1 = compute_marginal_wasserstein(real, gen)
        assert abs(w1 - shift) < 0.1, f"W1 = {w1:.4f}, expected ~= {shift}"

    def test_positive(self):
        real = torch.randn(100, 64)
        gen = torch.randn(100, 64) + 1.0
        w1 = compute_marginal_wasserstein(real, gen)
        assert w1 > 0


# ---------------------------------------------------------------------------
# compute_higher_moments
# ---------------------------------------------------------------------------


class TestComputeHigherMoments:
    def test_gaussian_skewness_near_zero(self):
        torch.manual_seed(1)
        data = torch.randn(500, 256)
        m = compute_higher_moments(data)
        assert abs(m["skewness"]) < 0.2, f"skewness = {m['skewness']:.4f}"

    def test_gaussian_kurtosis_near_zero(self):
        torch.manual_seed(2)
        data = torch.randn(500, 256)
        m = compute_higher_moments(data)
        assert abs(m["excess_kurtosis"]) < 0.5, f"excess kurtosis = {m['excess_kurtosis']:.4f}"

    def test_skewed_distribution(self):
        """Exponential distribution has positive skewness."""
        torch.manual_seed(3)
        data = torch.distributions.Exponential(1.0).sample((300, 256))
        m = compute_higher_moments(data)
        assert m["skewness"] > 1.0, f"skewness of Exp(1) = {m['skewness']:.4f}, expected > 1"

    def test_return_keys(self):
        data = torch.randn(50, 64)
        m = compute_higher_moments(data)
        assert "skewness" in m
        assert "excess_kurtosis" in m


# ---------------------------------------------------------------------------
# StatisticsEvaluator.compute_extended_metrics
# ---------------------------------------------------------------------------


class TestComputeExtendedMetrics:
    @pytest.fixture
    def evaluator(self):
        torch.manual_seed(7)
        real = torch.randn(80, 128)
        gen = torch.randn(80, 128)
        return StatisticsEvaluator(real, gen)

    def test_returns_dict(self, evaluator):
        metrics = evaluator.compute_extended_metrics(dt=0.01, max_lag=16)
        assert isinstance(metrics, dict)

    def test_expected_keys(self, evaluator):
        metrics = evaluator.compute_extended_metrics(dt=0.01, max_lag=16)
        for key in ("acf_mse", "psd_log_mae", "wasserstein_1"):
            assert key in metrics, f"missing key: {key}"

    def test_identical_data_small_acf_mse(self):
        data = torch.randn(80, 128)
        ev = StatisticsEvaluator(data, data.clone())
        metrics = ev.compute_extended_metrics(dt=0.01, max_lag=16)
        assert metrics["acf_mse"] < 1e-6

    def test_identical_data_small_wasserstein(self):
        data = torch.randn(80, 128)
        ev = StatisticsEvaluator(data, data.clone())
        metrics = ev.compute_extended_metrics(dt=0.01, max_lag=16)
        assert metrics["wasserstein_1"] < 0.05
