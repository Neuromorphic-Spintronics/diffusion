"""Evaluation utilities for diffusion models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sample_acf(trajectories: torch.Tensor, max_lag: int) -> torch.Tensor:
    r"""Per-trajectory autocorrelation for lags 0..max_lag.

    Args:
        trajectories: [n_trajectories, n_timesteps]
        max_lag: Maximum lag index (must be < n_timesteps)

    Returns:
        acf: [n_trajectories, max_lag + 1], lag-0 entry is always 1.0
    """
    mu = trajectories.mean(dim=1, keepdim=True)
    sigma2 = trajectories.var(dim=1, keepdim=True).clamp(min=1e-8)
    normalised = (trajectories - mu) / sigma2.sqrt()
    acf_values = [torch.ones(trajectories.shape[0], device=trajectories.device)]
    for lag in range(1, max_lag + 1):
        acf_values.append((normalised[:, :-lag] * normalised[:, lag:]).mean(dim=1))
    return torch.stack(acf_values, dim=1)


def compute_psd(
    trajectories: torch.Tensor,
    dt: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensemble-averaged power spectral density via Welch's method.

    Args:
        trajectories: [n_trajectories, n_timesteps]
        dt: Sampling interval in physical time units
        nperseg: Welch segment length; defaults to min(n_timesteps, 256)

    Returns:
        freqs: [n_freq] frequency array in units of 1/dt
        psd_mean: [n_freq] mean PSD averaged over trajectories
    """
    from scipy.signal import welch

    data = trajectories.cpu().numpy()
    n = data.shape[1]
    seg = nperseg or min(n, 256)
    psds = []
    for traj in data:
        f, pxx = welch(traj, fs=1.0 / dt, nperseg=seg)
        psds.append(pxx)
    return f, np.stack(psds).mean(axis=0)


def compute_marginal_wasserstein(
    real: torch.Tensor,
    generated: torch.Tensor,
    tail_fraction: float = 0.5,
) -> float:
    """Wasserstein-1 distance between marginal distributions on the stationary tail.

    Pools all values from the last ``tail_fraction`` of each trajectory and
    computes the 1-Wasserstein (Earth Mover's) distance between the two empirical
    distributions.

    Args:
        real: [n_traj, n_timesteps]
        generated: [n_traj, n_timesteps]
        tail_fraction: Fraction of each trajectory to treat as stationary

    Returns:
        Wasserstein-1 distance as a float
    """
    from scipy.stats import wasserstein_distance

    n_tail = max(1, int(real.shape[1] * tail_fraction))
    u = real[:, -n_tail:].flatten().cpu().numpy()
    v = generated[:, -n_tail:].flatten().cpu().numpy()
    return float(wasserstein_distance(u, v))


def compute_higher_moments(
    data: torch.Tensor,
    tail_fraction: float = 0.5,
) -> dict[str, float]:
    """Skewness and excess kurtosis of the pooled stationary-tail marginal.

    Args:
        data: [n_traj, n_timesteps]
        tail_fraction: Fraction of each trajectory to treat as stationary

    Returns:
        dict with keys ``skewness`` and ``excess_kurtosis``
    """
    from scipy.stats import kurtosis, skew

    n_tail = max(1, int(data.shape[1] * tail_fraction))
    tail = data[:, -n_tail:].flatten().cpu().numpy()
    return {
        "skewness": float(skew(tail)),
        "excess_kurtosis": float(kurtosis(tail, fisher=True)),
    }


class StatisticsEvaluator:
    r"""Evaluate generated data statistics against ground truth.

    Computes and compares $\mu(t)$ and $\sigma(t)$ curves between real and
    generated data.

    Args:
        real_data: Real trajectories [n_runs, n_timesteps]
        generated_data: Generated trajectories [n_runs, n_timesteps]
    """

    def __init__(self, real_data: torch.Tensor, generated_data: torch.Tensor) -> None:
        """Initialise evaluator with real and generated data."""
        self.real_data = real_data.cpu()
        self.generated_data = generated_data.cpu()

    @staticmethod
    def compute_statistics(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Compute $\mu(t)$ and $\sigma(t)$ for trajectories.

        Args:
            data: Trajectories of shape [n_runs, n_timesteps]

        Returns:
            Tuple of ($\mu(t)$, $\sigma(t)$) each of shape [n_timesteps]
        """
        mu_t = data.mean(dim=0)
        sigma_t = data.std(dim=0)
        return mu_t, sigma_t

    def compute_metrics(self) -> dict[str, float]:
        """Compute comparison metrics between real and generated.

        Returns:
            Dictionary with metrics like MSE, MAE, correlation, etc.
        """
        # Compute statistics
        real_mu, real_sigma = self.compute_statistics(self.real_data)
        gen_mu, gen_sigma = self.compute_statistics(self.generated_data)

        # Compute metrics
        metrics = {
            "mu_mse": F.mse_loss(gen_mu, real_mu).item(),
            "sigma_mse": F.mse_loss(gen_sigma, real_sigma).item(),
            "mu_mae": F.l1_loss(gen_mu, real_mu).item(),
            "sigma_mae": F.l1_loss(gen_sigma, real_sigma).item(),
            "mu_correlation": torch.corrcoef(torch.stack([real_mu, gen_mu]))[0, 1].item(),
            "sigma_correlation": torch.corrcoef(torch.stack([real_sigma, gen_sigma]))[0, 1].item(),
        }

        return metrics

    def compute_acf_metrics(
        self,
        max_lag: int = 64,
        tail_fraction: float = 1.0,
    ) -> dict[str, float | np.ndarray]:
        """ACF comparison between real and generated trajectories.

        Args:
            max_lag: Maximum lag index
            tail_fraction: Fraction of each trajectory to use (1.0 = all)

        Returns:
            dict with scalar metrics and ACF curve arrays
        """
        if tail_fraction < 1.0:
            n_tail = max(max_lag + 2, int(self.real_data.shape[1] * tail_fraction))
            real = self.real_data[:, -n_tail:]
            gen = self.generated_data[:, -n_tail:]
        else:
            real = self.real_data
            gen = self.generated_data

        max_lag = min(max_lag, real.shape[1] - 2)
        real_acf = sample_acf(real, max_lag)
        gen_acf = sample_acf(gen, max_lag)

        real_mean = real_acf.mean(dim=0).numpy()
        gen_mean = gen_acf.mean(dim=0).numpy()

        acf_mae = float(np.mean(np.abs(real_mean - gen_mean)))
        acf_mse = float(np.mean((real_mean - gen_mean) ** 2))

        return {
            "real_acf_mean": real_mean,
            "generated_acf_mean": gen_mean,
            "real_acf_std": real_acf.std(dim=0).numpy(),
            "generated_acf_std": gen_acf.std(dim=0).numpy(),
            "acf_mae": acf_mae,
            "acf_mse": acf_mse,
        }

    def compute_psd_metrics(
        self,
        dt: float,
        nperseg: int | None = None,
        tail_fraction: float = 1.0,
    ) -> dict[str, float | np.ndarray]:
        """PSD comparison via Welch's method.

        Args:
            dt: Sampling interval in physical time units
            nperseg: Welch segment length
            tail_fraction: Fraction of each trajectory to use

        Returns:
            dict with scalar metrics and PSD curve arrays
        """
        if tail_fraction < 1.0:
            n_tail = max(2, int(self.real_data.shape[1] * tail_fraction))
            real = self.real_data[:, -n_tail:]
            gen = self.generated_data[:, -n_tail:]
        else:
            real = self.real_data
            gen = self.generated_data

        freqs, real_psd = compute_psd(real, dt, nperseg)
        _, gen_psd = compute_psd(gen, dt, nperseg)

        # Log-domain MAE (avoids DC-component dominance)
        eps = np.finfo(float).tiny
        log_mae = float(np.mean(np.abs(np.log10(real_psd + eps) - np.log10(gen_psd + eps))))

        dominant_freq_real = float(freqs[np.argmax(real_psd[1:]) + 1])
        dominant_freq_gen = float(freqs[np.argmax(gen_psd[1:]) + 1])

        return {
            "freqs": freqs,
            "real_psd_mean": real_psd,
            "generated_psd_mean": gen_psd,
            "psd_log_mae": log_mae,
            "dominant_freq_real": dominant_freq_real,
            "dominant_freq_gen": dominant_freq_gen,
        }

    def compute_distributional_metrics(
        self,
        tail_fraction: float = 0.5,
    ) -> dict[str, float]:
        """Higher-order moments and Wasserstein-1 on the stationary marginal.

        Args:
            tail_fraction: Fraction of each trajectory to treat as stationary

        Returns:
            dict with wasserstein_1, skewness, excess_kurtosis for both real and generated
        """
        w1 = compute_marginal_wasserstein(self.real_data, self.generated_data, tail_fraction)
        real_moments = compute_higher_moments(self.real_data, tail_fraction)
        gen_moments = compute_higher_moments(self.generated_data, tail_fraction)

        return {
            "wasserstein_1": w1,
            "real_skewness": real_moments["skewness"],
            "generated_skewness": gen_moments["skewness"],
            "real_excess_kurtosis": real_moments["excess_kurtosis"],
            "generated_excess_kurtosis": gen_moments["excess_kurtosis"],
            "skewness_error": abs(real_moments["skewness"] - gen_moments["skewness"]),
            "kurtosis_error": abs(real_moments["excess_kurtosis"] - gen_moments["excess_kurtosis"]),
        }

    def compute_extended_metrics(
        self,
        *,
        dt: float,
        max_lag: int = 64,
        tail_fraction: float = 0.5,
        nperseg: int | None = None,
    ) -> dict[str, float | np.ndarray]:
        """All scalar fidelity metrics in a single flat dict (suitable for CSV).

        Args:
            dt: Sampling interval in physical time units
            max_lag: Maximum ACF lag
            tail_fraction: Fraction of trajectory treated as stationary
            nperseg: Welch segment length for PSD

        Returns:
            Flat dict of scalar metrics (excludes array-valued outputs)
        """
        out: dict[str, float | np.ndarray] = {}
        out.update(self.compute_metrics())
        acf = self.compute_acf_metrics(max_lag=max_lag, tail_fraction=tail_fraction)
        out["acf_mae"] = acf["acf_mae"]
        out["acf_mse"] = acf["acf_mse"]
        psd = self.compute_psd_metrics(dt=dt, nperseg=nperseg, tail_fraction=tail_fraction)
        out["psd_log_mae"] = psd["psd_log_mae"]
        out["dominant_freq_real"] = psd["dominant_freq_real"]
        out["dominant_freq_gen"] = psd["dominant_freq_gen"]
        out.update(self.compute_distributional_metrics(tail_fraction=tail_fraction))
        return out
