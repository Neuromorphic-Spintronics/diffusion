"""Plotting utilities for diffusion models.

This module provides consistent styling and reusable plotting functions
for visualising diffusion model results.

Example usage:
    >>> from diffusionsde.plotting import configure_matplotlib
    >>> configure_matplotlib()  # Set up LaTeX fonts
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# =============================================================================
# Colour Palette
# =============================================================================

COLOURS: list[str] = [
    "#ffbe0b",  # Yellow
    "#fb5607",  # Orange
    "#ff006e",  # Pink
    "#8338ec",  # Purple
    "#3a86ff",  # Blue
    "#06d6a0",  # Green
    "#390099",  # Deep purple
    "#ef476f",  # Coral red
    "#61E8E1",  # Teal
    "#00D4FF",  # Cyan
]

# Default colours for reference vs generated comparison
REFERENCE_COLOUR = "#6f6f6f"
GENERATED_COLOUR = COLOURS[3]  # Purple
ANALYTIC_COLOUR = "black"

# =============================================================================
# Matplotlib Configuration
# =============================================================================

_LATEX_CONFIG = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,  # Use proper LaTeX minus sign
}

_FALLBACK_CONFIG = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
}


def configure_matplotlib(use_latex: bool = True) -> bool:
    """Configure matplotlib

    The LaTeX renderability probe runs on the non-interactive Agg backend;
    afterwards the caller's original backend is restored so figures still
    display interactively (e.g. in Jupyter).

    Args:
        use_latex: Whether to attempt LaTeX rendering (default: True)

    Returns:
        True if LaTeX rendering is enabled, False otherwise
    """
    previous_backend = matplotlib.get_backend()
    try:
        plt.switch_backend("Agg")
        if use_latex:
            try:
                plt.rcParams.update(_LATEX_CONFIG)  # ty: ignore[no-matching-overload]
                # Test LaTeX rendering
                fig, ax = plt.subplots(1, 1)
                ax.text(0.5, 0.5, r"$\mu$")
                fig.canvas.draw()
                plt.close(fig)
                return True
            except (RuntimeError, FileNotFoundError):
                pass

        plt.rcParams.update(_FALLBACK_CONFIG)  # ty: ignore[no-matching-overload]
        return False
    finally:
        plt.switch_backend(previous_backend)


def _to_numpy(data: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def save_figure(fig: plt.Figure, save_path: Path | str | None) -> None:
    """Save figure to PDF.

    Args:
        fig: Matplotlib figure to save
        save_path: Path to save figure (without extension), or None to show
    """
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path.with_suffix('.pdf')}")
        plt.close(fig)
    else:
        plt.show()


def clean_axes(ax: plt.Axes) -> None:
    """Remove top and right spines from axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Generic Plotting Functions
# =============================================================================


def plot_trajectory_statistics(
    reference: torch.Tensor | np.ndarray,
    generated: torch.Tensor | np.ndarray,
    time_array: np.ndarray,
    reference_label: str = "Reference",
    generated_label: str = "Generated",
    ylabel_mean: str = r"$\mu(t)$",
    ylabel_var: str = r"$\sigma^2(t)$",
    xlabel: str = r"$t$",
    reference_band_mode: str | None = "std",
    generated_band_mode: str | None = "std",
    analytic_mean: torch.Tensor | np.ndarray | None = None,
    analytic_var: torch.Tensor | np.ndarray | None = None,
    analytic_label: str = "Analytic",
    save_path: Path | str | None = None,
) -> None:
    """Plot comparison of trajectory statistics (mean and variance).

    Creates two subplots showing mean and variance comparison.

    Args:
        reference: Reference trajectories [n_trajectories, n_steps]
        generated: Generated trajectories [n_trajectories, n_steps]
        time_array: Time values [n_steps]
        reference_label: Label for reference data
        generated_label: Label for generated data
        ylabel_mean: Y-axis label for mean plot
        ylabel_var: Y-axis label for variance plot
        xlabel: X-axis label
        reference_band_mode: Spread band for reference curve: ``"std"``, ``"sem"``, or ``None``
        generated_band_mode: Spread band for generated curve: ``"std"``, ``"sem"``, or ``None``
        analytic_mean: Optional analytic mean curve [n_steps]
        analytic_var: Optional analytic variance curve [n_steps]
        analytic_label: Label used for analytic overlays
        save_path: Path to save figure (without extension)
    """

    reference = _to_numpy(reference)
    generated = _to_numpy(generated)
    analytic_mean = None if analytic_mean is None else _to_numpy(analytic_mean)
    analytic_var = None if analytic_var is None else _to_numpy(analytic_var)

    fig, axes = plt.subplots(2, 1, figsize=(6, 4.8), sharex=True)

    def compute_mean_band(data: np.ndarray, mode: str | None) -> np.ndarray | None:
        if mode is None:
            return None
        if mode == "std":
            return data.std(axis=0)
        if mode == "sem":
            return data.std(axis=0) / np.sqrt(data.shape[0])
        raise ValueError(f"Unknown band mode {mode!r}. Expected 'std', 'sem', or None.")

    def compute_variance_band(data: np.ndarray, mode: str | None) -> np.ndarray | None:
        if mode is None:
            return None

        centred = data - data.mean(axis=0, keepdims=True)
        squared = centred**2

        if mode == "std":
            return squared.std(axis=0)
        if mode == "sem":
            return squared.std(axis=0) / np.sqrt(data.shape[0])
        raise ValueError(f"Unknown band mode {mode!r}. Expected 'std', 'sem', or None.")

    # Compute statistics
    ref_mean = reference.mean(axis=0)
    gen_mean = generated.mean(axis=0)
    ref_mean_band = compute_mean_band(reference, reference_band_mode)
    gen_mean_band = compute_mean_band(generated, generated_band_mode)

    # Subplot 1: Mean comparison
    ax = axes[0]
    ax.plot(
        time_array,
        ref_mean,
        color=REFERENCE_COLOUR,
        linewidth=1.5,
        label=f"{reference_label} (mean)",
    )
    if ref_mean_band is not None:
        band_label = (
            rf"{reference_label} ($\pm 1\sigma$)"
            if reference_band_mode == "std"
            else rf"{reference_label} (SEM)"
        )
        ax.fill_between(
            time_array,
            ref_mean - ref_mean_band,
            ref_mean + ref_mean_band,
            color="grey",
            alpha=0.3,
            label=band_label,
        )
    ax.plot(
        time_array,
        gen_mean,
        color=GENERATED_COLOUR,
        linewidth=1.5,
        label=f"{generated_label} (mean)",
    )
    if gen_mean_band is not None:
        band_label = (
            rf"{generated_label} ($\pm 1\sigma$)"
            if generated_band_mode == "std"
            else rf"{generated_label} (SEM)"
        )
        ax.fill_between(
            time_array,
            gen_mean - gen_mean_band,
            gen_mean + gen_mean_band,
            color=GENERATED_COLOUR,
            alpha=0.3,
            label=band_label,
        )
    if analytic_mean is not None:
        ax.plot(
            time_array,
            analytic_mean,
            color=ANALYTIC_COLOUR,
            linewidth=1.2,
            linestyle="--",
            label=analytic_label,
        )
    ax.set_ylabel(ylabel_mean)
    ax.set_xlim(time_array[0], time_array[-1])
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=3,
        bbox_to_anchor=(0.0, 1.04),
        borderaxespad=0.0,
    )
    clean_axes(ax)

    # Subplot 2: Variance comparison with standard error shading
    ax = axes[1]
    ref_var = reference.var(axis=0)
    gen_var = generated.var(axis=0)
    ref_var_band = compute_variance_band(reference, reference_band_mode)
    gen_var_band = compute_variance_band(generated, generated_band_mode)

    ax.plot(time_array, ref_var, color=REFERENCE_COLOUR, linewidth=1.5, label=reference_label)
    ax.plot(time_array, gen_var, color=GENERATED_COLOUR, linewidth=1.5, label=generated_label)
    if ref_var_band is not None:
        band_label = (
            rf"{reference_label} variance ($\pm 1\sigma$)"
            if reference_band_mode == "std"
            else rf"{reference_label} variance (SEM)"
        )
        ax.fill_between(
            time_array,
            np.clip(ref_var - ref_var_band, a_min=0.0, a_max=None),
            ref_var + ref_var_band,
            color="grey",
            alpha=0.2,
            label=band_label,
        )
    if gen_var_band is not None:
        band_label = (
            rf"{generated_label} variance ($\pm 1\sigma$)"
            if generated_band_mode == "std"
            else rf"{generated_label} variance (SEM)"
        )
        ax.fill_between(
            time_array,
            np.clip(gen_var - gen_var_band, a_min=0.0, a_max=None),
            gen_var + gen_var_band,
            color=GENERATED_COLOUR,
            alpha=0.2,
            label=band_label,
        )
    if analytic_var is not None:
        ax.plot(
            time_array,
            analytic_var,
            color=ANALYTIC_COLOUR,
            linewidth=1.2,
            linestyle="--",
            label=analytic_label,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_var)
    ax.set_xlim(time_array[0], time_array[-1])
    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=3,
        bbox_to_anchor=(0.0, 1.04),
        borderaxespad=0.0,
    )
    clean_axes(ax)

    fig.subplots_adjust(top=0.84, hspace=0.7)
    save_figure(fig, save_path)


def plot_sample_trajectories(
    reference: torch.Tensor | np.ndarray,
    generated: torch.Tensor | np.ndarray,
    time_array: np.ndarray,
    n_samples: int = 5,
    reference_label: str = "Reference",
    generated_label: str = "Generated",
    ylabel_ref: str | None = None,
    ylabel_gen: str | None = None,
    xlabel: str = r"$t$",
    save_path: Path | str | None = None,
) -> None:
    """Plot sample individual trajectories.

    Creates two subplots showing sample reference and generated trajectories.

    Args:
        reference: Reference trajectories [n_trajectories, n_steps]
        generated: Generated trajectories [n_trajectories, n_steps]
        time_array: Time values [n_steps]
        n_samples: Number of sample trajectories to plot
        reference_label: Label for reference data
        generated_label: Label for generated data
        ylabel_ref: Y-axis label for reference plot (default: uses reference_label)
        ylabel_gen: Y-axis label for generated plot (default: uses generated_label)
        xlabel: X-axis label
        save_path: Path to save figure (without extension)
    """

    reference = _to_numpy(reference)
    generated = _to_numpy(generated)

    if ylabel_ref is None:
        ylabel_ref = rf"$x_{{\mathrm{{{reference_label}}}}}(t)$"
    if ylabel_gen is None:
        ylabel_gen = rf"$x_{{\mathrm{{{generated_label}}}}}(t)$"

    fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    # Plot reference trajectories
    ax = axes[0]
    for i in range(min(n_samples, len(reference))):
        ax.plot(time_array, reference[i], alpha=0.6, linewidth=1, color=REFERENCE_COLOUR)
    ax.set_ylabel(ylabel_ref)
    ax.set_xlim(time_array[0], time_array[-1])
    clean_axes(ax)

    # Plot generated trajectories
    ax = axes[1]
    for i in range(min(n_samples, len(generated))):
        ax.plot(time_array, generated[i], alpha=0.6, linewidth=1, color=GENERATED_COLOUR)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel_gen)
    ax.set_xlim(time_array[0], time_array[-1])
    clean_axes(ax)

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_acf_comparison(
    real_acf: np.ndarray,
    generated_acf: np.ndarray | None,
    lag_axis: np.ndarray,
    real_acf_std: np.ndarray | None = None,
    generated_acf_std: np.ndarray | None = None,
    analytic_acf: np.ndarray | None = None,
    reference_label: str = "Reference",
    generated_label: str = "Generated",
    analytic_label: str = "Analytic",
    xlabel: str = r"Lag $\tau$",
    ylabel: str = r"$\rho(\tau)$",
    save_path: Path | str | None = None,
) -> None:
    """Plot autocorrelation function comparison.

    Args:
        real_acf: Mean ACF of reference trajectories [max_lag+1]
        generated_acf: Mean ACF of generated trajectories [max_lag+1], or None to omit
        lag_axis: Lag values in physical time units [max_lag+1]
        real_acf_std: Optional std of reference ACF for shading
        generated_acf_std: Optional std of generated ACF for shading
        analytic_acf: Optional analytic ACF curve [max_lag+1]
        reference_label: Label for reference curve
        generated_label: Label for generated curve
        analytic_label: Label for analytic curve
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (without extension)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))

    ax.plot(lag_axis, real_acf, color=REFERENCE_COLOUR, linewidth=1.5, label=reference_label)
    if real_acf_std is not None:
        ax.fill_between(
            lag_axis,
            real_acf - real_acf_std,
            real_acf + real_acf_std,
            color=REFERENCE_COLOUR,
            alpha=0.25,
        )

    if generated_acf is not None:
        ax.plot(
            lag_axis, generated_acf, color=GENERATED_COLOUR, linewidth=1.5, label=generated_label
        )
        if generated_acf_std is not None:
            ax.fill_between(
                lag_axis,
                generated_acf - generated_acf_std,
                generated_acf + generated_acf_std,
                color=GENERATED_COLOUR,
                alpha=0.25,
            )

    if analytic_acf is not None:
        ax.plot(
            lag_axis,
            analytic_acf,
            color=ANALYTIC_COLOUR,
            linewidth=1.2,
            linestyle="--",
            label=analytic_label,
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(lag_axis[0], lag_axis[-1])
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_psd_comparison(
    freqs: np.ndarray,
    real_psd: np.ndarray,
    generated_psd: np.ndarray | None,
    reference_label: str = "Reference",
    generated_label: str = "Generated",
    xlabel: str = "Frequency",
    ylabel: str = "PSD",
    log_scale: bool = True,
    freq_limit: float | None = None,
    reference_freq: float | None = None,
    save_path: Path | str | None = None,
) -> None:
    """Plot power spectral density comparison.

    Args:
        freqs: Frequency array [n_freq]
        real_psd: Mean PSD of reference trajectories [n_freq]
        generated_psd: Mean PSD of generated trajectories [n_freq], or None to omit
        reference_label: Label for reference curve
        generated_label: Label for generated curve
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: If True, use log scale on y-axis
        freq_limit: Optional upper frequency limit for x-axis
        reference_freq: Optional vertical dashed line at a known drive frequency
        save_path: Path to save figure (without extension)
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))

    mask = freqs > 0  # Exclude DC component
    if freq_limit is not None:
        mask = mask & (freqs <= freq_limit)

    ax.plot(
        freqs[mask], real_psd[mask], color=REFERENCE_COLOUR, linewidth=1.5, label=reference_label
    )
    if generated_psd is not None:
        ax.plot(
            freqs[mask],
            generated_psd[mask],
            color=GENERATED_COLOUR,
            linewidth=1.5,
            label=generated_label,
        )

    if reference_freq is not None:
        ax.axvline(
            reference_freq,
            color=ANALYTIC_COLOUR,
            linewidth=1.0,
            linestyle="--",
            label="Drive freq.",
            alpha=0.7,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_marginal_distribution(
    real: np.ndarray | torch.Tensor,
    generated: np.ndarray | torch.Tensor | None,
    reference_label: str = "Reference",
    generated_label: str = "Generated",
    analytic_label: str = "Analytic",
    xlabel: str = "Value",
    n_bins: int = 80,
    tail_fraction: float = 0.5,
    analytic_pdf: tuple[np.ndarray, np.ndarray] | None = None,
    save_path: Path | str | None = None,
) -> None:
    """Plot marginal distribution comparison as overlaid histograms.

    Pools the stationary tail of all trajectories and overlays normalised
    histograms for reference and generated data.

    Args:
        real: Reference trajectories [n_traj, n_timesteps] or flat array
        generated: Generated trajectories [n_traj, n_timesteps] or flat array, or None to omit
        reference_label: Label for reference histogram
        generated_label: Label for generated histogram
        analytic_label: Label for analytic PDF curve
        xlabel: X-axis label
        n_bins: Number of histogram bins
        tail_fraction: Fraction of each trajectory to use (last portion)
        analytic_pdf: Optional (x, pdf) tuple to overlay as a dashed analytic curve
        save_path: Path to save figure (without extension)
    """

    def _extract_tail(data: np.ndarray | torch.Tensor, frac: float) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if data.ndim == 2:
            n_tail = max(1, int(data.shape[1] * frac))
            data = data[:, -n_tail:]
        return data.flatten()

    real_vals = _extract_tail(real, tail_fraction)

    if generated is not None:
        gen_vals = _extract_tail(generated, tail_fraction)
        all_vals = np.concatenate([real_vals, gen_vals])
    else:
        gen_vals = None
        all_vals = real_vals

    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))

    ax.hist(
        real_vals,
        bins=bin_edges,
        density=True,
        color=REFERENCE_COLOUR,
        alpha=0.55,
        label=reference_label,
    )
    if gen_vals is not None:
        ax.hist(
            gen_vals,
            bins=bin_edges,
            density=True,
            color=GENERATED_COLOUR,
            alpha=0.55,
            label=generated_label,
        )

    if analytic_pdf is not None:
        x_analytic, pdf_analytic = analytic_pdf
        ax.plot(
            x_analytic,
            pdf_analytic,
            color=ANALYTIC_COLOUR,
            linewidth=1.4,
            linestyle="--",
            label=analytic_label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right")
    clean_axes(ax)

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_loss_curve(
    losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Path | str | None = None,
) -> None:
    """Plot training loss curve with optional validation loss.

    Args:
        losses: List of training loss values per epoch
        val_losses: Optional list of validation loss values per epoch
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (without extension)
    """

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    epochs = np.arange(1, len(losses) + 1)
    ax.plot(epochs, losses, color=REFERENCE_COLOUR, linewidth=1, label="Train")

    if val_losses is not None and len(val_losses) > 0:
        val_epochs = np.arange(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, color=GENERATED_COLOUR, linewidth=1, label="Val")
        ax.legend(frameon=False, loc="upper right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, len(losses))
    clean_axes(ax)

    plt.tight_layout()
    save_figure(fig, save_path)


def plot_phase_space(
    trajectories: np.ndarray,
    trajectories_2: np.ndarray | None = None,
    n_samples: int = 10,
    xlabel: str = r"$q$",
    ylabel: str = r"$v$",
    label_1: str = "Reference",
    label_2: str = "Generated",
    x_idx: int = 0,
    y_idx: int = 1,
    save_path: Path | str | None = None,
) -> None:
    """Plot phase space trajectories.

    Args:
        trajectories: First set of trajectories [n_trajectories, n_steps, n_dim]
        trajectories_2: Optional second set for comparison
        n_samples: Number of trajectories to plot
        xlabel: X-axis label
        ylabel: Y-axis label
        label_1: Label for first set
        label_2: Label for second set
        x_idx: Index of x-coordinate in state vector
        y_idx: Index of y-coordinate in state vector
        save_path: Path to save figure (without extension)
    """

    trajectories = _to_numpy(trajectories)
    if trajectories_2 is not None:
        trajectories_2 = _to_numpy(trajectories_2)

    if trajectories_2 is None:
        # Single subplot
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        for i in range(min(n_samples, len(trajectories))):
            x = trajectories[i, :, x_idx]
            y = trajectories[i, :, y_idx]
            ax.plot(x, y, alpha=0.6, linewidth=0.5, color=REFERENCE_COLOUR)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clean_axes(ax)
    else:
        # Two subplots
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        ax = axes[0]
        for i in range(min(n_samples, len(trajectories))):
            x = trajectories[i, :, x_idx]
            y = trajectories[i, :, y_idx]
            ax.plot(x, y, alpha=0.6, linewidth=0.5, color=REFERENCE_COLOUR)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clean_axes(ax)

        ax = axes[1]
        for i in range(min(n_samples, len(trajectories_2))):
            x = trajectories_2[i, :, x_idx]
            y = trajectories_2[i, :, y_idx]
            ax.plot(x, y, alpha=0.6, linewidth=0.5, color=GENERATED_COLOUR)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        clean_axes(ax)

    plt.tight_layout()
    save_figure(fig, save_path)
