"""Inverse problem solver for diffusion models.

Given observed data and a trained DDPM, find the parameter(s) that best reproduce
the observations by gradient-based optimisation through the sampling process.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusionsde.distances import moment_matching_loss

if TYPE_CHECKING:
    from diffusionsde import DDPM

BuildConditioningFn = Callable[
    [torch.Tensor, int, int, str, int],
    torch.Tensor,
]


def _physical_to_bounded_raw(
    values: np.ndarray,
    lower: float,
    upper: float,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """Map bounded physical values to unconstrained raw values."""
    if upper <= lower:
        msg = "upper must exceed lower for bounded reparameterisation"
        raise ValueError(msg)
    unit = np.clip((values - lower) / (upper - lower), eps, 1.0 - eps)
    return np.log(unit / (1.0 - unit)).astype(np.float32)


def _bounded_raw_to_physical(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """Map unconstrained raw values back to the physical interval."""
    return lower + (upper - lower) * torch.sigmoid(raw)


@dataclass
class InverseProblemConfig:
    """Configuration for inverse problem optimisation.

    Args:
        n_iterations: Number of optimisation iterations.
        lr: Peak learning rate for Adam.
        grad_steps: Number of DDIM denoising steps that retain gradients.
            ``None`` means all steps are differentiable.
        grad_steps_spread: If ``True``, spread gradient-carrying steps linearly
            across the full denoising chain instead of concentrating them at the end.
        use_best_iterate: If ``True``, return the iterate with lowest loss
            rather than the final iterate.
        tail_start: Time-step index from which to compute matching statistics,
            skipping any initial transient.
            ``None`` uses all time steps.
        mean_weight: Weight for the ensemble mean term.
        variance_weight: Weight for the pointwise ensemble variance term.
        n_gen: Number of samples to generate per run per iteration.  ``None``
            falls back to the number of observed trajectories.  Increasing this
            (e.g. to 64) reduces noise in the moment-matching loss gradient at
            the cost of extra memory and compute.
        parameterisation: Optimisation-space parameterisation. ``"physical"``
            optimises directly in physical units with a soft boundary penalty.
            ``"sigmoid_bounded"`` optimises an unconstrained raw variable and
            maps it smoothly into the training interval.
        lr_floor_ratio: Fraction of the peak learning rate used as the cosine
            schedule floor.
    """

    n_iterations: int = 50
    lr: float = 0.1
    grad_steps: int | None = 20
    grad_steps_spread: bool = False
    use_best_iterate: bool = True
    tail_start: int | None = None
    mean_weight: float = 1.0
    variance_weight: float = 1.0
    n_gen: int | None = None  # number of generated samples per run; None falls back to n_obs
    parameterisation: str = "physical"
    lr_floor_ratio: float = 0.1


@dataclass
class InverseResult:
    """Results from an inverse problem solve."""

    estimates: np.ndarray
    parameter_inits: np.ndarray
    parameter_range: tuple[float, float]
    losses: np.ndarray
    wall_time: float = 0.0
    trajectories: list[list[float]] | None = None
    loss_trajectories: list[list[float]] | None = None

    def errors(self, true_value: float) -> np.ndarray:
        return np.abs(self.estimates - true_value)

    def mean_error(self, true_value: float) -> float:
        return float(self.errors(true_value).mean())

    def to_dataframe(
        self,
        true_value: float,
        num_steps: int | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "parameter_true": true_value,
                "parameter_init": self.parameter_inits,
                "parameter_est": self.estimates,
                "abs_error": self.errors(true_value),
                "loss": self.losses,
            }
        )
        if num_steps is not None:
            df["num_steps"] = num_steps
        return df

    def evolution_dataframe(
        self,
        true_value: float,
        num_steps: int | None = None,
    ) -> pd.DataFrame:
        if self.trajectories is None:
            msg = "No trajectories recorded — call solve() with return_trajectories=True"
            raise ValueError(msg)
        rows: list[dict] = []
        for run_idx, traj in enumerate(self.trajectories):
            for it_idx, val in enumerate(traj):
                row: dict = {
                    "run_idx": run_idx,
                    "iteration": it_idx + 1,
                    "parameter_true": true_value,
                    "parameter_init": float(self.parameter_inits[run_idx]),
                    "parameter_est": val,
                    "abs_error": abs(val - true_value),
                }
                if num_steps is not None:
                    row["num_steps"] = num_steps
                rows.append(row)
        return pd.DataFrame(rows)


class InverseProblem:
    """Gradient-based inverse problem solver using a trained DDPM.

    Optimises scalar parameters by back-propagating through the DDIM sampling
    chain. The optimisation variable can live either in physical space or in a
    smoothly bounded raw space, depending on ``InverseProblemConfig``.
    """

    def __init__(
        self,
        ddpm: DDPM,
        observed: torch.Tensor,
        build_conditioning: BuildConditioningFn,
        parameter_range: tuple[float, float],
        config: InverseProblemConfig | None = None,
        sample_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.ddpm = ddpm
        self.observed = observed
        self.build_conditioning = build_conditioning
        self.p_min, self.p_max = parameter_range
        if self.p_max <= self.p_min:
            msg = "parameter_range must satisfy max > min"
            raise ValueError(msg)
        self.p_scale = max(0.5 * (self.p_max - self.p_min), 1e-8)
        self.config = config or InverseProblemConfig()
        self.sample_transform = sample_transform

        if observed.ndim == 2:
            self.per_run_obs = False
            self.n_obs, self.T = observed.shape
        elif observed.ndim == 3:
            self.per_run_obs = True
            _, self.n_obs, self.T = observed.shape
        else:
            msg = f"observed must be 2-D or 3-D, got {observed.ndim}-D"
            raise ValueError(msg)

    def solve(
        self,
        n_runs: int | None = None,
        parameter_inits: np.ndarray | None = None,
        return_trajectories: bool = False,
        true_value: float | None = None,
        runs_per_chunk: int = 5,
    ) -> InverseResult:
        """Optimise all runs simultaneously in a batched loop.

        Args:
            n_runs: Number of independent runs.
            parameter_inits: Initial physical-space values ``[n_runs]``.
            return_trajectories: Record per-iteration trajectories.
            true_value: True parameter value for display.
            runs_per_chunk: Runs per backward pass to bound memory.

        Returns:
            :class:`InverseResult`.
        """
        if n_runs is None and parameter_inits is not None:
            n_runs = len(parameter_inits)
        elif n_runs is None:
            n_runs = 1

        if parameter_inits is None:
            rng = np.random.default_rng()
            parameter_inits = rng.uniform(
                self.p_min,
                self.p_max,
                size=n_runs,
            ).astype(np.float32)

        device = self.ddpm.config.device
        cfg = self.config

        if runs_per_chunk < 1:
            raise ValueError("runs_per_chunk must be at least 1")
        if cfg.n_gen is not None and cfg.n_gen < 1:
            raise ValueError("config.n_gen must be at least 1 when provided")
        if cfg.lr_floor_ratio < 0.0:
            raise ValueError("config.lr_floor_ratio must be non-negative")
        if self.per_run_obs and self.observed.shape[0] != n_runs:
            msg = (
                "Per-run observations must match the number of optimisation runs: "
                f"{self.observed.shape[0]} observations vs {n_runs} runs."
            )
            raise ValueError(msg)

        # Slice to the stationary tail once; the differentiable ensemble loss
        # computes matching statistics internally.
        observed_dev = self.observed.to(device)
        obs_tail = observed_dev if cfg.tail_start is None else observed_dev[..., cfg.tail_start :]
        obs_means: torch.Tensor | None = None
        obs_vars: torch.Tensor | None = None
        obs_stats: tuple[torch.Tensor, torch.Tensor] | None = None
        if self.per_run_obs:
            obs_means = obs_tail.mean(dim=1).detach()
            obs_vars = obs_tail.var(dim=1).detach()
        else:
            obs_stats = (
                obs_tail.mean(dim=0).detach(),
                obs_tail.var(dim=0).detach(),
            )

        if cfg.parameterisation == "physical":
            parameters = torch.tensor(
                parameter_inits,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            optimiser_lr = cfg.lr * self.p_scale
            grad_clip_norm = 1.0 / self.p_scale
        elif cfg.parameterisation == "sigmoid_bounded":
            raw_inits = _physical_to_bounded_raw(parameter_inits, self.p_min, self.p_max)
            parameters = torch.tensor(
                raw_inits,
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            optimiser_lr = 2.0 * cfg.lr
            grad_clip_norm = 1.0
        else:
            raise ValueError(
                f"Unknown parameterisation {cfg.parameterisation!r}. "
                "Choose 'physical' or 'sigmoid_bounded'."
            )

        optimiser = torch.optim.Adam([parameters], lr=optimiser_lr)
        scheduler = CosineAnnealingLR(
            optimiser,
            T_max=cfg.n_iterations,
            eta_min=optimiser_lr * cfg.lr_floor_ratio,
        )

        chunk_starts = list(range(0, n_runs, runs_per_chunk))

        trajs: list[list[float]] | None = (
            [[] for _ in range(n_runs)] if return_trajectories else None
        )
        loss_trajs: list[list[float]] | None = (
            [[] for _ in range(n_runs)] if return_trajectories else None
        )

        if cfg.parameterisation == "sigmoid_bounded":
            best_p_phys = _bounded_raw_to_physical(parameters.detach(), self.p_min, self.p_max)
        else:
            best_p_phys = parameters.detach().clone()
        best_losses = torch.full((n_runs,), float("inf"), device=device)

        # Pre-sample fixed initial noise per run so each run's loss landscape is
        # deterministic across iterations but independent across runs.
        # Shape [n_runs, n_gen, trajectory_length].
        n_gen = cfg.n_gen if cfg.n_gen is not None else self.n_obs
        fixed_noises = torch.randn(n_runs, n_gen, self.T, device=device)

        t0 = time.perf_counter()

        for it in range(cfg.n_iterations):
            optimiser.zero_grad()
            total_loss = 0.0
            iter_run_losses = [0.0] * n_runs

            for c_start in chunk_starts:
                c_end = min(c_start + runs_per_chunk, n_runs)
                chunk_loss = torch.tensor(0.0, device=device)

                for r in range(c_start, c_end):
                    if cfg.parameterisation == "sigmoid_bounded":
                        p_phys = _bounded_raw_to_physical(parameters[r], self.p_min, self.p_max)
                    else:
                        p_phys = parameters[r]

                    conditioning = self.build_conditioning(
                        p_phys.unsqueeze(0),
                        n_gen,
                        self.T,
                        device,
                        r,
                    )
                    generated = self.ddpm.generate(
                        conditioning,
                        n_samples=1,
                        require_grad=True,
                        show_progress=False,
                        initial_noise=fixed_noises[r],
                        grad_steps=cfg.grad_steps,
                        grad_steps_spread=cfg.grad_steps_spread,
                    ).squeeze(1)

                    if self.sample_transform is not None:
                        generated = self.sample_transform(generated, p_phys)

                    gen_tail = (
                        generated if cfg.tail_start is None else generated[..., cfg.tail_start :]
                    )
                    if self.per_run_obs:
                        assert obs_means is not None
                        assert obs_vars is not None
                        obs_stats_for_run = (obs_means[r], obs_vars[r])
                    else:
                        assert obs_stats is not None
                        obs_stats_for_run = obs_stats
                    run_loss = moment_matching_loss(
                        gen_tail,
                        obs_tail[r] if self.per_run_obs else obs_tail,
                        mean_weight=cfg.mean_weight,
                        variance_weight=cfg.variance_weight,
                        precomputed_obs_stats=obs_stats_for_run,
                    )
                    if cfg.parameterisation == "physical":
                        run_loss = (
                            run_loss + 0.1 * (F.relu(p_phys - self.p_max) / self.p_scale) ** 2
                        )
                        run_loss = (
                            run_loss + 0.1 * (F.relu(self.p_min - p_phys) / self.p_scale) ** 2
                        )
                    chunk_loss = chunk_loss + run_loss
                    iter_run_losses[r] = run_loss.item()

                    if cfg.use_best_iterate:
                        with torch.no_grad():
                            if run_loss.item() < best_losses[r].item():
                                best_losses[r] = run_loss.detach()
                                best_p_phys[r] = p_phys.detach()

                chunk_loss.backward()
                total_loss += chunk_loss.item()

            torch.nn.utils.clip_grad_norm_([parameters], max_norm=grad_clip_norm)
            optimiser.step()
            scheduler.step()

            if trajs is not None:
                with torch.no_grad():
                    if cfg.parameterisation == "sigmoid_bounded":
                        p_phys_all = _bounded_raw_to_physical(parameters, self.p_min, self.p_max)
                    else:
                        p_phys_all = parameters.detach()
                    for r in range(n_runs):
                        trajs[r].append(float(p_phys_all[r]))
            if loss_trajs is not None:
                for r in range(n_runs):
                    loss_trajs[r].append(iter_run_losses[r])

            if it % 10 == 0:
                with torch.no_grad():
                    if cfg.parameterisation == "sigmoid_bounded":
                        p_phys_all = _bounded_raw_to_physical(parameters, self.p_min, self.p_max)
                    else:
                        p_phys_all = parameters.detach()
                target_str = f"  target={true_value:.4f}" if true_value is not None else ""
                print(
                    f"  iter {it:3d}/{cfg.n_iterations}: "
                    f"loss={total_loss:.6f}, "
                    f"range=[{p_phys_all.min():.4f}, {p_phys_all.max():.4f}]"
                    f"{target_str}"
                )

        wall_time = time.perf_counter() - t0

        with torch.no_grad():
            if cfg.use_best_iterate:
                final_phys = best_p_phys.cpu().numpy()
            elif cfg.parameterisation == "sigmoid_bounded":
                final_phys = (
                    _bounded_raw_to_physical(parameters, self.p_min, self.p_max).cpu().numpy()
                )
            else:
                final_phys = parameters.detach().cpu().numpy()

        final_losses = (
            best_losses.cpu().numpy()
            if cfg.use_best_iterate
            else np.full(n_runs, total_loss / n_runs)
        )

        print(
            f"\n  {n_runs} runs done in {wall_time:.1f}s.  "
            f"median={float(np.median(final_phys)):.4f}, "
            f"mean={float(np.mean(final_phys)):.4f}"
        )

        return InverseResult(
            estimates=final_phys,
            parameter_inits=parameter_inits.copy(),
            parameter_range=(self.p_min, self.p_max),
            losses=final_losses,
            wall_time=wall_time,
            trajectories=trajs,
            loss_trajectories=loss_trajs,
        )
