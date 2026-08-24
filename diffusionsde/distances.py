"""Distance metrics for comparing trajectory ensembles."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def moment_matching_loss(
    generated: torch.Tensor,
    observed: torch.Tensor,
    *,
    mean_weight: float = 1.0,
    variance_weight: float = 1.0,
    precomputed_obs_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    r"""Pointwise moment-matching cost between two trajectory ensembles.

    Computes:

    .. math::
        \mathcal{C} = w_\mu \frac{1}{T}\sum_{k=1}^{T}
            \bigl[\mu_{\mathrm{gen}}(t_k) - \mu_{\mathrm{obs}}(t_k)\bigr]^2
        + w_\sigma \frac{1}{T}\sum_{k=1}^{T}
            \bigl[\sigma^2_{\mathrm{gen}}(t_k) - \sigma^2_{\mathrm{obs}}(t_k)\bigr]^2

    This treats each time point independently — it cannot detect differences
    in temporal correlation structure (e.g. autocorrelation, smoothness).

    Args:
        generated: Generated trajectories ``[n_gen, T]``.
        observed: Observed trajectories ``[n_obs, T]``.  Ignored when
            ``precomputed_obs_stats`` is provided.
        mean_weight: Weight for the mean term ($w_\mu$).
        variance_weight: Weight for the variance term ($w_\sigma$).
        precomputed_obs_stats: Optional ``(obs_mean, obs_var)`` each of shape
            ``[T]``, pre-computed from the observed data.  When supplied the
            ``observed`` argument is not used, avoiding redundant computation
            across optimisation iterations.

    Returns:
        Scalar loss tensor (differentiable).
    """
    gen_mean = generated.mean(dim=0)
    gen_var = generated.var(dim=0)

    if precomputed_obs_stats is not None:
        obs_mean, obs_var = precomputed_obs_stats
    else:
        obs_mean = observed.mean(dim=0)
        obs_var = observed.var(dim=0)

    mean_loss = F.mse_loss(gen_mean, obs_mean)
    var_loss = F.mse_loss(gen_var, obs_var)

    return mean_weight * mean_loss + variance_weight * var_loss
