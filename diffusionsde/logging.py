"""Weights & Biases logging utilities.

W&B is enabled automatically if a .env file exists in the project root
with WANDB_ENTITY set. No CLI arguments needed.

Usage:
    from diffusionsde.logging import wandb_init, wandb_log, wandb_finish

    # At start of script
    wandb_init(config, run_name="my-run")  # or let it auto-generate

    # During training
    wandb_log({"loss": 0.5, "epoch": 1})

    # At end
    wandb_finish()
"""

import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Try to load .env from project root
_PROJECT_ROOT = Path(__file__).parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"

if _ENV_FILE.exists():
    from dotenv import load_dotenv

    load_dotenv(_ENV_FILE)

# Check if W&B should be enabled
_WANDB_ENTITY = os.getenv("WANDB_ENTITY")
_WANDB_ENABLED = _WANDB_ENTITY is not None

if _WANDB_ENABLED:
    import wandb
else:
    wandb = None


def is_wandb_enabled() -> bool:
    """Check if W&B logging is enabled."""
    return _WANDB_ENABLED


def _detect_project_name() -> str:
    """Auto-detect project name from the calling script's location.

    If running from examples/duffing/train.py → returns "duffing"
    If running from examples/foo/bar.py → returns "foo"
    Otherwise → returns "ddpm"
    """
    import inspect

    # Walk up the call stack to find the main script
    for frame_info in inspect.stack():
        filepath = Path(frame_info.filename)

        # Check if we're in an examples subdirectory
        parts = filepath.parts
        if "examples" in parts:
            examples_idx = parts.index("examples")
            if examples_idx + 1 < len(parts):
                return parts[examples_idx + 1]

    return "ddpm"  # Default project name


def _generate_run_name(prefix: str | None = None, params: dict | None = None) -> str:
    """Generate a descriptive run name.

    Format: {prefix}_{key-params}_{timestamp}
    Example: train_gamma-0.2-0.8_20250121-143052
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    parts = []
    if prefix:
        parts.append(prefix)

    # Add key parameters to name if provided
    if params:
        param_strs = []
        for key, value in params.items():
            if isinstance(value, float):
                param_strs.append(f"{key}-{value:.2f}")
            elif isinstance(value, int | str):
                param_strs.append(f"{key}-{value}")
        if param_strs:
            parts.append("_".join(param_strs[:3]))  # Limit to 3 params in name

    parts.append(timestamp)

    return "_".join(parts)


def _flatten_config(config: Any, prefix: str = "") -> dict[str, Any]:
    """Convert a config object to a flat dict for W&B logging."""
    if is_dataclass(config) and not isinstance(config, type):
        d = asdict(config)
    elif isinstance(config, dict):
        d = config
    else:
        return {}
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_config(v, key))
        else:
            flat[key] = v
    return flat


def wandb_init(
    config: Any = None,
    run_name: str | None = None,
    run_prefix: str | None = None,
    name_params: dict | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    extra_config: dict | None = None,
) -> bool:
    """Initialise W&B run if enabled.

    Args:
        config: Main config object (dataclass or dict) to log
        run_name: Explicit run name (if None, auto-generated)
        run_prefix: Prefix for auto-generated name (e.g., "train", "eval")
        name_params: Key params to include in auto-generated name
        tags: Optional tags for the run
        notes: Optional notes/description
        extra_config: Additional config to log (merged with main config)

    Returns:
        True if W&B initialised, False otherwise

    Example:
        wandb_init(
            config=diffusion_config,
            run_prefix="train",
            name_params={"gamma": 0.5},
            tags=["baseline"],
            extra_config={"duffing/alpha": -1.0, "duffing/beta": 1.0},
        )
    """
    if not _WANDB_ENABLED:
        return False

    try:
        assert wandb is not None  # Type guard for type checker
        project = _detect_project_name()

        if run_name is None:
            run_name = _generate_run_name(run_prefix, name_params)

        # Build config dict
        run_config = {}
        if config is not None:
            run_config.update(_flatten_config(config))
        if extra_config is not None:
            run_config.update(extra_config)

        wandb.init(
            project=project,
            entity=_WANDB_ENTITY,
            name=run_name,
            tags=tags,
            notes=notes,
            config=run_config,
        )

        print(f"W&B: Logging to {_WANDB_ENTITY}/{project}/{run_name}")
        return True

    except Exception as e:
        print(f"W&B initialisation failed: {e}")
        print("Continuing without W&B logging.")
        return False


def wandb_log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to W&B if active.

    Args:
        metrics: Dictionary of metric names to values
        step: Optional step number (e.g., epoch)

    Example:
        wandb_log({"train/loss": 0.5, "train/lr": 1e-4}, step=epoch)
    """
    if _WANDB_ENABLED and wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def wandb_log_figure(
    fig: plt.Figure,
    name: str,
    step: int | None = None,
    close: bool = True,
) -> None:
    """Log matplotlib figure to W&B.

    Args:
        fig: Matplotlib figure
        name: Name for the logged image
        step: Optional step number
        close: Whether to close the figure after logging
    """
    if _WANDB_ENABLED and wandb is not None and wandb.run is not None:
        wandb.log({name: wandb.Image(fig)}, step=step)
        if close:
            plt.close(fig)


def wandb_log_artifact(
    filepath: Path | str,
    name: str | None = None,
    artifact_type: str = "model",
    metadata: dict | None = None,
) -> None:
    """Log file as W&B artifact.

    Args:
        filepath: Path to file to log
        name: Artifact name (defaults to filename)
        artifact_type: Type of artifact ("model", "dataset", "samples", etc.)
        metadata: Optional metadata dict
    """
    if _WANDB_ENABLED and wandb is not None and wandb.run is not None:
        filepath = Path(filepath)
        if name is None:
            name = filepath.stem

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata,
        )
        artifact.add_file(str(filepath))
        wandb.log_artifact(artifact)


def wandb_finish() -> None:
    """Finish W&B run if active."""
    if _WANDB_ENABLED and wandb is not None and wandb.run is not None:
        wandb.finish()


# Convenience: context manager for W&B runs
