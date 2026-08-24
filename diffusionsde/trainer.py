"""Training utilities for diffusion models."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusionsde.config import DiffusionConfig
from diffusionsde.diffusion import DiffusionModel
from diffusionsde.logging import wandb_log


class DiffusionTrainer:
    """Trainer for the diffusion model.

    Handles training loop, optimisation, and checkpointing.

    Args:
        model: Diffusion model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Training configuration
    """

    def __init__(
        self,
        model: DiffusionModel,
        train_dataset: Any,
        val_dataset: Any | None = None,
        config: DiffusionConfig | None = None,
    ) -> None:
        """Initialise trainer with model and dataset."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or DiffusionConfig()

        # Create dataloader
        use_pin_memory = self.config.device == "cuda"  # Only pin memory for CUDA
        num_workers = 0  # MPS works better with num_workers=0
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
            )
        else:
            self.val_loader = None

        # Optimiser
        self.optimiser = torch.optim.Adam(model.model.parameters(), lr=self.config.learning_rate)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimiser, T_max=self.config.n_epochs
        )

        # Metrics tracking
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss: float | None = None
        self.best_epoch: int | None = None
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def train(self) -> dict[str, list[float]]:
        """Train the diffusion model.

        Returns:
            Dictionary of training metrics
        """
        self.model.model.train()

        for epoch in range(self.config.n_epochs):
            # Training phase
            epoch_train_losses = []

            with tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.n_epochs}") as pbar:
                for data_batch, cond_batch in pbar:
                    data_batch = data_batch.to(self.config.device)
                    cond_batch = cond_batch.to(self.config.device)

                    # Compute loss
                    loss = self.model.compute_loss(data_batch, cond_batch)

                    # Backward pass
                    self.optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                    self.optimiser.step()

                    # Track loss
                    epoch_train_losses.append(loss.item())
                    pbar.set_postfix({"train_loss": loss.item()})

            # Update scheduler
            self.scheduler.step()

            # Record epoch average
            avg_train_loss = float(np.mean(epoch_train_losses))
            self.train_losses.append(avg_train_loss)

            # Validation phase
            val_loss = None
            if self.val_loader is not None:
                val_loss = self._validate()
                self.val_losses.append(val_loss)
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    self.best_state_dict = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in self.model.model.state_dict().items()
                    }

            # W&B logging (automatic if enabled)
            wandb_log(
                {
                    "train/loss": avg_train_loss,
                    "train/epoch": epoch + 1,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    **({"val/loss": val_loss} if val_loss is not None else {}),
                },
                step=epoch + 1,
            )

            # Print epoch summary
            if val_loss is not None:
                print(
                    f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}")

            # Save periodic checkpoints only when requested.
            checkpoint_interval = self.config.checkpoint_interval
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
                checkpoint_path = (
                    Path(self.config.checkpoint_dir) / checkpoint_name
                    if self.config.checkpoint_dir is not None
                    else Path(checkpoint_name)
                )
                self.save_checkpoint(checkpoint_path)

        if self.best_state_dict is not None:
            self.model.model.load_state_dict(self.best_state_dict)
            print(
                "Restored best validation weights: "
                f"epoch {self.best_epoch}, val loss = {self.best_val_loss:.4f}"
            )

        metrics = {"train_loss": self.train_losses}
        if self.val_losses:
            metrics["val_loss"] = self.val_losses
        if self.best_val_loss is not None:
            metrics["best_val_loss"] = [self.best_val_loss]
        if self.best_epoch is not None:
            metrics["best_epoch"] = [float(self.best_epoch)]
        return metrics

    def _validate(self) -> float:
        """Run validation and return average loss."""
        assert self.val_loader is not None, "Validation loader must be set"
        self.model.model.eval()
        val_losses = []

        with torch.no_grad():
            for data_batch, cond_batch in self.val_loader:
                data_batch = data_batch.to(self.config.device)
                cond_batch = cond_batch.to(self.config.device)

                loss = self.model.compute_loss(data_batch, cond_batch)
                val_losses.append(loss.item())

        self.model.model.train()
        return float(np.mean(val_losses))

    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Full path to save the checkpoint, or just a filename for default location
        """
        # If path has a parent directory specified or is absolute, use it directly
        if path.parent != Path(".") or path.is_absolute():
            checkpoint_path = path
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Use default checkpoints directory for simple filenames
            timestep_str = f"{self.config.trajectory_length}_timesteps"
            checkpoints_dir = Path("checkpoints") / timestep_str
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoints_dir / path.name

        torch.save(
            {
                "model_state_dict": self.model.model.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "best_val_loss": self.best_val_loss,
                "best_epoch": self.best_epoch,
                "config": self.config,
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss")
        self.best_epoch = checkpoint.get("best_epoch")
        print(f"Checkpoint loaded from {path}")
