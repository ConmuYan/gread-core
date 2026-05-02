"""Checkpoint save/load with metadata for GReaD-Core training pipeline.

Checkpoint metadata schema:
{
    "experiment_id": "...",
    "git_commit": "...",
    "config_hash": "...",
    "seed": 1,
    "stage": 1,
    "created_at": "..."
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints with metadata.

    Args:
        output_dir: Root directory for checkpoints.
        experiment_id: Unique identifier for the experiment.
        seed: Random seed used for training.
        config: Configuration dict (hashed for reproducibility tracking).
    """

    def __init__(
        self,
        output_dir: str | Path,
        experiment_id: str,
        seed: int,
        config: dict[str, Any],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.seed = seed
        self.config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]
        self._git_commit = self._get_git_commit()

    @staticmethod
    def _get_git_commit() -> str:
        """Get current git commit hash, or 'unknown' if not in a git repo."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _build_metadata(self, stage: int) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "git_commit": self._git_commit,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "stage": stage,
            "created_at": datetime.now(tz=UTC).isoformat(),
        }

    def save(
        self,
        model: nn.Module,
        stage: int,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save model checkpoint with metadata.

        Args:
            model: The model to save.
            stage: Training stage (1, 2, or 3).
            epoch: Current epoch number.
            optimizer: Optional optimizer state to save.
            extra: Optional extra data to include in checkpoint.

        Returns:
            Path to the saved checkpoint directory.
        """
        ckpt_dir = self.output_dir / f"stage{stage}" / f"epoch_{epoch:04d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

        # Save optimizer if provided
        if optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        # Save metadata
        metadata = self._build_metadata(stage)
        metadata["epoch"] = epoch
        if extra:
            metadata["extra"] = extra

        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved checkpoint: %s", ckpt_dir)
        return ckpt_dir

    def load(
        self,
        model: nn.Module,
        stage: int,
        epoch: int,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, Any]:
        """Load model checkpoint with metadata.

        Args:
            model: The model to load weights into.
            stage: Training stage (1, 2, or 3).
            epoch: Epoch number of the checkpoint to load.
            optimizer: Optional optimizer to load state into.

        Returns:
            Metadata dict from the checkpoint.
        """
        ckpt_dir = self.output_dir / f"stage{stage}" / f"epoch_{epoch:04d}"

        model_path = ckpt_dir / "model.pt"
        if not model_path.exists():
            msg = f"Checkpoint not found: {model_path}"
            raise FileNotFoundError(msg)

        model.load_state_dict(torch.load(model_path, weights_only=True))

        if optimizer is not None:
            opt_path = ckpt_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, weights_only=True))

        with open(ckpt_dir / "metadata.json") as f:
            metadata: dict[str, Any] = json.load(f)

        logger.info("Loaded checkpoint: %s", ckpt_dir)
        return metadata

    def list_checkpoints(self, stage: int) -> list[Path]:
        """List all checkpoint directories for a given stage."""
        stage_dir = self.output_dir / f"stage{stage}"
        if not stage_dir.exists():
            return []
        return sorted(d for d in stage_dir.iterdir() if d.is_dir())

    def get_latest_epoch(self, stage: int) -> int | None:
        """Get the latest epoch number for a given stage, or None if no checkpoints."""
        checkpoints = self.list_checkpoints(stage)
        if not checkpoints:
            return None
        # Parse epoch from directory name "epoch_NNNN"
        epochs = []
        for cp in checkpoints:
            try:
                epochs.append(int(cp.name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(epochs) if epochs else None
