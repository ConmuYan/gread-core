"""Experiment registry with manifest generation for GReaD-Core."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExperimentRegistry:
    """Track experiment configuration and write manifests for reproducibility.

    Args:
        experiment_id: Unique identifier for the experiment.
        config: Configuration dictionary (hashed for identity).
        config_path: Path to the config file used.
        dataset: Name of the dataset.
        seed: Random seed used for the experiment.
        output_dir: Root directory for manifest output.
    """

    def __init__(
        self,
        experiment_id: str,
        config: dict[str, Any],
        config_path: str,
        dataset: str,
        seed: int,
        output_dir: str | Path,
    ) -> None:
        self.experiment_id = experiment_id
        self.config = config
        self.config_path = config_path
        self.dataset = dataset
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]
        self._git_commit = self._get_git_commit()
        self._manifest: dict[str, Any] | None = None

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

    def _get_software_versions(self) -> dict[str, str]:
        """Collect software version info."""
        versions: dict[str, str] = {"python": sys.version}
        try:
            import torch

            versions["torch"] = torch.__version__
        except ImportError:
            versions["torch"] = "not installed"
        try:
            import torch_geometric

            versions["torch_geometric"] = torch_geometric.__version__
        except ImportError:
            versions["torch_geometric"] = "not installed"
        return versions

    def write_manifest(self) -> Path:
        """Write experiment manifest to output_dir.

        Returns:
            Path to the written manifest file.
        """
        manifest = self.get_manifest()
        manifest_dir = self.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"experiment_{self.experiment_id}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

    def get_manifest(self) -> dict[str, Any]:
        """Return the experiment manifest dict."""
        if self._manifest is None:
            self._manifest = {
                "experiment_id": self.experiment_id,
                "git_commit": self._git_commit,
                "config_path": self.config_path,
                "config_hash": self.config_hash,
                "dataset": self.dataset,
                "seed": self.seed,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "software": self._get_software_versions(),
            }
        return self._manifest
