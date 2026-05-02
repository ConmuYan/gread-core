"""Experiment logging setup for GReaD-Core."""

from __future__ import annotations

import logging
from pathlib import Path


class ExperimentLogger:
    """Configure file-based logging for an experiment.

    Args:
        experiment_id: Unique identifier for the experiment.
        log_dir: Directory where log files are written.
    """

    def __init__(self, experiment_id: str, log_dir: str | Path) -> None:
        self.experiment_id = experiment_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = logging.getLogger(f"gread_core.experiment.{experiment_id}")
        self._logger.setLevel(logging.DEBUG)

        if not self._logger.handlers:
            fh = logging.FileHandler(self.log_dir / f"{experiment_id}.log")
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

    def get_logger(self, name: str) -> logging.Logger:
        """Return a child logger under the experiment logger.

        Args:
            name: Name suffix for the child logger.

        Returns:
            A child Logger instance.
        """
        return self._logger.getChild(name)
