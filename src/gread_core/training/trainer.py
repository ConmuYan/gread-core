"""Unified trainer: orchestrates the 3-stage GReaD-Core training pipeline.

Stage 1: Base detector warm-up (no LLM)
Stage 2: Offline ERR generation + verification (LLM only here)
Stage 3: Reasoner distillation (accepted ERRs only)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

from gread_core.training.checkpointing import CheckpointManager
from gread_core.training.stage1_train_detector import train_detector
from gread_core.training.stage3_train_reasoner import train_reasoner

logger = logging.getLogger(__name__)


class Trainer:
    """Unified 3-stage trainer for GReaD-Core.

    Args:
        config: Full configuration dict (YAML-driven).
        output_dir: Root directory for checkpoints and artifacts.
        experiment_id: Unique experiment identifier.
        seed: Random seed.
        device: Torch device.
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: str | Path = "artifacts",
        experiment_id: str = "default",
        seed: int = 1,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.checkpoint_manager = CheckpointManager(
            output_dir=output_dir,
            experiment_id=experiment_id,
            seed=seed,
            config=config,
        )

    def run_stage1(
        self,
        detector: Any,
        data: Data,
    ) -> Any:
        """Stage 1: Train base detector.

        CRITICAL: Does NOT import or use LLM.
        """
        logger.info("=== Stage 1: Base Detector Warm-up ===")
        return train_detector(
            detector=detector,
            data=data,
            config=self.config,
            checkpoint_manager=self.checkpoint_manager,
        )

    def run_stage2(
        self,
        detector: Any,
        data: Data,
        adapter: Any,
        teacher: Any,
        verifier: Any,
    ) -> Any:
        """Stage 2: Generate ERRs via LLM.

        CRITICAL: This is the ONLY stage that calls LLM.
        """
        from gread_core.training.stage2_generate_err import generate_errs

        logger.info("=== Stage 2: Offline ERR Generation ===")
        result = generate_errs(
            detector=detector,
            data=data,
            adapter=adapter,
            teacher=teacher,
            verifier=verifier,
            config=self.config,
            seed=self.seed,
        )

        # Save ERRs
        err_dir = self.checkpoint_manager.output_dir / "stage2"
        result.save(err_dir)

        return result

    def run_stage3(
        self,
        reasoner: Any,
        detector: Any,
        data: Data,
        accepted_errs: list[dict[str, Any]],
    ) -> Any:
        """Stage 3: Train reasoner using accepted ERRs only.

        CRITICAL: Rejected ERRs are excluded from reasoning loss.
        """
        logger.info("=== Stage 3: Reasoner Distillation ===")
        return train_reasoner(
            reasoner=reasoner,
            detector=detector,
            data=data,
            accepted_errs=accepted_errs,
            config=self.config,
            checkpoint_manager=self.checkpoint_manager,
            device=self.device,
        )
