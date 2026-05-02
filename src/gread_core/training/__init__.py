"""Training pipeline for GReaD-Core: 3-stage training with checkpointing."""

from gread_core.training.checkpointing import CheckpointManager
from gread_core.training.stage1_train_detector import train_detector
from gread_core.training.stage3_train_reasoner import train_reasoner

__all__ = ["CheckpointManager", "train_detector", "train_reasoner"]
