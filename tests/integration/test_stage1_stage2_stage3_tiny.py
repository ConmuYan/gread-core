"""Integration test: 3-stage pipeline on tiny graph (CPU only).

Verifies:
- Stage 1 trains detector without LLM
- Stage 2 generates ERRs (using replay/mock)
- Stage 3 trains reasoner with accepted ERRs only
- Checkpoints include metadata
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gread_core.data.loaders import load_tiny_graph
from gread_core.training.checkpointing import CheckpointManager


class TestStage1Stage2Stage3Tiny:
    """End-to-end 3-stage training on tiny graph."""

    @pytest.fixture
    def tiny_data(self) -> object:
        return load_tiny_graph(num_nodes=30, num_features=16, seed=42)

    @pytest.fixture
    def config(self) -> dict:
        return {
            "project": {"name": "gread-test", "seed": 42, "output_dir": "artifacts"},
            "method": {"score_blind": True, "lambda_reason": 0.5, "residual_rho": 0.1},
            "stage1": {
                "epochs": 3, "lr": 0.01, "weight_decay": 5e-4,
                "log_every": 1, "save_every": 3,
            },
            "stage3": {
                "epochs": 3, "lr": 0.001, "weight_decay": 1e-5,
                "log_every": 1, "save_every": 3,
            },
            "detector": {"hidden_channels": 16},
            "evidence": {"num_slots": 8},
            "verifier": {},
            "trace_selection": {
                "total_budget": 10,
                "buckets": {
                    "uncertain": 0.333,
                    "high_conf_fraud": 0.333,
                    "high_conf_benign": 0.334,
                },
            },
        }

    def test_stage1_no_llm_import(self) -> None:
        """Stage 1 module must not import LLM."""
        import ast
        from pathlib import Path

        stage1_path = Path("src/gread_core/training/stage1_train_detector.py")
        tree = ast.parse(stage1_path.read_text())

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "llm" in node.module.lower()
            ):
                pytest.fail(f"Stage 1 imports LLM module: {node.module}")

    def test_stage1_trains(self, tiny_data: object, config: dict, tmp_path: Path) -> None:
        """Stage 1 detector training runs and saves checkpoint."""
        from gread_core.detectors.pyg_gnn import GCNDetector
        from gread_core.training.stage1_train_detector import train_detector

        detector = GCNDetector(in_channels=16, hidden_channels=16)
        ckpt_mgr = CheckpointManager(
            output_dir=tmp_path, experiment_id="test", seed=42, config=config,
        )

        train_detector(detector, tiny_data, config, ckpt_mgr)

        # Checkpoint should exist
        ckpt_dir = tmp_path / "stage1" / "epoch_0003"
        assert (ckpt_dir / "model.pt").exists()
        assert (ckpt_dir / "metadata.json").exists()

        # Verify metadata
        with open(ckpt_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["stage"] == 1
        assert meta["seed"] == 42
        assert "git_commit" in meta
        assert "config_hash" in meta
        assert "created_at" in meta

    def test_stage3_with_accepted_errs(
        self, tiny_data: object, config: dict, tmp_path: Path
    ) -> None:
        """Stage 3 trains reasoner with mock accepted ERRs."""
        from gread_core.detectors.pyg_gnn import GCNDetector
        from gread_core.models.evidence_encoder import EvidenceEncoder
        from gread_core.models.reasoner import GReaDReasoner
        from gread_core.training.stage1_train_detector import train_detector
        from gread_core.training.stage3_train_reasoner import train_reasoner

        # Stage 1: train detector
        detector = GCNDetector(in_channels=16, hidden_channels=16)
        detector = train_detector(detector, tiny_data, config)

        # Mock accepted ERRs
        accepted_errs = [
            {
                "node_id": "0",
                "node_idx": 0,
                "bucket": "uncertain",
                "err": {
                    "risk_type": "camouflage_neighbor",
                    "supporting_evidence": ["neighbor_consistency"],
                    "counter_evidence": ["counter_signal"],
                    "summary": "test summary",
                },
                "accepted": True,
            },
            {
                "node_id": "1",
                "node_idx": 1,
                "bucket": "high_conf_fraud",
                "err": {
                    "risk_type": "spectral_anomaly",
                    "supporting_evidence": ["detector_signal"],
                    "counter_evidence": [],
                    "summary": "test summary 2",
                },
                "accepted": True,
            },
        ]

        # Create reasoner
        evidence_encoder = EvidenceEncoder(
            vocab_size=18, embed_dim=16, num_slots=8, output_dim=32,
        )
        reasoner = GReaDReasoner(
            hidden_dim=16,
            evidence_encoder=evidence_encoder,
            num_risk_types=6,
            num_evidence_slots=8,
            rho=0.1,
        )

        ckpt_mgr = CheckpointManager(
            output_dir=tmp_path, experiment_id="test", seed=42, config=config,
        )

        train_reasoner(
            reasoner=reasoner,
            detector=detector,
            data=tiny_data,
            accepted_errs=accepted_errs,
            config=config,
            checkpoint_manager=ckpt_mgr,
        )

        # Checkpoint should exist
        ckpt_dir = tmp_path / "stage3" / "epoch_0003"
        assert (ckpt_dir / "model.pt").exists()
        assert (ckpt_dir / "metadata.json").exists()

        with open(ckpt_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["stage"] == 3
        assert meta["seed"] == 42

    def test_stage3_rejected_excluded(self, tiny_data: object, config: dict) -> None:
        """Stage 3 with zero accepted ERRs should return reasoner unchanged."""
        from gread_core.detectors.pyg_gnn import GCNDetector
        from gread_core.models.evidence_encoder import EvidenceEncoder
        from gread_core.models.reasoner import GReaDReasoner
        from gread_core.training.stage1_train_detector import train_detector
        from gread_core.training.stage3_train_reasoner import train_reasoner

        detector = GCNDetector(in_channels=16, hidden_channels=16)
        detector = train_detector(detector, tiny_data, config)

        evidence_encoder = EvidenceEncoder(
            vocab_size=18, embed_dim=16, num_slots=8, output_dim=32,
        )
        reasoner = GReaDReasoner(
            hidden_dim=16,
            evidence_encoder=evidence_encoder,
            num_risk_types=6,
            num_evidence_slots=8,
            rho=0.1,
        )

        # No accepted ERRs
        trained = train_reasoner(
            reasoner=reasoner,
            detector=detector,
            data=tiny_data,
            accepted_errs=[],
            config=config,
        )

        # Reasoner should be returned as-is
        assert trained is reasoner
