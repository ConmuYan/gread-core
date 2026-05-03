"""Regression test for GReaDInferencePipeline.predict().

Verifies that predict() does not crash on a tiny graph and that returned
evidence slot names are all from the canonical EVIDENCE_SLOTS_ORDERED list.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gread_core.adapters.base import EvidenceAdapter
from gread_core.inference.predictor import GReaDInferencePipeline
from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.reasoner import GReaDReasoner
from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)
from gread_core.schemas.risk_taxonomy import EVIDENCE_SLOTS_ORDERED

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_NODES = 8
_FEAT_DIM = 16
_HIDDEN_DIM = 32
_NUM_SLOTS = 16
_NUM_RISK_TYPES = 6


class _MockDetector(nn.Module):
    """Minimal detector exposing ``forward_with_embedding``."""

    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feat_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward_with_embedding(
        self, graph: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = graph.x  # type: ignore[union-attr]
        embedding = self.linear(x)  # [N, H]
        logits = self.head(embedding).squeeze(-1)  # [N]
        return logits, embedding


class _MockAdapter(EvidenceAdapter):
    """Adapter that returns a valid MEP for every requested node."""

    detector_name = "mock"

    def __init__(self) -> None:
        pass

    def extract(self, node_ids: list[int]) -> list[MinimalEvidencePackage]:
        meps: list[MinimalEvidencePackage] = []
        for nid in node_ids:
            meps.append(
                MinimalEvidencePackage(
                    node_id=str(nid),
                    detector_name=self.detector_name,
                    calibration=CalibrationChannel(
                        prediction_score=0.42,
                        uncertainty=0.1,
                    ),
                    reasoning=ReasoningChannel(
                        uncertainty_level="low",
                        degree_level="moderate",
                        neighbor_consistency="consistent",
                        feature_neighbor_discrepancy="none",
                        detector_signal="positive",
                        detector_signal_strength="moderate",
                        counter_signal="none",
                        allowed_support_ids=[
                            "uncertainty_level",
                            "degree_level",
                            "neighbor_consistency",
                            "feature_neighbor_discrepancy",
                            "detector_signal",
                            "detector_signal_strength",
                        ],
                        allowed_counter_ids=["counter_signal"],
                    ),
                )
            )
        return meps

    def supports_detector_signal(self) -> bool:
        return True


def _make_reasoner() -> GReaDReasoner:
    encoder = EvidenceEncoder(
        vocab_size=_NUM_SLOTS + 1,
        embed_dim=8,
        num_slots=_NUM_SLOTS,
        output_dim=32,
    )
    return GReaDReasoner(
        hidden_dim=_HIDDEN_DIM,
        evidence_encoder=encoder,
        num_risk_types=_NUM_RISK_TYPES,
        num_evidence_slots=_NUM_SLOTS,
        rho=0.1,
    )


def _make_graph() -> object:
    """Create a tiny PyG-like Data graph (8 nodes, 16 features, random edges)."""

    class _Data:
        pass

    data = _Data()
    data.x = torch.randn(_NUM_NODES, _FEAT_DIM)
    data.edge_index = torch.randint(0, _NUM_NODES, (2, 20))
    return data


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestInferenceRegression:
    """GReaDInferencePipeline regression: predict() smoke + evidence slot names."""

    def test_predict_returns_correct_count(self) -> None:
        pipeline = GReaDInferencePipeline(
            detector=_MockDetector(_FEAT_DIM, _HIDDEN_DIM),
            reasoner=_make_reasoner(),
            adapter=_MockAdapter(),
            config={
                "evidence": {
                    "num_slots": _NUM_SLOTS,
                    "evidence_slot_names": EVIDENCE_SLOTS_ORDERED,
                },
            },
        )
        graph = _make_graph()
        node_ids = list(range(_NUM_NODES))

        results = pipeline.predict(graph, node_ids)

        assert len(results) == _NUM_NODES

    def test_predict_evidence_slots_are_canonical(self) -> None:
        pipeline = GReaDInferencePipeline(
            detector=_MockDetector(_FEAT_DIM, _HIDDEN_DIM),
            reasoner=_make_reasoner(),
            adapter=_MockAdapter(),
            config={
                "evidence": {
                    "num_slots": _NUM_SLOTS,
                    "evidence_slot_names": EVIDENCE_SLOTS_ORDERED,
                },
            },
        )
        graph = _make_graph()
        node_ids = list(range(_NUM_NODES))

        results = pipeline.predict(graph, node_ids)

        canonical = set(EVIDENCE_SLOTS_ORDERED)
        for r in results:
            for name in r.supporting_evidence:
                assert name in canonical, f"Unexpected supporting slot: {name}"
            for name in r.counter_evidence:
                assert name in canonical, f"Unexpected counter slot: {name}"

    def test_predict_fraud_scores_in_range(self) -> None:
        pipeline = GReaDInferencePipeline(
            detector=_MockDetector(_FEAT_DIM, _HIDDEN_DIM),
            reasoner=_make_reasoner(),
            adapter=_MockAdapter(),
            config={
                "evidence": {
                    "num_slots": _NUM_SLOTS,
                    "evidence_slot_names": EVIDENCE_SLOTS_ORDERED,
                },
            },
        )
        graph = _make_graph()
        node_ids = list(range(_NUM_NODES))

        results = pipeline.predict(graph, node_ids)

        for r in results:
            assert 0.0 <= r.fraud_score <= 1.0, (
                f"fraud_score {r.fraud_score} outside [0, 1]"
            )

    def test_predict_risk_type_is_valid(self) -> None:
        pipeline = GReaDInferencePipeline(
            detector=_MockDetector(_FEAT_DIM, _HIDDEN_DIM),
            reasoner=_make_reasoner(),
            adapter=_MockAdapter(),
            config={
                "evidence": {
                    "num_slots": _NUM_SLOTS,
                    "evidence_slot_names": EVIDENCE_SLOTS_ORDERED,
                },
            },
        )
        graph = _make_graph()
        node_ids = list(range(_NUM_NODES))

        results = pipeline.predict(graph, node_ids)

        from gread_core.schemas.risk_taxonomy import RISK_TYPES

        for r in results:
            assert r.risk_type in RISK_TYPES, (
                f"Unexpected risk_type: {r.risk_type}"
            )

    def test_predict_explanation_is_nonempty(self) -> None:
        pipeline = GReaDInferencePipeline(
            detector=_MockDetector(_FEAT_DIM, _HIDDEN_DIM),
            reasoner=_make_reasoner(),
            adapter=_MockAdapter(),
            config={
                "evidence": {
                    "num_slots": _NUM_SLOTS,
                    "evidence_slot_names": EVIDENCE_SLOTS_ORDERED,
                },
            },
        )
        graph = _make_graph()
        node_ids = list(range(_NUM_NODES))

        results = pipeline.predict(graph, node_ids)

        for r in results:
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 0
