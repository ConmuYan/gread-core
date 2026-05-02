"""Tests for EvidenceAdapter protocol and adapter implementations.

Validates:
- Every adapter returns valid MEPs (Pydantic validation)
- Every adapter output includes all three evidence categories
- Adapter outputs do not leak prediction_score to ReasoningChannel
- Missing detector-native evidence produces detector_signal=unavailable
- EvidenceAdapter ABC conformance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import Tensor

from gread_core.adapters.base import EvidenceAdapter
from gread_core.schemas.evidence import MinimalEvidencePackage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeGraph:
    """Minimal PyG-like graph object."""
    edge_index: Tensor
    x: Tensor
    y: Tensor


def _make_tiny_graph(num_nodes: int = 20, num_features: int = 8) -> FakeGraph:
    torch.manual_seed(42)
    x = torch.randn(num_nodes, num_features)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    labels[:4] = 1

    srcs: list[int] = []
    dsts: list[int] = []
    for i in range(num_nodes):
        for j in range(i + 1, min(i + 3, num_nodes)):
            srcs.extend([i, j])
            dsts.extend([j, i])
    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

    return FakeGraph(edge_index=edge_index, x=x, y=labels)


def _make_logits(n: int = 20) -> Tensor:
    torch.manual_seed(123)
    return torch.randn(n, 2)


# ---------------------------------------------------------------------------
# ABC Tests
# ---------------------------------------------------------------------------

class TestEvidenceAdapterABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            EvidenceAdapter()  # type: ignore[abstract]

    def test_subclass_must_implement_extract(self) -> None:
        class Incomplete(EvidenceAdapter):
            detector_name = "inc"

            def supports_detector_signal(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_supports(self) -> None:
        class Incomplete(EvidenceAdapter):
            detector_name = "inc"

            def extract(self, node_ids: list[int]) -> list[MinimalEvidencePackage]:
                return []

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# BWGNNAdapter
# ---------------------------------------------------------------------------

class TestBWGNNAdapter:
    def _make_adapter(self, spectral_responses: Tensor | None = None) -> Any:
        from gread_core.adapters.bwgnn_adapter import BWGNNAdapter

        graph = _make_tiny_graph()
        return BWGNNAdapter(
            detector=None,
            graph=graph,
            logits=_make_logits(),
            spectral_responses=spectral_responses,
        )

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(self._make_adapter(), EvidenceAdapter)

    def test_detector_name(self) -> None:
        assert self._make_adapter().detector_name == "bwgnn"

    def test_supports_detector_signal(self) -> None:
        sr = torch.randn(20, 8)
        assert self._make_adapter(spectral_responses=sr).supports_detector_signal() is True

    def test_no_spectral_unavailable(self) -> None:
        adapter = self._make_adapter(spectral_responses=None)
        assert adapter.supports_detector_signal() is False

    def test_extract_returns_list(self) -> None:
        result = self._make_adapter().extract([0, 1, 2])
        assert isinstance(result, list)
        assert len(result) == 3

    def test_extract_returns_valid_meps(self) -> None:
        for mep in self._make_adapter().extract([0, 1]):
            assert isinstance(mep, MinimalEvidencePackage)

    def test_mep_has_all_evidence_categories(self) -> None:
        mep = self._make_adapter().extract([0])[0]
        assert mep.reasoning.degree_level != ""
        assert mep.reasoning.neighbor_consistency != ""
        assert mep.reasoning.feature_neighbor_discrepancy != ""
        assert mep.reasoning.uncertainty_level in ("low", "medium", "high")
        assert mep.reasoning.detector_signal != ""
        strengths = ("weak", "moderate", "strong", "unavailable")
        assert mep.reasoning.detector_signal_strength in strengths
        assert mep.reasoning.counter_signal != ""

    def test_no_score_leakage_to_reasoning(self) -> None:
        mep = self._make_adapter().extract([0])[0]
        assert 0.0 <= mep.calibration.prediction_score <= 1.0
        assert "prediction_score" not in str(mep.reasoning.model_dump())

    def test_to_teacher_payload_excludes_calibration(self) -> None:
        mep = self._make_adapter().extract([0])[0]
        assert "prediction_score" not in str(mep.to_teacher_payload())

    def test_empty_node_ids(self) -> None:
        assert self._make_adapter().extract([]) == []


# ---------------------------------------------------------------------------
# CAREGNNAdapter
# ---------------------------------------------------------------------------

class TestCAREGNNAdapter:
    def _make_adapter(self, filter_weights: dict[int, Tensor] | None = None) -> Any:
        from gread_core.adapters.caregnn_adapter import CAREGNNAdapter

        graph = _make_tiny_graph()
        return CAREGNNAdapter(
            detector=None,
            graph=graph,
            logits=_make_logits(),
            filter_weights=filter_weights,
        )

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(self._make_adapter(), EvidenceAdapter)

    def test_detector_name(self) -> None:
        assert self._make_adapter().detector_name == "caregnn"

    def test_extract_returns_valid_meps(self) -> None:
        for mep in self._make_adapter().extract([0, 1]):
            assert isinstance(mep, MinimalEvidencePackage)

    def test_mep_has_all_evidence_categories(self) -> None:
        mep = self._make_adapter().extract([0])[0]
        assert mep.reasoning.detector_signal != ""
        assert mep.reasoning.counter_signal != ""
        assert mep.reasoning.degree_level != ""

    def test_no_score_leakage(self) -> None:
        mep = self._make_adapter().extract([0])[0]
        assert "prediction_score" not in str(mep.to_teacher_payload())

    def test_no_filter_weights_unavailable(self) -> None:
        adapter = self._make_adapter(filter_weights=None)
        assert adapter.supports_detector_signal() is False
        mep = adapter.extract([0])[0]
        assert mep.reasoning.detector_signal == "unavailable"

    def test_with_filter_weights(self) -> None:
        fw = {i: torch.rand(3) for i in range(20)}
        adapter = self._make_adapter(filter_weights=fw)
        assert adapter.supports_detector_signal() is True


# ---------------------------------------------------------------------------
# PyGGNNAdapter
# ---------------------------------------------------------------------------

class TestPyGGNNAdapter:
    def _make_adapter(self, embeddings: Tensor | None = None) -> Any:
        from gread_core.adapters.pyg_gnn_adapter import PyGGNNAdapter

        graph = _make_tiny_graph()
        return PyGGNNAdapter(
            detector=None,
            graph=graph,
            logits=_make_logits(),
            embeddings=embeddings if embeddings is not None else torch.randn(20, 16),
        )

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(self._make_adapter(), EvidenceAdapter)

    def test_detector_name(self) -> None:
        assert self._make_adapter().detector_name == "pyg_gnn"

    def test_extract_returns_valid_meps(self) -> None:
        for mep in self._make_adapter().extract([0, 1]):
            assert isinstance(mep, MinimalEvidencePackage)

    def test_no_score_leakage(self) -> None:
        mep = self._make_adapter().extract([0])[0]
        assert "prediction_score" not in str(mep.to_teacher_payload())

    def test_no_embeddings_unavailable(self) -> None:
        adapter = self._make_adapter(embeddings=torch.empty(0))
        assert adapter.supports_detector_signal() is False
        mep = adapter.extract([0])[0]
        assert mep.reasoning.detector_signal == "unavailable"


# ---------------------------------------------------------------------------
# TreeAdapter
# ---------------------------------------------------------------------------

_UNSET = object()


class TestTreeAdapter:
    def _make_adapter(self, feature_importance: object = _UNSET) -> Any:
        from gread_core.adapters.tree_adapter import TreeAdapter

        graph = _make_tiny_graph()
        fi = torch.rand(8) if feature_importance is _UNSET else feature_importance
        return TreeAdapter(
            detector=None,
            graph=graph,
            logits=_make_logits(),
            feature_importance=fi,  # type: ignore[arg-type]
        )

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(self._make_adapter(), EvidenceAdapter)

    def test_detector_name(self) -> None:
        assert self._make_adapter().detector_name == "tree"

    def test_extract_returns_valid_meps(self) -> None:
        for mep in self._make_adapter().extract([0, 1]):
            assert isinstance(mep, MinimalEvidencePackage)

    def test_no_score_leakage(self) -> None:
        mep = self._make_adapter().extract([0])[0]
        assert "prediction_score" not in str(mep.to_teacher_payload())

    def test_no_feature_importance_unavailable(self) -> None:
        adapter = self._make_adapter(feature_importance=None)
        assert adapter.supports_detector_signal() is False
        mep = adapter.extract([0])[0]
        assert mep.reasoning.detector_signal == "unavailable"


# ---------------------------------------------------------------------------
# MEP Builder
# ---------------------------------------------------------------------------

class TestMEPBuilder:
    def test_build_valid_mep(self) -> None:
        from gread_core.evidence.mep_builder import build_mep

        mep = build_mep(
            node_id="n1",
            detector_name="test",
            prediction_score=0.8,
            uncertainty=0.3,
            uncertainty_level="low",
            degree_level="high",
            neighbor_consistency="medium",
            feature_neighbor_discrepancy="low",
            detector_signal="test_signal_high",
            detector_signal_strength="strong",
            counter_signal="benign_neighbor_signal_low",
            allowed_support_ids=["degree_level", "detector_signal"],
            allowed_counter_ids=["counter_signal"],
        )
        assert isinstance(mep, MinimalEvidencePackage)
        assert mep.node_id == "n1"

    def test_build_mep_default_ids(self) -> None:
        from gread_core.evidence.mep_builder import build_mep

        mep = build_mep(
            node_id=0,
            detector_name="test",
            prediction_score=0.5,
            uncertainty=0.5,
            uncertainty_level="medium",
            degree_level="medium",
            neighbor_consistency="medium",
            feature_neighbor_discrepancy="medium",
            detector_signal="test",
            detector_signal_strength="moderate",
            counter_signal="test",
        )
        assert isinstance(mep, MinimalEvidencePackage)

    def test_score_not_in_reasoning(self) -> None:
        from gread_core.evidence.mep_builder import build_mep

        mep = build_mep(
            node_id="n1",
            detector_name="test",
            prediction_score=0.95,
            uncertainty=0.1,
            uncertainty_level="low",
            degree_level="high",
            neighbor_consistency="high",
            feature_neighbor_discrepancy="low",
            detector_signal="test_signal",
            detector_signal_strength="strong",
            counter_signal="benign_neighbor_signal_low",
        )
        assert "prediction_score" not in str(mep.to_teacher_payload())

    def test_forbidden_support_id_raises(self) -> None:
        from gread_core.evidence.mep_builder import build_mep

        with pytest.raises(ValueError):
            build_mep(
                node_id="n1",
                detector_name="test",
                prediction_score=0.8,
                uncertainty=0.3,
                uncertainty_level="low",
                degree_level="high",
                neighbor_consistency="medium",
                feature_neighbor_discrepancy="low",
                detector_signal="test_signal",
                detector_signal_strength="strong",
                counter_signal="benign_neighbor_signal_low",
                allowed_support_ids=["prediction_score"],
                allowed_counter_ids=["counter_signal"],
            )


# ---------------------------------------------------------------------------
# Cross-adapter score-blindness
# ---------------------------------------------------------------------------

class TestScoreBlindnessAcrossAdapters:
    def _check(self, mep: MinimalEvidencePackage) -> None:
        assert "prediction_score" not in str(mep.reasoning.model_dump())

    def test_bwgnn(self) -> None:
        from gread_core.adapters.bwgnn_adapter import BWGNNAdapter

        graph = _make_tiny_graph()
        adapter = BWGNNAdapter(
            detector=None, graph=graph, logits=_make_logits(),
            spectral_responses=torch.randn(20, 8),
        )
        for mep in adapter.extract([0, 1, 2, 3, 4]):
            self._check(mep)

    def test_caregnn(self) -> None:
        from gread_core.adapters.caregnn_adapter import CAREGNNAdapter

        graph = _make_tiny_graph()
        adapter = CAREGNNAdapter(
            detector=None, graph=graph, logits=_make_logits(),
        )
        for mep in adapter.extract([0, 1, 2, 3, 4]):
            self._check(mep)

    def test_pyg_gnn(self) -> None:
        from gread_core.adapters.pyg_gnn_adapter import PyGGNNAdapter

        graph = _make_tiny_graph()
        adapter = PyGGNNAdapter(
            detector=None, graph=graph, logits=_make_logits(),
            embeddings=torch.randn(20, 16),
        )
        for mep in adapter.extract([0, 1, 2, 3, 4]):
            self._check(mep)

    def test_tree(self) -> None:
        from gread_core.adapters.tree_adapter import TreeAdapter

        graph = _make_tiny_graph()
        adapter = TreeAdapter(
            detector=None, graph=graph, logits=_make_logits(),
            feature_importance=torch.rand(8),
        )
        for mep in adapter.extract([0, 1, 2, 3, 4]):
            self._check(mep)
