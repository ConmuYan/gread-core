from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
from torch import Tensor

from gread_core.adapters.diagnostics import build_native_evidence_distribution_report


@dataclass
class FakeGraph:
    edge_index: Tensor
    x: Tensor
    y: Tensor


def _make_graph() -> FakeGraph:
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ],
        dtype=torch.long,
    )
    return FakeGraph(edge_index=edge_index, x=x, y=y)


def _logits() -> Tensor:
    return torch.tensor(
        [
            [0.2, 1.0],
            [1.0, 0.1],
            [0.0, 0.0],
            [0.3, 0.7],
        ]
    )


def test_distribution_report_includes_raw_metric_quantiles() -> None:
    from gread_core.adapters.bwgnn_adapter import BWGNNAdapter

    graph = _make_graph()
    spectral = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.4, 0.4, 0.4],
            [0.8, 0.8, 0.8],
            [0.2, 0.2, 0.2],
        ]
    )
    adapter = BWGNNAdapter(
        detector=None,
        graph=graph,
        logits=_logits(),
        spectral_responses=spectral,
    )
    node_ids = [0, 1, 2]
    meps = adapter.extract(node_ids)

    report = build_native_evidence_distribution_report(
        adapter=adapter,
        meps=meps,
        node_ids=node_ids,
        source_metadata={"dataset": "tiny", "detector": "bwgnn"},
    )

    assert report["ok"] is True
    assert report["source_metadata"]["dataset"] == "tiny"
    assert report["source_metadata"]["supports_detector_signal"] is True
    assert report["raw_metric"]["available"] is True
    assert report["raw_metric"]["metric_name"] == "spectral_energy_ratio"
    assert report["raw_metric"]["count"] == 3
    assert report["raw_metric"]["quantiles"]["min"] == pytest.approx(0.1)
    assert report["raw_metric"]["quantiles"]["q50"] == pytest.approx(0.4)
    assert report["raw_metric"]["quantiles"]["max"] == pytest.approx(0.8)
    assert report["distributions"]["detector_signal"]["counts"] == {
        "bandpass_response_high": 1,
        "high_frequency_response_high": 1,
        "spectral_energy_shift_high": 1,
    }
    assert report["distributions"]["detector_signal_strength"]["counts"] == {
        "moderate": 1,
        "strong": 1,
        "weak": 1,
    }
    assert report["degeneracy_flags"]["raw_metric_unavailable"] is False
    assert report["score_blind_audit"]["calibration_channel_excluded"] is True
    assert report["score_blind_audit"]["reasoning_violations"] == []


def test_distribution_report_marks_unavailable_native_evidence() -> None:
    from gread_core.adapters.tree_adapter import TreeAdapter

    adapter = TreeAdapter(
        detector=None,
        graph=_make_graph(),
        logits=_logits(),
        feature_importance=None,
    )
    node_ids = [0, 1, 2]
    meps = adapter.extract(node_ids)

    report = build_native_evidence_distribution_report(
        adapter=adapter,
        meps=meps,
        node_ids=node_ids,
    )

    assert report["source_metadata"]["supports_detector_signal"] is False
    assert report["raw_metric"]["available"] is False
    assert report["raw_metric"]["quantiles"] == {}
    assert report["distributions"]["detector_signal"]["counts"] == {"unavailable": 3}
    assert report["degeneracy_flags"]["unsupported_detector_signal"] is True
    assert report["degeneracy_flags"]["all_detector_signal_unavailable"] is True
    assert report["degeneracy_flags"]["single_detector_signal"] is True
    assert report["degeneracy_flags"]["raw_metric_unavailable"] is True
    assert report["score_blind_audit"]["ok"] is True


def test_distribution_report_rejects_mismatched_nodes() -> None:
    class EmptyAdapter:
        detector_name = "empty"

        def supports_detector_signal(self) -> bool:
            return False

    with pytest.raises(ValueError, match="same length"):
        build_native_evidence_distribution_report(
            adapter=EmptyAdapter(),
            meps=[],
            node_ids=[0],
        )
