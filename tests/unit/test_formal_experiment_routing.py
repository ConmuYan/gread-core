"""Regression tests for formal experiment data/API routing."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def test_resolve_data_root_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gread_core.data.loaders import resolve_data_root

    cli_root = tmp_path / "cli"
    config_root = tmp_path / "config"
    env_root = tmp_path / "env"
    monkeypatch.setenv("GREAD_DATA_ROOT", str(env_root))

    assert resolve_data_root(
        cli_data_root=str(cli_root),
        config={"data": {"root": str(config_root)}},
    ) == cli_root
    assert resolve_data_root(config={"data": {"root": str(config_root)}}) == config_root
    assert resolve_data_root() == env_root

    monkeypatch.delenv("GREAD_DATA_ROOT")
    assert resolve_data_root() == Path("data/raw")


def test_load_graph_dataset_passes_resolved_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import gread_core.data.loaders as loaders

    captured: dict[str, object] = {}
    expected = object()

    def fake_loader(*, data_root: str | Path, seed: int) -> object:
        captured["data_root"] = Path(data_root)
        captured["seed"] = seed
        return expected

    monkeypatch.setattr(loaders, "load_real_yelpchi", fake_loader)

    result = loaders.load_graph_dataset("yelpchi", root=str(tmp_path), seed=7)

    assert result is expected
    assert captured == {"data_root": tmp_path, "seed": 7}


def test_stage2_runtime_uses_config_and_rejects_stub_for_real_data() -> None:
    from gread_core.cli.generate_err import resolve_stage2_runtime

    config = {
        "stage2": {
            "llm_backend": "stub",
            "cache_dir": ".cache/from-config",
            "llm_model": "example-model",
            "temperature": 0.2,
        }
    }

    tiny_runtime = resolve_stage2_runtime(config, dataset="tiny")
    assert tiny_runtime.llm_backend == "stub"
    assert tiny_runtime.cache_dir == ".cache/from-config"
    assert tiny_runtime.llm_model == "example-model"
    assert tiny_runtime.temperature == 0.2

    with pytest.raises(ValueError, match=r"stub.*non-tiny"):
        resolve_stage2_runtime(config, dataset="yelpchi")


def test_stage2_cli_override_wins_over_config_for_backend() -> None:
    from gread_core.cli.generate_err import resolve_stage2_runtime

    runtime = resolve_stage2_runtime(
        {"stage2": {"llm_backend": "stub", "cache_dir": ".cache/from-config"}},
        dataset="yelpchi",
        cli_llm_backend="replay",
        cli_cache_dir=".cache/from-cli",
    )

    assert runtime.llm_backend == "replay"
    assert runtime.cache_dir == ".cache/from-cli"


def test_stage2_runtime_local_backend_requires_model_path_and_reads_overrides() -> None:
    from gread_core.cli.generate_err import resolve_stage2_runtime

    with pytest.raises(ValueError, match="local LLM backend requires"):
        resolve_stage2_runtime(
            {"stage2": {"llm_backend": "local"}},
            dataset="yelpchi",
        )

    runtime = resolve_stage2_runtime(
        {
            "stage2": {
                "llm_backend": "local",
                "llm_model": "Qwen3-4B-Instruct-2507",
                "llm_model_path": "/data1/mq/models/Qwen3-4B-Instruct-2507",
                "llm_device": "cuda:2",
                "llm_batch_size": 4,
                "llm_max_new_tokens": 256,
            }
        },
        dataset="yelpchi",
    )

    assert runtime.llm_backend == "local"
    assert runtime.llm_model == "Qwen3-4B-Instruct-2507"
    assert runtime.model_path == "/data1/mq/models/Qwen3-4B-Instruct-2507"
    assert runtime.device == "cuda:2"
    assert runtime.batch_size == 4
    assert runtime.max_new_tokens == 256


def test_evaluate_requires_real_inputs_unless_synthetic_is_explicit() -> None:
    from gread_core.cli.evaluate import resolve_evaluation_mode

    missing = Namespace(
        dataset=None,
        detector=None,
        detector_checkpoint=None,
        err_dir=None,
        synthetic=False,
    )
    with pytest.raises(ValueError, match="requires --dataset"):
        resolve_evaluation_mode(missing)

    synthetic = Namespace(
        dataset=None,
        detector=None,
        detector_checkpoint=None,
        err_dir=None,
        synthetic=True,
    )
    assert resolve_evaluation_mode(synthetic) == "synthetic"

    real_missing_errs = Namespace(
        dataset="yelpchi",
        detector="gcn",
        detector_checkpoint="artifacts/exp/stage1/epoch_0010",
        err_dir=None,
        synthetic=False,
    )
    with pytest.raises(ValueError, match="--err-dir"):
        resolve_evaluation_mode(real_missing_errs)

    real = Namespace(
        dataset="yelpchi",
        detector="gcn",
        detector_checkpoint="artifacts/exp/stage1/epoch_0010",
        err_dir="artifacts/exp/stage2",
        synthetic=False,
    )
    assert resolve_evaluation_mode(real) == "real"


def _tiny_graph() -> Any:
    from torch_geometric.data import Data

    return Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long),
        y=torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )


def test_adapter_factory_dispatches_detector_specific_adapters() -> None:
    from gread_core.adapters.bwgnn_adapter import BWGNNAdapter
    from gread_core.adapters.caregnn_adapter import CAREGNNAdapter
    from gread_core.adapters.factory import create_evidence_adapter
    from gread_core.adapters.pyg_gnn_adapter import PyGGNNAdapter
    from gread_core.adapters.tree_adapter import TreeAdapter

    graph = _tiny_graph()
    logits = torch.randn(4)
    embeddings = torch.randn(4, 8)

    class NativeDetector:
        def __init__(self) -> None:
            self.filter_weights = {i: torch.rand(3) for i in range(4)}
            self.feature_importance = torch.rand(4, 8)
            self.layer_scores = [torch.rand(4), torch.rand(4)]
            self.gammas = torch.nn.Parameter(torch.tensor([0.2, 0.3, -0.1, 0.4]))
            self.layer_deltas = [torch.rand(4), torch.rand(4)]

    cases = {
        "gcn": PyGGNNAdapter,
        "gat": PyGGNNAdapter,
        "sage": PyGGNNAdapter,
        "bwgnn": BWGNNAdapter,
        "caregnn": CAREGNNAdapter,
        "tree_neighbor": TreeAdapter,
        "pc_gnn": PyGGNNAdapter,
        "gpr_gnn": PyGGNNAdapter,
        "h2gcn": PyGGNNAdapter,
        "gin": PyGGNNAdapter,
    }

    for detector_type, expected_cls in cases.items():
        adapter = create_evidence_adapter(
            detector_type=detector_type,
            detector=NativeDetector(),
            graph=graph,
            logits=logits,
            embeddings=embeddings,
            strict_detector_signal=True,
        )
        assert isinstance(adapter, expected_cls)
        mep = adapter.extract([0])[0]
        assert mep.detector_name == detector_type
        assert mep.reasoning.detector_signal != "unavailable"
        assert "prediction_score" not in str(mep.to_teacher_payload())


def test_adapter_factory_fails_closed_when_detector_signal_unavailable() -> None:
    from gread_core.adapters.factory import create_evidence_adapter

    graph = _tiny_graph()

    with pytest.raises(RuntimeError, match="detector-native evidence"):
        create_evidence_adapter(
            detector_type="caregnn",
            detector=object(),
            graph=graph,
            logits=torch.randn(4),
            embeddings=torch.randn(4, 8),
            strict_detector_signal=True,
        )


def test_detector_specific_adapters_build_sparse_adjacency() -> None:
    from gread_core.adapters.bwgnn_adapter import _build_adj as build_bwgnn_adj
    from gread_core.adapters.caregnn_adapter import _build_adj as build_caregnn_adj
    from gread_core.adapters.tree_adapter import _build_adj as build_tree_adj

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    assert build_bwgnn_adj(edge_index, 3).is_sparse
    assert build_caregnn_adj(edge_index, 3).is_sparse
    assert build_tree_adj(edge_index, 3).is_sparse


def test_formal_scripts_use_configurable_llm_backend_and_real_evaluation_args() -> None:
    for script_name in (
        "run_full_experiments.sh",
        "run_validation.sh",
        "run_main_table.sh",
        "run_ablations.sh",
    ):
        text = (Path("scripts") / script_name).read_text()
        assert '--llm-backend "$LLM_BACKEND"' in text
        assert '--dataset "$DATASET"' in text
        assert '--detector "$DETECTOR"' in text
        assert '--detector-checkpoint "$STAGE1_CKPT"' in text
        assert '--err-dir "$OUTPUT_DIR/stage2"' in text


def test_export_results_excludes_synthetic_and_unknown_results(tmp_path: Path) -> None:
    from scripts.export_results import build_main_table

    real_path = tmp_path / "artifacts" / "full_gcn_yelpchi" / "metrics" / "evaluation_results.json"
    synthetic_path = tmp_path / "artifacts" / "smoke" / "metrics" / "evaluation_results.json"
    unknown_path = (
        tmp_path / "artifacts" / "full_gcn_unknown" / "metrics" / "evaluation_results.json"
    )

    real_data = {
        "evaluation_mode": "real",
        "dataset": "yelpchi",
        "detector": "gcn",
        "detection": {"auc": 0.7, "auprc": 0.2, "f1": 0.3},
    }
    synthetic_data = {
        "evaluation_mode": "synthetic",
        "dataset": "synthetic",
        "detector": "gcn",
        "detection": {"auc": 0.9, "auprc": 0.9, "f1": 0.9},
    }
    unknown_data = {
        "evaluation_mode": "real",
        "dataset": "unknown",
        "detector": "gcn",
        "detection": {"auc": 0.1, "auprc": 0.1, "f1": 0.1},
    }

    rows = build_main_table(
        [
            (real_path, real_data),
            (synthetic_path, synthetic_data),
            (unknown_path, unknown_data),
        ]
    )

    assert rows == [
        {
            "experiment": "full_gcn_yelpchi",
            "dataset": "yelpchi",
            "detector": "gcn",
            "AUC": "0.7000",
            "AUPRC": "0.2000",
            "F1": "0.3000",
        }
    ]
