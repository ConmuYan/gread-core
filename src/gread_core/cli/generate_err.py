"""Stage 2 CLI: Generate ERRs via LLM.

CRITICAL: This is the ONLY stage that calls LLM.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from gread_core.detectors.factory import ALL_DETECTOR_TYPES, create_detector
from gread_core.experiment.seed import set_seed


@dataclass(frozen=True)
class Stage2Runtime:
    """Resolved Stage 2 LLM runtime settings."""

    llm_backend: str
    cache_dir: str
    llm_model: str
    temperature: float
    base_url: str | None
    model_path: str | None
    device: str
    batch_size: int
    max_new_tokens: int
    show_progress: bool


def resolve_stage2_runtime(
    config: dict[str, Any],
    *,
    dataset: str,
    cli_llm_backend: str | None = None,
    cli_cache_dir: str | None = None,
    cli_llm_model: str | None = None,
    cli_temperature: float | None = None,
    cli_base_url: str | None = None,
    cli_model_path: str | None = None,
    cli_device: str | None = None,
    cli_batch_size: int | None = None,
    cli_max_new_tokens: int | None = None,
) -> Stage2Runtime:
    """Resolve Stage 2 runtime from CLI overrides and config.

    Non-tiny formal datasets must not use the stub backend.
    """
    stage2_cfg = config.get("stage2", {})
    llm_backend = cli_llm_backend or stage2_cfg.get("llm_backend", "replay")
    cache_dir = cli_cache_dir or stage2_cfg.get("cache_dir", ".cache/llm")
    llm_model = (
        cli_llm_model
        or stage2_cfg.get("llm_model")
        or stage2_cfg.get("model")
        or "gpt-4o-mini"
    )
    temperature_raw = (
        cli_temperature
        if cli_temperature is not None
        else stage2_cfg.get("temperature", 0.0)
    )
    temperature = float(temperature_raw)
    base_url = (
        cli_base_url
        or stage2_cfg.get("llm_base_url")
        or stage2_cfg.get("base_url")
    )
    model_path = cli_model_path or stage2_cfg.get("llm_model_path")
    device = str(cli_device or stage2_cfg.get("llm_device") or "cpu")
    batch_size_raw = cli_batch_size or stage2_cfg.get("llm_batch_size", 16)
    max_new_tokens_raw = (
        cli_max_new_tokens or stage2_cfg.get("llm_max_new_tokens", 384)
    )
    show_progress = bool(stage2_cfg.get("show_progress", True))

    if dataset != "tiny" and llm_backend == "stub":
        raise ValueError(
            "stub LLM backend is allowed only for tiny/smoke runs; "
            "non-tiny formal datasets must use replay or openai."
        )

    if llm_backend == "local" and not model_path:
        raise ValueError(
            "local LLM backend requires stage2.llm_model_path or --llm-model-path"
        )

    return Stage2Runtime(
        llm_backend=str(llm_backend),
        cache_dir=str(cache_dir),
        llm_model=str(llm_model),
        temperature=temperature,
        base_url=str(base_url) if base_url else None,
        model_path=str(model_path) if model_path else None,
        device=device,
        batch_size=max(1, int(batch_size_raw)),
        max_new_tokens=max(1, int(max_new_tokens_raw)),
        show_progress=show_progress,
    )


def main(argv: list[str] | None = None) -> None:
    """Stage 2: Generate ERRs CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GReaD-Core Stage 2: Generate ERRs"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config path"
    )
    parser.add_argument(
        "--dataset", type=str, default="tiny", help="Dataset name"
    )
    parser.add_argument(
        "--detector", type=str, default="gcn",
        choices=ALL_DETECTOR_TYPES,
        help="Detector type",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Stage 1 checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts", help="Output dir"
    )
    parser.add_argument(
        "--experiment-id", type=str, default="stage2", help="Experiment ID"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Real dataset root; overrides config data.root and GREAD_DATA_ROOT",
    )
    parser.add_argument(
        "--llm-backend", type=str, default=None,
        choices=["replay", "openai", "stub", "local"], help="LLM backend",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="Cache dir"
    )
    parser.add_argument(
        "--llm-model", type=str, default=None, help="Live LLM model name"
    )
    parser.add_argument(
        "--llm-temperature", type=float, default=None, help="Live LLM temperature"
    )
    parser.add_argument(
        "--llm-base-url", type=str, default=None, help="OpenAI-compatible base URL"
    )
    parser.add_argument(
        "--llm-model-path", type=str, default=None, help="Local model path"
    )
    parser.add_argument(
        "--llm-device", type=str, default=None, help="Device for local LLM backend"
    )
    parser.add_argument(
        "--llm-batch-size", type=int, default=None, help="Batch size for LLM generation"
    )
    parser.add_argument(
        "--llm-max-new-tokens", type=int, default=None, help="Max new tokens per LLM response"
    )
    parser.add_argument(
        "--trace-budget", type=int, default=None, help="Override trace selection total budget"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.trace_budget is not None:
        config.setdefault("trace_selection", {})["total_budget"] = args.trace_budget

    set_seed(args.seed)
    runtime = resolve_stage2_runtime(
        config,
        dataset=args.dataset,
        cli_llm_backend=args.llm_backend,
        cli_cache_dir=args.cache_dir,
        cli_llm_model=args.llm_model,
        cli_temperature=args.llm_temperature,
        cli_base_url=args.llm_base_url,
        cli_model_path=args.llm_model_path,
        cli_device=args.llm_device or args.device,
        cli_batch_size=args.llm_batch_size,
        cli_max_new_tokens=args.llm_max_new_tokens,
    )
    stage2_cfg = config.setdefault("stage2", {})
    stage2_cfg["llm_backend"] = runtime.llm_backend
    stage2_cfg["llm_device"] = runtime.device
    stage2_cfg["llm_batch_size"] = runtime.batch_size
    stage2_cfg["llm_max_new_tokens"] = runtime.max_new_tokens
    stage2_cfg.setdefault(
        "release_detector_before_llm",
        runtime.llm_backend == "local",
    )

    # Load dataset
    from gread_core.data.loaders import load_graph_dataset
    data = load_graph_dataset(
        args.dataset, root=args.data_root, seed=args.seed, config=config
    )
    device = torch.device(args.device)
    data = data.to(device)

    # Create and load detector
    in_channels = data.x.shape[1]
    hidden_channels = config.get("detector", {}).get("hidden_channels", 64)

    detector = create_detector(
        args.detector,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        config=config,
    )

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    model_path = checkpoint_path / "model.pt"
    if model_path.exists():
        detector.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info("Loaded detector checkpoint from %s", model_path)
    detector = detector.to(device)

    # Create adapter (need logits + embeddings for ALL nodes)
    from gread_core.adapters.factory import create_evidence_adapter
    from gread_core.training.stage2_generate_err import _strip_masks
    no_mask_data = _strip_masks(data)
    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(no_mask_data)  # type: ignore[operator]
    adapter_thresholds = config.get("adapter", {}).get("thresholds", {})
    adapter = create_evidence_adapter(
        detector_type=args.detector,
        detector=detector,
        graph=data,
        logits=logits,
        embeddings=embeddings,
        thresholds=adapter_thresholds,
        strict_detector_signal=args.dataset != "tiny",
    )

    # Create LLM client
    from gread_core.llm.clients import LLMClient
    if runtime.llm_backend == "stub":
        from gread_core.llm.clients import StubClient
        client: LLMClient = StubClient()
    elif runtime.llm_backend == "replay":
        from gread_core.llm.clients import ReplayClient
        client = ReplayClient(runtime.cache_dir)
    elif runtime.llm_backend == "local":
        from gread_core.llm.clients import LocalTransformersClient
        client = LocalTransformersClient(
            model_path=runtime.model_path or "",
            model=runtime.llm_model,
            temperature=runtime.temperature,
            device=runtime.device,
            batch_size=runtime.batch_size,
            max_new_tokens=runtime.max_new_tokens,
        )
    else:
        from gread_core.llm.clients import OpenAIClient
        client = OpenAIClient(
            model=runtime.llm_model,
            temperature=runtime.temperature,
            base_url=runtime.base_url,
        )

    # Create teacher and verifier
    from gread_core.llm.teacher import LLMTeacher
    from gread_core.verification.verifier import EvidenceContractVerifier

    # Load contract YAML — fail closed if missing
    verifier_cfg = config.get("verifier", {})
    contract_path = verifier_cfg.get("contract_path")
    if contract_path:
        contract_file = Path(contract_path)
        if not contract_file.exists():
            raise FileNotFoundError(
                f"Contract YAML not found: {contract_path}"
            )
        with open(contract_file) as cf:
            contract_yaml = yaml.safe_load(cf)
        # Deep merge: YAML contract + config overrides
        contract_config: dict = {}
        contract_config.update(contract_yaml)
        for key, val in verifier_cfg.items():
            is_nested = (key in contract_config
                         and isinstance(contract_config[key], dict)
                         and isinstance(val, dict))
            if is_nested:
                contract_config[key] = {**contract_config[key], **val}
            elif key == "label_compatibility" and isinstance(val, bool):
                if isinstance(contract_config.get(key), dict):
                    contract_config[key]["enabled"] = val
                else:
                    contract_config[key] = {"enabled": val}
            else:
                contract_config[key] = val
    else:
        raise ValueError(
            "verifier.contract_path is required in config. "
            "Set configs/contracts/gread_v1.yaml or an ablation contract."
        )
    # Inject contract_version into config for cache metadata downstream.
    cv = contract_yaml.get("contract_version")
    if cv:
        config.setdefault("verifier", {})["contract_version"] = cv

    # Create experiment registry (AFTER contract loading so contract_version is resolved)
    from gread_core.experiment.registry import ExperimentRegistry
    _cv = config.get("verifier", {}).get("contract_version")
    registry = ExperimentRegistry(
        experiment_id=args.experiment_id,
        config=config,
        config_path=args.config,
        dataset=args.dataset,
        seed=args.seed,
        output_dir=args.output_dir,
        contract_version=_cv,
        detector_checkpoint=args.checkpoint,
    )

    score_blind = config.get("method", {}).get("score_blind", True)
    verifier = EvidenceContractVerifier(contract_config, score_blind=score_blind)
    teacher = LLMTeacher(
        client,
        verifier,
        runtime.cache_dir,
        score_blind=score_blind,
        model_name=runtime.llm_model,
        batch_size=runtime.batch_size,
        show_progress=runtime.show_progress,
    )

    # Generate ERRs
    from gread_core.training.stage2_generate_err import generate_errs

    result = generate_errs(
        detector=detector,
        data=data,
        adapter=adapter,
        teacher=teacher,
        verifier=verifier,
        config=config,
        seed=args.seed,
    )

    # Save
    out_dir = Path(args.output_dir) / "stage2"
    result.save(out_dir)
    manifest_path = registry.write_manifest()
    logger.info(
        "Stage 2 complete: %d accepted, %d rejected saved to %s. Manifest: %s",
        result.num_accepted, result.num_rejected, out_dir, manifest_path,
    )


if __name__ == "__main__":
    main()
