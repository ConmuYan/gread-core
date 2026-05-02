"""Ablation runner: disable model components and measure impact.

Supported ablation configs:
    - no_residual_readout: set rho=0
    - no_evidence: zero out evidence token IDs
    - no_counter_evidence: zero out counter evidence tokens
    - no_type_head: zero out type logits

All ablations are deterministic given the same input. No LLM dependencies.
"""

from __future__ import annotations

from typing import Any

import torch

from gread_core.models.reasoner import GReaDReasoner


def _run_forward_with_ablation(
    model: GReaDReasoner,
    z_v: torch.Tensor,
    base_logit: torch.Tensor,
    evidence_token_ids: torch.Tensor,
    ablation: str,
) -> dict[str, torch.Tensor]:
    """Run forward pass with a specific ablation applied.

    Args:
        model: GReaDReasoner model.
        z_v: [B, H] node embeddings.
        base_logit: [B] base detector logits.
        evidence_token_ids: [B, K] evidence token IDs.
        ablation: Name of ablation to apply.

    Returns:
        Dict of model outputs.
    """
    original_rho = model.rho

    result: dict[str, torch.Tensor]

    if ablation == "no_residual_readout":
        model.rho = 0.0
        result = model(z_v, base_logit, evidence_token_ids)
        model.rho = original_rho
        return result

    if ablation == "no_evidence":
        zero_ids = torch.zeros_like(evidence_token_ids)
        result = model(z_v, base_logit, zero_ids)
        return result

    if ablation == "no_counter_evidence":
        # Zero out the last half of evidence slots (counter evidence)
        modified_ids = evidence_token_ids.clone()
        half = modified_ids.shape[1] // 2
        modified_ids[:, half:] = 0
        result = model(z_v, base_logit, modified_ids)
        return result

    if ablation == "no_type_head":
        result = model(z_v, base_logit, evidence_token_ids)
        result["type_logits"] = torch.zeros_like(result["type_logits"])
        return result

    msg = f"Unknown ablation: {ablation}"
    raise ValueError(msg)


def run_ablation(
    model: GReaDReasoner,
    z_v: torch.Tensor,
    base_logit: torch.Tensor,
    evidence_token_ids: torch.Tensor,
    ablation_config: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Run ablation experiments by disabling components.

    Args:
        model: GReaDReasoner model.
        z_v: [B, H] node embeddings.
        base_logit: [B] base detector logits.
        evidence_token_ids: [B, K] evidence token IDs.
        ablation_config: Dict mapping ablation name to enabled flag.
            Defaults to all ablations enabled.

    Returns:
        Dict mapping ablation name to output tensors.
    """
    if ablation_config is None:
        ablation_config = {
            "no_residual_readout": True,
            "no_evidence": True,
            "no_counter_evidence": True,
            "no_type_head": True,
        }

    # Always include baseline (no ablation)
    model.eval()
    results: dict[str, Any] = {}

    with torch.no_grad():
        baseline = model(z_v, base_logit, evidence_token_ids)
        results["baseline"] = {
            k: v.detach().clone() for k, v in baseline.items()
        }

        for ablation_name, enabled in ablation_config.items():
            if not enabled:
                continue
            ablated = _run_forward_with_ablation(
                model, z_v, base_logit, evidence_token_ids, ablation_name
            )
            results[ablation_name] = {
                k: v.detach().clone() for k, v in ablated.items()
            }

    return results
