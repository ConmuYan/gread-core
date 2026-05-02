"""Stage 2: Offline ERR generation + verification.

CRITICAL CONSTRAINT: Stage 2 is the ONLY stage that calls LLM.
This stage:
1. Runs base detector to get embeddings and scores.
2. Selects trace nodes using TraceSelector.
3. Builds MEPs using detector adapter.
4. Generates ERRs via LLMTeacher.
5. Verifies ERRs via EvidenceContractVerifier.
6. Saves accepted (and rejected) ERRs for Stage 3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

from gread_core.tracing.selector import TraceSelector

logger = logging.getLogger(__name__)


def _strip_masks(data: Data) -> Data:
    """Remove all masks so forward_with_embedding returns logits for all nodes."""
    try:
        import copy
        d = copy.copy(data)
    except Exception:
        d = data
    for attr in ("train_mask", "val_mask", "test_mask"):
        if hasattr(d, attr):
            setattr(d, attr, None)
    return d


class Stage2Result:
    """Holds all ERR results from Stage 2 generation."""

    def __init__(
        self,
        accepted_errs: list[dict[str, Any]],
        rejected_errs: list[dict[str, Any]],
    ) -> None:
        self.accepted_errs = accepted_errs
        self.rejected_errs = rejected_errs

    @property
    def num_accepted(self) -> int:
        return len(self.accepted_errs)

    @property
    def num_rejected(self) -> int:
        return len(self.rejected_errs)

    def save(self, output_dir: str | Path) -> Path:
        """Save ERR results to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "accepted_errs.json", "w") as f:
            json.dump(self.accepted_errs, f, indent=2)

        with open(out / "rejected_errs.json", "w") as f:
            json.dump(self.rejected_errs, f, indent=2)

        logger.info(
            "Saved ERRs: %d accepted, %d rejected to %s",
            self.num_accepted, self.num_rejected, out,
        )
        return out

    @classmethod
    def load(cls, output_dir: str | Path) -> Stage2Result:
        """Load ERR results from disk."""
        out = Path(output_dir)
        with open(out / "accepted_errs.json") as f:
            accepted = json.load(f)
        with open(out / "rejected_errs.json") as f:
            rejected = json.load(f)
        return cls(accepted_errs=accepted, rejected_errs=rejected)


def generate_errs(
    detector: Any,
    data: Data,
    adapter: Any,
    teacher: Any,
    verifier: Any,
    config: dict[str, Any],
    seed: int = 1,
) -> Stage2Result:
    """Stage 2: Generate and verify ERRs via LLM.

    Args:
        detector: Trained base detector (Stage 1 output).
        data: PyG Data object.
        adapter: Evidence adapter for building MEPs.
        teacher: LLMTeacher instance for generating ERRs.
        verifier: EvidenceContractVerifier for checking ERRs.
        config: Configuration dict.
        seed: Random seed for trace selection.

    Returns:
        Stage2Result with accepted and rejected ERRs.
    """
    detector.eval()

    # Get detector outputs for ALL nodes (strip masks so no filtering)
    no_mask_data = _strip_masks(data)
    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(no_mask_data)
        scores = torch.sigmoid(logits).cpu()
        uncertainties = _compute_uncertainties(scores, embeddings).cpu()

    labels = data.y.cpu() if hasattr(data, "y") else None

    # Select trace nodes
    meps = adapter.extract(list(range(data.num_nodes)))
    selector = TraceSelector(config, seed)
    selection = selector.select(scores, uncertainties, labels, meps)

    logger.info("Selected %d trace nodes for ERR generation", len(selection.node_ids))

    # Generate ERRs via LLM
    selected_meps = [meps[i] for i in selection.node_ids]
    selected_labels = (
        [int(labels[i]) for i in selection.node_ids] if labels is not None else None
    )

    err_results = teacher.generate_err(selected_meps, selected_labels)

    # Separate accepted vs rejected
    # teacher.generate_err already returns only accepted, but we also need
    # to track rejected ones for the loss masking contract
    accepted_errs: list[dict[str, Any]] = []
    rejected_errs: list[dict[str, Any]] = []

    for i, mep in enumerate(selected_meps):
        # Try to find matching accepted result
        accepted_match = None
        for result in err_results:
            if result.node_id == mep.node_id:
                accepted_match = result
                break

        if accepted_match is not None:
            accepted_errs.append({
                "node_id": mep.node_id,
                "node_idx": selection.node_ids[i],
                "bucket": selection.bucket_labels[i],
                "err": accepted_match.err.model_dump(),
                "accepted": True,
            })
        else:
            # This MEP was rejected or failed parsing
            rejected_errs.append({
                "node_id": mep.node_id,
                "node_idx": selection.node_ids[i],
                "bucket": selection.bucket_labels[i],
                "accepted": False,
            })

    logger.info(
        "ERR generation complete: %d accepted, %d rejected",
        len(accepted_errs), len(rejected_errs),
    )

    return Stage2Result(accepted_errs=accepted_errs, rejected_errs=rejected_errs)


def _compute_uncertainties(scores: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    """Compute per-node uncertainty from scores and embeddings.

    Uses entropy-like measure: uncertainty = 1 - |2*score - 1|.
    High uncertainty when score is near 0.5.
    """
    return 1.0 - torch.abs(2.0 * scores - 1.0)
