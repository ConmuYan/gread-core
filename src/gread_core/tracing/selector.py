"""Main trace node selection interface.

Combines 3-bucket assignment with evidence diversity sampling to select
trace nodes for offline LLM ERR generation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.tracing.buckets import assign_buckets
from gread_core.tracing.diversity import diversity_sample

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of trace node selection.

    prediction_score is NOT included -- it is used only internally
    for bucket assignment and must never be exposed to prompts.
    """

    node_ids: list[int] = field(default_factory=list)
    bucket_labels: list[str] = field(default_factory=list)
    diversity_scores: list[float] = field(default_factory=list)


class TraceSelector:
    """Select trace nodes using 3-bucket assignment and evidence diversity sampling.

    Buckets:
    - uncertain: near decision boundary or high uncertainty
    - high_conf_fraud: label=1, high score, low uncertainty
    - high_conf_benign: label=0, low score, low uncertainty

    prediction_score is used only for bucket assignment and is NOT
    exposed in the SelectionResult or any downstream prompt.
    """

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        ts = config.get("trace_selection", {})
        self.total_budget: int = ts.get("total_budget", 1000)
        buckets_cfg = ts.get("buckets", {})
        self.bucket_fractions: dict[str, float] = {
            "uncertain": buckets_cfg.get("uncertain", 0.333),
            "high_conf_fraud": buckets_cfg.get("high_conf_fraud", 0.333),
            "high_conf_benign": buckets_cfg.get("high_conf_benign", 0.334),
        }
        self.diversity_sampling: bool = ts.get("diversity_sampling", True)
        self.seed = seed

    def _bucket_budgets(self) -> dict[str, int]:
        """Compute per-bucket budget from fractions and total_budget."""
        budgets: dict[str, int] = {}
        remaining = self.total_budget
        names = list(self.bucket_fractions.keys())
        for name in names[:-1]:
            b = int(self.total_budget * self.bucket_fractions[name])
            budgets[name] = b
            remaining -= b
        # Last bucket gets the remainder to avoid rounding loss
        budgets[names[-1]] = remaining
        return budgets

    def select(
        self,
        scores: Tensor,
        uncertainties: Tensor,
        labels: Tensor | None,
        meps: list[MinimalEvidencePackage],
    ) -> SelectionResult:
        """Select trace nodes with bucket assignment and diversity sampling.

        Args:
            scores: prediction scores, shape [N], values in [0, 1].
            uncertainties: uncertainty values, shape [N], values in [0, 1].
            labels: ground-truth labels (0/1), shape [N], or None.
            meps: list of MinimalEvidencePackage, one per node.

        Returns:
            SelectionResult with node_ids, bucket_labels, and diversity_scores.
            prediction_score is NOT included in the result.
        """
        n = scores.shape[0]
        if n == 0:
            logger.warning("Empty input: no nodes to select")
            return SelectionResult()

        # Assign buckets
        bucket_assignments = assign_buckets(scores, uncertainties, labels)

        # Group nodes by bucket
        bucket_nodes: dict[str, list[int]] = {
            "uncertain": [],
            "high_conf_fraud": [],
            "high_conf_benign": [],
        }
        for i, bl in enumerate(bucket_assignments):
            if bl is not None:
                bucket_nodes[bl].append(i)

        # Compute per-bucket budgets
        budgets = self._bucket_budgets()

        # Select from each bucket
        rng = random.Random(self.seed)
        all_node_ids: list[int] = []
        all_bucket_labels: list[str] = []
        all_diversity_scores: list[float] = []

        for bucket_name in ("uncertain", "high_conf_fraud", "high_conf_benign"):
            candidates = bucket_nodes[bucket_name]
            budget = budgets[bucket_name]

            if not candidates:
                logger.warning("Bucket '%s' has no eligible nodes", bucket_name)
                continue

            if self.diversity_sampling:
                selected, div_scores = diversity_sample(candidates, meps, budget, rng)
            else:
                # Fallback: random sampling
                shuffled = list(candidates)
                rng.shuffle(shuffled)
                selected = shuffled[:budget]
                div_scores = [0.0] * len(selected)

            all_node_ids.extend(selected)
            all_bucket_labels.extend([bucket_name] * len(selected))
            all_diversity_scores.extend(div_scores)

        # Enforce total budget cap
        if len(all_node_ids) > self.total_budget:
            all_node_ids = all_node_ids[: self.total_budget]
            all_bucket_labels = all_bucket_labels[: self.total_budget]
            all_diversity_scores = all_diversity_scores[: self.total_budget]

        logger.info(
            "Selected %d trace nodes (budget=%d): "
            "uncertain=%d, high_conf_fraud=%d, high_conf_benign=%d",
            len(all_node_ids),
            self.total_budget,
            all_bucket_labels.count("uncertain"),
            all_bucket_labels.count("high_conf_fraud"),
            all_bucket_labels.count("high_conf_benign"),
        )

        return SelectionResult(
            node_ids=all_node_ids,
            bucket_labels=all_bucket_labels,
            diversity_scores=all_diversity_scores,
        )
