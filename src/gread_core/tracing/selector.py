"""Main trace node selection interface.

Combines 3-bucket assignment with evidence diversity sampling to select
trace nodes for offline LLM ERR generation.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from torch import Tensor

from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.tracing.buckets import BucketLabel, BucketPolicy, assign_buckets
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
        bucket_policy_cfg = ts.get("bucket_policy", "fixed")
        if isinstance(bucket_policy_cfg, str):
            self.bucket_policy: BucketPolicy = self._validate_bucket_policy(bucket_policy_cfg)
            policy_cfg: dict[str, Any] = {}
        elif isinstance(bucket_policy_cfg, dict):
            self.bucket_policy = self._validate_bucket_policy(
                str(bucket_policy_cfg.get("mode", "fixed"))
            )
            policy_cfg = bucket_policy_cfg
        else:
            raise TypeError("trace_selection.bucket_policy must be a string or mapping")
        self.bucket_policy_params: dict[str, Any] = {
            "percentile_uncertain_fraction": policy_cfg.get(
                "uncertain_fraction", policy_cfg.get("percentile_uncertain_fraction", 0.2)
            ),
            "percentile_high_conf_fraction": policy_cfg.get(
                "high_conf_fraction", policy_cfg.get("percentile_high_conf_fraction", 0.2)
            ),
            "percentile_low_uncertainty_fraction": policy_cfg.get(
                "low_uncertainty_fraction",
                policy_cfg.get("percentile_low_uncertainty_fraction", 0.5),
            ),
        }
        self.fallback_budget_reallocation: bool = ts.get(
            "fallback_budget_reallocation", False
        )
        self.seed = seed

    @staticmethod
    def _validate_bucket_policy(value: str) -> BucketPolicy:
        if value not in ("fixed", "percentile"):
            raise ValueError(f"Unsupported trace_selection.bucket_policy: {value}")
        return cast(BucketPolicy, value)

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
        meps: Sequence[MinimalEvidencePackage],
        bucket_assignments: Sequence[BucketLabel | None] | None = None,
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
        if bucket_assignments is None:
            bucket_assignments = assign_buckets(
                scores,
                uncertainties,
                labels,
                bucket_policy=self.bucket_policy,
                **self.bucket_policy_params,
            )
        assigned_buckets = list(bucket_assignments)

        # Group nodes by bucket
        bucket_nodes: dict[str, list[int]] = {
            "uncertain": [],
            "high_conf_fraud": [],
            "high_conf_benign": [],
        }
        for i, bl in enumerate(assigned_buckets):
            if bl is not None:
                bucket_nodes[bl].append(i)

        eligible_uncertain = len(bucket_nodes["uncertain"])
        eligible_fraud = len(bucket_nodes["high_conf_fraud"])
        eligible_benign = len(bucket_nodes["high_conf_benign"])
        unassigned = n - eligible_uncertain - eligible_fraud - eligible_benign
        empty_buckets = [
            name for name, count in (
                ("uncertain", eligible_uncertain),
                ("high_conf_fraud", eligible_fraud),
                ("high_conf_benign", eligible_benign),
            ) if count == 0
        ]
        if empty_buckets:
            logger.warning(
                "Empty trace buckets=%s | eligible uncertain=%d sample=%s "
                "high_conf_fraud=%d sample=%s high_conf_benign=%d sample=%s "
                "unassigned=%d",
                empty_buckets,
                eligible_uncertain,
                bucket_nodes["uncertain"][:5],
                eligible_fraud,
                bucket_nodes["high_conf_fraud"][:5],
                eligible_benign,
                bucket_nodes["high_conf_benign"][:5],
                unassigned,
            )

        # Compute per-bucket budgets
        budgets = self._bucket_budgets()
        metadata: dict[str, Any] = {
            "bucket_policy": self.bucket_policy,
            "eligible_counts": {
                "uncertain": eligible_uncertain,
                "high_conf_fraud": eligible_fraud,
                "high_conf_benign": eligible_benign,
                "unassigned": unassigned,
            },
            "initial_budgets": dict(budgets),
            "fallback_budget_reallocation": self.fallback_budget_reallocation,
            "reallocated_budget": 0,
            "reallocated_to": {
                "uncertain": 0,
                "high_conf_fraud": 0,
                "high_conf_benign": 0,
            },
        }

        # Select from each bucket
        rng = random.Random(self.seed)
        all_node_ids: list[int] = []
        all_bucket_labels: list[str] = []
        all_diversity_scores: list[float] = []
        selected_by_bucket: dict[str, set[int]] = {
            "uncertain": set(),
            "high_conf_fraud": set(),
            "high_conf_benign": set(),
        }

        for bucket_name in ("uncertain", "high_conf_fraud", "high_conf_benign"):
            candidates = bucket_nodes[bucket_name]
            budget = budgets[bucket_name]

            if not candidates:
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
            selected_by_bucket[bucket_name].update(selected)

        if self.fallback_budget_reallocation and len(all_node_ids) < self.total_budget:
            remaining_budget = self.total_budget - len(all_node_ids)
            for bucket_name in ("uncertain", "high_conf_fraud", "high_conf_benign"):
                if remaining_budget <= 0:
                    break
                candidates = [
                    node_id
                    for node_id in bucket_nodes[bucket_name]
                    if node_id not in selected_by_bucket[bucket_name]
                ]
                if not candidates:
                    continue

                budget = min(remaining_budget, len(candidates))
                if self.diversity_sampling:
                    selected, div_scores = diversity_sample(candidates, meps, budget, rng)
                else:
                    shuffled = list(candidates)
                    rng.shuffle(shuffled)
                    selected = shuffled[:budget]
                    div_scores = [0.0] * len(selected)

                all_node_ids.extend(selected)
                all_bucket_labels.extend([bucket_name] * len(selected))
                all_diversity_scores.extend(div_scores)
                selected_by_bucket[bucket_name].update(selected)
                remaining_budget -= len(selected)
                metadata["reallocated_to"][bucket_name] += len(selected)
                metadata["reallocated_budget"] += len(selected)

        # Enforce total budget cap
        if len(all_node_ids) > self.total_budget:
            all_node_ids = all_node_ids[: self.total_budget]
            all_bucket_labels = all_bucket_labels[: self.total_budget]
            all_diversity_scores = all_diversity_scores[: self.total_budget]

        logger.info(
            "Selected %d trace nodes (budget=%d): "
            "eligible uncertain=%d, high_conf_fraud=%d, high_conf_benign=%d, unassigned=%d | "
            "selected uncertain=%d, high_conf_fraud=%d, high_conf_benign=%d",
            len(all_node_ids),
            self.total_budget,
            eligible_uncertain,
            eligible_fraud,
            eligible_benign,
            unassigned,
            all_bucket_labels.count("uncertain"),
            all_bucket_labels.count("high_conf_fraud"),
            all_bucket_labels.count("high_conf_benign"),
        )
        metadata["selected_counts"] = {
            "uncertain": all_bucket_labels.count("uncertain"),
            "high_conf_fraud": all_bucket_labels.count("high_conf_fraud"),
            "high_conf_benign": all_bucket_labels.count("high_conf_benign"),
        }
        metadata["unfilled_budget"] = self.total_budget - len(all_node_ids)

        return SelectionResult(
            node_ids=all_node_ids,
            bucket_labels=all_bucket_labels,
            diversity_scores=all_diversity_scores,
            metadata=metadata,
        )
