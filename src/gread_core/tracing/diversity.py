"""Evidence diversity sampling within buckets.

Uses farthest-first traversal over evidence signatures to maximize
coverage of different detector signals, degree levels, and patterns.
"""

from __future__ import annotations

import random

from gread_core.schemas.evidence import MinimalEvidencePackage


def _evidence_signature(mep: MinimalEvidencePackage) -> tuple[str, ...]:
    """Build a hashable signature from MEP reasoning slots for diversity comparison."""
    r = mep.reasoning
    return (
        r.uncertainty_level,
        r.degree_level,
        r.neighbor_consistency,
        r.feature_neighbor_discrepancy,
        r.detector_signal,
        r.detector_signal_strength,
        r.counter_signal,
    )


def diversity_sample(
    candidates: list[int],
    meps: list[MinimalEvidencePackage],
    budget: int,
    rng: random.Random,
) -> tuple[list[int], list[float]]:
    """Select nodes from candidates maximizing evidence diversity.

    Uses greedy farthest-first traversal over evidence signatures.
    Diversity score for each selected node = minimum Hamming-like distance
    to any previously selected node's evidence signature (1.0 for first).

    Args:
        candidates: indices of candidate nodes within this bucket.
        meps: full MEP list (indexed by global node id).
        budget: maximum number of nodes to select.
        rng: seeded Random instance for deterministic tie-breaking.

    Returns:
        Tuple of (selected node ids, diversity scores per selected node).
    """
    if not candidates or budget <= 0:
        return [], []

    budget = min(budget, len(candidates))

    # Build signatures for candidates
    sigs: dict[int, tuple[str, ...]] = {}
    for c in candidates:
        sigs[c] = _evidence_signature(meps[c])

    # Shuffle candidates deterministically for tie-breaking
    shuffled = list(candidates)
    rng.shuffle(shuffled)

    selected: list[int] = []
    diversity_scores: list[float] = []

    # First pick: random from shuffled
    first = shuffled[0]
    selected.append(first)
    diversity_scores.append(1.0)

    if budget == 1:
        return selected, diversity_scores

    remaining = [c for c in shuffled if c != first]

    for _ in range(budget - 1):
        if not remaining:
            break

        best_node = remaining[0]
        best_score = -1.0

        for cand in remaining:
            cand_sig = sigs[cand]
            # Minimum distance to any already-selected node
            min_dist = float("inf")
            for sel in selected:
                sel_sig = sigs[sel]
                dist = sum(1 for a, b in zip(cand_sig, sel_sig, strict=False) if a != b)
                min_dist = min(min_dist, dist)

            # Normalize by signature length
            norm_dist = min_dist / len(cand_sig) if cand_sig else 0.0

            if norm_dist > best_score:
                best_score = norm_dist
                best_node = cand

        selected.append(best_node)
        diversity_scores.append(best_score)
        remaining.remove(best_node)

    return selected, diversity_scores
