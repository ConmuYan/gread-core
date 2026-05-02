"""Reasoning quality metrics: acceptance rate, evidence F1, risk-type agreement.

All metrics are deterministic given the same input. No LLM dependencies.
"""

from __future__ import annotations


def compute_acceptance_rate(accepted: int, total: int) -> float:
    """Compute ERR acceptance rate.

    Args:
        accepted: Number of accepted ERRs.
        total: Total number of ERRs evaluated.

    Returns:
        Acceptance rate in [0, 1]. Returns 0.0 if total is 0.
    """
    if total <= 0:
        return 0.0
    return float(accepted / total)


def compute_evidence_f1(
    predicted_evidence: list[str],
    reference_evidence: list[str],
) -> float:
    """Compute F1 between predicted and reference evidence sets.

    Treats evidence as sets of strings. Precision = |predicted ∩ reference| / |predicted|.
    Recall = |predicted ∩ reference| / |reference|.

    Args:
        predicted_evidence: List of predicted evidence IDs.
        reference_evidence: List of reference evidence IDs.

    Returns:
        F1 score in [0, 1]. Returns 0.0 if both lists are empty.
    """
    pred_set = set(predicted_evidence)
    ref_set = set(reference_evidence)

    if not pred_set and not ref_set:
        return 1.0  # both empty: perfect agreement

    if not pred_set or not ref_set:
        return 0.0

    intersection = pred_set & ref_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(ref_set)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_risk_type_agreement(
    predicted_type: str,
    reference_type: str,
) -> bool:
    """Check if predicted risk type matches reference.

    Args:
        predicted_type: Predicted risk type string.
        reference_type: Reference risk type string.

    Returns:
        True if types match exactly.
    """
    return predicted_type == reference_type


def compute_all_reasoning_metrics(
    predictions: list[dict[str, object]],
    references: list[dict[str, object]],
) -> dict[str, float]:
    """Compute aggregate reasoning metrics over a batch.

    Each prediction/reference dict should have:
        - "accepted": bool (whether ERR was accepted)
        - "evidence": list[str] (evidence IDs)
        - "risk_type": str

    Args:
        predictions: List of prediction dicts.
        references: List of reference dicts.

    Returns:
        Dict with "acceptance_rate", "evidence_f1", "risk_type_accuracy".
    """
    if not predictions:
        return {
            "acceptance_rate": 0.0,
            "evidence_f1": 0.0,
            "risk_type_accuracy": 0.0,
        }

    accepted = sum(1 for p in predictions if p.get("accepted", False))
    total = len(predictions)

    evidence_f1s: list[float] = []
    type_matches: list[bool] = []

    for pred, ref in zip(predictions, references, strict=False):
        pred_evidence = pred.get("evidence", [])
        ref_evidence = ref.get("evidence", [])
        if not isinstance(pred_evidence, list):
            pred_evidence = []
        if not isinstance(ref_evidence, list):
            ref_evidence = []
        evidence_f1s.append(compute_evidence_f1(pred_evidence, ref_evidence))

        type_matches.append(
            compute_risk_type_agreement(
                str(pred.get("risk_type", "")),
                str(ref.get("risk_type", "")),
            )
        )

    return {
        "acceptance_rate": compute_acceptance_rate(accepted, total),
        "evidence_f1": float(sum(evidence_f1s) / len(evidence_f1s)),
        "risk_type_accuracy": float(sum(type_matches) / len(type_matches)),
    }
