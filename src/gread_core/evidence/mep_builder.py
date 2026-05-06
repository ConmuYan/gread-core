"""MEP builder: assembles MinimalEvidencePackage from signal components.

This builder enforces score-blind construction: prediction_score only enters
the CalibrationChannel, never the ReasoningChannel.
"""

from __future__ import annotations

from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)


def build_mep(
    node_id: int | str,
    detector_name: str,
    prediction_score: float,
    uncertainty: float,
    uncertainty_level: str,
    degree_level: str,
    neighbor_consistency: str,
    feature_neighbor_discrepancy: str,
    detector_signal: str,
    detector_signal_strength: str,
    counter_signal: str,
    allowed_support_ids: list[str] | None = None,
    allowed_counter_ids: list[str] | None = None,
) -> MinimalEvidencePackage:
    """Build a score-blind MinimalEvidencePackage.

    prediction_score enters only CalibrationChannel. All reasoning fields
    are string-valued and never reference the raw score.

    Args:
        node_id: Node identifier.
        detector_name: Name of the detector that produced this evidence.
        prediction_score: Calibrated fraud score (calibration-only).
        uncertainty: Raw uncertainty value in [0, 1].
        uncertainty_level: Discretized uncertainty (low/medium/high).
        degree_level: Discretized degree level.
        neighbor_consistency: Discretized neighbor label consistency.
        feature_neighbor_discrepancy: Discretized feature-neighbor discrepancy.
        detector_signal: Detector-native signal string.
        detector_signal_strength: Strength of detector signal (weak/moderate/strong/unavailable).
        counter_signal: Counter-evidence signal string.
        allowed_support_ids: IDs allowed in supporting evidence.
        allowed_counter_ids: IDs allowed in counter evidence.

    Returns:
        Validated MinimalEvidencePackage.
    """
    if allowed_support_ids is None:
        allowed_support_ids = [
            "degree_level",
            "neighbor_consistency",
            "feature_neighbor_discrepancy",
            "detector_signal",
            "uncertainty_level",
        ]
    if allowed_counter_ids is None:
        allowed_counter_ids = ["counter_signal", "uncertainty_level"]

    calibration = CalibrationChannel(
        prediction_score=prediction_score,
        uncertainty=uncertainty,
    )

    reasoning = ReasoningChannel(
        uncertainty_level=uncertainty_level,  # type: ignore[arg-type]
        degree_level=degree_level,
        neighbor_consistency=neighbor_consistency,
        feature_neighbor_discrepancy=feature_neighbor_discrepancy,
        detector_signal=detector_signal,
        detector_signal_strength=detector_signal_strength,  # type: ignore[arg-type]
        counter_signal=counter_signal,
        allowed_support_ids=allowed_support_ids,
        allowed_counter_ids=allowed_counter_ids,
    )

    return MinimalEvidencePackage(
        node_id=str(node_id),
        detector_name=detector_name,
        calibration=calibration,
        reasoning=reasoning,
    )
