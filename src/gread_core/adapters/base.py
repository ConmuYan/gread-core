"""EvidenceAdapter abstract base class for GReaD-Core.

All detector adapters must inherit from EvidenceAdapter and implement:
- extract(node_ids) -> list[MinimalEvidencePackage]
- supports_detector_signal() -> bool

The adapter converts detector-specific outputs into score-blind
MinimalEvidencePackage objects that pass the Evidence Contract Verifier.

Research constraint: prediction_score must NEVER appear in ReasoningChannel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from gread_core.schemas.evidence import MinimalEvidencePackage


class EvidenceAdapter(ABC):
    """Abstract base class for detector-evidence adapters.

    Each adapter extracts three categories of evidence:
    - generic: degree_level, neighbor_consistency, feature_neighbor_discrepancy, uncertainty_level
    - detector_native: detector_signal, detector_signal_strength
    - counter: counter_signal

    The adapter assembles these into a MinimalEvidencePackage with both
    CalibrationChannel (prediction_score, uncertainty) and ReasoningChannel
    (all evidence fields).
    """

    detector_name: str

    @abstractmethod
    def extract(self, node_ids: list[int]) -> list[MinimalEvidencePackage]:
        """Extract score-blind detector-native MEPs for given nodes.

        Args:
            node_ids: List of node indices to extract evidence for.

        Returns:
            List of MinimalEvidencePackage, one per node.
            Each package must pass Pydantic validation and the Evidence Contract Verifier.
        """
        ...

    @abstractmethod
    def supports_detector_signal(self) -> bool:
        """Return whether this detector exposes detector-native evidence.

        Returns:
            True if the adapter provides detector_signal and detector_signal_strength.
            False if only generic signals are available.
        """
        ...
