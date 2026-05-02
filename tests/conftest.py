from pathlib import Path
from typing import Any

import pytest
import yaml

from gread_core.schemas.evidence import (
    CalibrationChannel,
    MinimalEvidencePackage,
    ReasoningChannel,
)
from gread_core.verification.verifier import EvidenceContractVerifier


@pytest.fixture
def contract_config() -> dict[str, Any]:
    return yaml.safe_load(Path("configs/contracts/gread_v1.yaml").read_text())


@pytest.fixture
def verifier(contract_config: dict[str, Any]) -> EvidenceContractVerifier:
    return EvidenceContractVerifier(contract_config)


@pytest.fixture
def sample_mep() -> MinimalEvidencePackage:
    return MinimalEvidencePackage(
        node_id="n1",
        detector_name="bwgnn",
        calibration=CalibrationChannel(prediction_score=0.9, uncertainty=0.1),
        reasoning=ReasoningChannel(
            uncertainty_level="low",
            degree_level="high",
            neighbor_consistency="low",
            feature_neighbor_discrepancy="high",
            detector_signal="high_frequency_response_high",
            detector_signal_strength="strong",
            counter_signal="benign_neighbor_signal_low",
            allowed_support_ids=[
                "degree_level",
                "neighbor_consistency",
                "feature_neighbor_discrepancy",
                "detector_signal",
                "detector_signal_strength",
            ],
            allowed_counter_ids=["counter_signal", "uncertainty_level"],
        ),
    )
