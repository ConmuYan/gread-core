import logging
from dataclasses import dataclass
from typing import Any

from gread_core.schemas.err import EvidenceRationaleRecord
from gread_core.schemas.evidence import MinimalEvidencePackage
from gread_core.verification.availability import check_availability
from gread_core.verification.contract import check_contract
from gread_core.verification.label_compatibility import check_label_compatibility
from gread_core.verification.role_consistency import check_role_consistency
from gread_core.verification.schema import check_schema
from gread_core.verification.score_blindness import check_score_blindness

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reasons: list[str]


class EvidenceContractVerifier:
    def __init__(
        self,
        contract_config: dict[str, Any],
        score_blind: bool = True,
    ) -> None:
        self.contract_config = contract_config
        self._score_blind = score_blind
        if not score_blind:
            logger.warning(
                "ABLATION: score_blind=False — score-blindness check disabled in verifier"
            )

    def verify(
        self,
        err: EvidenceRationaleRecord,
        mep: MinimalEvidencePackage,
        label: int | None = None,
    ) -> VerificationResult:
        checks = [
            check_schema(err, mep, self.contract_config),
            check_availability(err, mep, self.contract_config),
            check_role_consistency(err, mep, self.contract_config),
            check_contract(err, mep, self.contract_config),
        ]
        if self._score_blind:
            checks.append(check_score_blindness(err, mep, self.contract_config))
        checks.append(check_label_compatibility(err, mep, self.contract_config, label))
        reasons = [reason for passed, reason in checks if not passed]
        return VerificationResult(accepted=len(reasons) == 0, reasons=reasons)
