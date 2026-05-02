"""Student reasoner components for GReaD-Core.

Modules:
- EvidenceEncoder: embed discrete evidence tokens into dense vectors
- RiskTypeHead: classify risk type from concatenated [z_v; g_v]
- SignedEvidenceHead: independent positive and negative evidence mask heads
- EvidenceGatedResidualReadout: evidence-gated residual logit adjustment
- GReaDReasoner: orchestrator combining all components
"""

from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.heads import RiskTypeHead, SignedEvidenceHead
from gread_core.models.reasoner import GReaDReasoner
from gread_core.models.residual_readout import EvidenceGatedResidualReadout

__all__ = [
    "EvidenceEncoder",
    "EvidenceGatedResidualReadout",
    "GReaDReasoner",
    "RiskTypeHead",
    "SignedEvidenceHead",
]
