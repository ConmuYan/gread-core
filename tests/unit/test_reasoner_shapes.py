"""Test output shapes of GReaDReasoner forward pass.

Verifies that all 5 outputs match the tensor shape contract:
    base_logit:        [B]
    final_logit:       [B]
    type_logits:       [B, T]
    pos_mask_logits:   [B, K]
    neg_mask_logits:   [B, K]
"""

from __future__ import annotations

import torch

from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.reasoner import GReaDReasoner


def _make_reasoner(
    hidden_dim: int = 32,
    vocab_size: int = 50,
    embed_dim: int = 16,
    num_slots: int = 7,
    evidence_dim: int = 64,
    num_risk_types: int = 6,
    rho: float = 0.1,
) -> GReaDReasoner:
    encoder = EvidenceEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_slots=num_slots,
        output_dim=evidence_dim,
    )
    return GReaDReasoner(
        hidden_dim=hidden_dim,
        evidence_encoder=encoder,
        num_risk_types=num_risk_types,
        num_evidence_slots=num_slots,
        rho=rho,
    )


def _run_reasoner(
    reasoner: GReaDReasoner,
    batch_size: int,
    hidden_dim: int,
    num_slots: int,
) -> dict:
    z_v = torch.randn(batch_size, hidden_dim)
    base_logit = torch.randn(batch_size)
    evidence_ids = torch.randint(0, 50, (batch_size, num_slots))
    return reasoner(z_v, base_logit, evidence_ids)


class TestReasonerShapes:
    """Verify output shapes match the tensor contract."""

    def test_output_keys(self) -> None:
        reasoner = _make_reasoner()
        outputs = _run_reasoner(reasoner, batch_size=4, hidden_dim=32, num_slots=7)
        expected = {
            "base_logit",
            "final_logit",
            "type_logits",
            "pos_mask_logits",
            "neg_mask_logits",
        }
        assert set(outputs.keys()) == expected

    def test_base_logit_shape(self) -> None:
        reasoner = _make_reasoner()
        outputs = _run_reasoner(reasoner, batch_size=8, hidden_dim=32, num_slots=7)
        assert outputs["base_logit"].shape == (8,)

    def test_final_logit_shape(self) -> None:
        reasoner = _make_reasoner()
        outputs = _run_reasoner(reasoner, batch_size=8, hidden_dim=32, num_slots=7)
        assert outputs["final_logit"].shape == (8,)

    def test_type_logits_shape(self) -> None:
        num_types = 6
        reasoner = _make_reasoner(num_risk_types=num_types)
        outputs = _run_reasoner(reasoner, batch_size=8, hidden_dim=32, num_slots=7)
        assert outputs["type_logits"].shape == (8, num_types)

    def test_pos_mask_logits_shape(self) -> None:
        num_slots = 7
        reasoner = _make_reasoner(num_slots=num_slots)
        outputs = _run_reasoner(reasoner, batch_size=8, hidden_dim=32, num_slots=num_slots)
        assert outputs["pos_mask_logits"].shape == (8, num_slots)

    def test_neg_mask_logits_shape(self) -> None:
        num_slots = 7
        reasoner = _make_reasoner(num_slots=num_slots)
        outputs = _run_reasoner(reasoner, batch_size=8, hidden_dim=32, num_slots=num_slots)
        assert outputs["neg_mask_logits"].shape == (8, num_slots)

    def test_batch_size_1(self) -> None:
        reasoner = _make_reasoner()
        outputs = _run_reasoner(reasoner, batch_size=1, hidden_dim=32, num_slots=7)
        assert outputs["base_logit"].shape == (1,)
        assert outputs["final_logit"].shape == (1,)
        assert outputs["type_logits"].shape == (1, 6)
        assert outputs["pos_mask_logits"].shape == (1, 7)
        assert outputs["neg_mask_logits"].shape == (1, 7)

    def test_large_batch(self) -> None:
        reasoner = _make_reasoner()
        outputs = _run_reasoner(reasoner, batch_size=256, hidden_dim=32, num_slots=7)
        assert outputs["base_logit"].shape == (256,)
        assert outputs["final_logit"].shape == (256,)
        assert outputs["type_logits"].shape == (256, 6)
        assert outputs["pos_mask_logits"].shape == (256, 7)
        assert outputs["neg_mask_logits"].shape == (256, 7)

    def test_gradient_flows_through_evidence(self) -> None:
        """Evidence tokens affect type/evidence outputs (gradient flows)."""
        reasoner = _make_reasoner()
        z_v = torch.randn(4, 32)
        base_logit = torch.randn(4)
        evidence_ids = torch.randint(0, 50, (4, 7))

        outputs = reasoner(z_v, base_logit, evidence_ids)
        loss = outputs["type_logits"].sum() + outputs["pos_mask_logits"].sum()
        loss.backward()

        grad = reasoner.evidence_encoder.embedding.weight.grad
        assert grad is not None
        assert grad.abs().sum() > 0

    def test_signed_masks_are_independent(self) -> None:
        """Pos and neg mask heads produce different outputs."""
        reasoner = _make_reasoner()
        z_v = torch.randn(4, 32)
        base_logit = torch.randn(4)
        evidence_ids = torch.randint(0, 50, (4, 7))

        outputs = reasoner(z_v, base_logit, evidence_ids)
        pos = outputs["pos_mask_logits"]
        neg = outputs["neg_mask_logits"]
        assert not torch.allclose(pos, neg, atol=1e-6)
