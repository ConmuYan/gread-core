"""Test that rho=0 recovers base detector logits exactly.

Critical ablation: when rho=0, the residual contribution
vanishes and final_logit must equal base_logit within float tolerance.
Tested with multiple random seeds for robustness.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from gread_core.models.evidence_encoder import EvidenceEncoder
from gread_core.models.reasoner import GReaDReasoner


def _make_reasoner(
    hidden_dim: int = 32,
    vocab_size: int = 50,
    embed_dim: int = 16,
    num_slots: int = 7,
    evidence_dim: int = 64,
    num_risk_types: int = 6,
    rho: float = 0.0,
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


class TestRhoZeroAblation:
    """Verify rho=0 gives final_logit == base_logit across multiple seeds."""

    def test_rho_zero_single_seed(self) -> None:
        """Basic rho=0 test with a single seed."""
        torch.manual_seed(0)
        reasoner = _make_reasoner(rho=0.0)
        outputs = _run_reasoner(reasoner, batch_size=8, hidden_dim=32, num_slots=7)
        max_diff = (
            outputs["final_logit"] - outputs["base_logit"]
        ).abs().max().item()
        assert torch.allclose(
            outputs["final_logit"], outputs["base_logit"], atol=1e-6
        ), f"rho=0 must recover base_logit. Max diff: {max_diff}"

    def test_rho_zero_multiple_seeds(self) -> None:
        """rho=0 must recover base_logit across multiple random seeds."""
        seeds = [0, 1, 42, 123, 999, 2024, 31415, 271828]

        for seed in seeds:
            torch.manual_seed(seed)
            reasoner = _make_reasoner(rho=0.0)
            outputs = _run_reasoner(
                reasoner, batch_size=8, hidden_dim=32, num_slots=7
            )
            max_diff = (
                outputs["final_logit"] - outputs["base_logit"]
            ).abs().max().item()
            assert torch.allclose(
                outputs["final_logit"], outputs["base_logit"], atol=1e-6
            ), f"Seed {seed}: rho=0 must recover base_logit. Max diff: {max_diff}"

    def test_rho_zero_different_batch_sizes(self) -> None:
        """rho=0 must work for various batch sizes."""
        batch_sizes = [1, 2, 4, 16, 64, 128]

        for batch_size in batch_sizes:
            torch.manual_seed(42)
            reasoner = _make_reasoner(rho=0.0)
            outputs = _run_reasoner(
                reasoner,
                batch_size=batch_size,
                hidden_dim=32,
                num_slots=7,
            )
            max_diff = (
                outputs["final_logit"] - outputs["base_logit"]
            ).abs().max().item()
            assert torch.allclose(
                outputs["final_logit"], outputs["base_logit"], atol=1e-6
            ), f"B={batch_size}: rho=0 must recover base_logit. Max diff: {max_diff}"

    def test_rho_zero_with_varying_evidence(self) -> None:
        """rho=0 gives same final_logit regardless of evidence tokens."""
        torch.manual_seed(42)
        reasoner = _make_reasoner(rho=0.0)
        z_v = torch.randn(4, 32)
        base_logit = torch.randn(4)

        evidence_a = torch.randint(0, 50, (4, 7))
        evidence_b = torch.randint(0, 50, (4, 7))

        out_a = reasoner(z_v, base_logit, evidence_a)
        out_b = reasoner(z_v, base_logit, evidence_b)

        assert torch.allclose(out_a["final_logit"], base_logit, atol=1e-6)
        assert torch.allclose(out_b["final_logit"], base_logit, atol=1e-6)
        assert torch.allclose(
            out_a["final_logit"], out_b["final_logit"], atol=1e-6
        )

    def test_rho_nonzero_diverges_from_base(self) -> None:
        """With rho != 0, final_logit should generally differ from base."""
        torch.manual_seed(42)
        reasoner = _make_reasoner(rho=0.1)
        outputs = _run_reasoner(reasoner, batch_size=8, hidden_dim=32, num_slots=7)
        assert not torch.allclose(
            outputs["final_logit"], outputs["base_logit"], atol=1e-6
        ), "With rho=0.1, final_logit should differ from base_logit"

    def test_rho_zero_eval_mode(self) -> None:
        """rho=0 should work in eval mode too."""
        torch.manual_seed(42)
        reasoner = _make_reasoner(rho=0.0)
        reasoner.eval()
        z_v = torch.randn(8, 32)
        base_logit = torch.randn(8)
        evidence_ids = torch.randint(0, 50, (8, 7))

        with torch.no_grad():
            outputs = reasoner(z_v, base_logit, evidence_ids)
        assert torch.allclose(
            outputs["final_logit"], outputs["base_logit"], atol=1e-6
        )
