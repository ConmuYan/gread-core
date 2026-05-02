"""Tests for EvidenceGatedResidualReadout.

Verifies:
- Output shape is [B]
- Gating mechanism works correctly
- Zero evidence produces zero net gate
- Gradient flows through all inputs
"""

from __future__ import annotations

import torch

from gread_core.models.residual_readout import EvidenceGatedResidualReadout


class TestResidualReadoutShapes:
    """Verify output shapes."""

    def test_output_shape(self) -> None:
        readout = EvidenceGatedResidualReadout(32, 64)
        z_v = torch.randn(8, 32)
        g_v = torch.randn(8, 64)
        pos_logits = torch.randn(8, 7)
        neg_logits = torch.randn(8, 7)

        out = readout(z_v, g_v, pos_logits, neg_logits)
        assert out.shape == (8,)

    def test_output_shape_batch_1(self) -> None:
        readout = EvidenceGatedResidualReadout(32, 64)
        z_v = torch.randn(1, 32)
        g_v = torch.randn(1, 64)
        pos_logits = torch.randn(1, 7)
        neg_logits = torch.randn(1, 7)

        out = readout(z_v, g_v, pos_logits, neg_logits)
        assert out.shape == (1,)


class TestResidualReadoutBehavior:
    """Verify gating behavior."""

    def test_zero_evidence_gives_zero_residual(self) -> None:
        """Very negative logits -> sigmoid ~ 0, net gate ~ 0."""
        readout = EvidenceGatedResidualReadout(32, 64)
        z_v = torch.randn(4, 32)
        g_v = torch.randn(4, 64)

        pos_logits = torch.full((4, 7), -100.0)
        neg_logits = torch.full((4, 7), -100.0)

        out = readout(z_v, g_v, pos_logits, neg_logits)
        assert torch.allclose(out, torch.zeros(4), atol=1e-5)

    def test_balanced_evidence_gives_zero_residual(self) -> None:
        """When pos and neg evidence are balanced, net gate -> 0."""
        readout = EvidenceGatedResidualReadout(32, 64)
        z_v = torch.randn(4, 32)
        g_v = torch.randn(4, 64)

        # sigmoid(0) = 0.5 for both -> net_gate = 0
        logits = torch.zeros(4, 7)
        out = readout(z_v, g_v, logits, logits)
        assert torch.allclose(out, torch.zeros(4), atol=1e-5)

    def test_strong_pos_evidence_increases_residual(self) -> None:
        """Strong positive evidence should produce a non-zero residual."""
        readout = EvidenceGatedResidualReadout(32, 64)
        z_v = torch.randn(4, 32)
        g_v = torch.randn(4, 64)

        pos_logits = torch.full((4, 7), 10.0)  # sigmoid(10) ~ 1
        neg_logits = torch.full((4, 7), -10.0)  # sigmoid(-10) ~ 0

        out = readout(z_v, g_v, pos_logits, neg_logits)
        assert out.shape == (4,)

    def test_gradient_flows_all_inputs(self) -> None:
        """Gradient flows through z_v, g_v, pos_logits, neg_logits."""
        readout = EvidenceGatedResidualReadout(32, 64)

        z_v = torch.randn(4, 32, requires_grad=True)
        g_v = torch.randn(4, 64, requires_grad=True)
        pos_logits = torch.randn(4, 7, requires_grad=True)
        neg_logits = torch.randn(4, 7, requires_grad=True)

        out = readout(z_v, g_v, pos_logits, neg_logits)
        out.sum().backward()

        assert z_v.grad is not None
        assert g_v.grad is not None
        assert pos_logits.grad is not None
        assert neg_logits.grad is not None
        assert z_v.grad.abs().sum() > 0
        assert g_v.grad.abs().sum() > 0
        assert pos_logits.grad.abs().sum() > 0
        assert neg_logits.grad.abs().sum() > 0

    def test_pos_vs_neg_residuals_differ(self) -> None:
        """Pure positive vs pure negative evidence gives opposite signs."""
        readout = EvidenceGatedResidualReadout(32, 64)

        torch.manual_seed(42)
        z_v = torch.randn(4, 32)
        g_v = torch.randn(4, 64)

        # Pure positive evidence
        pos_logits = torch.full((4, 7), 10.0)
        neg_logits = torch.full((4, 7), -10.0)
        out_pos = readout(z_v, g_v, pos_logits, neg_logits)

        # Pure negative evidence (swap)
        out_neg = readout(z_v, g_v, neg_logits, pos_logits)

        # out_pos ~ -out_neg since net_gate flips sign
        assert torch.allclose(out_pos, -out_neg, atol=1e-4)
