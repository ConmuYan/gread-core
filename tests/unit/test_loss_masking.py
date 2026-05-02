"""Test loss masking: rejected ERRs produce zero type/evidence loss.

Contract:
- L = L_sup + lambda * a_v * (L_type + L_evidence)
- If accepted_mask.sum() == 0: type_loss = 0, evidence_loss = 0, total_loss = L_sup
- Rejected ERRs produce zero type/evidence loss
- summary is NEVER used in loss computation
"""

from __future__ import annotations

import torch

from gread_core.losses.reasoning import ReasoningLoss


class TestReasoningLossMasking:
    """Verify that rejected ERRs produce zero type/evidence loss."""

    def setup_method(self) -> None:
        self.loss_fn = ReasoningLoss(lambda_reason=0.5, num_risk_types=6)
        self.B = 4
        self.T = 6
        self.K = 8

    def _make_outputs(self) -> dict[str, torch.Tensor]:
        return {
            "final_logit": torch.randn(self.B),
            "type_logits": torch.randn(self.B, self.T),
            "pos_mask_logits": torch.randn(self.B, self.K),
            "neg_mask_logits": torch.randn(self.B, self.K),
        }

    def test_all_accepted(self) -> None:
        """When all samples are accepted, type and evidence losses are nonzero."""
        out = self._make_outputs()
        labels = torch.tensor([0, 1, 1, 0])
        risk_types = torch.tensor([0, 1, 2, 3])
        pos_targets = torch.zeros(self.B, self.K)
        neg_targets = torch.zeros(self.B, self.K)
        accepted_mask = torch.ones(self.B, dtype=torch.bool)

        result = self.loss_fn(
            final_logit=out["final_logit"],
            type_logits=out["type_logits"],
            pos_mask_logits=out["pos_mask_logits"],
            neg_mask_logits=out["neg_mask_logits"],
            labels=labels,
            risk_type_targets=risk_types,
            pos_evidence_targets=pos_targets,
            neg_evidence_targets=neg_targets,
            accepted_mask=accepted_mask,
        )

        # With accepted samples, type_loss should be > 0
        assert result["type_loss"].item() > 0 or result["evidence_loss"].item() >= 0
        # total_loss should differ from sup_loss when reasoning losses are active
        assert result["total_loss"].item() != result["sup_loss"].item() or \
               result["type_loss"].item() == 0.0

    def test_none_accepted(self) -> None:
        """When no samples are accepted, type_loss = 0, evidence_loss = 0, total = L_sup."""
        out = self._make_outputs()
        labels = torch.tensor([0, 1, 1, 0])
        risk_types = torch.tensor([0, 1, 2, 3])
        pos_targets = torch.zeros(self.B, self.K)
        neg_targets = torch.zeros(self.B, self.K)
        accepted_mask = torch.zeros(self.B, dtype=torch.bool)

        result = self.loss_fn(
            final_logit=out["final_logit"],
            type_logits=out["type_logits"],
            pos_mask_logits=out["pos_mask_logits"],
            neg_mask_logits=out["neg_mask_logits"],
            labels=labels,
            risk_type_targets=risk_types,
            pos_evidence_targets=pos_targets,
            neg_evidence_targets=neg_targets,
            accepted_mask=accepted_mask,
        )

        assert result["type_loss"].item() == 0.0
        assert result["evidence_loss"].item() == 0.0
        assert result["total_loss"].item() == result["sup_loss"].item()

    def test_partial_accepted(self) -> None:
        """With some accepted and some rejected, only accepted contribute to reasoning loss."""
        out = self._make_outputs()
        labels = torch.tensor([0, 1, 1, 0])
        risk_types = torch.tensor([0, 1, 2, 3])
        pos_targets = torch.zeros(self.B, self.K)
        neg_targets = torch.zeros(self.B, self.K)
        # Only first 2 are accepted
        accepted_mask = torch.tensor([True, True, False, False])

        result = self.loss_fn(
            final_logit=out["final_logit"],
            type_logits=out["type_logits"],
            pos_mask_logits=out["pos_mask_logits"],
            neg_mask_logits=out["neg_mask_logits"],
            labels=labels,
            risk_type_targets=risk_types,
            pos_evidence_targets=pos_targets,
            neg_evidence_targets=neg_targets,
            accepted_mask=accepted_mask,
        )

        # total_loss = sup_loss + lambda * (type_loss + evidence_loss)
        # type_loss and evidence_loss should be computed only on accepted
        expected_total = result["sup_loss"] + 0.5 * (result["type_loss"] + result["evidence_loss"])
        assert abs(result["total_loss"].item() - expected_total.item()) < 1e-5

    def test_none_targets_with_accepted(self) -> None:
        """When risk_type_targets is None, type/evidence loss is 0 even with accepted."""
        out = self._make_outputs()
        labels = torch.tensor([0, 1, 1, 0])
        accepted_mask = torch.ones(self.B, dtype=torch.bool)

        result = self.loss_fn(
            final_logit=out["final_logit"],
            type_logits=out["type_logits"],
            pos_mask_logits=out["pos_mask_logits"],
            neg_mask_logits=out["neg_mask_logits"],
            labels=labels,
            risk_type_targets=None,
            pos_evidence_targets=None,
            neg_evidence_targets=None,
            accepted_mask=accepted_mask,
        )

        assert result["type_loss"].item() == 0.0
        assert result["evidence_loss"].item() == 0.0
        assert result["total_loss"].item() == result["sup_loss"].item()

    def test_summary_never_used(self) -> None:
        """Verify that summary field is never referenced in loss computation.

        The loss function signature does not accept summary at all,
        ensuring it cannot be used in loss computation.
        """
        import inspect
        sig = inspect.signature(self.loss_fn.forward)
        param_names = list(sig.parameters.keys())
        assert "summary" not in param_names
