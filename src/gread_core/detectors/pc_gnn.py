r"""PC-GNN detector (Pick-and-Choose GNN for imbalanced graph fraud detection).

Reference: Liu et al., "Pick and Choose: A GNN-based Imbalanced Learning
           Approach for Fraud Detection", WWW 2021.

PC-GNN tackles class imbalance in fraud graphs via two complementary sub-modules:
    * PICK   -- label-aware under/over-sampler that rebalances the minibatch
                drawn from the training graph (train-time concern, lives in
                the training loop, not in the detector module).
    * CHOOSE -- neighbor filter that, for each target node, selects neighbors
                whose predicted label-distribution is close to the target's.

This module implements a **simplified inference-time PC-GNN** that keeps the
CHOOSE sub-module as a per-layer auxiliary head. The auxiliary score
`s \in [0,1]^N` at each layer is interpreted as `P(fraud | h_l)` and is used
to gate the residual mix between the aggregated neighbor signal and the ego
signal. Exposing `self.layer_scores` gives GReaD-Core adapters a
per-layer, per-node signed evidence channel that directly reflects the
CHOOSE sub-module's decisions.

Scope of this implementation (explicit):
    [x] CHOOSE module (auxiliary per-layer fraud score + gated residual)
    [x] Evidence hook: `self.layer_scores : list[Tensor[N]]`
    [ ] PICK module (label-aware minibatch sampler)  -- lives in training loop
    [ ] Per-relation aggregation (multi-relation graphs) -- out of scope here
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
except ImportError:
    Data = Any
    SAGEConv = None


class PCGNNDetector(nn.Module):
    """PC-GNN base detector (simplified -- CHOOSE gate only).

    Architecture (per layer l):
        h'_l = SAGEConv(h_{l-1})           # inductive mean aggregation
        s_l  = sigmoid( W_aux h'_l )       # [N], auxiliary fraud score
        h_l  = (1 - s_l) * h'_l + s_l * h_{l-1}  # CHOOSE-gated ego residual

    Final classification head operates on h_L.

    Tensor shapes:
        x:              [N, F]
        edge_index:     [2, E]
        logit:          [N]
        embedding:      [N, hidden]
        layer_scores:   list of `num_layers` tensors, each [N]

    The per-layer auxiliary scores are the CHOOSE sub-module's decisions and
    can be surfaced as evidence by the PC-GNN adapter.
    """

    detector_name: str = "pc_gnn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        if SAGEConv is None:
            msg = "torch_geometric is required for PCGNNDetector"
            raise ImportError(msg)
        if num_layers < 1:
            msg = "num_layers must be >= 1"
            raise ValueError(msg)

        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.aux_heads = nn.ModuleList()
        self.ego_projs = nn.ModuleList()

        # First layer projects in_channels -> hidden; residual must also match.
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.aux_heads.append(nn.Linear(hidden_channels, 1))
        self.ego_projs.append(nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.aux_heads.append(nn.Linear(hidden_channels, 1))
            self.ego_projs.append(nn.Identity())

        self.head = nn.Linear(hidden_channels, 1)

        # Evidence hook: populated by forward_with_embedding().
        # Each entry is a detached [N] tensor of CHOOSE gate scores.
        self.layer_scores: list[Tensor] = []

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[N], node_embedding[N, H]).

        Side effect: populates `self.layer_scores` with per-layer CHOOSE
        scores (detached, on-device). Adapters read this attribute as the
        natively-signed evidence signal for PC-GNN.
        """
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index

        h_prev = x
        scores: list[Tensor] = []

        for conv, aux, ego_proj in zip(
            self.convs, self.aux_heads, self.ego_projs, strict=True
        ):
            h_agg = conv(h_prev, edge_index)
            h_agg = F.relu(h_agg)
            h_agg = F.dropout(h_agg, p=self.dropout, training=self.training)

            # CHOOSE gate: auxiliary fraud-probability at this layer.
            gate = torch.sigmoid(aux(h_agg)).squeeze(-1)  # [N] in [0, 1]
            scores.append(gate.detach())

            # Gated residual: mix aggregated neighbor signal with projected ego.
            h_ego = ego_proj(h_prev)
            gate_e = gate.unsqueeze(-1)  # [N, 1]
            h_prev = (1.0 - gate_e) * h_agg + gate_e * h_ego

        # Cache per-layer scores as the CHOOSE evidence channel.
        self.layer_scores = scores

        embedding = h_prev  # [N, hidden]
        target_mask = self._get_target_mask(graph)
        target_emb = embedding[target_mask] if target_mask is not None else embedding
        logit = self.head(target_emb).squeeze(-1)  # [B]
        return logit, embedding

    def _get_target_mask(self, graph: Data) -> Tensor | None:
        if hasattr(graph, "test_mask") and graph.test_mask is not None and graph.test_mask.any():
            return graph.test_mask  # type: ignore[no-any-return]
        if hasattr(graph, "val_mask") and graph.val_mask is not None and graph.val_mask.any():
            return graph.val_mask  # type: ignore[no-any-return]
        if hasattr(graph, "train_mask") and graph.train_mask is not None and graph.train_mask.any():
            return graph.train_mask  # type: ignore[no-any-return]
        return None
