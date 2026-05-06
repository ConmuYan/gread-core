"""Tree ensemble with neighbor aggregation detector for GReaD-Core.

Combines self-feature projection, mean neighbor aggregation, and an MLP head.

Research constraints:
- No LLM imports allowed in this module.
- No hidden constants - all hyperparameters are config-driven.
- forward_with_embedding() is the ONLY entry point.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any


class TreeNeighborDetector(nn.Module):
    """Tree ensemble with neighbor aggregation features.

    Architecture:
        self_proj: Linear(F, hidden)       -- project self features
        neigh_proj: Linear(F, hidden)      -- project neighbor features
        head: Linear(hidden*2, hidden) -> ReLU -> Linear(hidden, 1)

    Tensor shapes:
        x:              [N, F]   node features
        edge_index:     [2, E]   edge indices
        logit:          [B]      classification logits
        embedding:      [B, hidden*2]  node embeddings (pre-head)
    """

    detector_name: str = "tree_neighbor"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if Data is None:
            msg = "torch_geometric is required for TreeNeighborDetector"
            raise ImportError(msg)

        self.dropout = dropout

        # Self-feature projection
        self.self_proj = nn.Linear(in_channels, hidden_channels)

        # Neighbor-feature projection
        self.neigh_proj = nn.Linear(in_channels, hidden_channels)

        # MLP classification head (concatenated self + neighbor embeddings)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.feature_importance: Tensor | None = None

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[B], node_embedding[N, H*2]).

        Embeddings are for ALL nodes (adapters need full graph for neighbor
        lookups). Logits are for target nodes only (mask-selected).
        """
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index
        row, col = edge_index
        num_nodes = x.size(0)

        # Self features: [N, hidden]
        self_feat = F.relu(self.self_proj(x))
        self_feat = F.dropout(self_feat, p=self.dropout, training=self.training)

        # Mean-aggregated neighbor features: [N, hidden]
        neigh_agg = torch.zeros(num_nodes, x.size(-1), device=x.device)
        neigh_agg.scatter_add_(0, row.unsqueeze(-1).expand_as(x[col]), x[col])
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(-1)
        neigh_agg = neigh_agg / deg
        neigh_feat = F.relu(self.neigh_proj(neigh_agg))
        neigh_feat = F.dropout(neigh_feat, p=self.dropout, training=self.training)

        # Concatenate self + neighbor: [N, hidden*2]
        full_embedding = torch.cat([self_feat, neigh_feat], dim=-1)
        self.feature_importance = (self_feat - neigh_feat).abs().detach()

        # Select target nodes for logits only
        target_mask = self._get_target_mask(graph)
        target_emb = full_embedding[target_mask] if target_mask is not None else full_embedding

        # Classification head: [B, hidden*2] -> [B]
        logit = self.head(target_emb).squeeze(-1)

        return logit, full_embedding

    def _get_target_mask(self, graph: Data) -> Tensor | None:
        """Select target nodes based on available masks.

        Priority: test_mask > val_mask > train_mask > None (all nodes).
        """
        if hasattr(graph, "test_mask") and graph.test_mask is not None and graph.test_mask.any():
            return graph.test_mask  # type: ignore[no-any-return]
        if hasattr(graph, "val_mask") and graph.val_mask is not None and graph.val_mask.any():
            return graph.val_mask  # type: ignore[no-any-return]
        if hasattr(graph, "train_mask") and graph.train_mask is not None and graph.train_mask.any():
            return graph.train_mask  # type: ignore[no-any-return]
        return None
