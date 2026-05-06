"""CARE-GNN (Similarity-Aware Neighbor Selection) detector for GReaD-Core.

Uses per-relation weight matrices and similarity-based neighbor filtering.

Research constraints:
- No LLM imports allowed in this module.
- No hidden constants - all hyperparameters are config-driven.
- forward_with_embedding() is the ONLY entry point.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any


class CARELayer(nn.Module):
    """CARE aggregation layer with similarity-aware neighbor selection.

    For each relation, computes a similarity score between node and neighbor
    embeddings, then filters/aggregates neighbors based on a learnable threshold.
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_relations: int, sim_threshold: float
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.sim_threshold = sim_threshold

        # Per-relation weight matrices
        self.relation_weights = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=False) for _ in range(num_relations)]
        )
        # Per-relation similarity scorer
        self.sim_linear = nn.ModuleList(
            [nn.Linear(in_channels * 2, 1) for _ in range(num_relations)]
        )
        self.attn = nn.Linear(out_channels, 1)
        self.last_filter_weights: dict[int, Tensor] = {}

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_type: Tensor | None = None
    ) -> Tensor:
        """Aggregate neighbor features with similarity filtering.

        Args:
            x: [N, F] node features.
            edge_index: [2, E] edge indices.
            edge_type: [E] relation type per edge (0..num_relations-1).
                If None, all edges use relation 0.

        Returns:
            [N, out_channels] aggregated features.
        """
        row, col = edge_index
        num_nodes: int = x.size(0)
        first_linear: nn.Linear = self.relation_weights[0]  # type: ignore[assignment]
        out_channels: int = first_linear.out_features

        if edge_type is None:
            edge_type = torch.zeros(row.size(0), dtype=torch.long, device=x.device)

        agg: Tensor = torch.zeros(num_nodes, out_channels, device=x.device)
        node_weights: dict[int, list[Tensor]] = {}

        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any():
                continue

            r_row = row[mask]
            r_col = col[mask]

            # Transform neighbor features
            nbr_feat = self.relation_weights[r](x[r_col])

            # Similarity scoring
            pair = torch.cat([x[r_row], x[r_col]], dim=-1)
            sim_score = torch.sigmoid(self.sim_linear[r](pair)).squeeze(-1)
            for nid in r_row.unique():
                nid_int = int(nid.item())
                node_weights.setdefault(nid_int, []).append(sim_score[r_row == nid].detach())

            # Filter by threshold: keep neighbors above similarity threshold
            weight = (sim_score >= self.sim_threshold).float()
            weighted = weight.unsqueeze(-1) * nbr_feat

            # Attention-weighted aggregation
            attn_score = torch.sigmoid(self.attn(weighted)).squeeze(-1)
            weighted = attn_score.unsqueeze(-1) * weighted

            agg.scatter_add_(0, r_row.unsqueeze(-1).expand_as(weighted), weighted)

        self.last_filter_weights = {
            node_id: torch.cat(values) for node_id, values in node_weights.items()
        }
        return agg


class CAREGNNDetector(nn.Module):
    """CARE-GNN detector with similarity-aware neighbor selection.

    Architecture:
        CARELayer(F, hidden) -> ReLU -> CARELayer(hidden, hidden) -> Linear(hidden, 1)

    Tensor shapes:
        x:              [N, F]   node features
        edge_index:     [2, E]   edge indices
        logit:          [B]      classification logits
        embedding:      [B, hidden]  node embeddings (pre-head)
    """

    detector_name: str = "caregnn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_relations: int = 2,
        sim_threshold: float = 0.5,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if Data is None:
            msg = "torch_geometric is required for CAREGNNDetector"
            raise ImportError(msg)

        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.filter_weights: dict[int, Tensor] = {}

        # First layer
        self.layers.append(
            CARELayer(in_channels, hidden_channels, num_relations, sim_threshold)
        )

        # Second layer
        self.layers.append(
            CARELayer(hidden_channels, hidden_channels, num_relations, sim_threshold)
        )

        # Classification head
        self.head = nn.Linear(hidden_channels, 1)

    def forward_with_embedding(
        self, graph: Data, edge_type: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[B], node_embedding[N, H]).

        Embeddings are for ALL nodes (adapters need full graph for neighbor
        lookups). Logits are for target nodes only (mask-selected).
        """
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index

        # CARE message passing — produces embeddings for all N nodes
        for module in self.layers:
            layer = cast(CARELayer, module)
            x = layer(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        collected: dict[int, list[Tensor]] = {}
        for module in self.layers:
            layer = cast(CARELayer, module)
            for node_id, values in layer.last_filter_weights.items():
                collected.setdefault(node_id, []).append(values)
        self.filter_weights = {
            node_id: torch.cat(values) for node_id, values in collected.items()
        }

        # Full embeddings: [N, hidden]
        full_embedding = x

        # Select target nodes for logits only
        target_mask = self._get_target_mask(graph)
        target_emb = full_embedding[target_mask] if target_mask is not None else full_embedding

        # Classification head: [B, hidden] -> [B]
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
