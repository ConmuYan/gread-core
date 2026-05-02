"""GCN and GAT base detector implementations for GReaD-Core.

These are standard PyG GNN detectors that implement the DetectorProtocol.
They serve as the base detector for the evidence-conditioned reasoner.

Research constraints:
- No LLM imports allowed in this module.
- No hidden constants - all hyperparameters are config-driven.
- forward_with_embedding() is the ONLY entry point.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, GCNConv
except ImportError:
    Data = Any
    GCNConv = None
    GATConv = None


class GCNDetector(nn.Module):
    """Graph Convolutional Network base detector.

    Architecture:
        GCNConv(F, hidden) -> ReLU -> GCNConv(hidden, hidden) -> Linear(hidden, 1)

    Tensor shapes:
        x:              [N, F]   node features
        edge_index:     [2, E]   edge indices
        logit:          [B]      classification logits
        embedding:      [B, hidden]  node embeddings (pre-head)
    """

    detector_name: str = "gcn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if GCNConv is None:
            msg = "torch_geometric is required for GCNDetector"
            raise ImportError(msg)

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Classification head
        self.head = nn.Linear(hidden_channels, 1)

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[B], node_embedding[N, H]).

        Embeddings are for ALL nodes (adapters need full graph for neighbor
        lookups). Logits are for target nodes only (mask-selected).
        """
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index

        # GCN message passing — produces embeddings for all N nodes
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

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


class GATDetector(nn.Module):
    """Graph Attention Network base detector.

    Architecture:
        GATConv(F, hidden, heads=attention_heads) -> ReLU ->
        GATConv(hidden*heads, hidden, heads=1) -> Linear(hidden, 1)

    Tensor shapes:
        x:              [N, F]   node features
        edge_index:     [2, E]   edge indices
        logit:          [B]      classification logits
        embedding:      [B, hidden]  node embeddings (pre-head)
    """

    detector_name: str = "gat"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if GATConv is None:
            msg = "torch_geometric is required for GATDetector"
            raise ImportError(msg)

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First layer with multiple attention heads
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=attention_heads, dropout=dropout)
        )

        # Hidden layers: input is hidden_channels * attention_heads from previous layer
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(
                    hidden_channels * attention_heads,
                    hidden_channels,
                    heads=1,
                    dropout=dropout,
                )
            )

        # Classification head
        self.head = nn.Linear(hidden_channels, 1)

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[B], node_embedding[N, H]).

        Embeddings are for ALL nodes. Logits are for target nodes only.
        """
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index

        # GAT message passing — produces embeddings for all N nodes
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Full embeddings: [N, hidden]
        full_embedding = x

        # Select target nodes for logits only
        target_mask = self._get_target_mask(graph)
        target_emb = full_embedding[target_mask] if target_mask is not None else full_embedding

        # Classification head: [B, hidden] -> [B]
        logit = self.head(target_emb).squeeze(-1)

        return logit, full_embedding

    def _get_target_mask(self, graph: Data) -> Tensor | None:
        """Select target nodes based on available masks."""
        if hasattr(graph, "test_mask") and graph.test_mask is not None and graph.test_mask.any():
            return graph.test_mask  # type: ignore[no-any-return]
        if hasattr(graph, "val_mask") and graph.val_mask is not None and graph.val_mask.any():
            return graph.val_mask  # type: ignore[no-any-return]
        if hasattr(graph, "train_mask") and graph.train_mask is not None and graph.train_mask.any():
            return graph.train_mask  # type: ignore[no-any-return]
        return None
