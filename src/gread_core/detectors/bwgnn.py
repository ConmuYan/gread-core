"""BWGNN (Beta Wavelet GNN) detector implementation for GReaD-Core.

Uses polynomial spectral filters as an approximation of Bessel wavelet filters.

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
from torch.nn import ModuleList

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import degree
except ImportError:
    Data = Any
    degree = None


class BWGNNConv(nn.Module):
    """Polynomial spectral filter convolution.

    Approximates a Bessel wavelet filter using a learnable polynomial:
        out = sum_i coeffs_i * A_hat^i * x

    where A_hat is the symmetric-normalised adjacency.
    """

    def __init__(self, in_channels: int, out_channels: int, num_coeffs: int = 3) -> None:
        super().__init__()
        self.num_coeffs = num_coeffs
        self.linears = ModuleList(
            [nn.Linear(in_channels, out_channels) for _ in range(num_coeffs)]
        )
        self.coeffs = nn.Parameter(torch.ones(num_coeffs))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Apply polynomial spectral filter.

        Args:
            x: [N, F] node features.
            edge_index: [2, E] edge indices.

        Returns:
            [N, out_channels] filtered node features.
        """
        row, col = edge_index
        num_nodes = x.size(0)

        # Compute symmetric-normalised adjacency: D^{-1/2} A D^{-1/2}
        deg = degree(row, num_nodes, dtype=x.dtype).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Power iteration: A_hat^i * x
        out: Tensor = self.coeffs[0] * self.linears[0](x)
        x_prop = x  # A_hat^0 * x = x
        for i in range(1, self.num_coeffs):
            # Sparse propagation: x_prop = A_hat * x_prop
            x_prop = self._sparse_spmm(row, col, norm, x_prop, num_nodes)
            out = out + self.coeffs[i] * self.linears[i](x_prop)

        return out

    @staticmethod
    def _sparse_spmm(
        row: Tensor, col: Tensor, norm: Tensor, x: Tensor, num_nodes: int
    ) -> Tensor:
        """Sparse matrix multiplication: D^{-1/2} A D^{-1/2} * x."""
        out = torch.zeros_like(x)
        weighted = norm.unsqueeze(-1) * x[col]  # [E, D]
        out.index_add_(0, row, weighted)
        return out


class BWGNNDetector(nn.Module):
    """Beta Wavelet GNN detector.

    Architecture:
        BWGNNConv(F, hidden, num_coeffs) -> ReLU ->
        BWGNNConv(hidden, hidden, num_coeffs) -> Linear(hidden, 1)

    Tensor shapes:
        x:              [N, F]   node features
        edge_index:     [2, E]   edge indices
        logit:          [B]      classification logits
        embedding:      [B, hidden]  node embeddings (pre-head)
    """

    detector_name: str = "bwgnn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_coeffs: int = 3,
    ) -> None:
        super().__init__()
        if degree is None:
            msg = "torch_geometric is required for BWGNNDetector"
            raise ImportError(msg)

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(BWGNNConv(in_channels, hidden_channels, num_coeffs))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(BWGNNConv(hidden_channels, hidden_channels, num_coeffs))

        # Classification head
        self.head = nn.Linear(hidden_channels, 1)

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[B], node_embedding[N, H]).

        Embeddings are for ALL nodes (adapters need full graph for neighbor
        lookups). Logits are for target nodes only (mask-selected).
        """
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index

        # BWGNN message passing — produces embeddings for all N nodes
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
