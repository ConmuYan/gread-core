"""H2GCN detector (Heterophilic 2-hop GCN).

Reference: Zhu et al., "Beyond Homophily in Graph Neural Networks: Current
           Limitations and Effective Designs", NeurIPS 2020.

Key ideas (exactly three, per the paper):
    D1. Ego- and neighbor-embedding separation.
    D2. Higher-order neighborhoods (1-hop AND 2-hop, kept separate).
    D3. Combination of intermediate representations from every round.

Evidence axis for GReaD-Core:
    H2GCN produces distinct (1-hop, 2-hop) representations per round.
    Adapters can read the per-hop contributions as separate evidence
    channels -- this is the cleanest natively-signed evidence signal we
    have in the roster (hop-1 vs hop-2 disagreement is the direct
    heterophily marker).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import (
        add_self_loops,
        degree,
        remove_self_loops,
        to_edge_index,
        to_torch_csr_tensor,
    )
except ImportError:  # pragma: no cover - import guard only
    Data = Any
    add_self_loops = None
    degree = None
    remove_self_loops = None
    to_edge_index = None
    to_torch_csr_tensor = None


def _sym_norm_propagate(x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
    """Symmetric-normalized (D^-1/2 A D^-1/2) propagation -- no self-loops.

    H2GCN explicitly EXCLUDES self-loops in the neighbor aggregator (this is
    what enforces ego/neighbor separation).
    """
    ei, _ = remove_self_loops(edge_index)
    if ei.numel() == 0:
        return torch.zeros_like(x)
    row, col = ei[0], ei[1]
    deg = degree(col, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    # Build a normalized sparse adjacency and use a fused sparse matmul. This
    # avoids ever materializing the [E, hidden] message tensor (and its
    # autograd-tracked counterpart in backward), which is critical on dense
    # graphs such as YelpChi where the 2-hop edge_index has tens of millions
    # of edges.
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([col, row], dim=0)
    adj = torch.sparse_coo_tensor(
        indices, norm, (num_nodes, num_nodes)
    ).coalesce()
    return cast(Tensor, torch.sparse.mm(adj, x))


def _two_hop_edge_index(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Compute strict 2-hop edge_index (A^2 with self-loops removed).

    NOTE: per H2GCN paper, the 2-hop adjacency is computed over the raw
    adjacency (no self-loops), and self-loops are stripped from A^2 too.

    The A @ A product can materialize tens of GB of intermediate storage on
    dense graphs such as YelpChi, so we always build A^2 on CPU and only move
    the resulting edge_index back to the original device.
    """
    ei, _ = remove_self_loops(edge_index)
    if ei.numel() == 0:
        return cast(Tensor, ei)
    original_device = ei.device
    ei_cpu = ei.detach().cpu()
    a = to_torch_csr_tensor(ei_cpu, size=(num_nodes, num_nodes))
    a2 = a @ a
    ei2, _ = to_edge_index(a2.to_sparse_coo().coalesce())
    ei2, _ = remove_self_loops(ei2)
    ei2 = ei2.to(original_device)
    return cast(Tensor, ei2)


class H2GCNDetector(nn.Module):
    """H2GCN base detector (ego / 1-hop / 2-hop separated, K rounds, concat).

    Architecture:
        h_0 = ReLU(Linear(x))                                     # ego init
        h_k = concat[ N_1(h_{k-1}),  N_2(h_{k-1}) ]               # separated hops
        z   = concat[ h_0, h_1, ..., h_K ]                        # multi-round combine
        logit = Linear(Dropout(z), 1)

    Tensor shapes:
        x:              [N, F]
        edge_index:     [2, E]
        logit:          [N]
        embedding:      [N, hidden * (1 + sum_{k=1..K} 2^k)]

    Args:
        num_rounds: number of propagation rounds K (paper uses 1 or 2).
    """

    detector_name: str = "h2gcn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_rounds: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if remove_self_loops is None:
            msg = "torch_geometric is required for H2GCNDetector"
            raise ImportError(msg)
        if num_rounds < 1:
            msg = "num_rounds must be >= 1"
            raise ValueError(msg)

        self.num_rounds = num_rounds
        self.dropout = dropout
        self.hidden_channels = hidden_channels

        # Ego embedding (D1).
        self.ego = nn.Linear(in_channels, hidden_channels)

        # After K rounds: h_k has dim hidden * 2^k, concat'd h_0..h_K yields
        # hidden * (1 + 2 + 4 + ... + 2^K) = hidden * (2^{K+1} - 1).
        concat_dim = hidden_channels * ((1 << (num_rounds + 1)) - 1)
        self.head = nn.Linear(concat_dim, 1)

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[N], node_embedding[N, concat_dim])."""
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index
        num_nodes = x.size(0)

        # Build strict 2-hop edges at most once per graph; cache across epochs.
        graph_fingerprint = (int(edge_index.data_ptr()), int(edge_index.shape[1]))
        cached = getattr(self, "_cached_edge_index_2", None)
        fingerprint = getattr(self, "_cached_edge_index_id", None)
        if (
            cached is None
            or fingerprint != graph_fingerprint
            or cached.device != edge_index.device
        ):
            cached = _two_hop_edge_index(edge_index, num_nodes)
            self._cached_edge_index_2 = cached
            self._cached_edge_index_id = graph_fingerprint
        edge_index_2 = cached

        # h_0 : ego embedding, [N, hidden]
        h = F.relu(self.ego(x))
        rounds = [h]

        for _ in range(self.num_rounds):
            h_prev = rounds[-1]
            # N_1 aggregation over 1-hop neighbors (no self-loops).
            agg1 = _sym_norm_propagate(h_prev, edge_index, num_nodes)
            # N_2 aggregation over strict 2-hop neighbors.
            if edge_index_2.numel() == 0:
                agg2 = torch.zeros_like(h_prev)
            else:
                agg2 = _sym_norm_propagate(h_prev, edge_index_2, num_nodes)
            h_new = torch.cat([agg1, agg2], dim=-1)  # [N, 2 * prev_dim]
            rounds.append(h_new)

        # D3: concat intermediate representations from every round.
        embedding = torch.cat(rounds, dim=-1)
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        target_mask = self._get_target_mask(graph)
        target_emb = embedding[target_mask] if target_mask is not None else embedding
        logit = self.head(target_emb).squeeze(-1)
        return logit, embedding

    def _get_target_mask(self, graph: Data) -> Tensor | None:
        if hasattr(graph, "test_mask") and graph.test_mask is not None and graph.test_mask.any():
            return graph.test_mask  # type: ignore[no-any-return]
        if hasattr(graph, "val_mask") and graph.val_mask is not None and graph.val_mask.any():
            return graph.val_mask  # type: ignore[no-any-return]
        if hasattr(graph, "train_mask") and graph.train_mask is not None and graph.train_mask.any():
            return graph.train_mask  # type: ignore[no-any-return]
        return None
