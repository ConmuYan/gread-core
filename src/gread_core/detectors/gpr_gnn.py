"""GPR-GNN detector (Generalized PageRank GNN).

Reference: Chien et al., "Adaptive Universal Generalized PageRank Graph
           Neural Network", ICLR 2021.

Key idea: decouple feature transformation from propagation, and learn a
per-hop coefficient gamma_k that mixes h_0, h_1=Ah_0, ..., h_K=A^K h_0.
The learned gamma_k vector is a natural *multi-hop evidence weight* and
plays well with GReaD-Core's signed-evidence-mask abstraction.

Evidence axis:
    self.gammas : Parameter of shape [K+1]
    Positive gammas = low-pass (homophilic) contribution at hop k;
    negative gammas = high-pass (heterophilic) contribution at hop k.
    Adapters can read `self.gammas.detach()` to emit signed hop-wise evidence.

Research constraints:
- No LLM imports in this module.
- Propagation is parameter-free (uses PyG LGConv), so only the MLP and
  `gammas` vector are learned — gamma_k is therefore a clean attribution
  signal rather than an entangled weight.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import LGConv
except ImportError:
    Data = Any
    LGConv = None


class GPRGNNDetector(nn.Module):
    """Generalized PageRank GNN base detector.

    Architecture:
        h_0 = MLP(x)                     # feature transformation
        h_k = A_norm @ h_{k-1}           # parameter-free propagation (LGConv)
        z   = sum_{k=0..K} gamma_k * h_k # learnable per-hop mixing
        logit = Linear(z, 1)

    Tensor shapes:
        x:              [N, F]
        edge_index:     [2, E]
        logit:          [N]
        embedding:      [N, hidden]
    """

    detector_name: str = "gpr_gnn"

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_hops: int = 10,
        dropout: float = 0.5,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if LGConv is None:
            msg = "torch_geometric is required for GPRGNNDetector"
            raise ImportError(msg)
        if num_hops < 1:
            msg = "num_hops must be >= 1"
            raise ValueError(msg)

        self.num_hops = num_hops
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Parameter-free propagation step (GCN-style symmetric normalization).
        self.prop = LGConv()

        # Learnable per-hop weights; init with PPR-style decay gamma_k = alpha*(1-alpha)^k
        # (standard GPR-GNN init from the paper).
        gammas = torch.tensor(
            [alpha * (1.0 - alpha) ** k for k in range(num_hops + 1)],
            dtype=torch.float32,
        )
        gammas[-1] = (1.0 - alpha) ** num_hops  # ensure conservation of mass
        self.gammas = nn.Parameter(gammas)

        self.head = nn.Linear(hidden_channels, 1)

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[N], node_embedding[N, H])."""
        x: Tensor = graph.x
        edge_index: Tensor = graph.edge_index

        # Feature transformation with dropout.
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.mlp(h)  # h_0 : [N, hidden]

        # Generalized PageRank propagation.
        out = self.gammas[0] * h
        for k in range(1, self.num_hops + 1):
            h = self.prop(h, edge_index)
            out = out + self.gammas[k] * h

        embedding = out  # [N, hidden]
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
