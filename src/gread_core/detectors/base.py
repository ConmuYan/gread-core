"""DetectorProtocol: the abstract interface for base detectors.

All base detectors must implement forward_with_embedding(), which returns
both the classification logit and the node embedding. This is the ONLY
entry point for detector forward passes — there is no separate forward()
that skips the embedding.

Research constraint: prediction_score from forward_with_embedding is
calibration-only. It must never enter LLM prompts or evidence targets.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any


@runtime_checkable
class DetectorProtocol(Protocol):
    """Protocol for base detectors in GReaD-Core.

    Every base detector must expose:
    - detector_name: str — unique identifier for this detector
    - forward_with_embedding(graph) -> (logit, embedding) — the ONLY forward entry point

    The logit is the raw classification score (before sigmoid). The embedding
    is the node-level representation used by the downstream reasoner.

    Tensor shapes:
        graph.x:            [N, F]   node features
        graph.edge_index:   [2, E]   edge indices
        logit:              [B]      classification logit for target nodes
        embedding:          [B, H]   node embeddings for target nodes
    """

    detector_name: str

    def forward_with_embedding(self, graph: Data) -> tuple[Tensor, Tensor]:
        """Forward pass returning (base_logit[B], node_embedding[B, H]).

        Args:
            graph: PyG Data object with x, edge_index, and optionally
                   train_mask/val_mask/test_mask for specifying target nodes.

        Returns:
            Tuple of:
            - base_logit: [B] tensor of classification logits
            - node_embedding: [B, H] tensor of node embeddings

        Where B is the batch size (number of target nodes).
        """
        ...
