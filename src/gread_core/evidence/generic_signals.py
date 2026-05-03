"""Generic evidence signal computation for graph nodes.

These signals are detector-agnostic and derived purely from graph structure,
node features, and class labels. They populate the generic fields of the
ReasoningChannel in a MinimalEvidencePackage.

CRITICAL: prediction_score must NEVER be used as input to any signal here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gread_core.evidence.quantization import (
    quantize_consistency,
    quantize_degree_level,
    quantize_discrepancy,
    quantize_uncertainty,
)


def _safe_div(num: float, den: float) -> float:
    """Safe division returning 0.0 when denominator is zero."""
    if den == 0.0:
        return 0.0
    return num / den


def compute_degree_level(
    degrees: Tensor,
    quantization_thresholds: dict[str, float] | None = None,
    burst_percentile: float = 0.99,
) -> list[str]:
    """Compute degree level for each node.

    Normalizes degrees to [0, 1] relative to max degree, then quantizes.

    Args:
        degrees: Integer tensor of node degrees, shape [N].
        quantization_thresholds: Optional custom thresholds for quantization.
        burst_percentile: Percentile above which a node is "burst".

    Returns:
        List of degree level strings: isolated/low/medium/high/burst.
    """
    if degrees.numel() == 0:
        return []

    max_deg = degrees.max().item()
    if max_deg == 0:
        return ["isolated"] * degrees.numel()

    normalized = (degrees.float() / max_deg).tolist()
    return quantize_degree_level(
        normalized,
        thresholds=quantization_thresholds,
        burst_percentile=burst_percentile,
    )


def compute_neighbor_consistency(
    labels: Tensor,
    adj: Tensor,
    quantization_thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Compute neighbor label consistency for each node.

    Consistency = fraction of neighbors whose label matches the node's own label.
    Nodes with no neighbors get NaN → "unavailable".

    Args:
        labels: Integer label tensor, shape [N].
        adj: Sparse adjacency matrix (scipy sparse or torch sparse), shape [N, N].
            Supports scipy.sparse (CSR/CSC) and torch sparse_coo_tensor.
        quantization_thresholds: Optional custom thresholds.

    Returns:
        List of consistency level strings: unavailable/low/medium/high.
    """
    n = labels.shape[0]

    # Vectorized: use sparse matrix structure directly
    if adj.is_sparse:
        if not adj.is_coalesced():
            adj = adj.coalesce()
        indices = adj.indices()  # [2, E]
        row, col = indices[0], indices[1]

        # For each edge, check if neighbor label matches node label
        node_labels = labels[row]
        neighbor_labels = labels[col]
        matches = (node_labels == neighbor_labels).float()

        # Sum matches and degrees per node using scatter
        from torch_scatter import scatter_add

        match_sum = scatter_add(matches, row, dim=0, dim_size=n)
        degree = scatter_add(torch.ones_like(matches), row, dim=0, dim_size=n)

        # Consistency = match_sum / degree (NaN where degree=0)
        consistency_values = torch.where(
            degree > 0, match_sum / degree, torch.tensor(float("nan"))
        )
        values = consistency_values.tolist()
    else:
        # Dense fallback (for small graphs)
        values = []
        for i in range(n):
            row_dense = adj[i]
            neighbors = row_dense.nonzero(as_tuple=True)[0]
            if len(neighbors) == 0:
                values.append(float("nan"))
                continue
            node_label = labels[i].item()
            match_count = sum(1 for nb in neighbors if labels[nb].item() == node_label)
            values.append(_safe_div(float(match_count), float(len(neighbors))))

    return quantize_consistency(values, thresholds=quantization_thresholds)


def compute_feature_neighbor_discrepancy(
    x: Tensor,
    adj: Tensor,
    quantization_thresholds: dict[str, float] | None = None,
) -> list[str]:
    """Compute feature discrepancy between each node and its neighbors.

    Discrepancy = 1 - mean(cosine_similarity(node, neighbor)) over all neighbors.
    Higher values mean the node's features diverge more from its neighborhood.
    Nodes with no neighbors or zero-norm features get NaN → "unavailable".

    Args:
        x: Node feature matrix, shape [N, F].
        adj: Sparse adjacency matrix, shape [N, N].
        quantization_thresholds: Optional custom thresholds.

    Returns:
        List of discrepancy level strings: unavailable/low/medium/high.
    """
    n = x.shape[0]

    # Vectorized: use sparse matrix structure directly
    if adj.is_sparse:
        if not adj.is_coalesced():
            adj = adj.coalesce()
        indices = adj.indices()  # [2, E]
        row, col = indices[0], indices[1]

        # Normalize features for cosine similarity
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x_normed = x / norms

        # Compute cosine similarity for each edge: dot(x[row], x[col])
        cos_sims = (x_normed[row] * x_normed[col]).sum(dim=-1)

        # Handle zero-norm nodes: set their edges to NaN
        zero_norm_mask = (x.norm(dim=-1) == 0)
        if zero_norm_mask.any():
            edge_nan = zero_norm_mask[row]
            cos_sims = cos_sims.clone()
            cos_sims[edge_nan] = float("nan")

        from torch_scatter import scatter_add

        # Sum similarities and count valid edges per node
        valid_mask = ~torch.isnan(cos_sims)
        cos_sims_clean = torch.where(valid_mask, cos_sims, torch.zeros_like(cos_sims))
        valid_count = valid_mask.float()

        sim_sum = scatter_add(cos_sims_clean, row, dim=0, dim_size=n)
        edge_count = scatter_add(valid_count, row, dim=0, dim_size=n)

        # Mean similarity -> discrepancy
        mean_sim = torch.where(
            edge_count > 0, sim_sum / edge_count, torch.tensor(float("nan"))
        )
        discrepancy = (1.0 - mean_sim).clamp(0.0, 1.0)

        # Nodes with no neighbors -> NaN
        has_neighbors = edge_count > 0
        discrepancy = torch.where(has_neighbors, discrepancy, torch.tensor(float("nan")))
        values = discrepancy.tolist()
    else:
        # Dense fallback (for small graphs)
        values = []
        for i in range(n):
            row_dense = adj[i]
            neighbors = row_dense.nonzero(as_tuple=True)[0]
            if len(neighbors) == 0:
                values.append(float("nan"))
                continue
            node_feat = x[i]
            node_norm = node_feat.norm().item()
            if node_norm == 0.0:
                values.append(float("nan"))
                continue
            similarities: list[float] = []
            for nb in neighbors:
                nb_feat = x[nb]
                nb_norm = nb_feat.norm().item()
                if nb_norm == 0.0:
                    continue
                cos_sim = torch.dot(node_feat, nb_feat).item() / (node_norm * nb_norm)
                similarities.append(cos_sim)
            if not similarities:
                values.append(float("nan"))
            else:
                avg_sim: float = sum(similarities) / len(similarities)
                values.append(max(0.0, min(1.0, 1.0 - avg_sim)))

    return quantize_discrepancy(values, thresholds=quantization_thresholds)


def compute_uncertainty(
    logits: Tensor,
    quantization_thresholds: dict[str, float] | None = None,
) -> tuple[list[str], list[float]]:
    """Compute prediction uncertainty from logits.

    Uncertainty is derived from the entropy of the softmax distribution.
    For binary classification (2-class logits), normalized entropy is in [0, 1].

    Args:
        logits: Model output logits, shape [N, C] where C >= 2.
        quantization_thresholds: Optional custom thresholds.

    Returns:
        Tuple of (uncertainty_levels, raw_uncertainty_values).
        Levels are low/medium/high.
    """
    if logits.numel() == 0:
        return [], []

    # Handle binary case: logits shape [N] -> [N, 2]
    if logits.dim() == 1:
        logits = torch.stack([logits, -logits], dim=-1)

    # Softmax probabilities
    probs = torch.softmax(logits, dim=-1)

    # Entropy computation
    # H = -sum(p * log(p)) / log(C)  → normalized to [0, 1]
    num_classes = probs.shape[-1]
    max_entropy = math.log(num_classes) if num_classes > 1 else 1.0

    # Clamp probs to avoid log(0)
    probs_clamped = probs.clamp(min=1e-10)
    entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
    normalized_entropy = (entropy / max_entropy).clamp(0.0, 1.0)

    raw_values = normalized_entropy.tolist()
    levels = quantize_uncertainty(raw_values, thresholds=quantization_thresholds)
    return levels, raw_values


def _get_neighbors(adj: Tensor, node_idx: int) -> list[int]:
    """Extract neighbor indices for a node from an adjacency matrix.

    Supports:
    - torch sparse_coo_tensor (2D)
    - scipy sparse matrices (CSR/CSC)

    Args:
        adj: Adjacency matrix.
        node_idx: Index of the node.

    Returns:
        Sorted list of neighbor indices.
    """
    # Try torch sparse
    if adj.is_sparse:
        # Convert to COO if needed
        if not adj.is_coalesced():
            adj = adj.coalesce()
        indices = adj.indices()
        row_mask = indices[0] == node_idx
        neighbor_indices = indices[1][row_mask]
        return sorted(neighbor_indices.tolist())

    # Dense fallback
    row = adj[node_idx]
    neighbor_indices = row.nonzero(as_tuple=True)[0]
    return sorted(neighbor_indices.tolist())
