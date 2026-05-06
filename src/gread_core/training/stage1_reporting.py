from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any

from gread_core.evaluation.detection import compute_all_detection_metrics
from gread_core.losses.supervised import supervised_loss

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def resolve_stage1_split_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.get("data", {})
    split_cfg = data_cfg.get("split", {}) if isinstance(data_cfg, dict) else {}
    raw_ratios = split_cfg.get("ratios", (0.7, 0.1, 0.2))
    if not isinstance(raw_ratios, (list, tuple)) or len(raw_ratios) != 3:
        raise ValueError("data.split.ratios must contain exactly 3 values")
    ratios = tuple(float(value) for value in raw_ratios)
    return {
        "ratios": list(ratios),
        "stratified": bool(split_cfg.get("stratified", False)),
    }


def apply_configured_split(data: Data, config: dict[str, Any], seed: int) -> Data:
    from gread_core.data.splits import generate_masks, stratified_split

    split_cfg = resolve_stage1_split_config(config)
    ratios = tuple(split_cfg["ratios"])
    if split_cfg["stratified"] and hasattr(data, "y") and data.y is not None:
        return stratified_split(data, seed=seed, ratios=ratios)
    return generate_masks(data, seed=seed, ratios=ratios)


def _make_mask_view(data: Data, split_name: str) -> Data:
    view = data.clone()
    if split_name == "train":
        view.test_mask = None
        view.val_mask = None
    elif split_name == "val":
        view.test_mask = None
    elif split_name != "test":
        raise ValueError(f"Unsupported split: {split_name}")
    return view


def evaluate_detector_split(
    detector: Any,
    data: Data,
    split_name: str,
    threshold: float = 0.5,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    mask = getattr(data, f"{split_name}_mask")
    node_indices = mask.nonzero(as_tuple=True)[0]
    labels = data.y[mask]
    detector.eval()
    with torch.no_grad():
        logits, embeddings = detector.forward_with_embedding(_make_mask_view(data, split_name))
        scores = torch.sigmoid(logits)
        preds = (scores >= threshold).long()
        loss_value = float(supervised_loss(logits, labels).item()) if labels.numel() > 0 else 0.0
    if labels.numel() == 0:
        metrics = {
            "auc": 0.0,
            "auroc": 0.0,
            "auprc": 0.0,
            "f1": 0.0,
            "f1_macro": 0.0,
            "g_means": 0.0,
            "accuracy": 0.0,
        }
    else:
        metrics = compute_all_detection_metrics(
            labels.detach().cpu().numpy(),
            scores.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
        )
        metrics["accuracy"] = float((preds == labels).float().mean().item())
    metrics["loss"] = loss_value
    metrics["num_nodes"] = int(node_indices.numel())
    result = {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "node_indices": node_indices.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),
        "scores": scores.detach().cpu().numpy(),
        "predictions": preds.detach().cpu().numpy(),
        "logits": logits.detach().cpu().numpy(),
    }
    if include_embeddings:
        result["embeddings"] = embeddings.detach().cpu().numpy()
    return result


def _balanced_sample_indices(labels: IntArray, max_points: int, seed: int) -> IntArray:
    total = labels.shape[0]
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    pos_idx = np.asarray(np.flatnonzero(labels == 1), dtype=np.int64)
    neg_idx = np.asarray(np.flatnonzero(labels == 0), dtype=np.int64)
    if pos_idx.size == 0 or neg_idx.size == 0:
        return np.asarray(
            np.sort(rng.choice(total, size=max_points, replace=False)),
            dtype=np.int64,
        )
    pos_take = min(pos_idx.size, max_points // 2)
    neg_take = min(neg_idx.size, max_points - pos_take)
    if pos_take + neg_take < max_points:
        remainder = max_points - pos_take - neg_take
        extra_pool = neg_idx if neg_idx.size > neg_take else pos_idx
        extra_used = neg_take if extra_pool is neg_idx else pos_take
        extra_take = min(remainder, extra_pool.size - extra_used)
        if extra_take > 0:
            chosen_extra = rng.choice(
                extra_pool,
                size=extra_take,
                replace=False,
            )
        else:
            chosen_extra = np.empty(0, dtype=np.int64)
    else:
        chosen_extra = np.empty(0, dtype=np.int64)
    chosen_pos = rng.choice(pos_idx, size=pos_take, replace=False)
    chosen_neg = rng.choice(neg_idx, size=neg_take, replace=False)
    combined = np.concatenate([chosen_pos, chosen_neg, chosen_extra])
    if combined.shape[0] < max_points:
        remaining_pool = np.setdiff1d(np.arange(total), combined, assume_unique=False)
        fill = rng.choice(remaining_pool, size=max_points - combined.shape[0], replace=False)
        combined = np.concatenate([combined, fill])
    return np.asarray(np.sort(combined), dtype=np.int64)


def _pca_init(x: FloatArray) -> FloatArray:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    y = centered @ basis
    if y.shape[1] < 2:
        pad = np.zeros((y.shape[0], 2 - y.shape[1]), dtype=y.dtype)
        y = np.concatenate([y, pad], axis=1)
    scale = np.std(y)
    if scale > 0:
        y = y / scale
    return np.asarray(y * 1e-4, dtype=np.float64)


def _entropy_and_probabilities(distances: FloatArray, beta: float) -> tuple[float, FloatArray]:
    probs = np.exp(-distances * beta)
    total = probs.sum()
    if total <= 1e-12:
        uniform = np.full_like(distances, 1.0 / max(1, distances.shape[0]), dtype=np.float64)
        return 0.0, uniform
    entropy = np.log(total) + beta * float(np.sum(distances * probs) / total)
    return float(entropy), np.asarray(probs / total, dtype=np.float64)


def _joint_probabilities(x: FloatArray, perplexity: float) -> FloatArray:
    from scipy.spatial.distance import pdist, squareform

    n = x.shape[0]
    distances = np.asarray(squareform(pdist(x, metric="sqeuclidean")), dtype=np.float64)
    target_entropy = np.log(perplexity)
    conditional = np.zeros((n, n), dtype=np.float64)
    for idx in range(n):
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        row_distances = distances[idx, mask]
        beta = 1.0
        beta_min: float | None = None
        beta_max: float | None = None
        for _ in range(60):
            entropy, row_probabilities = _entropy_and_probabilities(row_distances, beta)
            diff = entropy - target_entropy
            if abs(diff) <= 1e-5:
                break
            if diff > 0.0:
                beta_min = beta
                beta = beta * 2.0 if beta_max is None else (beta + beta_max) * 0.5
            else:
                beta_max = beta
                beta = beta * 0.5 if beta_min is None else (beta + beta_min) * 0.5
        conditional[idx, mask] = row_probabilities
    joint = (conditional + conditional.T) / (2.0 * n)
    return np.asarray(np.maximum(joint, 1e-12), dtype=np.float64)


def compute_tsne(
    x: FloatArray,
    perplexity: float,
    iterations: int,
    seed: int,
) -> FloatArray:
    from scipy.spatial.distance import pdist, squareform

    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = _pca_init(x)
    rng = np.random.default_rng(seed)
    y += rng.normal(scale=1e-4, size=y.shape)
    p = _joint_probabilities(x, perplexity)
    gains = np.ones_like(y)
    velocity = np.zeros_like(y)
    learning_rate = max(50.0, float(x.shape[0]) / 12.0)
    for iteration in range(iterations):
        distances = np.asarray(squareform(pdist(y, metric="sqeuclidean")), dtype=np.float64)
        numerator = 1.0 / (1.0 + distances)
        np.fill_diagonal(numerator, 0.0)
        denominator = numerator.sum()
        if denominator <= 0.0:
            break
        q = np.maximum(numerator / denominator, 1e-12)
        exaggeration = 4.0 if iteration < 100 else 1.0
        pq = ((p * exaggeration) - q) * numerator
        grad = 4.0 * ((pq.sum(axis=1, keepdims=True) * y) - (pq @ y))
        momentum = 0.5 if iteration < 100 else 0.8
        same_direction = np.sign(grad) == np.sign(velocity)
        gains = np.where(same_direction, gains * 0.8, gains + 0.2)
        gains = np.clip(gains, 0.01, None)
        velocity = momentum * velocity - learning_rate * gains * grad
        y = y + velocity
        y = y - y.mean(axis=0, keepdims=True)
    return np.asarray(y, dtype=np.float64)


def _normalize_points(points: FloatArray, width: int, height: int, margin: int) -> FloatArray:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    normalized = (points - mins) / spans
    normalized[:, 0] = margin + normalized[:, 0] * (width - 2 * margin)
    normalized[:, 1] = margin + normalized[:, 1] * (height - 2 * margin)
    normalized[:, 1] = height - normalized[:, 1]
    return np.asarray(normalized, dtype=np.float64)


def save_tsne_plot(
    output_dir: str | Path,
    embeddings: FloatArray,
    labels: IntArray,
    scores: FloatArray,
    node_indices: IntArray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    stage_cfg = config.get("stage1", {})
    tsne_cfg = stage_cfg.get("tsne", {})
    enabled = bool(tsne_cfg.get("enabled", True))
    if not enabled:
        return {"enabled": False}
    from PIL import Image, ImageDraw

    max_points = int(tsne_cfg.get("max_points", 1000))
    perplexity = float(tsne_cfg.get("perplexity", 30.0))
    iterations = int(tsne_cfg.get("iterations", 250))
    sample_idx = _balanced_sample_indices(labels, max_points=max_points, seed=seed)
    sampled_embeddings = embeddings[sample_idx]
    sampled_labels = labels[sample_idx]
    sampled_scores = scores[sample_idx]
    sampled_nodes = node_indices[sample_idx]
    effective_perplexity = min(perplexity, max(5.0, float(sampled_embeddings.shape[0] - 1) / 3.0))
    points = compute_tsne(
        sampled_embeddings,
        perplexity=effective_perplexity,
        iterations=iterations,
        seed=seed,
    )
    width = 960
    height = 720
    margin = 60
    canvas_points = _normalize_points(points, width=width, height=height, margin=margin)
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (margin, margin, width - margin, height - margin),
        outline=(180, 180, 180),
        width=1,
    )
    colors = {0: (37, 99, 235), 1: (220, 38, 38)}
    for idx, (x_pos, y_pos) in enumerate(canvas_points):
        label = int(sampled_labels[idx])
        score = float(sampled_scores[idx])
        radius = 2 if score < 0.5 else 3
        color = colors.get(label, (107, 114, 128))
        draw.ellipse(
            (x_pos - radius, y_pos - radius, x_pos + radius, y_pos + radius),
            fill=color,
            outline=color,
        )
    draw.text((margin, 20), "Stage1 t-SNE (test split)", fill=(0, 0, 0))
    draw.text(
        (margin, 40),
        f"points={sampled_embeddings.shape[0]} perplexity={effective_perplexity:.1f}",
        fill=(0, 0, 0),
    )
    draw.rectangle((width - 200, 24, width - 188, 36), fill=colors[0], outline=colors[0])
    draw.text((width - 182, 20), "benign", fill=(0, 0, 0))
    draw.rectangle((width - 200, 48, width - 188, 60), fill=colors[1], outline=colors[1])
    draw.text((width - 182, 44), "fraud", fill=(0, 0, 0))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    image_path = output_root / "tsne_test.png"
    coords_path = output_root / "tsne_test.npz"
    image.save(image_path)
    np.savez(
        coords_path,
        node_indices=sampled_nodes,
        labels=sampled_labels,
        scores=sampled_scores,
        coords=points,
    )
    return {
        "enabled": True,
        "image_path": str(image_path),
        "coords_path": str(coords_path),
        "num_points": int(sampled_embeddings.shape[0]),
        "perplexity": float(effective_perplexity),
        "iterations": iterations,
    }


def save_stage1_artifacts(
    output_dir: str | Path,
    detector: Any,
    data: Data,
    config: dict[str, Any],
    dataset: str,
    detector_name: str,
    seed: int,
) -> Path:
    stage_cfg = config.get("stage1", {})
    threshold = float(stage_cfg.get("decision_threshold", 0.5))
    stage1_dir = Path(output_dir) / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    split_results = {
        split: evaluate_detector_split(
            detector,
            data,
            split,
            threshold=threshold,
            include_embeddings=(split == "test"),
        )
        for split in ("train", "val", "test")
    }
    summary: dict[str, Any] = {
        "dataset": dataset,
        "detector": detector_name,
        "seed": seed,
        "split": resolve_stage1_split_config(config),
        "training": dict(getattr(detector, "stage1_training_info", {})),
        "decision_threshold": threshold,
        "splits": {
            split: result["metrics"]
            for split, result in split_results.items()
        },
    }
    for split_name, result in split_results.items():
        np.savez(
            stage1_dir / f"predictions_{split_name}.npz",
            node_indices=result["node_indices"],
            labels=result["labels"],
            scores=result["scores"],
            predictions=result["predictions"],
            logits=result["logits"],
        )
    test_result = split_results["test"]
    test_embeddings = test_result["embeddings"][test_result["node_indices"]]
    summary["tsne"] = save_tsne_plot(
        stage1_dir,
        embeddings=test_embeddings,
        labels=test_result["labels"],
        scores=test_result["scores"],
        node_indices=test_result["node_indices"],
        config=config,
        seed=seed,
    )
    summary_path = stage1_dir / "metrics_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary_path
