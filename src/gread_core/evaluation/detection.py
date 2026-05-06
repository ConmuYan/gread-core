"""Detection metrics: AUC, AUPRC, F1, Precision@K, Recall@K.

Pure numpy implementation -- no sklearn dependency.
All metrics are deterministic given the same input. No LLM dependencies.
"""

from __future__ import annotations

import numpy as np


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve using the trapezoidal rule.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.

    Returns:
        AUC score in [0, 1].

    Raises:
        ValueError: If y_true contains only one class.
    """
    if len(np.unique(y_true)) < 2:
        msg = "AUC requires at least two classes in y_true"
        raise ValueError(msg)

    # Sort by descending score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Trapezoidal rule: prepend (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc_val = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * 0.5))
    return auc_val


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.

    Returns:
        AUPRC score in [0, 1].
    """
    if len(np.unique(y_true)) < 2:
        if np.sum(y_true) == 0:
            return 0.0
        return 1.0

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)

    precision = tps / (tps + fps)
    recall = tps / np.sum(y_true)

    # Prepend (recall=0, precision=1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])

    auprc_val = float(np.sum(np.diff(recall) * precision[:-1]))
    return auprc_val


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F1 score from binary predictions.

    Args:
        y_true: Binary ground-truth labels.
        y_pred: Binary predicted labels.

    Returns:
        F1 score in [0, 1].
    """
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    positive_f1 = compute_f1(y_true, y_pred)
    negative_f1 = compute_f1(1 - y_true, 1 - y_pred)
    return float((positive_f1 + negative_f1) * 0.5)


def compute_g_means(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0.0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    return float(np.sqrt(sensitivity * specificity))


def compute_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute Precision@K: fraction of top-k scored items that are positive.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.
        k: Number of top items to consider.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0:
        return 0.0
    n = len(y_true)
    k = min(k, n)
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[top_k_idx]) / k)


def compute_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute Recall@K: fraction of positives captured in top-k scored items.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.
        k: Number of top items to consider.

    Returns:
        Recall@K in [0, 1].
    """
    total_positives = int(np.sum(y_true))
    if total_positives == 0 or k <= 0:
        return 0.0
    n = len(y_true)
    k = min(k, n)
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[top_k_idx]) / total_positives)


def compute_all_detection_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all detection metrics.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.
        y_pred: Binary predicted labels (thresholded).
        k_values: List of K values for Precision@K and Recall@K.
            Defaults to [10, 50, 100].

    Returns:
        Dict with metric names as keys.
    """
    if k_values is None:
        k_values = [10, 50, 100]

    try:
        auc = compute_auc(y_true, y_score)
    except ValueError:
        auc = 0.0
    metrics: dict[str, float] = {
        "auc": auc,
        "auroc": auc,
        "auprc": compute_auprc(y_true, y_score),
        "f1": compute_f1(y_true, y_pred),
        "f1_macro": compute_f1_macro(y_true, y_pred),
        "g_means": compute_g_means(y_true, y_pred),
    }

    for k in k_values:
        metrics[f"precision@{k}"] = compute_precision_at_k(y_true, y_score, k)
        metrics[f"recall@{k}"] = compute_recall_at_k(y_true, y_score, k)

    return metrics
