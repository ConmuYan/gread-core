"""Non-redundancy evaluation: nested logistic regression models.

Pure numpy implementation of logistic regression via gradient descent -- no sklearn.

Compares three nested models:
    Y ~ P                    (score only)
    Y ~ P + T               (score + risk type)
    Y ~ P + T + M           (score + risk type + evidence masks)

Improvement in AUC/AUPRC at each level demonstrates that each component
adds non-redundant information beyond what the score provides.

All computations are deterministic given the same input. No LLM dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    result: np.ndarray = 1.0 / (1.0 + np.exp(-z))
    return result


def _logistic_regression_fit(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    max_iter: int = 1000,
    seed: int = 42,
) -> tuple[list[float], float]:
    """Fit logistic regression via gradient descent.

    Args:
        x: [N, D] feature matrix.
        y: [N] binary labels.
        lr: Learning rate.
        max_iter: Max iterations.
        seed: Random seed.

    Returns:
        Tuple of (weights [D], bias scalar).
    """
    rng = np.random.RandomState(seed)
    n, d = x.shape
    w: Any = rng.normal(0, 0.01, size=d)
    b = 0.0

    for _ in range(max_iter):
        z = x @ w + b
        p = _sigmoid(z)
        grad_w = x.T @ (p - y) / n
        grad_b = float(np.mean(p - y))
        w = w - lr * grad_w
        b = b - lr * grad_b

    w_list: list[float] = [float(v) for v in w]
    return w_list, b


def _logistic_regression_predict(
    x: np.ndarray,
    w: list[float],
    b: float,
) -> np.ndarray:
    """Predict probabilities."""
    w_arr = np.array(w)
    return _sigmoid(x @ w_arr + b)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC, returning 0.5 if only one class present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    # Reuse detection module's AUC
    from gread_core.evaluation.detection import compute_auc
    return compute_auc(y_true, y_score)


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUPRC, returning 0.0 if only one class present."""
    if len(np.unique(y_true)) < 2:
        if np.sum(y_true) == 0:
            return 0.0
        return 1.0
    from gread_core.evaluation.detection import compute_auprc
    return compute_auprc(y_true, y_score)


def fit_score_only_model(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Fit logistic regression: Y ~ P (score only).

    Args:
        scores: [N] array of fraud scores.
        labels: [N] binary ground-truth labels.

    Returns:
        Dict with "auc", "auprc", "coef", "model_name".
    """
    x = scores.reshape(-1, 1)
    w, b = _logistic_regression_fit(x, labels)
    y_prob = _logistic_regression_predict(x, w, b)

    return {
        "model_name": "Y~P",
        "auc": _safe_auc(labels, y_prob),
        "auprc": _safe_auprc(labels, y_prob),
        "coef": list(w),
    }


def fit_score_type_model(
    scores: np.ndarray,
    types_encoded: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Fit logistic regression: Y ~ P + T (score + risk type).

    Args:
        scores: [N] array of fraud scores.
        types_encoded: [N, T] one-hot encoded risk types.
        labels: [N] binary ground-truth labels.

    Returns:
        Dict with "auc", "auprc", "coef", "model_name".
    """
    x = np.hstack([scores.reshape(-1, 1), types_encoded])
    w, b = _logistic_regression_fit(x, labels)
    y_prob = _logistic_regression_predict(x, w, b)

    return {
        "model_name": "Y~P+T",
        "auc": _safe_auc(labels, y_prob),
        "auprc": _safe_auprc(labels, y_prob),
        "coef": list(w),
    }


def fit_score_type_evidence_model(
    scores: np.ndarray,
    types_encoded: np.ndarray,
    evidence_features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Fit logistic regression: Y ~ P + T + M (score + type + evidence masks).

    Args:
        scores: [N] array of fraud scores.
        types_encoded: [N, T] one-hot encoded risk types.
        evidence_features: [N, E] evidence mask features.
        labels: [N] binary ground-truth labels.

    Returns:
        Dict with "auc", "auprc", "coef", "model_name".
    """
    x = np.hstack([scores.reshape(-1, 1), types_encoded, evidence_features])
    w, b = _logistic_regression_fit(x, labels)
    y_prob = _logistic_regression_predict(x, w, b)

    return {
        "model_name": "Y~P+T+M",
        "auc": _safe_auc(labels, y_prob),
        "auprc": _safe_auprc(labels, y_prob),
        "coef": list(w),
    }


def compute_non_redundancy(
    scores: np.ndarray,
    types_encoded: np.ndarray,
    evidence_features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Compute non-redundancy metrics by comparing nested models.

    Args:
        scores: [N] fraud scores.
        types_encoded: [N, T] one-hot encoded risk types.
        evidence_features: [N, E] evidence mask features.
        labels: [N] binary labels.

    Returns:
        Dict with:
            - "models": list of per-model results
            - "auc_improvement": {"type_over_score": float, "evidence_over_type": float}
            - "auprc_improvement": {"type_over_score": float, "evidence_over_type": float}
    """
    m1 = fit_score_only_model(scores, labels)
    m2 = fit_score_type_model(scores, types_encoded, labels)
    m3 = fit_score_type_evidence_model(scores, types_encoded, evidence_features, labels)

    return {
        "models": [m1, m2, m3],
        "auc_improvement": {
            "type_over_score": m2["auc"] - m1["auc"],
            "evidence_over_type": m3["auc"] - m2["auc"],
        },
        "auprc_improvement": {
            "type_over_score": m2["auprc"] - m1["auprc"],
            "evidence_over_type": m3["auprc"] - m2["auprc"],
        },
    }
