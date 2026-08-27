from __future__ import annotations

import numpy as np


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    error = prediction - target
    residual = float(np.sum(error * error))
    centered = target - target.mean()
    total = float(np.sum(centered * centered))
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target.size > 1 and target.std() > 0.0 and prediction.std() > 0.0 else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - residual / total) if total > 0.0 else float("nan"),
        "pearson": correlation,
        "count": int(target.size),
    }

def fit_linear_probe(features: np.ndarray, targets: np.ndarray, ridge: float = 1.0e-4) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    augmented = np.column_stack((features, np.ones(len(features))))
    regulariser = ridge * np.eye(augmented.shape[1])
    regulariser[-1, -1] = 0.0
    weights = np.linalg.solve(
        augmented.T @ augmented + regulariser,
        augmented.T @ targets,
    )
    return weights


def apply_linear_probe(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    augmented = np.column_stack((features, np.ones(len(features))))
    return augmented @ weights


def effective_rank(features: np.ndarray) -> float:
    features = np.asarray(features, dtype=np.float64)
    features = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(features, compute_uv=False)
    probabilities = singular_values**2
    total = probabilities.sum()
    if total <= 0:
        return 0.0
    probabilities /= total
    entropy = -np.sum(probabilities * np.log(probabilities + 1.0e-12))
    return float(np.exp(entropy))