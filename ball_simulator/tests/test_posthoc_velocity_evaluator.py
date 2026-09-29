import numpy as np
import torch

from ball_world_model.evaluation.posthoc_velocity_evaluator import (
    _fit_mlp,
    _metric_rows,
)

def test_metric_rows_report_all_components():
    target = np.arange(30, dtype=float).reshape(10, 3)
    rows = _metric_rows("example", "ridge", target, target)
    assert [row["component"] for row in rows] == ["x", "y", "z"]
    assert min(row["r2"] for row in rows) > 0.999999


def test_small_mlp_recovers_simple_mapping():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(512, 8)).astype(np.float32)
    weights = rng.normal(size=(8, 3)).astype(np.float32)
    y = x @ weights
    estimate, history, state = _fit_mlp(
        x[:400], y[:400], x[400:],
        device=torch.device("cpu"),
        epochs=80,
        hidden_dim=32,
        batch_size=64,
        learning_rate=3.0e-3,
        seed=7,
    )
    error = np.mean((estimate - y[400:]) ** 2)
    assert error < 0.05
    assert len(history) == 80
    assert state["input_dim"] == 8
