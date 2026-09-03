import numpy as np

from ball_world_model.evaluation.metrics import (
    apply_linear_probe,
    effective_rank,
    fit_linear_probe,
    regression_metrics,
)


def test_exact_regression_metrics():
    target = np.linspace(-1.0, 1.0, 100)
    metrics = regression_metrics(target, target)
    assert metrics["rmse"] == 0.0
    assert np.isclose(metrics["r2"], 1.0)
    assert np.isclose(metrics["pearson"], 1.0)


def test_ridge_probe_recovers_linear_state():
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(500, 16))
    target = latent @ rng.normal(size=(16, 3)) + np.array([0.2, -0.4, 0.7])
    weights = fit_linear_probe(latent[:400], target[:400], ridge=1.0e-8)
    estimate = apply_linear_probe(latent[400:], weights)
    np.testing.assert_allclose(estimate, target[400:], atol=1.0e-5)


def test_effective_rank_detects_collapsed_representation():
    collapsed = np.ones((100, 32))
    rich = np.random.default_rng(1).normal(size=(100, 32))
    assert effective_rank(collapsed) == 0.0
    assert effective_rank(rich) > 10.0
