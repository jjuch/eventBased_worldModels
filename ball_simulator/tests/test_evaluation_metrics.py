import numpy as np

from ball_world_model.evaluation.metrics import (
    apply_linear_probe,
    fit_linear_probe,
    regression_metrics,
)


def test_regression_metrics_for_exact_prediction():
    target = np.linspace(-2.0, 2.0, 100)
    metrics = regression_metrics(target, target.copy())
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["bias"] == 0.0
    assert np.isclose(metrics["r2"], 1.0)
    assert np.isclose(metrics["pearson"], 1.0)


def test_linear_probe_recovers_linear_state():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(500, 12))
    true_weights = rng.normal(size=(12, 3))
    targets = features @ true_weights + np.asarray([0.2, -0.1, 0.5])
    weights = fit_linear_probe(features[:400], targets[:400], ridge=1.0e-8)
    estimate = apply_linear_probe(features[400:], weights)
    np.testing.assert_allclose(estimate, targets[400:], atol=1.0e-6)
