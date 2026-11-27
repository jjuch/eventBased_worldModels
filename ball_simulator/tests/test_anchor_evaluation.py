"""Viewing-ray decomposition and the depth-scale fit of the evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from ball_world_model.evaluation.evaluator import depth_scale, viewing_ray_errors


def _points(count: int = 200) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack(
        (rng.uniform(-0.5, 0.5, count), rng.uniform(-0.5, 0.5, count), rng.uniform(1.5, 3.0, count))
    )


def test_pure_depth_error_lies_along_the_ray():
    target = _points()
    prediction = 1.1 * target  # 10 % too far away, same bearing
    errors = viewing_ray_errors(prediction, target)
    np.testing.assert_allclose(errors["along"], 0.1 * np.linalg.norm(target, axis=-1), atol=1e-12)
    np.testing.assert_allclose(errors["across"], 0.0, atol=1e-12)


def test_sideways_error_lies_across_the_ray():
    target = np.array([[0.0, 0.0, 2.0]])
    errors = viewing_ray_errors(target + [0.01, 0.0, 0.0], target)
    assert errors["along"][0] == pytest.approx(0.0, abs=1e-12)
    assert errors["across"][0] == pytest.approx(0.01)


def test_depth_scale_recovers_a_global_size_error():
    target = _points()
    noise = np.random.default_rng(1).normal(scale=1e-3, size=target.shape)
    scale = depth_scale(target / 1.25 + noise, target)
    assert scale == pytest.approx(1.25, rel=2e-3)
