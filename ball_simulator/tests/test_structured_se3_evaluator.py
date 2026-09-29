import numpy as np
from ball_world_model.evaluation.metrics import (
    effective_rank,
    fit_linear_probe,
    apply_linear_probe,
    regression_metrics
)
from ball_world_model.evaluation.structured_se3_evaluator import _regression_rows

def test_effective_rank():
    rng = np.random.default_rng(1)
    assert effective_rank(rng.normal(size=(200, 16))) > 8

def test_sector_regression_probe_recovers_linear_target():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(500, 20))
    matrix = rng.normal(size=(20, 3))
    y = x @ matrix
    rows = _regression_rows(
        "sector",
        "linear_velocity",
        x[:350],
        y[:350],
        x[350:],
        y[350:],
    )
    assert len(rows) == 3
    assert min(row["r2"] for row in rows) > 0.999999
