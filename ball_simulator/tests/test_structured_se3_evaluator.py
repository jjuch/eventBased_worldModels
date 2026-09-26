import numpy as np
from ball_world_model.evaluation.metrics import (
    effective_rank,
    fit_linear_probe,
    apply_linear_probe,
    regression_metrics
)

def test_effective_rank():
    rng = np.random.default_rng(1)
    assert effective_rank(rng.normal(size=(200, 16))) > 8

# def test_probe_recovers_linear_map():
#     rng = np.random.default_rng(2)
#     x = rng.normal(size=(300, 20))
#     y = x @ rng.normal(size=(20, 3))
#     rows = regression_metrics(x[:200], y[:200], x[200:], y[200:])
#     assert min(row["r2"] for row in rows) > .999