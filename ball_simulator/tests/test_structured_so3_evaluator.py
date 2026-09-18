import numpy as np
from ball_world_model.evaluation.structured_so3_evaluator import _effective_rank, _regression

def test_effective_rank():
    rng = np.random.default_rng(1)
    assert _effective_rank(rng.normal(size=(200, 16))) > 8

def test_probe_recovers_linear_map():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 20))
    y = x @ rng.normal(size=(20, 3))
    rows = _regression(x[:200], y[:200], x[200:], y[200:])
    assert min(row["r2"] for row in rows) > .999