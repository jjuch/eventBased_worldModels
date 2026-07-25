import numpy as np
from ball_simulator.rendering.resampling import resample_states


def test_resampling_preserves_endpoints_and_unit_quaternions():
    time = np.array([0.0, 0.5, 1.0])
    position = np.column_stack([time, np.zeros(3), np.ones(3)])
    quaternion = np.array([[0, 0, 0, 1], [0, 0, 0.38268343, 0.92387953], [0, 0, 0.70710678, 0.70710678]])
    velocity = np.tile([1.0, 0.0, 0.0], (3, 1))
    omega = np.tile([0.0, 0.0, np.pi / 2], (3, 1))
    result = resample_states(time, position, quaternion, velocity, omega, 0.25)
    assert result["position"].shape == (5, 3)
    assert np.allclose(result["position"][[0, -1]], position[[0, -1]])
    assert np.allclose(np.linalg.norm(result["quaternion_xyzw"], axis=1), 1.0)
