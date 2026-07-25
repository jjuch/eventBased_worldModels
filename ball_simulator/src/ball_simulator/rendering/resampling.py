from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def resample_states(
    time: np.ndarray,
    position: np.ndarray,
    quaternion_xyzw: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    frame_dt: float | None,
) -> dict[str, np.ndarray]:
    """Resample smooth rigid state. Use high-rate input around impacts."""
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be a strictly increasing 1D array with >=2 samples")
    if frame_dt is None:
        target_time = time.copy()
    else:
        count = int(np.floor((time[-1] - time[0]) / frame_dt + 1e-10)) + 1
        target_time = time[0] + np.arange(count, dtype=np.float64) * frame_dt
        target_time = target_time[target_time <= time[-1] + 1e-10]

    p = np.column_stack([
        np.interp(target_time, time, position[:, component]) for component in range(3)
    ])
    v = np.column_stack([
        np.interp(target_time, time, linear_velocity[:, component]) for component in range(3)
    ])
    w = np.column_stack([
        np.interp(target_time, time, angular_velocity[:, component]) for component in range(3)
    ])
    q = Slerp(time, Rotation.from_quat(quaternion_xyzw))(target_time).as_quat(canonical=False)
    return {
        "time": target_time,
        "position": p,
        "quaternion_xyzw": q,
        "linear_velocity": v,
        "angular_velocity": w,
    }