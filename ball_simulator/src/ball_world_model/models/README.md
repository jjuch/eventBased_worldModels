# SE(3) observer

Estimates the rigid-body state of the marked ball from a window `T` RGB frames.

```text
frames [B,T,3,H,W]
  └─ CoordinateAwareFrameEncoder ── feature maps F [B,T,C,h,w]
       ├─ SpatialKeypointEncoder (float32) ── frame descriptor ── pose_head ──> pose per frame       [B,T]
       └─ IntervalMotionEncoder: (F[k], F[k+1], standardised (F[k+1]-F[k])/dt) ──> motion per interval [B,T-1,D]
            └─ IntervalTemporalMixer (1-D convs over intervals)
                 ├─ twist_head        ──> mean twist per interval         [B,T-1,6]
                 └─ acceleration_head ──> mean acceleration per interval  [B,T-1,6]
current state at t_last = (pose[T-1], twist[last] + ½·dt·acceleration[last], acceleration[last])

```

## Why interval means

Two frames reveal the *mean* motion between them, not the instantaneous motion at either frame.
The mean makes one SE(3) step exact:
```text
p[k+1] = p[k] + dt · v_mean            R[k+1] = exp(dt · ω_mean) R[k]
```

The targets (`training/targets.py`) are computed from the simulator states without approximation:
`v_mean = Δp/dt`, `ω_mean = log(R[k+1] R[k]ᵀ)/dt`, `a_mean = Δv/dt`, `α_mean = Δω/dt`. No accelerations have to be stored in the dataset.

## Losses (`training/losses.py`)
| term | definition | scale |
|---|---|---|
| position | per frame | training std of position |
| orientation | per frame, chordal ≈ angle² | `orientation_scale_deg` |
| linear / angular velocity | per interval mean | training std |
| linear / angular acceleration | per interval mean | training std (floored at 1e-3) |
| rollout position / orientation | current state propagated with constant acceleration to `rollout_steps` future frames | same as pose terms |

Every term is dimensionless, so the weights in `training.yaml` are directly comparable.

## Notes

- Use `precision: 32-true`. Frame differences are smaller than bfloat16 resolution.
- If the trajectories have no gravity, drag or Magnus effect, acceleration is identically zero. Training prints this, and the evaluator reports R² as undefined for those components.
- Carrier points (`geometry/se3.py: carrier_state/velocity/acceleration`) are kept as tested geometry utilities for the contact stage; they are no longer trained, since they are exact functions of pose and twist.