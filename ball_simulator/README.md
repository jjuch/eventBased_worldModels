# Ball Simulator

Object-oriented Python package for generating synthetic sphere-wall trajectories for fixed-depth and adaptive-depth world-model research.

## Design

- `SphereWallSimulator` owns simulation orchestration.
- `ForceModel` is an extension point for gravity, aerodynamics, walls, floors, and future force fields.
- `Integrator` is replaceable; the baseline is semi-implicit Euler.
- `ParameterSampler` separates domain randomization from mechanics.
- `HDF5TrajectoryWriter` separates storage from simulation.
- Observation-rate data and optional high-rate diagnostics are stored separately.

The contact law is a non-adhesive Hunt-Crossley-style Hertz normal force and a history-dependent tangential spring-damper capped by Coulomb friction. Quaternions use SciPy's scalar-last `xyzw` convention.

## Install

```bash
cd ball_simulator
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run

```bash
ball_simulator validate-config configs/poc.yaml
ball_simulator generate configs/poc.yaml --output trajectories.h5
pytest -q
```

For a fast smoke test, change `trajectories` to 2 and `duration` to 0.2 in a copy of the YAML file.

## HDF5 layout

```text
/trajectories/00000000/
  observations/{time, position, quaternion_xyzw, linear_velocity, ...}
  high_rate/{...}                         # optional
  parameters                              # attributes
```

Diagnostic labels include contact activity, mode, penetration, contact velocity, and normal/tangential forces. They need not be exposed to the learned world model.

## Future extensions

1. Add `AdaptiveIntegrator` without changing dataset APIs.
2. Add floors or arbitrary planes through additional `ForceModel` instances.
3. Replace HDF5 storage with Zarr by implementing another writer.
4. Add per-step difficulty labels from force, penetration, Jacobian estimates, or solver effort.
5. Add a PyTorch `Dataset` that emits fixed-rate transitions and event-balanced batches.
6. Add MuJoCo/Drake generators behind the same trajectory schema for cross-simulator testing.

## Important numerical note

The default parameter ranges are proof-of-concept values, not calibrated material data. Always run time-step convergence studies before generating a large corpus. Increase stiffness only while reducing `internal_dt` sufficiently.
