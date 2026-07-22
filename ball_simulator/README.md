# Ball Simulator

Object-oriented Python package for generating synthetic sphere-wall trajectories for fixed-depth and adaptive-depth world-model research.

## Design

- `BallSimulator` owns simulation orchestration.
- `ForceModel` is an extension point for gravity, aerodynamics, walls, floors, and future force fields.
- `Integrator` is replaceable; the baseline is semi-implicit Euler.
- `ParameterSampler` separates domain randomization from mechanics.
- `HDF5TrajectoryWriter` separates storage from simulation.
- Observation-rate data and optional high-rate diagnostics are stored separately.

The contact law is a non-adhesive Hunt-Crossley-style Hertz normal force and a history-dependent tangential spring-damper capped by Coulomb friction. Quaternions use SciPy's scalar-last `xyzw` convention.

 ### IMPORTANT Note
Contact evaluation mutates tangential-memory state.This is valid for the current one-evaluation-per-step integrator. Multi-stage or adaptive integrators require transactional contact state. This is an issue when the ball hits two walls at the same time (corners).


## Install

```bash
cd ball_simulator
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Run

### Simulate
```bash
ball_simulator validate-config configs/poc.yaml
ball_simulator generate configs/poc.yaml --environment single-wall --output single_wall.h5
ball_simulator generate configs/poc.yaml --environment u-box --output u_box.h5
pytest -q
```

For a fast smoke test, change `trajectories` to 2 and `duration` to 0.2 in a copy of the YAML file. Set the `default_environment` parameter in the YAML file and omit the `--environment` flag.

### Analyze
#### Display the initial condition coverage
```bash
ball_simulator plot-initial-conditions trajectories.h5 --output initial_conditions.png --no-show
```

For larger datasets the number of intial conditions van be limited with the `--max-points` flag.

#### List trajectory IDs
```bash
ball_simulator list-trajectories trajectories.h5
```

#### Plot trajectory 17
```bash
ball_simulator plot-trajectory trajectories.h5 17 --output trajectory_17.png --diagnostics-output trajectory_17_diagnostics.png --no-show
```

Use high-rate samples by adding the `--high-rate` flag.

#### Stratified inspection
Inspect several regimes in the large set of trajectories to verify the generated trajectories.
```bash
ball_simulator inspect-regimes trajectories.h5 -o validation/regimes
ball_simulator inspect-regimes trajectories.h5 -o validation/regimes --high-rate
ball_simulator inspect-regimes trajectories.h5 -o validation/regimes --no-diagnostics
```
The output directory contains `manifest.csv`, one 3D plot per selected regime, and optional physics-diagnostic plots. The selector uses dataset-relative deciles, so it continues to work when YAML ranges change. It reads only the fields needed for selection and only loads full trajectories after selection.

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
