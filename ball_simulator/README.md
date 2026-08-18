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
python -m pip install -e ".[world-model,dev]"
```

## Generate trajectories

### Simulate
```bash
ball_simulator validate-config configs/poc.yaml
ball_simulator generate configs/poc.yaml --environment single-wall --output single_wall.h5 --workers 0
ball_simulator generate configs/poc.yaml --environment u-box --output u_box.h5 --workers 0
pytest -q
```

For a fast smoke test, change `trajectories` to 2 and `duration` to 0.2 in a copy of the YAML file. Set the `default_environment` parameter in the YAML file and omit the `--environment` flag. The `--workers` flag can be used to parallelise the task by distributing the work over several CPU cores (0: all available cores | 1: serial execution).

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

## Render trajectories
Once you are satisfied with the trajectories rendered with the `ball_simulator`, you can render the trajectories into a pixel-based output using Blender with the `ball_renderer`.

### Blender installation
Make sure Blender and FFMPEG is installed and added to the path:
```bash
ball_renderer blender-info
ffmpeg -version
```
This will show you the version and the path. If it isn't on the path you can use `--blender-executable` followed by the absolute path to blender, but it is preferred to have it installed on the path.

### Render a single path
To run a single trajectory render do:
```bash
ball_renderer render-trajectory trajectories.h5 0 --config configs/render_debug.yaml --output-directory rendered/debug 
```
or render all trajectories a h5-file:
```bash
ball_renderer render-trajectory trajectories.h5 --all --workers 2 --threads-per-blender 0 --resume --config configs/render_debug.yaml --output-directory rendered/debug
```

There is the option to use multiple `workers`. But note that `BLENDER_EEVEE_NEXT`, the default engine, already uses the graphics device selected by Blender/driver configuration. It does not behave like a CPU workload where 32 workers necessarily means 32 times more throughput.

With one GPU:
* `--workers 1` may already saturate it;
* `--workers 2` can hide CPU setup and PNG-writing latency;
* `--workers 3` may improve throughput further;
* too many workers will compete for GPU memory, disk bandwidth and shader resources.

Practical reports of multiple EEVEE instances show good improvement initially, followed by flattening or regression as instance count rises.

## Train model
### Preparing the dataset
Create a manifest using the configs in `ubox_temporal.yaml`:
```bash
ball_world_model build-manifest \
  -c ball_simulator/src/ball_world_model/configs/data/ubox_temporal.yaml
```
Then the data in the config can be validated:
```bash
ball_world_model validate-data \
  ball_simulator/rendered/training/manifest.parquet
```
A final check can be done by creating an inspection figure:
```bash
ball_world_model inspect-data \
  -c ball_simulator/src/ball_world_model/configs/data/ubox_temporal.yaml \
  -m ball_simulator/rendered/training/manifest.parquet \
  -o inspection/data_batch.png
```


### Train observability
Train the ten-frame visual state-estimation prerequisite baseline. It tests whether the ball’s position, orientation, linear velocity, and angular velocity can actually be inferred from the rendered video.

1. Translation-only dataset
Use:
* no collisions during the complete context;
* zero angular velocity;
* fixed orientation;
* initially zero gravity and constant linear velocity;
* fixed ball radius for the first experiment;
* oblique fixed camera;
* checkerboard reference planes with known square size;
* the ball completely visible in all frames.

```bash
ball_world_model train-kinematic \
  -c src/ball_world_model/configs/experiment/translation_observer.yaml
```
Important TensorBoard metrics:
```text
validation/position_rmse_m
validation/velocity_rmse_mps
validation/translation_consistency_m
```

2. Rotation-only dataset
Use:
* fixed ball center;
* zero linear velocity;
* zero gravity;
* constant random angular velocity;
* random initial orientation;
* four unique tetrahedral markers;
* a large apparent ball diameter in the 256x256 model input.

```bash
ball_world_model train-kinematic \
  -c src/ball_world_model/configs/experiment/rotation_observer.yaml
```
Important metrics:
```text
validation/orientation_deg
validation/omega_rmse_radps
validation/rotation_consistency_deg
```

3. Combined dataset
Use translation and rotation together, still without contact. Add gravity only after the zero-gravity combined experiment succeeds.
Update the `root` and `manifest_path` fields in the three data YAML files if your actual locations differ.

```bash
ball_world_model train-kinematic \
  -c src/ball_world_model/configs/experiment/combined_observer.yaml
```
Open TensorBoard with:
```bash
tensorboard --logdir outputs
```

#### Geometric bias used by this patch
The observer predicts a six-dimensional continuous rotation representation and maps it to a valid rotation matrix. Orientation is evaluated intrinsically on SO(3). It also predicts a state at every context frame. Adjacent predictions are regularized using:
```text
p[t+1] approximately p[t] + dt * v[t]
R[t+1] approximately Exp(dt * omega[t]) @ R[t]
```
The second convention matches the current simulator's world-coordinate angular velocity. If that simulator convention changes, set `world_angular_velocity: false` in the experiment YAML.
The consistency losses are intentionally soft, with weight `0.1`. The model remains free to fit rendering and numerical imperfections. This is not a hard TSE(3) architecture.

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


