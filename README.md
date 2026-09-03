# eventBased_worldModels

A research repository for studying predictive world models of physical systems, with a current focus on rendered rigid-ball dynamics and future adaptive-computation world models.

The software is organized in two layers:

```text
eventBased_worldModels/
├── ball_simulator/          Python package and current ball experiment backend
├── external/le-wm/          Pinned LeWorldModel research dependency
└── README.md
```

Experiment data should normally live **outside this Git repository** in portable project workspaces created with `ball_project` or `create_world_model`. This keeps source code, scientific configuration, large datasets, rendered frames, and model outputs clearly separated.

## Current capabilities

- deterministic and parallel rigid-ball trajectory generation;
- single-wall, U-box, and controlled free-flight environments;
- batched Blender rendering with fixed semantic camera configuration;
- deterministic camera proposal from complete trajectory bounds;
- validated temporal RGB/state datasets;
- translation-only, rotation-only, and combined kinematic observers;
- an extensible project-workspace layer for future experiment types.

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/jjuch/eventBased_worldModels.git
cd eventBased_worldModels/ball_simulator
```

Create and activate a Python environment, then install the project:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the simulator, project tooling, world-model dependencies, and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[world-model,dev]"
```

Install the pinned LeWorldModel core when JEPA development requires it:

```bash
python -m pip install -e ../external/le-wm
```

Blender is discovered separately by `ball_renderer`. Verify it with:

```bash
ball_renderer blender-info
```

## Create an experiment workspace

From the parent directory where experiment data should live, run:

```bash
create_world_model translation_trial \
  --type ball \
  --subtype free-flight \
  --mode translation
```

Equivalent canonical command:

```bash
ball_project create translation_trial \
  --type ball \
  --subtype free-flight \
  --mode translation
```

Available initial modes are:

```text
translation
rotation
combined
```

Then enter the workspace:

```bash
cd translation_trial
ball_project run
```

`ball_project run` opens a terminal menu for:

- editing configurations;
- generating trajectories;
- proposing a camera;
- rendering all trajectories;
- building and inspecting the manifest;
- training the kinematic observer.

Activation is not required. Project commands discover the nearest `project.yaml` by searching the current directory and its parents.

## Direct noninteractive workflow

Every menu operation is also a direct command:

```bash
ball_project trajectories
ball_project camera
ball_project render
ball_project build-manifest
ball_project inspect-data
ball_project train
```

Inspect state:

```bash
ball_project status
```

Preview an automatic pipeline without executing it:

```bash
ball_project pipeline --dry-run
```

Run the missing stages automatically:

```bash
ball_project pipeline
```

## Workspace structure

A created project contains:

```text
translation_trial/
├── project.yaml             Central paths, identity, and runtime settings
├── configs/                 Human-editable scientific configurations
├── data/                    HDF5 trajectories, rendered frames, manifest
├── outputs/                 Training, checkpoints, evaluation, inspection
├── reports/                 Camera and future stage reports
├── logs/                    Stage-specific logs
└── .ball_project/           Generated effective configs and execution records
```

`project.yaml` owns data and output paths. Scientific config files remain independent:

- `configs/trajectories.yaml` defines simulation and initial-state ranges;
- `configs/rendering.yaml` defines camera, appearance, lighting, and output settings;
- `configs/data.yaml` defines temporal windows and loader behavior;
- `configs/training.yaml` defines model and optimizer settings.

The project runner writes effective path-resolved files under `.ball_project/effective/`. Therefore, changing a scientific parameter in one YAML does not silently rewrite another YAML.

Edit from the terminal:

```bash
ball_project config edit trajectories
ball_project config edit rendering
ball_project config edit data
ball_project config edit training
ball_project config edit project
```

Set an editor through `BALL_PROJECT_EDITOR`, `VISUAL`, or `EDITOR`, or pass `--editor`.

## Copy a ball_project folder
Do not copy a existing ball_project folder in the naive way, as this could cause conflicts in the hardcoded paths. There is a command to create a new project, using the configs, trajectories and renders from an existing source project.

Suppose the source experiment is:
```text
local_experiments/translation_trial/
```
and contains completed trajectories, rendered frames, and a manifest.
From `local_experiments/`:
```bash
ball_project derive translation_bidirectional \
  --from translation_trial
```
This produces:
```text
local_experiments/translation_bidirectional/
```
Then:
```bash
cd translation_bidirectional
ball_project run
```
To place the derived project in another existing parent directory:
```bash
ball_project derive translation_bidirectional \
  --from /path/to/translation_trial \
  --parent /path/to/other_experiments
```

## Repository development

Run tests from `eventBased_worldModels/ball_simulator`:

```bash
pytest -q
```

The experiment-type registry currently maps `ball/free-flight` to the ball simulator backend. Future systems, such as boiling-water experiments, should add a new template and stage adapter without changing existing ball workspaces.

## Reproducibility principles

- complete trajectories, never frames, define train/validation/test splits;
- project configurations use relative paths;
- camera selection is fixed per dataset, not per trajectory;
- physical parameters are fixed in the first observability experiments;
- stage commands write execution records under `.ball_project/records/`;
- generated data and outputs are ignored by each workspace's `.gitignore`;
- source code and experiment data remain separate.

## License

MIT. See `LICENSE`.
