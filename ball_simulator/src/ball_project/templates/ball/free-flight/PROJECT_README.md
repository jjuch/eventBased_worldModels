# {{PROJECT_NAME}}

A `ball / free-flight / {{MODE}}` world-model experiment workspace.

```text
datasets/<name>/   configs/, dataset.yaml, trajectories.h5, rendered/, manifest.parquet
runs/<name>/       training.yaml (its `dataset:` names the training data), checkpoints,
                   dataset_snapshot/, evaluations/<dataset>/
reports/           comparison.csv from `ball_project report`
```

A dataset does not change once a run has used it. New shapes, markers or physics get a new
dataset: `ball_project datasets new NAME --from REFERENCE` copies the configs and keeps the
reference camera, so trained models see the same view.

Typical sequence:

```bash
ball_project prepare --dataset default        # trajectories, camera, renders, manifest
ball_project train --run default
ball_project datasets new cube --from default # edit datasets/cube/configs/, then:
ball_project prepare --dataset cube
ball_project evaluate --run default --dataset default --dataset cube
ball_project report
```

`ball_project status` shows progress; `ball_project run` opens the interactive menu. Use
`--dry-run` before an expensive command.
