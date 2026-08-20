# {{PROJECT_NAME}}

A `ball / free-flight / {{MODE}}` world-model experiment workspace.

Run `ball_project status` to inspect progress and `ball_project run` for the interactive menu. Scientific settings live in `configs/`; central paths and runtime worker settings live in `project.yaml`. Generated path fields are injected into `.ball_project/effective/`, so scientific config files do not need duplicated output paths.

Typical sequence:

```bash
ball_project trajectories
ball_project camera
ball_project render
ball_project build-manifest
ball_project inspect-data
ball_project train
```

Use `ball_project pipeline --dry-run` before an expensive run.
