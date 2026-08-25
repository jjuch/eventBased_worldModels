# Kinematic observer evaluation and representation-analysis
This code section adds the first dedicated validation and interpretability suite for the trained translation, rotation, and combined kinematic observers.
The immediate goal is to answer four questions more clearly than TensorBoard loss curves can:
1. Do predicted states follow the true state trajectories over the complete context window?
2. Does the model beat a mean-state baseline?
3. Does changing temporal order change the inferred velocity in a physically meaningful way?
4. At which representation layer do position and velocity become linearly decodable?

## Outputs
Running the evaluator creates:
```text
outputs/evaluation/
├── summary.json
├── aggregate_metrics.csv
├── position_component_scatter.png
├── velocity_component_scatter.png
├── interventions.csv
├── trajectories/
│   ├── best_*.png
│   ├── median_*.png
│   ├── worst_*.png
│   └── random_*.png
└── probes/
    ├── layerwise_linear_probes.csv
    └── layerwise_linear_probes.png
```
Each trajectory plot shows true and predicted position and velocity over time, their error norms, and predicted kinematic consistency:
```text
p[t+1] - p[t] - dt * v[t]
```
Aggregate scatter plots compare every predicted component with ground truth and report RMSE and R2.
The intervention report evaluates:
```text
forward frames
reversed frames
repeated final frame
randomly shuffled frames
```
The probe report compares:
```text
mean_state_baseline
frame_last
frame_sequence_mean
frame_difference
 temporal_last
```
`frame_last` tests what one spatial frame embedding contains. `frame_difference` tests whether simple change in spatial features exposes velocity before the temporal Conv1D. `temporal_last` tests what becomes available after temporal processing.

## How to interpret the reports
### Aggregate scatter plots
Good predictions lie near the identity line. A horizontal prediction band indicates a
mean-state collapse. A line with the correct sign but too little slope indicates systematic
underestimation. One poor component with two good components often indicates a camera-depth
observability problem.

### Trajectory plots
Look for:
* constant offset in position;
* correct velocity direction but wrong magnitude;
* temporal lag;
* noisy predictions between frames;
* low consistency error despite incorrect absolute state;
* good image-plane components but poor camera-depth component.

### Interventions
For a genuine constant-velocity temporal representation:
* reversed frames should tend to reverse inferred velocity;
* repeated frames should reduce inferred speed toward zero;
* shuffled frames should disturb velocity more than position.
The CSV contains `mean_reversal_error`. For the reversed intervention, smaller values indicate that reversed predictions more closely approximate the negative of forward predictions. These interventions are diagnostic, not formal proofs, because the network was trained only on physically ordered sequences. Reversed and shuffled inputs are outside the training distribution.

### Layerwise probes
Expected progression:
```text
position:
  frame_last              reasonably decodable
  temporal_last           at least as good

velocity:
  mean_state_baseline     poor
  frame_last              weak
  frame_difference        improved
  temporal_last           strongest
```
If `frame_last` predicts velocity very well, inspect the dataset for static correlations between state and velocity. If `temporal_last` does not improve over `frame_last` or `frame_difference`, the temporal Conv1D may not be contributing useful information. A linear probe measures decodability, not causal usage. The next interpretability increment should add latent interventions and soft-keypoint overlays after these first reports identify which layer is worth investigating.
