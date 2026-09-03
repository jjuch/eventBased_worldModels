# Training loop
The Lightning loss is now decomposed into:
```text
position_loss_normalised
velocity_loss_normalised
forward_latent_loss
backward_latent_loss
latent_prediction_loss
reverse_velocity_loss
kinematic_loss_m2
variance_loss
```
The total objective is:
```text
position_weight * position_loss
+ velocity_weight * velocity_loss
+ latent_prediction_weight * latent_prediction_loss
+ kinematic_weight * kinematic_loss
+ reverse_weight * reverse_velocity_loss
+ variance_weight * variance_loss
```
The position and velocity heads still produce normalised state values. Iterative correction is performed in physical metres and metres per second, then converted back to normalised outputs. This avoids mixing normalised states with the dimensional kinematic residual.

## Difference normalisation
The model computes a spatial feature rate:
```text
feature_rate[t] = (F[t+1] - F[t]) / dt
```
A running per-channel mean and variance standardise this rate across batch, time, and spatial locations. This emphasizes small but repeatable motion signals while preserving relative magnitude.
The implementation deliberately does not normalise every individual difference to unit norm. Per-sample unit normalisation would erase the distinction between slow and fast motion. The running statistics are buffers in the checkpoint and are frozen during validation and test evaluation.

## Bidirectional training
For each adjacent pair, the same motion encoder receives:
```text
forward:
  F[t], F[t+1], normalised feature rate

backward:
  F[t+1], F[t], negative normalised feature rate
```
The shared velocity decoder predicts normalised forward and backward velocities. The reversal loss enforces:
```text
v_backward approximately -v_forward
```
The shared latent transition predicts both the next and previous content embeddings. The content targets are stop-gradient values, while variance regularisation discourages content or motion collapse.

## Iterative correction
`refinement_iterations` controls repeated application of one weight-shared corrector.
```yaml
refinement_iterations: 0  # no correction baseline
refinement_iterations: 1  # initial recommended run
refinement_iterations: 2  # later ablation
refinement_iterations: 3  # later ablation
```
The final corrector layer is initialised to zero. Therefore, iteration begins as an identity operation and cannot damage initial predictions before learning useful residual corrections. Do not immediately interpret `K=3` as superior. First compare `K=0` and `K=1` at equal seeds. Iteration is useful only if residuals and physical metrics improve.

## TensorBoard diagnostics
Monitor these quantities separately:
```text
validation/position_rmse_m
validation/velocity_rmse_mps
validation/latent_prediction_loss
validation/reverse_velocity_loss
validation/kinematic_consistency_m
validation/feature_delta_abs_mean
validation/motion_std
```
Collapse diagnostics:
```text
validation/position_std_ratio_x
validation/position_std_ratio_y
validation/position_std_ratio_z
validation/velocity_std_ratio_x
validation/velocity_std_ratio_y
validation/velocity_std_ratio_z
```
Interpretation:
```text
std ratio near 0:
  nearly constant predictions

std ratio near 1:
  predicted variation is comparable to target variation

large std ratio:
  unstable or excessively variable predictions
```
For iterative correction, compare:
```text
validation/refinement_residual_0_m
validation/refinement_residual_1_m
validation/refinement_residual_2_m
```
Only fields corresponding to configured iterations are emitted.