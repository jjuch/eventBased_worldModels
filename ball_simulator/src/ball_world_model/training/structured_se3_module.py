"""Unified training module for translation, rotation and combined rigid motion."""
from __future__ import annotations

from dataclasses import asdict
import lightning as L

import torch
import torch.nn.functional as F

from ball_world_model.models.rotation import quaternion_xyzw_to_matrix, rotation_geodesic_error
from ball_world_model.models.structured_se3_estimator import StructuredSE3StateEstimator, SE3Prediction
from ball_world_model.models.structured_se3 import SE3PhysicalTeacher
from ball_world_model.models.artifact_residual import cross_covariance_loss
from .kinematic_module import KinematicStatistics


class StructuredSE3ObservabilityModule(L.LightningModule):
    # The structured estimator exposes omega directly in physical rad/s.
    outputs_physical_units = True # TODO: find a more elegant to circumvent this; hack for model_loader > denormalised_prediction

    def __init__(
        self, 
        statistics: KinematicStatistics, 
        *, 
        task: str = "combined", 
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        position_weight: float = 1.0,
        velocity_weight: float = 1.0, 
        orientation_weight: float = 1.0, 
        omega_weight: float = 1.0, 
        carrier_weight: float = 0.25,
        tangent_weight: float = 0.05, 
        group_weight: float = 0.1, 
        reverse_weight: float = 0.1, 
        artifact_weight: float = 0.01,
        amplitude_weight: float = 0.01,
        invariant_prediction_weight: float = 0.01,
        invariant_variance_weight: float = 0.005,
        variance_weight: float = 0.01,
        physical_feature_rate_weight: float = 0.01,
        artifact_residual_weight: float = 0.01,
        artifact_cross_covariance_weight: float = 0.001,
        geometric_channels: int = 24,
        physical_invariant_dim: int = 16,
        latent_architecture: str = "structured_se3_artifacts",
        **model: object
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["statistics"])
        self.model = StructuredSE3StateEstimator(
            task=task,
            geometric_channels=geometric_channels,
            physical_invariant_dim=physical_invariant_dim,
            **model
        )

        self.teacher = SE3PhysicalTeacher(geometric_channels)
        for name, value in asdict(statistics).items():
            self.register_buffer(name, value.float())

    @staticmethod
    def _denormalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return value * std + mean

    @staticmethod
    def _variance_floor(value: torch.Tensor, floor: float = 0.25) -> torch.Tensor:
        if value.shape[-1] == 0:
            return value.new_zeros(())
        flattened = value.flatten(0, 1)
        std = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1.0e-4)
        return torch.relu(floor - std).mean()

    def _losses(self, prediction: SE3Prediction, batch: dict):
        context = prediction.context_sectors
        motion = prediction.motion
        mask = context.mask
        zero = prediction.frame_latent.new_zeros(())

        target_position = batch["context_position"]
        target_velocity = batch["context_linear_velocity"]
        target_rotation = quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"])
        target_omega = batch["context_angular_velocity"]
        teacher_translation_amplitudes, teacher_rotation_amplitudes, teacher_translation_carrier, teacher_rotation_carrier = self.teacher.context(target_position, target_rotation, mask)
        teacher_translation_carrier_rate, teacher_rotation_carrier_rate = self.teacher.motion(
            target_position[:, :-1], target_rotation[:, :-1], target_velocity[:, :-1], target_omega[:, :-1], mask
        )

        position_loss = ((context.position - target_position) / self.position_std).square().mean() if mask.translation else zero # lp
        velocity_loss = ((motion.forward_sectors.linear_velocity - target_velocity[:, :-1]) / self.linear_velocity_std).square().mean() if mask.translation else zero # lv

        orientation_error = rotation_geodesic_error(context.rotation, target_rotation) if mask.rotation else zero
        orientation_loss = orientation_error.mean() # lr
        omega_loss = ((motion.forward_sectors.angular_velocity - target_omega[:, :-1]) / self.angular_velocity_std).square().mean() if mask.rotation else zero # lw
        
        carrier_loss = (
            (F.mse_loss(context.translation_carrier, teacher_translation_carrier) if mask.translation else zero) +
            (F.mse_loss(context.rotation_carrier, teacher_rotation_carrier) if mask.rotation else zero)
        ) # lc
        tangent_loss = (
            (F.mse_loss(motion.forward_sectors.translation_carrier_tangents, teacher_translation_carrier_rate) if mask.translation else zero) +
            (F.mse_loss(motion.forward_sectors.rotation_carrier_tangents, teacher_rotation_carrier_rate) if mask.rotation else zero)
        ) # lt

        amplitude_loss = (
            (F.mse_loss(context.translation_amplitudes, teacher_translation_amplitudes) if mask.translation else zero) +
            (F.mse_loss(context.rotation_amplitudes, teacher_rotation_amplitudes) if mask.rotation else zero)
        ) # la


        fwd_group_loss = (
            (F.mse_loss(motion.predicted_next_position, context.position[:, 1:].detach()) if mask.translation else zero) +
            (rotation_geodesic_error(
            motion.predicted_next_rotation, context.rotation[:, 1:].detach()).square().mean() if mask.rotation else zero)
        )
        bwd_group_loss = (
            (F.mse_loss(motion.predicted_previous_position, context.position[:, :-1].detach()) if mask.translation else zero) +
            (rotation_geodesic_error(
            motion.predicted_previous_rotation, context.rotation[:, :-1].detach()).square().mean() if mask.rotation else zero)
        )
        group_prediction = 0.5 * (fwd_group_loss + bwd_group_loss) # lg

        reverse_loss = (
            (F.mse_loss(motion.backward_sectors.linear_velocity, -motion.forward_sectors.linear_velocity) if mask.translation else zero) +
            (F.mse_loss(motion.backward_sectors.angular_velocity, -motion.forward_sectors.angular_velocity) if mask.rotation else zero)
        ) # lr

        forward_invariant_loss = (
            F.mse_loss(motion.predicted_next_invariants, context.physical_scalars[:, 1:].detach())
        )
        backward_invariant_loss = (
            F.mse_loss(motion.predicted_previous_invariants, context.physical_scalars[:, :-1].detach())
        )
        invariant_prediction = 0.5 * (forward_invariant_loss + backward_invariant_loss) # li

        fwd_artifact = F.mse_loss(motion.predicted_next_artifacts, context.artifacts[:, 1:].detach())
        bwd_artifact = F.mse_loss(motion.predicted_previous_artifacts, context.artifacts[:, :-1].detach())
        artifact_prediction = 0.5 * (fwd_artifact + bwd_artifact) # lz

        artifact_variance = self._variance_floor(context.artifacts) + self._variance_floor(motion.forward_sectors.artifacts) # lvar

        invariant_variance = self._variance_floor(context.physical_scalars) + self._variance_floor(motion.forward_sectors.physical_scalars) # livar

        forward_physical_feature_rate_loss = F.smooth_l1_loss(
            motion.physical_feature_rate_forward,
            motion.normalised_forward_difference.detach(),
        )
        backward_physical_feature_rate_loss = F.smooth_l1_loss(
            motion.physical_feature_rate_backward,
            -motion.normalised_forward_difference.detach(),
        )
        physical_feature_rate_loss = 0.5 * (forward_physical_feature_rate_loss + backward_physical_feature_rate_loss) # lphys

        forward_residual_loss = F.smooth_l1_loss(
            motion.artifact_residual_reconstruction_forward, 
            motion.artifact_residual_target_forward.detach()
        )
        backward_residual_loss = F.smooth_l1_loss(
            motion.artifact_residual_reconstruction_backward,
            motion.artifact_residual_target_backward.detach(),
        )
        residual_reconstruction = 0.5 * (forward_residual_loss + backward_residual_loss) # lres

        physical_motion = torch.cat(
            (   
                motion.forward_sectors.linear_velocity,
                motion.forward_sectors.angular_velocity,
                motion.forward_sectors.translation_carrier_tangents.flatten(-2),
                motion.forward_sectors.rotation_carrier_tangents.flatten(-2),
                motion.forward_sectors.physical_scalars,
            ),
            dim=-1,
        )

        crs_var_loss = cross_covariance_loss(physical_motion, motion.forward_sectors.artifacts) # lxc

        # Lie exponentiation uses physical omega. Only the supervised residual is standardised so it remains numerically balanced with the other losses.
        # omega_rmse_radps = torch.sqrt(
        #     F.mse_loss(motion.forward_sectors.omega, teacher_omega)
        # )


        # strength = adversary_strength(
        #     int(self.current_epoch),
        #     int(self.hparams.artifact_adversary_warmup_epochs),
        #     float(self.hparams.artifact_adversary_max_strength),
        # )
        # adversary_omega, adversary_relative_rotation = self.artifact_adversary(
        #     motion.forward_sectors.artifacts, strength
        # )

        # target_relative_rotation = (
        #     target_rotation[:, 1:] @ target_rotation[:, :-1].transpose(-1, -2)
        # )
        # target_omega_normalised = (
        #     teacher_omega - self.angular_velocity_mean
        # ) / self.angular_velocity_std

        

        # artifact_losses = artifact_disentanglement_loss(
        #     artifact=motion.forward_sectors.artifacts,
        #     reconstructed_residual=motion.artifact_residual_reconstruction_forward,
        #     residual_target=motion.artifact_residual_target_forward,
        #     adversary_omega=adversary_omega,
        #     adversary_relative_rotation_6d=adversary_relative_rotation,
        #     target_omega_normalised=target_omega_normalised,
        #     target_relative_rotation=target_relative_rotation,
        #     physical_motion=physical_motion,
        # )


        total = (
            self.hparams.position_weight * position_loss
            + self.hparams.velocity_weight * velocity_loss
            + self.hparams.orientation_weight * orientation_loss
            + self.hparams.omega_weight * omega_loss
            + self.hparams.carrier_weight * carrier_loss
            + self.hparams.tangent_weight * tangent_loss
            + self.hparams.amplitude_weight * amplitude_loss
            + self.hparams.group_weight * group_prediction
            + self.hparams.reverse_weight * reverse_loss
            + self.hparams.invariant_prediction_weight * invariant_prediction
            + self.hparams.artifact_weight * artifact_prediction
            + self.hparams.variance_weight * artifact_variance
            + self.hparams.invariant_variance_weight * invariant_variance
            + self.hparams.physical_feature_rate_weight * physical_feature_rate_loss
            + self.hparams.artifact_residual_weight * residual_reconstruction
            + self.hparams.artifact_cross_covariance_weight * crs_var_loss
        )

        metrics = {
            "position_rmse_m": torch.sqrt(F.mse_loss(context.position, target_position)) if mask.translation else zero,
            "velocity_rmse_mps": torch.sqrt(F.mse_loss(motion.forward_sectors.linear_velocity, target_velocity[:, :-1])) if mask.translation else zero,
            "orientation_deg": torch.rad2deg(orientation_error).mean(),
            "omega_rmse_radps": torch.sqrt(F.mse_loss(motion.forward_sectors.angular_velocity, target_omega[:, :-1])) if mask.rotation else zero,
            "group_loss": group_prediction,
            "reverse_loss": reverse_loss,
            "carrier_loss": carrier_loss,
            "tangent_loss": tangent_loss,
            "artifact_prediction_loss": artifact_prediction,
            "artifact_residual_loss": residual_reconstruction,
            "artifact_cross_covariance_loss": crs_var_loss,            
        }

        return total, metrics

    def _step(self, batch, stage: str) -> float:
        prediction = self.model(batch["context_rgb"], batch["context_time"])
        loss, metrics = self._losses(prediction, batch)
        self.log_dict(
            {f"{stage}/{key}": value for key, value in {**metrics, "loss": loss}.items()},
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["context_rgb"].shape[0],
        )
        return loss


    def training_step(self, batch, batch_index):
        del batch_index
        return self._step(batch, "train")

    def validation_step(self, batch, batch_index):
        del batch_index
        self._step(batch, "validation")

    def test_step(self, batch, batch_index):
        del batch_index
        self._step(batch, "test")

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=max(1, self.trainer.max_epochs),
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }