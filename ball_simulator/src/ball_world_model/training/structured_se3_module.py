"""Unified, dimensionless training module for translation, rotation and combined rigid motion."""
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
from .se3_balanced_losses import (
    balanced_mean,
    camera_basis,
    component_rmse,
    dimensionless_group_loss,
    dimensionless_reverse_loss,
    dimensionless_twist_loss,
    gradient_norm,
    rms_normalised_loss,
    safe_scale,
    world_to_camera_vector,
)


class StructuredSE3ObservabilityModule(L.LightningModule):
    """Train one fixed-width latent with one dimensionless six-dimensional twist."""

    # The structured SE(3) estimator exposes every canonical physical state directly in physical units:
    # position [m], linear velocity [m/s],rotation matrix, and angular velocity [rad/s].
    outputs_physical_units = True

    def __init__(
        self, 
        statistics: KinematicStatistics, 
        *, 
        task: str = "combined", 
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        configuration_weight: float = 1.0,
        twist_weight: float = 1.0,
        orientation_scale_deg: float = 5.0,
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
        diagnostic_gradient_interval: int = 100,
        camera_location: tuple[float, float, float] | list[float] | None = None,
        camera_target: tuple[float, float, float] | list[float] | None = None,
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
            
        if (camera_location is None) != (camera_target is None):
            raise ValueError(
                "camera_location and camera_target must either both be provided "
                "or both be omitted."
            )
        if camera_location is None:
            basis = torch.empty(0, 3, dtype=torch.float32)
        else:
            basis = camera_basis(
                torch.tensor(camera_location, dtype=torch.float32),
                torch.tensor(camera_target, dtype=torch.float32),
            )
        self.register_buffer("camera_velocity_basis", basis)

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
        dt = torch.diff(batch["context_time"], dim=1).unsqueeze(-1).clamp_min(1.0e-8)

        (
            teacher_translation_amplitudes, 
            teacher_rotation_amplitudes, 
            teacher_translation_carrier, 
            teacher_rotation_carrier
        ) = self.teacher.context(target_position, target_rotation, mask)
        (
            teacher_translation_carrier_rate, 
            teacher_rotation_carrier_rate 
        ) = self.teacher.motion(
            target_position[:, :-1], target_rotation[:, :-1], target_velocity[:, :-1], target_omega[:, :-1], mask
        )

        position_loss = (
            ((context.position - target_position) / safe_scale(self.position_std).to(context.position)).square().mean() if mask.translation else zero # lp
        )

        orientation_error = rotation_geodesic_error(context.rotation, target_rotation) if mask.rotation else zero
        orientation_scale = torch.deg2rad(context.position.new_tensor(float(self.hparams.orientation_scale_deg)))
        orientation_loss = (
            (orientation_error / orientation_scale.clamp_min(1.0e-6)).square().mean()
            if mask.rotation else zero # lr
        )
        configuration_loss = balanced_mean(
            position_loss if mask.translation else None,
            orientation_loss if mask.rotation else None,
        )

        twist_loss = dimensionless_twist_loss(
            motion.forward_sectors.linear_velocity if mask.translation else None,
            target_velocity[:, :-1] if mask.translation else None,
            self.linear_velocity_std if mask.translation else None,
            motion.forward_sectors.angular_velocity if mask.rotation else None,
            target_omega[:, :-1] if mask.rotation else None,
            self.angular_velocity_std if mask.rotation else None,
        )

        translation_carrier_loss = translation_carrier_scale = None
        rotation_carrier_loss = rotation_carrier_scale = None
        if mask.translation:
            translation_carrier_loss, translation_carrier_scale = rms_normalised_loss(
                context.translation_carrier, teacher_translation_carrier
            )
        if mask.rotation:
            rotation_carrier_loss, rotation_carrier_scale = rms_normalised_loss(
                context.rotation_carrier, teacher_rotation_carrier
            )
        carrier_loss = balanced_mean(translation_carrier_loss, rotation_carrier_loss) # lc

        translation_tangent_loss = translation_tangent_scale = None
        rotation_tangent_loss = rotation_tangent_scale = None
        if mask.translation:
            translation_tangent_loss, translation_tangent_scale = rms_normalised_loss(
                motion.forward_sectors.translation_carrier_tangents,
                teacher_translation_carrier_rate,
            )
        if mask.rotation:
            rotation_tangent_loss, rotation_tangent_scale = rms_normalised_loss(
                motion.forward_sectors.rotation_carrier_tangents,
                teacher_rotation_carrier_rate,
            )
        tangent_loss = balanced_mean(translation_tangent_loss, rotation_tangent_loss) # lt

        translation_amplitude_loss = (
            F.mse_loss(context.translation_amplitudes, teacher_translation_amplitudes) 
            if mask.translation else None
        )
        rotation_amplitude_loss = (
            F.mse_loss(context.rotation_amplitudes, teacher_rotation_amplitudes) 
            if mask.rotation else None
        )
        amplitude_loss = balanced_mean(translation_amplitude_loss, rotation_amplitude_loss) # la

        position_step_scale = (
            dt.mean() * torch.sqrt(safe_scale(self.linear_velocity_std).square().mean()) 
            if mask.translation else None
        )
        rotation_step_scale = (
            dt.mean() * torch.sqrt(safe_scale(self.angular_velocity_std).square().mean())
            if mask.rotation else None
        )

        group_loss, translation_group_loss, rotation_group_loss = dimensionless_group_loss(
            motion.predicted_next_position if mask.translation else None,
            context.position[:, 1:].detach() if mask.translation else None,
            motion.predicted_previous_position if mask.translation else None,
            context.position[:, :-1].detach() if mask.translation else None,
            position_step_scale,
            motion.predicted_next_rotation if mask.rotation else None,
            context.rotation[:, 1:].detach() if mask.rotation else None,
            motion.predicted_previous_rotation if mask.rotation else None,
            context.rotation[:, :-1].detach() if mask.rotation else None,
            rotation_step_scale,
        )
        reverse_loss = dimensionless_reverse_loss(
            motion.forward_sectors.linear_velocity if mask.translation else None,
            motion.backward_sectors.linear_velocity if mask.translation else None,
            self.linear_velocity_std if mask.translation else None,
            motion.forward_sectors.angular_velocity if mask.rotation else None,
            motion.backward_sectors.angular_velocity if mask.rotation else None,
            self.angular_velocity_std if mask.rotation else None,
        )


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

        physical_parts = []
        if mask.translation:
            physical_parts.extend((
                motion.forward_sectors.linear_velocity / safe_scale(self.linear_velocity_std).to(motion.forward_sectors.linear_velocity),
                motion.forward_sectors.translation_carrier_tangents / safe_scale(translation_tangent_scale).to(motion.forward_sectors.translation_carrier_tangents),
            ))
        if mask.rotation:
            physical_parts.extend((
                motion.forward_sectors.angular_velocity / safe_scale(rotation_tangent_scale).to(motion.forward_sectors.rotation_carrier_tangents),
            ))
        physical_motion = torch.cat(
            tuple(part.flatten(-2) if part.ndim == 4 else part for part in physical_parts)
            + (motion.forward_sectors.physical_scalars,),
            dim=-1,
        )

        cross_covariance = cross_covariance_loss(physical_motion, motion.forward_sectors.artifacts) # lxc


        total = (
            self.self.hparams.configuration_weight * configuration_loss
            + self.hparams.twist_weight * twist_loss
            + self.hparams.carrier_weight * carrier_loss
            + self.hparams.tangent_weight * tangent_loss
            + self.hparams.amplitude_weight * amplitude_loss
            + self.hparams.group_weight * group_loss
            + self.hparams.reverse_weight * reverse_loss
            + self.hparams.invariant_prediction_weight * invariant_prediction
            + self.hparams.artifact_weight * artifact_prediction
            + self.hparams.variance_weight * artifact_variance
            + self.hparams.invariant_variance_weight * invariant_variance
            + self.hparams.physical_feature_rate_weight * physical_feature_rate_loss
            + self.hparams.artifact_residual_weight * residual_reconstruction
            + self.hparams.artifact_cross_covariance_weight * cross_covariance
        )

        metrics = {
            "configuration_loss": configuration_loss.detach(),
            "position_loss_normalised": position_loss.detach(),
            "orientation_loss_normalised": orientation_loss.detach(),
            "twist_loss": twist_loss.detach(),
            "carrier_loss_balanced": carrier_loss.detach(),
            "tangent_loss_balanced": tangent_loss.detach(),
            "group_loss_balanced": group_loss.detach(),
            "reverse_loss_balanced": reverse_loss.detach(),
            "position_rmse_m": torch.sqrt(F.mse_loss(context.position, target_position)) if mask.translation else zero,
            "velocity_rmse_mps": torch.sqrt(F.mse_loss(motion.forward_sectors.linear_velocity, target_velocity[:, :-1])) if mask.translation else zero,
            "orientation_deg": torch.rad2deg(orientation_error).mean() if mask.rotation else zero,
            "omega_rmse_radps": torch.sqrt(F.mse_loss(motion.forward_sectors.angular_velocity, target_omega[:, :-1])) if mask.rotation else zero,
            "translation_carrier_loss_normalised": translation_carrier_loss.detach() if mask.translation else zero,
            "rotation_carrier_loss_normalised": rotation_carrier_loss.detach() if mask.rotation else zero,
            "translation_tangent_loss_normalised": translation_tangent_loss.detach() if mask.translation else zero,
            "rotation_tangent_loss_normalised": rotation_tangent_loss.detach() if mask.rotation else zero,
            "translation_carrier_target_rms": translation_carrier_scale.detach() if mask.translation else zero,
            "rotation_carrier_target_rms": rotation_carrier_scale.detach() if mask.rotation else zero,
            "translation_tangent_target_rms": translation_tangent_scale.detach() if mask.translation else zero,
            "rotation_tangent_target_rms": rotation_tangent_scale.detach() if mask.rotation else zero,
            "translation_group_loss_normalised": translation_group_loss.detach() if mask.translation else zero,
            "rotation_group_loss_normalised": rotation_group_loss.detach() if mask.rotation else zero,
            "artifact_prediction_loss": artifact_prediction.detach(),
            "artifact_residual_loss": residual_reconstruction.detach(),
            "artifact_cross_covariance_loss": cross_covariance.detach(),            
        }

        if mask.translation and self.camera_velocity_basis.numel() != 0:
            velocity_error_camera = world_to_camera_vector(
                motion.forward_sectors.linear_velocity - target_velocity[:, :-1],
                self.camera_velocity_basis,
            )
            camera_rmse = component_rmse(velocity_error_camera)
            metrics.update({
                "velocity_rmse_camera_right_mps": camera_rmse[0],
                "velocity_rmse_camera_up_mps": camera_rmse[1],
                "velocity_rmse_camera_depth_mps": camera_rmse[2],
            })

        diagnostic_losses = {
            "twist": twist_loss,
            "translation_tangent": translation_tangent_loss,
            "rotation_tangent": rotation_tangent_loss,
            "translation_group": translation_group_loss,
            "rotation_group": rotation_group_loss,
        }
        return total, metrics, diagnostic_losses

    def _shared_motion_parameters(self) -> list[torch.nn.Parameter]:
        modules = (self.model.motion_encoder, self.model.motion_head)
        return [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]

    def _gradient_diagnostics(self, losses: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
        interval = int(self.hparams.diagnostic_gradient_interval)
        if interval <= 0 or not self.training or self.global_step % interval != 0:
            return {}
        parameters = self._shared_motion_parameters()
        return {
            f"gradient_norm_{name}": gradient_norm(loss, parameters)
            for name, loss in losses.items()
            if loss is not None and loss.requires_grad
        }

    def _step(self, batch, stage: str) -> float:
        prediction = self.model(batch["context_rgb"], batch["context_time"])
        loss, metrics, diagnostic_losses = self._losses(prediction, batch)
        if stage == "train":
            metrics.update(self._gradient_diagnostics(diagnostic_losses))
        self.log_dict(
            {f"{stage}/{key}": value for key, value in {**metrics, "loss": loss}.items()},
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["context_rgb"].shape[0],
        )

        self.log(f"{stage}/loss_progress", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["context_rgb"].shape[0])
        self.log(f"{stage}/twist_progress", metrics["twist_loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["context_rgb"].shape[0])
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