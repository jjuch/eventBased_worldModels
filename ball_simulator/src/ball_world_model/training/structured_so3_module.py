from __future__ import annotations

from dataclasses import asdict
import lightning as L

import torch
import torch.nn.functional as F

from ball_world_model.models.rotation import quaternion_xyzw_to_matrix, rotation_geodesic_error
from ball_world_model.models.structured_kinematic_estimator import StructuredSO3StateEstimator
from ball_world_model.models.structured_so3 import RichPhysicalStateTeacher
from ball_world_model.models.artifact_residual import PhysicsAdversary, adversary_strength
from .kinematic_module import KinematicStatistics
from .artifact_disentanglement import artifact_disentanglement_loss


class StructuredSO3ObservabilityModule(L.LightningModule):
    # The structured estimator exposes omega directly in physical rad/s.
    outputs_physical_units = True # TODO: find a more elegant to circumvent this; hack for model_loader > denormalised_prediction

    def __init__(
        self, 
        statistics: KinematicStatistics, 
        *, 
        task: str = "rotation", 
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05, 
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
        geometric_channels: int = 24,
        physical_invariant_dim: int = 16,
        latent_architecture: str = "structured_so3_artifacts",
        artifact_mode: str = "physics_residual_adversarial",
        physical_feature_rate_weight: float = 0.01,
        artifact_residual_weight: float = 0.01,
        artifact_cross_covariance_weight: float = 0.001,
        artifact_adversary_omega_weight: float = 0.01,
        artifact_adversary_rotation_weight: float = 0.005,
        artifact_adversary_max_strength: float = 0.02,
        artifact_adversary_warmup_epochs: int = 10,
        artifact_residual_hidden_channels: int = 128,
        artifact_adversary_hidden_dim: int = 128,
        **model: object
    ) -> None:
        super().__init__()
        if task != "rotation":
            raise ValueError("The current patch only supports task='rotation' for now.")
        
        self.save_hyperparameters(ignore=["statistics"])
        self.model = StructuredSO3StateEstimator(
            geometric_channels=geometric_channels,
            physical_invariant_dim=physical_invariant_dim,
            artifact_mode=artifact_mode,
            artifact_residual_hidden_channels=artifact_residual_hidden_channels,
            **model
        )
        artifact_dim = self.model.motion_head.layout.motion_artifact_dim
        self.artifact_adversary = PhysicsAdversary(
            artifact_dim, artifact_adversary_hidden_dim
        )
        self.teacher = RichPhysicalStateTeacher(geometric_channels)
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

    def _losses(self, prediction, batch):
        target_rotation = quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"])
        target_omega = batch["context_angular_velocity"]
        teacher_amplitudes, teacher_orbit = self.teacher.context(target_rotation)
        teacher_omega, teacher_tangent = self.teacher.motion(
            target_rotation[:, :-1], target_omega[:, :-1]
        )
        context = prediction.context_sectors
        motion = prediction.motion

        orientation_error = rotation_geodesic_error(context.rotation, target_rotation)
        orientation_loss = orientation_error.mean()

        carrier_loss = F.mse_loss(context.carrier, teacher_orbit)

        # Lie exponentiation uses physical omega. Only the supervised residual is standardised so it remains numerically balanced with the other losses.
        omega_error_normalised = (motion.forward_sectors.omega - teacher_omega) / self.angular_velocity_std
        omega_loss = omega_error_normalised.square().mean()
        omega_rmse_radps = torch.sqrt(
            F.mse_loss(motion.forward_sectors.omega, teacher_omega)
        )
        tangent_loss = F.mse_loss(motion.forward_sectors.carrier_tangent, teacher_tangent)
        amplitude_loss = F.mse_loss(context.amplitudes, teacher_amplitudes)

        fwd_group_loss = rotation_geodesic_error(
            motion.predicted_next_rotation, context.rotation[:, 1:].detach()
        ).square().mean()
        bwd_group_loss = rotation_geodesic_error(
            motion.predicted_previous_rotation, context.rotation[:, :-1].detach()
        ).square().mean()
        group_prediction = 0.5 * (fwd_group_loss + bwd_group_loss)

        reverse_loss = F.mse_loss(motion.backward_sectors.omega, -motion.forward_sectors.omega)

        forward_invariant_loss = F.mse_loss(motion.predicted_next_invariants, context.physical_invariants[:, 1:].detach())
        backward_invariant_loss = F.mse_loss(motion.predicted_previous_invariants, context.physical_invariants[:, :-1].detach())
        invariant_prediction = 0.5 * (forward_invariant_loss + backward_invariant_loss)

        fwd_artifact = F.mse_loss(motion.predicted_next_artifacts, context.artifacts[:, 1:].detach())
        bwd_artifact = F.mse_loss(motion.predicted_previous_artifacts, context.artifacts[:, :-1].detach())
        artifact_prediction = 0.5 * (fwd_artifact + bwd_artifact)
        artifact_variance = self._variance_floor(context.artifacts) + self._variance_floor(motion.forward_sectors.artifacts)

        invariant_variance = self._variance_floor(context.physical_invariants) + self._variance_floor(motion.forward_sectors.physical_invariants)

        forward_physical_feature_rate_loss = F.smooth_l1_loss(
            motion.physical_feature_rate_forward,
            motion.normalised_forward_difference.detach(),
        )
        backward_physical_feature_rate_loss = F.smooth_l1_loss(
            motion.physical_feature_rate_backward,
            -motion.normalised_forward_difference.detach(),
        )
        physical_feature_rate_loss = 0.5 * (forward_physical_feature_rate_loss + backward_physical_feature_rate_loss)

        strength = adversary_strength(
            int(self.current_epoch),
            int(self.hparams.artifact_adversary_warmup_epochs),
            float(self.hparams.artifact_adversary_max_strength),
        )
        adversary_omega, adversary_relative_rotation = self.artifact_adversary(
            motion.forward_sectors.artifacts, strength
        )

        target_relative_rotation = (
            target_rotation[:, 1:] @ target_rotation[:, :-1].transpose(-1, -2)
        )
        target_omega_normalised = (
            teacher_omega - self.angular_velocity_mean
        ) / self.angular_velocity_std

        physical_motion = torch.cat(
            (
                omega_error_normalised + target_omega_normalised,
                motion.forward_sectors.amplitude_rate,
                motion.forward_sectors.carrier_tangent.flatten(-2) / self.angular_velocity_std.norm().clamp_min(1.0),
                motion.forward_sectors.physical_invariants,
            ),
            dim=-1,
        )

        artifact_losses = artifact_disentanglement_loss(
            artifact=motion.forward_sectors.artifacts,
            reconstructed_residual=motion.artifact_residual_reconstruction_forward,
            residual_target=motion.artifact_residual_target_forward,
            adversary_omega=adversary_omega,
            adversary_relative_rotation_6d=adversary_relative_rotation,
            target_omega_normalised=target_omega_normalised,
            target_relative_rotation=target_relative_rotation,
            physical_motion=physical_motion,
        )
        backward_residual_loss = F.smooth_l1_loss(
            motion.artifact_residual_reconstruction_backward,
            motion.artifact_residual_target_backward.detach(),
        )
        residual_reconstruction = 0.5 * (
            artifact_losses["artifact_residual_reconstruction_loss"]
            + backward_residual_loss
        )

        total = (
            self.hparams.orientation_weight * orientation_loss
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
            + self.hparams.artifact_cross_covariance_weight
            * artifact_losses["artifact_cross_covariance_loss"]
            + self.hparams.artifact_adversary_omega_weight
            * artifact_losses["artifact_adversary_omega_loss"]
            + self.hparams.artifact_adversary_rotation_weight
            * artifact_losses["artifact_adversary_rotation_loss"]

        )

        metrics = {
            "orientation_deg": torch.rad2deg(orientation_error).mean(),
            "omega_rmse_radps": omega_rmse_radps,
            "orientation_loss_rad": orientation_loss,
            "omega_loss_normalised": omega_loss,
            "carrier_loss": carrier_loss,
            "tangent_loss": tangent_loss,
            "group_loss_rad2":group_prediction,
            "reverse_loss": reverse_loss,
            "artifact_prediction_loss": artifact_prediction,
            "invariant_prediction_loss": invariant_prediction,
            "amplitude_loss": amplitude_loss,
            "artifact_variance_loss": artifact_variance,
            "invariant_variance_loss": invariant_variance,
            "physical_feature_rate_loss": physical_feature_rate_loss,
            "artifact_residual_reconstruction_loss": residual_reconstruction,
            "artifact_cross_covariance_loss": artifact_losses["artifact_cross_covariance_loss"],
            "artifact_adversary_omega_loss": artifact_losses["artifact_adversary_omega_loss"],
            "artifact_adversary_rotation_loss": artifact_losses["artifact_adversary_rotation_loss"],
            "artifact_adversary_strength": torch.as_tensor(strength, device=self.device),
            "context_invariant_std": context.physical_invariants.std(unbiased=False),
            "motion_invariant_std": motion.forward_sectors.physical_invariants.std(unbiased=False),
            "motion_artifact_std": motion.forward_sectors.artifacts.std(unbiased=False),
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