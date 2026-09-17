from __future__ import annotations

from dataclasses import asdict
import lightning as L

import torch
import torch.nn.functional as F

from ball_world_model.models.rotation import quaternion_xyzw_to_matrix, rotation_geodesic_error
from ball_world_model.models.structured_kinematic_estimator import StructuredSO3StateEstimator
from ball_world_model.models.structured_so3 import RichPhysicalStateTeacher
from .kinematic_module import KinematicStatistics


class StructuredSO3ObservabilityModule(L.LightningModule):
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
        variance_weight: float = 0.01, 
        geometric_channels: int = 24,
        latent_architecture: str = "structured_so3_artifacts", 
        **model: object
    ) -> None:
        super().__init__()
        if task != "rotation":
            raise ValueError("The current patch only supports task='rotation' for now.")
        
        self.save_hyperparameters(ignore=["statistics"])
        self.model = StructuredSO3StateEstimator(
            geometric_channels=geometric_channels,
            **model
        )
        self.teacher = RichPhysicalStateTeacher(geometric_channels)
        for n, v in asdict(statistics).items():
            self.register_buffer(n, v.float())

    @staticmethod
    def _denormalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return value * std + mean

    @staticmethod
    def _variance_floor(value: torch.Tensor, floor: float = 0.25) -> torch.Tensor:
        flat = value.flatten(0, 1)
        std = torch.sqrt(flat.var(0, unbiased=False) + 1e-4)
        return torch.relu(floor - std).mean()

    def _losses(self, prediction, batch):
        target_rotation = quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"])
        target_omega = batch["context_angular_velocity"]
        teacher_rotation, teacher_orbit = self.teacher.context(target_rotation)
        teacher_omega, teacher_tangent = self.teacher.motion(
            target_rotation[:, :-1], target_omega[:, :-1]
        )
        context = prediction.context_sectors
        motion = prediction.motion

        orientation_error = rotation_geodesic_error(context.rotation, teacher_rotation)
        orientation_loss = orientation_error.mean()

        marker_loss = F.mse_loss(context.carrier, teacher_orbit)
        omega_loss = F.mse_loss(motion.forward_sectors.omega, teacher_omega)
        tangent_loss = F.mse_loss(motion.forward_sectors.carrier_tangent, teacher_tangent)
        amplitude_loss = F.mse_loss(context.amplitudes, teacher_rotation)

        fwd_group = rotation_geodesic_error(
            motion.predicted_next_rotation, context.rotation[:, 1:].detach()
        ).square().mean()
        bwd_group = rotation_geodesic_error(
            motion.predicted_previous_rotation, context.rotation[:, :-1].detach()
        ).square().mean()
        group_prediction = 0.5 * (fwd_group + bwd_group)

        reverse_loss = F.mse_loss(motion.backward_sectors.omega, -motion.forward_sectors.omega)
        artifact_size = context.artifact.shape[-1]
        fwd_artifact = F.mse_loss(motion.predicted_next_embedding[..., -artifact_size:], context.artifact[:, 1:].detach())
        bwd_artifact = F.mse_loss(motion.predicted_previous_embedding[..., -artifact_size:], context.artifact[:, :-1].detach())
        artifact_prediction = 0.5 * (fwd_artifact + bwd_artifact)

        variance = self._variance_floor(context.artifact) + self._variance_floor(motion.forward_sectors.artifact)

        total = (
            self.hparams.orientation_weight * orientation_loss
            + self.hparams.omega_weight * omega_loss
            + self.hparams.carrier_weight * marker_loss
            + self.hparams.tangent_weight * tangent_loss
            + self.hparams.amplitude_weight * amplitude_loss
            + self.hparams.group_weight * group_prediction
            + self.hparams.artifact_weight * artifact_prediction
            + self.hparams.reverse_weight * reverse_loss
            + self.hparams.variance_weight * variance
        )

        metrics = {
            "orientation_deg": torch.rad2deg(orientation_error).mean(),
            "omega_rmse_radps":torch.sqrt(omega_loss),
            "carrier_loss": marker_loss,
            "tangent_loss": tangent_loss,
            "group_loss_rad2":group_prediction,
            "reverse_loss": reverse_loss,
            "artifact_prediction_loss": artifact_prediction,
            "amplitude_loss":amplitude_loss,
            "variance_loss": variance}

        return total, metrics

    def _step(self, batch, stage: str) -> float:
        prediction = self.model(batch["context_rgb"], batch["context_time"])
        loss, metrics = self._losses(prediction, batch)
        self.log_dict(
            {f"{stage} / {key}": value for key, value in {**metrics, "loss": loss}.items()},
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