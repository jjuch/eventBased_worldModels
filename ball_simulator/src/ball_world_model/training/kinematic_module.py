from __future__ import annotations

from dataclasses import asdict, dataclass

import lightning as L
import torch
import torch.nn.functional as F

from ball_world_model.models.kinematic_estimator import KinematicStateEstimator, TaskKind
from ball_world_model.models.rotation import (
    quaternion_xyzw_to_matrix,
    rotation_geodesic_error,
    so3_exp,
)


@dataclass(frozen=True)
class KinematicStatistics:
    position_mean: torch.Tensor
    position_std: torch.Tensor
    linear_velocity_mean: torch.Tensor
    linear_velocity_std: torch.Tensor
    angular_velocity_mean: torch.Tensor
    angular_velocity_std: torch.Tensor


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    return values.std(dim=(0, 1), unbiased=False).clamp_min(1.0e-6)


def compute_kinematic_statistics(loader) -> KinematicStatistics:
    positions, velocities, omegas = [], [], []
    for batch in loader:
        positions.append(batch["context_position"].float())
        velocities.append(batch["context_linear_velocity"].float())
        omegas.append(batch["context_angular_velocity"].float())

    position = torch.cat(positions)
    velocity = torch.cat(velocities)
    omega = torch.cat(omegas)
    return KinematicStatistics(
        position.mean((0, 1)), _safe_std(position),
        velocity.mean((0, 1)), _safe_std(velocity),
        omega.mean((0, 1)), _safe_std(omega),
    )


class KinematicObservabilityModule(L.LightningModule):
    """
    Stage-wise observer with dense per-frame supervision and geometric consistency.

    The primary output is a temporal latent sequence. State heads supervise every observed frame, not only the last frame, which makes differential-state learning much easier to diagnose.
    """
    def __init__(
        self,
        statistics: KinematicStatistics,
        *,
        task: TaskKind,
        embedding_dim: int = 256,
        temporal_depth:int = 4,
        keypoints: int = 8,
        dropout: float = 0.0,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 5.0e-2,
        state_weight: float = 1.0,
        translation_consistency_weight: float = 0.1,
        rotation_consistency_weight: float = 0.1,
        world_angular_velocity: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("statistics",))
        self.model = KinematicStateEstimator(
            task=task,
            embedding_dim=embedding_dim,
            temporal_depth=temporal_depth,
            keypoints=keypoints,
            dropout=dropout,
        )

        for name, value in asdict(statistics).items():
            self.register_buffer(name, value.float())


    @staticmethod
    def _normalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (value - mean) / std

    @staticmethod
    def _denormalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return value * std + mean


    def _translation_losses(self, prediction, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_position = batch["context_position"]
        target_velocity = batch["context_linear_velocity"]
        position_loss = F.mse_loss(
            prediction.position,
            self._normalise(target_position, self.position_mean, self.position_std),
        )
        velocity_loss = F.mse_loss(
            prediction.linear_velocity,
            self._normalise(target_velocity, self.linear_velocity_mean, self.linear_velocity_std),
        )
        position = self._denormalise(
            prediction.position, self.position_mean, self.position_std,
        )
        velocity = self._denormalise(
            prediction.linear_velocity,
            self.linear_velocity_mean,
            self.linear_velocity_std,
        )

        dt = torch.diff(batch["context_time"], dim=1).unsqueeze(-1)
        expected_next_position = position[:, :-1] + dt * velocity[:, :-1]
        consistency = F.mse_loss(position[:, 1:], expected_next_position)
        metrics = {
            "position_rmse_m": torch.sqrt(F.mse_loss(position[:, -1], target_position[:, -1])),
            "velocity_rmse_mps": torch.sqrt(F.mse_loss(velocity[:, -1], target_velocity[:, -1])),
            "translation_consistency_m": torch.sqrt(consistency),
        }
        loss = position_loss + velocity_loss + self.hparams.translation_consistency_weight * consistency
        return loss, metrics

    def _rotation_losses(self, prediction, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_rotation = quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"])
        target_omega = batch["context_angular_velocity"]
        orientation_error = rotation_geodesic_error(prediction.rotation_matrix, target_rotation)
        orientation_loss = orientation_error.mean()
        omega_loss = F.mse_loss(
            prediction.angular_velocity,
            self._normalise(
                target_omega, self.angular_velocity_mean, self.angular_velocity_std
            ),
        )
        omega = self._denormalise(
            prediction.angular_velocity,
            self.angular_velocity_mean,
            self.angular_velocity_std,
        )
        dt = torch.diff(batch["context_time"], dim=1).unsqueeze(-1)
        increments = so3_exp(dt * omega[:, :-1])
        if self.hparams.world_angular_velocity:
            expected_next_rotation = increments @ prediction.rotation_matrix[:, :-1]
        else:
            expected_next_rotation = prediction.rotation_matrix[:, :-1] @ increments
        rotation_consistency = rotation_geodesic_error(
            prediction.rotation_matrix[:, 1:], expected_next_rotation
        ).mean()
        metrics = {
            "orientation_deg": torch.rad2deg(orientation_error[:, -1]).mean(),
            "omega_rmse_radps": torch.sqrt(F.mse_loss(omega[:, -1], target_omega[:, -1])),
            "rotation_consistency_deg": torch.rad2deg(rotation_consistency),
        }
        loss = orientation_loss + omega_loss + self.hparams.rotation_consistency_weight * rotation_consistency
        return loss, metrics
    

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        prediction = self.model(batch["context_rgb"])
        loss = torch.zeros((), device=self.device)
        metrics: dict[str, torch.Tensor] = {}

        if self.hparams.task in {"translation", "combined"}:
            translation_loss, translation_metrics = self._translation_losses(prediction, batch)
            loss = loss + translation_loss
            metrics.update(translation_metrics)
        if self.hparams.task in {"rotation", "combined"}:
            rotation_loss, rotation_metrics = self._rotation_losses(prediction, batch)
            loss = loss + rotation_loss
            metrics.update(rotation_metrics)

        loss = self.hparams.state_weight * loss

        logged = {f"{stage}/{name}": value for name, value, in metrics.items()}
        logged[f"{stage}/loss"] = loss
        self.log_dict(
            logged,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["context_rgb"].shape[0],
        )
        return loss


    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> torch.Tensor:
        del batch_index
        return self._step(batch, "train")


    def validation_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
        del batch_index
        self._step(batch, "validation")


    def test_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
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