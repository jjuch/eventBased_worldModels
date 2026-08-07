from __future__ import annotations

from dataclasses import asdict

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from ball_world_model.models.rotation import (
    quaternion_xyzw_to_matrix,
    rotation_geodesic_error
)
from ball_world_model.models.state_estimator import StateEstimator
from .state_normaliser import StateStatistics


class ObservabilityModule(L.LightningModule):
    def __init__(
        self,
        statistics: StateStatistics,
        *,
        embedding_dim: int = 384,
        temporal_depth:int = 4,
        temporal_heads: int = 6,
        maximum_frames: int = 32,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 5.0e-2,
        orientation_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("statistics",))
        self.model = StateEstimator(
            embedding_dim=embedding_dim,
            temporal_depth=temporal_depth,
            temporal_heads=temporal_heads,
            maximum_frames=maximum_frames,
        )

        for name, value in asdict(statistics).items():
            self.register_buffer(name, value.float())


    @staticmethod
    def _normalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (value - mean) / std

    @staticmethod
    def _denormalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return value * std + mean


    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        prediction = self.model(batch["context_rgb"])
        target_position = batch["context_position"][:, -1]
        target_velocity = batch["context_linear_velocity"][:, -1]
        target_angular_velocity = batch["context_angular_velocity"][:, -1]
        target_rotation = quaternion_xyzw_to_matrix(batch["context_quaternion_xyzw"][:, -1])

        position_loss = F.mse_loss(
            prediction.position,
            self._normalise(target_position, self.position_mean, self.position_std),
        )
        velocity_loss = F.mse_loss(
            prediction.linear_velocity,
            self._normalise(target_velocity, self.linear_velocity_mean, self.linear_velocity_std),
        )
        angular_velocity_loss = F.mse_loss(
            prediction.angular_velocity,
            self._normalise(target_angular_velocity, self.angular_velocity_mean, self.angular_velocity_std),
        )
        orientation_error = rotation_geodesic_error(prediction.rotation_matrix, target_rotation)
        orientation_loss = orientation_error.mean()

        loss = position_loss + velocity_loss + angular_velocity_loss + (
            self.hparams.orientation_weight * orientation_loss
        )

        predicted_position = self._denormalise(prediction.position, self.position_mean, self.position_std)
        predicted_velocity = self._denormalise(prediction.linear_velocity, self.linear_velocity_mean, self.linear_velocity_std)
        predicted_angular_velocity = self._denormalise(prediction.angular_velocity, self.angular_velocity_mean, self.angular_velocity_std)

        batch_size = batch["context_rgb"].shape[0]
        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/position_rmse_m": torch.sqrt(
                F.mse_loss(predicted_position, target_position)
            ),
            f"{stage}/velocity_rmse_mps": torch.sqrt(
                F.mse_loss(predicted_velocity, target_velocity)
            ),
            f"{stage}/omega_rmse_radps": torch.sqrt(
                F.mse_loss(predicted_angular_velocity, target_angular_velocity)
            ),
            f"{stage}/orientation_deg": torch.rad2deg(orientation_error).mean(),
        }
        self.log_dict(
            metrics,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
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


    def configure_optimiser(self):
        optimiser = torch.optim.AdamW(
            self.parameter(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=max(1, self.trainer.max_epochs),
        )
        return {
            "optimiser": optimiser,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }