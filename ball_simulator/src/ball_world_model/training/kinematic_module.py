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
    def __init__(
        self,
        statistics: KinematicStatistics,
        *,
        task: TaskKind,
        embedding_dim: int = 256,
        keypoints: int = 8,
        motion_dim: int = 256,
        decoder_hidden_dim: int = 256,
        refinement_hidden_dim: int = 128,
        refinement_iterations: int = 1,
        delta_momentum: float = 0.01,
        default_frame_dt: float = 0.01,
        dropout: float = 0.0,
        temporal_depth: int | None = None,
        learning_rate: float = 3.0e-4,
        weight_decay: float = 5.0e-2,
        position_weight: float = 1.0,
        velocity_weight: float = 1.0,
        latent_prediction_weight: float = 0.1,
        kinematic_weight: float = 0.01,
        reverse_weight: float = 0.1,
        variance_weight: float = 0.01,
        rotation_weight: float = 1.0,
        angular_velocity_weight: float = 1.0,
        state_weight: float | None = None,
        translation_consistency_weight: float | None = None,
        rotation_consistency_weight: float | None = None,
        world_angular_velocity: bool = True,
    ) -> None:
        super().__init__()
        del state_weight, translation_consistency_weight, rotation_consistency_weight
        self.save_hyperparameters(ignore=("statistics",))
        self.model = KinematicStateEstimator(
            task=task,
            embedding_dim=embedding_dim,
            keypoints=keypoints,
            motion_dim=motion_dim,
            decoder_hidden_dim=decoder_hidden_dim,
            refinement_hidden_dim=refinement_hidden_dim,
            refinement_iterations=refinement_iterations,
            delta_momentum=delta_momentum,
            default_frame_dt=default_frame_dt,
            dropout=dropout,
            temporal_depth=temporal_depth,
            position_mean=statistics.position_mean,
            position_std=statistics.position_std,
            velocity_mean=statistics.linear_velocity_mean,
            velocity_std=statistics.linear_velocity_std,
        )

        for name, value in asdict(statistics).items():
            self.register_buffer(name, value.float())


    @staticmethod
    def _normalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (value - mean) / std

    @staticmethod
    def _denormalise(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return value * std + mean

    @staticmethod
    def _variance_floor_loss(value: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
        # Applied to batch/time samples per feature. It prevents constant representations without forcing every feature to unit variance.
        flattened = value.flatten(0, 1)
        std = torch.sqrt(flattened.var(dim=0, unbiased=False) + 1.0e-4)
        return torch.relu(floor - std).mean()

    @staticmethod
    def _std_metrics(prediction, target, prefix: str):
        prediction_std = prediction.std(dim=(0, 1), unbiased=False)
        target_std = target.std(dim=(0, 1), unbiased=False)
        metrics = {}
        for index, axis in enumerate(("x", "y", "z")):
            metrics[f"{prefix}_predicted_std_{axis}"] = prediction_std[index]
            metrics[f"{prefix}_target_std_{axis}"] = target_std[index]
            metrics[f"{prefix}_std_ratio_{axis}"] = (
                prediction_std[index] / target_std[index].clamp_min(1.0e-8)
            )
        return metrics



    def _translation_losses(self, prediction, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target_position = batch["context_position"]
        target_velocity = batch["context_linear_velocity"]
        normalised_position = self._normalise(
            target_position, self.position_mean, self.position_std
        )
        normalised_velocity = self._normalise(
            target_velocity, self.linear_velocity_mean, self.linear_velocity_std
        )
        position_loss = F.mse_loss(prediction.position, normalised_position)
        velocity_loss = F.mse_loss(prediction.linear_velocity, normalised_velocity)

        motion = prediction.motion
        target_latent = prediction.frame_latent.detach()
        forward_latent_loss = F.mse_loss(motion.predicted_next_embedding, target_latent[:, 1:])
        backward_latent_loss = F.mse_loss(motion.predicted_previous_embedding, target_latent[:, :-1])
        latent_prediction_loss = 0.5 * (forward_latent_loss + backward_latent_loss)
        reverse_loss = F.mse_loss(motion.backward_velocity, -motion.forward_velocity)


        position = self._denormalise(
            prediction.position, self.position_mean, self.position_std,
        )
        velocity = self._denormalise(
            prediction.linear_velocity,
            self.linear_velocity_mean,
            self.linear_velocity_std,
        )

        dt = torch.diff(batch["context_time"], dim=1).unsqueeze(-1)
        kinematic_residual = position[:, 1:] - position[:, :-1] - velocity[:, :-1] * dt
        kinematic_loss = torch.mean(kinematic_residual**2)
        variance_loss = self._variance_floor_loss(prediction.frame_latent) + self._variance_floor_loss(motion.forward_motion)

        total = (
            self.hparams.position_weight * position_loss
            + self.hparams.velocity_weight * velocity_loss
            + self.hparams.latent_prediction_weight * latent_prediction_loss
            + self.hparams.kinematic_weight * kinematic_loss
            + self.hparams.reverse_weight * reverse_loss
            + self.hparams.variance_weight * variance_loss
        )
        metrics = {
            "position_loss_normalised": position_loss,
            "velocity_loss_normalised": velocity_loss,
            "latent_prediction_loss": latent_prediction_loss,
            "forward_latent_loss": forward_latent_loss,
            "backward_latent_loss": backward_latent_loss,
            "reverse_velocity_loss": reverse_loss,
            "kinematic_loss_m2": kinematic_loss,
            "variance_loss": variance_loss,
            "feature_delta_abs_mean": motion.normalized_forward_difference.abs().mean(),
            "motion_std": motion.forward_motion.std(unbiased=False),
            "position_rmse_m": torch.sqrt(F.mse_loss(position, target_position)),
            "velocity_rmse_mps": torch.sqrt(F.mse_loss(velocity, target_velocity)),
            "kinematic_consistency_m": torch.sqrt(kinematic_loss),
        }
        metrics.update(self._std_metrics(position, target_position, "position"))
        metrics.update(self._std_metrics(velocity, target_velocity, "velocity"))
        for iteration, residual in enumerate(motion.refinement_residuals):
            metrics[f"refinement_residual_{iteration}_m"] = torch.sqrt(torch.mean(residual**2))
        
        return total, metrics

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
        # TODO: implement rotation so3 loss again
        # dt = torch.diff(batch["context_time"], dim=1).unsqueeze(-1)
        # increments = so3_exp(dt * omega[:, :-1])
        # if self.hparams.world_angular_velocity:
        #     expected_next_rotation = increments @ prediction.rotation_matrix[:, :-1]
        # else:
        #     expected_next_rotation = prediction.rotation_matrix[:, :-1] @ increments
        # rotation_consistency = rotation_geodesic_error(
        #     prediction.rotation_matrix[:, 1:], expected_next_rotation
        # ).mean()
        total = (
            self.hparams.rotation_weight * orientation_loss
            + self.hparams.angular_velocity_weight * omega_loss
        )
        metrics = {
            "orientation_deg": torch.rad2deg(orientation_error).mean(),
            "omega_rmse_radps": torch.sqrt(F.mse_loss(omega, target_omega)),
            "orientation_loss": orientation_loss,
            "omega_loss_normalised": omega_loss,
        }
        return total, metrics
    

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