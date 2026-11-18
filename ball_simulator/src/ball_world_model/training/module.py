"""Lightning wrapper. Dataset construction remains owned by the existing project."""

from __future__ import annotations

import lightning as L
import torch

from ball_world_model.models.observer import SE3Observer, SE3ObserverConfig
from ball_world_model.models.teacher import SE3Teacher
from .losses import LossWeights, observer_losses
from .targets import current_targets

class SE3ObserverModule(L.LightningModule):
    def __init__(
        self, 
        model_config: dict, 
        loss_weights: dict | None = None,
        learning_rate: float = 3.0e-4, 
        weight_decay: float = 5.0e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        config = SE3ObserverConfig(**model_config)
        self.model = SE3Observer(config)
        self.teacher = SE3Teacher(config.carrier_points, config.carrier_radius)
        self.weights = LossWeights(**(loss_weights or {}))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _step(self, batch, stage: str):
        prediction = self.model(batch["context_rgb"], batch.get("context_time"))
        target = current_targets(batch)
        scales = {
            "position": self.model.position_std,
            "linear_velocity": self.model.linear_velocity_std,
            "angular_velocity": self.model.angular_velocity_std,
            "linear_acceleration": self.model.linear_acceleration_std,
            "angular_acceleration": self.model.angular_acceleration_std,
        }
        total, components = observer_losses(prediction, target, self.teacher, self.weights, scales)

        self.log(f"{stage}/loss", total, prog_bar=True, on_epoch=True, batch_size=len(batch["context_rgb"]))
        for name, value in components.items():
            self.log(f"{stage}/{name}", value, on_epoch=True, batch_size=len(batch["context_rgb"]))
        return total

    def training_step(self, batch: dict[str, torch.Tensor], batch_index: int) -> None:
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
