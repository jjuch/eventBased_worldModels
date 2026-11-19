"""Lightning wrapper around the SE(3) observer. Dataset construction lives in train_kinematic."""

from __future__ import annotations

import lightning as L
import torch

from ball_world_model.models.observer import SE3Observer, SE3ObserverConfig
from .losses import LossSettings, observer_losses, physical_metrics
from .targets import future_targets, window_targets

class SE3ObserverModule(L.LightningModule):
    def __init__(
        self, 
        model_config: dict, 
        loss_config: dict | None = None,
        learning_rate: float = 3.0e-4, 
        weight_decay: float = 5.0e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        config = SE3ObserverConfig(**model_config)
        self.model = SE3Observer(config)
        self.loss_settings = LossSettings(**(loss_config or {}))

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        prediction = self.model(batch["context_rgb"], batch["context_time"])
        targets = window_targets(batch)
        future = (
            future_targets(batch, self.loss_settings.rollout_steps)
            if self.self.loss_settings.rollout > 0
            else None
        )
        total, components = observer_losses(
            prediction, targets, self.model.scales(), self.loss_settings, future,
        )

        batch_size = len(batch["context_rgb"])
        self.log(f"{stage}/loss", total, prog_bar=True, on_epoch=True, batch_size=batch_size)
        logged = {f"{stage}/loss_{name}": value.detach() for name, value in components.items()}
        logged.update({f"{stage}/{name}": value for name, value in physical_metrics(prediction, targets).items()})
        self.log_dict(logged, on_step=False, on_epoch=True, batch_size=batch_size)
        return total

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
