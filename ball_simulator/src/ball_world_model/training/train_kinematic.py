from __future__ import annotations

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import yaml

from ball_world_model.config import DataConfig
from ball_world_model.data.dataset import RenderedTrajectoryDataset, build_dataloader
from .kinematic_module import (
    KinematicObservabilityModule,
    compute_kinematic_statistics,
)


def train_kinematic(config_path: str | Path) -> Path:
    path = Path(config_path)
    configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
    data_config_path = Path(configuration["data_config"])
    data_config = DataConfig.from_yaml(data_config_path)
    manifest = data_config.manifest_path or data_config.root / "manifest.parquet"

    train_data = RenderedTrajectoryDataset(manifest, data_config, split="train")
    validation_data = RenderedTrajectoryDataset(manifest, data_config, split="validation")
    test_data = RenderedTrajectoryDataset(manifest, data_config, split="test")

    train_loader = build_dataloader(train_data, data_config, shuffle=True)
    statistics_loader = build_dataloader(train_data, data_config, shuffle=False)
    validation_loader = build_dataloader(validation_data, data_config, shuffle=False)
    test_loader = build_dataloader(test_data, data_config, shuffle=False)

    statistics = compute_kinematic_statistics(statistics_loader)
    model_config = configuration.get("model", {})
    module = KinematicObservabilityModule(
        statistics,
        **model_config,
    )

    train_config = configuration.get("training", {})
    output = Path(train_config.get("output_directory", "outputs/observability"))
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best-{epoch:03d}",
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        default_root_dir=output,
        accelerator="auto",
        devices=train_config.get("devices", "auto"),
        max_epochs=int(train_config.get("epochs", 50)),
        precision=train_config.get(
            "precision", "bf16-mixed" if torch.cuda.is_available() else "32-true"
        ),
        gradient_clip_val=float(train_config.get("gradient_clip_norm", 1.0)),
        deterministic=bool(train_config.get("deterministic", True)),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        logger=TensorBoardLogger(output, name="tensorboard"),
        log_every_n_steps=int(train_config.get("log_every_n_steps", 10)),
    )
    trainer.fit(module, train_loader, validation_loader)
    trainer.test(module, test_loader, ckpt_path="best")
    if not checkpoint.best_model_path:
        raise RuntimeError("Training completed without producing a best checkpoint.")
    return Path(checkpoint.best_model_path)