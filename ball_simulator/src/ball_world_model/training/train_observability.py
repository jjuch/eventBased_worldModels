from __future__ import annotations

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import yaml

from ball_world_model.config import DataConfig
from ball_world_model.data.dataset import RenderedTrajectoryDataset, build_dataloader
from .observability_module import ObservabilityModule
from .state_normaliser import compute_state_statics


def train_observability(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config_path = Path(configuration["data_config"])
    data = DataConfig.from_yaml(data_config_path)
    manifest = data.manifest_path or data.root / "manifest.parquet"

    train_dataset = RenderedTrajectoryDataset(manifest, data, split="train")
    validation_dataset = RenderedTrajectoryDataset(manifest, data, split="validation")
    test_dataset = RenderedTrajectoryDataset(manifest, data, split="test")

    train_loader = build_dataloader(train_dataset, data, shuffle=True)
    statistics_loader = build_dataloader(train_dataset, data, shuffle=False)
    validation_loader = build_dataloader(validation_dataset, data, shuffle=False)
    test_loader = build_dataloader(test_dataset, data, shuffle=False)
    statistics = compute_state_statics(statistics_loader)

    model_config = configuration.get("model", {})
    training_config = configuration.get("training", {})
    module = ObservabilityModule(statistics, **model_config)

    output_directory = Path(training_config.get("output_directory", "outputs/observability"))
    checkpoint = ModelCheckpoint(
        dirpath=output_directory / "checkpoints",
        filename="best-{epoch:03d}",
        monitor="validation/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        default_root_dir=output_directory,
        accelerator="auto",
        devices=training_config.get("devices", "auto"),
        max_epochs=int(training_config.get("epochs", 50)),
        precision=training_config.get(
            "precision", "bf16-mixed" if torch.cuda.is_available() else "32-true"
        ),
        gradient_clip_val=float(training_config.get("gradient_clip_norm", 1.0)),
        deterministic=bool(training_config.get("deterministic", True)),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        logger=TensorBoardLogger(output_directory, name="tensorboard"),
        log_every_n_steps=int(training_config.get("log_every_n_steps", 10)),
    )
    trainer.fit(module, train_loader, validation_loader)
    trainer.test(module, test_loader, ckpt_path="best")
    if checkpoint.best_model_path == "":
        raise RuntimeError("Training completed without producing a best checkpoint.")
    return Path(checkpoint.best_model_path)





