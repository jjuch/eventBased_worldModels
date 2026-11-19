"""CLI-facing training entry point for the SE(3) observer."""

from __future__ import annotations

from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from ball_world_model.config import DataConfig, KinematicTrainingConfig
from ball_world_model.data.dataset import RenderedTrajectoryDataset, build_dataloader
from ball_world_model.models.observer import STANDARDISED_QUANTITIES
from .module import SE3ObserverModule
from .statistics import StateStatistics, compute_statistics


def _model_config_with_statistics(model_config: dict[str, object], statistics: StateStatistics) -> dict[str, object]:
    result = dict(model_config)
    for name in STANDARDISED_QUANTITIES:
        result[f"{name}_mean"] = statistics.means[name].tolist()
        result[f"{name}_std"] = statistics.stds[name].tolist()
    return result


def _report_statistics(statistics: StateStatistics) -> None:
    for name in STANDARDISED_QUANTITIES:
        std = ", ".join(f"{value:.4g}" for value in statistics.stds[name].tolist())
        note = "" if statistics.informative[name] else " <- constant in the data; its R^2 is undefined."
        print(f"[train_kinematic] {name:<21} std = ({std}){note}")


def train_kinematic(config_path: str | Path) -> Path:
    config = KinematicTrainingConfig.from_yaml(config_path)
    data_config = DataConfig.from_yaml(config.data_config)

    manifest = data_config.manifest_path or data_config.root / "manifest.parquet"

    train_data = RenderedTrajectoryDataset(manifest, data_config, split="train")
    validation_data = RenderedTrajectoryDataset(manifest, data_config, split="validation")
    test_data = RenderedTrajectoryDataset(manifest, data_config, split="test")

    train_loader = build_dataloader(train_data, data_config, shuffle=True)
    statistics_loader = build_dataloader(train_data, data_config, shuffle=False)
    validation_loader = build_dataloader(validation_data, data_config, shuffle=False)
    test_loader = build_dataloader(test_data, data_config, shuffle=False)

    if config.loss.rollout > 0 and config.loss.rollout_steps > data_config.temporal.prediction_frames:
        raise ValueError(
            f"loss.rollout_steps ({config.loss.rollout_steps}) exceeds the "
            f"{data_config.temporal.prediction_frames} prediction frames in the data config."
        )

    statistics = compute_statistics(statistics_loader)
    _report_statistics(statistics)

    module = SE3ObserverModule(
        model_config=_model_config_with_statistics(config.model.model_dump(), statistics),
        loss_config=config.loss.model_dump(),
        learning_rate=config.optimiser.learning_rate,
        weight_decay=config.optimiser.weight_decay,
    )

    output = config.training.output_directory
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", filename="best-{epoch:03d}",
        monitor=config.checkpoint.monitor, mode=config.checkpoint.mode,
        save_top_k=config.checkpoint.save_top_k, save_last=config.checkpoint.save_last,
        auto_insert_metric_name=False,
    )

    precision = config.training.precision
    if precision == "bf16-mixed" and not torch.cuda.is_available(): 
        precision = "32-true"

    trainer = L.Trainer(
        default_root_dir=output, accelerator=config.training.accelerator,
        devices=config.training.devices, max_epochs=config.training.epochs,
        precision=precision, gradient_clip_val=config.training.gradient_clip_norm,
        deterministic=config.training.deterministic,
        num_sanity_val_steps=config.training.num_sanity_val_steps,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        logger=TensorBoardLogger(output, name=config.tensorboard.name, version=config.tensorboard.version),
        log_every_n_steps=config.training.log_every_n_steps,
    )

    trainer.fit(module, train_loader, validation_loader)
    if not checkpoint.best_model_path: 
        raise RuntimeError("Training completed without producing a best checkpoint.")
    
    trainer.test(module, test_loader, ckpt_path="best")
    return Path(checkpoint.best_model_path)
