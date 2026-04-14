"""Standardized run-artifact layout for PERI experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime as dt
import re

from peri.analysis import ensure_dir, write_json

from .config import TrainingConfig


@dataclass(frozen=True)
class RunArtifacts:
    root: Path
    run_config_path: Path
    dataset_summary_path: Path
    training_history_path: Path
    final_metrics_path: Path
    summary_path: Path
    checkpoints_dir: Path
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    plots_dir: Path
    tensorboard_dir: Path
    loss_curve_path: Path
    map_curve_path: Path
    vad_curve_path: Path
    per_class_ap_path: Path
    lr_curve_path: Path


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _default_run_name(config: TrainingConfig) -> str:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_sanitize_token(config.model_name)}"


def infer_run_root_from_checkpoint(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    parent = checkpoint.parent
    if parent.name == "checkpoints":
        return parent.parent
    return parent


def prepare_run_artifacts(config: TrainingConfig) -> RunArtifacts:
    run_name = config.run_name or _default_run_name(config)
    root = infer_run_root_from_checkpoint(config.resume_from) if config.resume_from is not None else (
        config.output_root / config.mode / _sanitize_token(config.experiment_name) / _sanitize_token(run_name)
    )
    checkpoints_dir = ensure_dir(root / "checkpoints")
    plots_dir = ensure_dir(root / "plots")
    tensorboard_dir = ensure_dir(root / "tensorboard")
    ensure_dir(root)
    return RunArtifacts(
        root=root,
        run_config_path=root / "run_config.json",
        dataset_summary_path=root / "dataset_summary.json",
        training_history_path=root / "training_history.json",
        final_metrics_path=root / "final_metrics.json",
        summary_path=root / "summary.json",
        checkpoints_dir=checkpoints_dir,
        best_checkpoint_path=checkpoints_dir / "best.pt",
        last_checkpoint_path=checkpoints_dir / "last.pt",
        plots_dir=plots_dir,
        tensorboard_dir=tensorboard_dir,
        loss_curve_path=plots_dir / "loss_curve.png",
        map_curve_path=plots_dir / "map_curve.png",
        vad_curve_path=plots_dir / "vad_curve.png",
        per_class_ap_path=plots_dir / "per_class_ap.png",
        lr_curve_path=plots_dir / "lr_curve.png",
    )


def write_run_config(config: TrainingConfig, artifacts: RunArtifacts) -> None:
    payload = config.to_dict()
    payload["run_root"] = str(artifacts.root)
    write_json(payload, artifacts.run_config_path)
