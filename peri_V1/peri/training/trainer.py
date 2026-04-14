"""Single-entry training runner for PERI."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
import time

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from peri.analysis import BatchMetricAccumulator, append_jsonl, write_json
from peri.data import assert_dataset_summary_ok, build_dataset_summary
from peri.models import PERIModel

from .config import TrainingConfig, set_global_seed
from .dataloaders import DataLoaderBundle, build_dataloaders
from .logging import RunArtifacts, prepare_run_artifacts, write_run_config
from .losses import MultiTaskLoss, build_loss_module
from .plots import write_training_plots


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_epoch: int = 0
    best_metric: float | None = None
    best_vad_error: float | None = None
    best_vad_epoch: int = 0
    history: list[dict[str, object]] = field(default_factory=list)


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.artifacts: RunArtifacts = prepare_run_artifacts(config)
        if self.config.pas_debug and self.config.pas_debug_dir is None:
            self.config.pas_debug_dir = self.artifacts.plots_dir
        write_run_config(config, self.artifacts)

        set_global_seed(config.seed)
        self.device = torch.device(config.device)
        self.amp_enabled = bool(config.use_amp and self.device.type == "cuda")
        self.state = TrainerState()
        self.dataset_summary: dict[str, object] | None = None
        self.last_training_seconds: float = 0.0
        self.tensorboard_writer: SummaryWriter | None = None

        self.model: nn.Module = PERIModel(
            pretrained=config.pretrained,
            pas_fusion_mode=config.pas_fusion_mode,
            cont_in_stages=config.cont_in_stages,
            cont_in_variant=config.cont_in_variant,
        ).to(self.device)
        self.loss_module: MultiTaskLoss = build_loss_module(
            vad_weight=config.vad_weight,
            label_smoothing=config.label_smoothing,
            emotion_loss_name=config.emotion_loss_name,
            focal_gamma=config.focal_gamma,
        ).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.dataloaders: DataLoaderBundle | None = None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self, *, steps_per_epoch: int | None = None) -> torch.optim.lr_scheduler._LRScheduler | None:
        if self.config.scheduler_name == "none":
            return None
        if self.config.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
        if self.config.scheduler_name == "onecycle":
            if steps_per_epoch is None:
                # Return None dummy during __init__, will be rebuilt in fit()
                return None
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=self.config.epochs,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1000.0,
            )
        if self.config.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=max(self.config.learning_rate * 0.01, 1e-6),
            )
        raise ValueError(f"Unsupported scheduler_name={self.config.scheduler_name!r}")

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
            self.tensorboard_writer = None
        if self.dataloaders is not None:
            self.dataloaders.close()

    def _ensure_tensorboard_writer(self) -> None:
        if not self.config.tensorboard_enabled or self.tensorboard_writer is not None:
            return
        purge_step = self.state.epoch + 1 if self.state.epoch > 0 else None
        self.tensorboard_writer = SummaryWriter(
            log_dir=str(self.artifacts.tensorboard_dir),
            purge_step=purge_step,
        )

    def _log_tensorboard_epoch(self, epoch_record: dict[str, object]) -> None:
        if self.tensorboard_writer is None:
            return
        epoch = int(epoch_record["epoch"])
        writer = self.tensorboard_writer
        writer.add_scalar("loss/train", float(epoch_record["train_loss"]), epoch)
        writer.add_scalar("loss/val", float(epoch_record["val_loss"]), epoch)
        writer.add_scalar("map/train", float(epoch_record["train_map"]), epoch)
        writer.add_scalar("map/val", float(epoch_record["val_map"]), epoch)
        writer.add_scalar("vad_error/train", float(epoch_record["train_vad_error"]), epoch)
        writer.add_scalar("vad_error/val", float(epoch_record["val_vad_error"]), epoch)
        writer.add_scalar("learning_rate", float(epoch_record["learning_rate"]), epoch)

    def _log_tensorboard_test_metrics(self, metrics: dict[str, object], *, epoch: int) -> None:
        if self.tensorboard_writer is None:
            return
        writer = self.tensorboard_writer
        if "map" in metrics:
            writer.add_scalar("test/map", float(metrics["map"]), epoch)
        if "vad_error" in metrics:
            writer.add_scalar("test/vad_error", float(metrics["vad_error"]), epoch)
        per_class_ap = metrics.get("per_class_ap")
        if isinstance(per_class_ap, dict):
            for class_name, value in per_class_ap.items():
                if value is not None:
                    writer.add_scalar(f"per_class_ap/{class_name}", float(value), epoch)
        writer.flush()

    def _move_batch_to_device(self, batch: dict[str, object]) -> dict[str, object]:
        moved: dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.device)
            elif isinstance(value, dict):
                nested: dict[str, object] = {}
                for nested_key, nested_value in value.items():
                    nested[nested_key] = nested_value.to(self.device) if isinstance(nested_value, torch.Tensor) else nested_value
                moved[key] = nested
            else:
                moved[key] = value
        return moved

    def _validate_batch(self, batch: dict[str, object], *, split: str, epoch: int, step: int) -> bool:
        tensors = {key: value for key, value in batch.items() if isinstance(value, torch.Tensor)}
        issues = [name for name, value in tensors.items() if not torch.isfinite(value).all()]
        if not issues:
            return True
        payload = {"split": split, "epoch": epoch, "step": step, "issues": issues}
        if self.config.save_failed_batches:
            append_jsonl(payload, self.artifacts.root / "failed_batches.jsonl")
        if self.config.allow_invalid_batches:
            return False
        raise RuntimeError(f"Invalid batch encountered: {payload}")

    def _run_epoch(
        self,
        *,
        split: str,
        loader,
        training: bool,
        epoch: int,
        include_per_class_ap: bool = False,
    ) -> dict[str, object]:
        assert self.dataloaders is not None
        if training:
            self.model.train()
        else:
            self.model.eval()
        accumulator = BatchMetricAccumulator()
        loss_sums = {"total_loss": 0.0, "emotion_loss": 0.0, "vad_loss": 0.0}
        processed = 0
        skipped = 0
        start = time.perf_counter()
        phase = "Eğitim" if training else "Doğrulama"
        progress_bar = tqdm(
            enumerate(loader, start=1),
            total=len(loader),
            desc=f"Epoch {epoch} [{phase}]",
            unit="batch",
            ncols=120,
            leave=True,
        )

        for step, batch in progress_bar:
            batch = self._move_batch_to_device(batch)
            if not self._validate_batch(batch, split=split, epoch=epoch, step=step):
                skipped += 1
                continue

            autocast_context = (
                torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp_enabled)
                if self.amp_enabled
                else nullcontext()
            )
            with torch.set_grad_enabled(training):
                with autocast_context:
                    outputs = self.model(batch)
                    losses = self.loss_module(
                        outputs, 
                        {"emotion": batch["emotion"], "vad": batch["vad"]},
                    )
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.amp_enabled:
                        self.scaler.scale(losses["total_loss"]).backward()
                        if self.config.grad_clip is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        losses["total_loss"].backward()
                        if self.config.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                        self.optimizer.step()
                    
                    if training and self.config.scheduler_name == "onecycle" and self.scheduler is not None:
                        self.scheduler.step()
                        
                    self.state.global_step += 1

            processed += 1
            for key in loss_sums:
                loss_sums[key] += float(losses[key].detach().item())
            accumulator.update(
                emotion_probabilities=outputs["emotion_probs"],
                emotion_targets=batch["emotion"],
                vad_predictions=outputs["vad"],
                vad_targets=batch["vad"],
            )
            avg_loss = loss_sums["total_loss"] / processed
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        elapsed = time.perf_counter() - start
        metrics = accumulator.compute(include_per_class_ap=include_per_class_ap)
        summary: dict[str, object] = {
            "split": split,
            "epoch": epoch,
            "processed_batches": processed,
            "skipped_batches": skipped,
            "elapsed_seconds": elapsed,
            "total_loss": loss_sums["total_loss"] / max(processed, 1),
            "emotion_loss": loss_sums["emotion_loss"] / max(processed, 1),
            "vad_loss": loss_sums["vad_loss"] / max(processed, 1),
        }
        summary.update(metrics)
        return summary

    def _checkpoint_payload(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.amp_enabled else None,
            "state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_epoch": self.state.best_epoch,
                "best_metric": self.state.best_metric,
                "best_vad_error": self.state.best_vad_error,
                "best_vad_epoch": self.state.best_vad_epoch,
                "history": self.state.history,
            },
        }

    def _save_checkpoint(self, path: Path) -> None:
        torch.save(self._checkpoint_payload(), path)

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if self.amp_enabled and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
        state = checkpoint.get("state", {})
        self.state.epoch = int(state.get("epoch", 0))
        self.state.global_step = int(state.get("global_step", 0))
        self.state.best_epoch = int(state.get("best_epoch", 0))
        self.state.best_metric = None if state.get("best_metric") is None else float(state.get("best_metric"))
        self.state.best_vad_error = None if state.get("best_vad_error") is None else float(state.get("best_vad_error"))
        self.state.best_vad_epoch = int(state.get("best_vad_epoch", 0))
        self.state.history = list(state.get("history", []))

    def _current_learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _write_history(self) -> None:
        write_json(self.state.history, self.artifacts.training_history_path)

    def _summary_payload(self, *, status: str, reason: str) -> dict[str, object]:
        return {
            "status": status,
            "best_checkpoint_path": str(self.artifacts.best_checkpoint_path),
            "last_checkpoint_path": str(self.artifacts.last_checkpoint_path),
            "reason_training_ended": reason,
            "resume_used": self.config.resume_from is not None,
            "resumed_from": str(self.config.resume_from) if self.config.resume_from is not None else None,
            "final_epoch": self.state.epoch,
            "global_step": self.state.global_step,
        }

    def write_summary(self, *, status: str, reason: str) -> dict[str, object]:
        payload = self._summary_payload(status=status, reason=reason)
        write_json(payload, self.artifacts.summary_path)
        return payload

    def prepare(self) -> dict[str, object]:
        self.dataset_summary = build_dataset_summary(
            data_root=self.config.data_root,
            backend=self.config.dataset_backend,
            annotations_root=self.config.annotations_root,
            images_root=self.config.images_root,
            annotations_mat_path=self.config.annotations_mat_path,
            jpg_root=self.config.jpg_root,
            npy_manifest_root=self.config.npy_manifest_root,
            include_extra_train=self.config.include_extra_train,
            validate_images=True,
            mediapipe_asset_root=self.config.mediapipe_asset_root,
        )
        write_json(self.dataset_summary, self.artifacts.dataset_summary_path)
        assert_dataset_summary_ok(self.dataset_summary)
        self.dataloaders = build_dataloaders(self.config)
        if self.config.scheduler_name == "onecycle":
            self.scheduler = self._build_scheduler(steps_per_epoch=len(self.dataloaders.train_loader))
        if self.config.resume_from is not None:
            self._load_checkpoint(self.config.resume_from)
        return self.dataset_summary

    def fit(self) -> dict[str, object]:
        if self.dataloaders is None:
            self.prepare()
        assert self.dataloaders is not None
        self._ensure_tensorboard_writer()

        training_start = time.perf_counter()
        for epoch in range(self.state.epoch + 1, self.config.epochs + 1):
            train_metrics = self._run_epoch(split="train", loader=self.dataloaders.train_loader, training=True, epoch=epoch)
            val_metrics = self._run_epoch(split="val", loader=self.dataloaders.val_loader, training=False, epoch=epoch)
            current_metric = float(val_metrics[self.config.primary_metric])
            current_vad_error = float(val_metrics["vad_error"])
            if not torch.isfinite(torch.tensor(current_metric)):
                current_metric = float("-inf")

            checkpoint_saved = False
            if self.state.best_metric is None or current_metric > float(self.state.best_metric):
                self.state.best_metric = current_metric
                self.state.best_epoch = epoch
                checkpoint_saved = True
            if self.state.best_vad_error is None or current_vad_error < float(self.state.best_vad_error):
                self.state.best_vad_error = current_vad_error
                self.state.best_vad_epoch = epoch

            epoch_record = {
                "epoch": epoch,
                "global_step": self.state.global_step,
                "train_loss": float(train_metrics["total_loss"]),
                "val_loss": float(val_metrics["total_loss"]),
                "train_map": float(train_metrics["map"]),
                "val_map": float(val_metrics["map"]),
                "train_vad_error": float(train_metrics["vad_error"]),
                "val_vad_error": float(val_metrics["vad_error"]),
                "train_vad_valence_l1": float(train_metrics["vad_valence_l1"]),
                "train_vad_arousal_l1": float(train_metrics["vad_arousal_l1"]),
                "train_vad_dominance_l1": float(train_metrics["vad_dominance_l1"]),
                "val_vad_valence_l1": float(val_metrics["vad_valence_l1"]),
                "val_vad_arousal_l1": float(val_metrics["vad_arousal_l1"]),
                "val_vad_dominance_l1": float(val_metrics["vad_dominance_l1"]),
                "learning_rate": self._current_learning_rate(),
                "checkpoint_saved": checkpoint_saved,
            }
            self.state.history.append(epoch_record)
            self.state.epoch = epoch
            if self.scheduler is not None and self.config.scheduler_name != "onecycle":
                self.scheduler.step()
            if checkpoint_saved:
                self._save_checkpoint(self.artifacts.best_checkpoint_path)
            self._save_checkpoint(self.artifacts.last_checkpoint_path)
            self._write_history()
            self._log_tensorboard_epoch(epoch_record)

        self.last_training_seconds = time.perf_counter() - training_start
        best_metrics = next((row for row in self.state.history if int(row["epoch"]) == self.state.best_epoch), None)
        test_metrics: dict[str, object] | None = None
        if self.config.evaluate_test_after_train:
            test_metrics = self.evaluate(split="test", checkpoint_path=self.artifacts.best_checkpoint_path, include_per_class_ap=True)
            self._log_tensorboard_test_metrics(test_metrics, epoch=self.state.epoch)

        final_metrics: dict[str, object] = {
            "best_epoch": self.state.best_epoch,
            "best_map": self.state.best_metric,
            "best_vad_error": self.state.best_vad_error,
            "best_vad_epoch": self.state.best_vad_epoch,
            "final_epoch": self.state.epoch,
            "total_training_seconds": self.last_training_seconds,
            "best_checkpoint_path": str(self.artifacts.best_checkpoint_path),
            "last_checkpoint_path": str(self.artifacts.last_checkpoint_path),
        }
        if best_metrics is not None:
            final_metrics["best_epoch_metrics"] = best_metrics
        if test_metrics is not None:
            final_metrics["test_metrics"] = test_metrics
            if "per_class_ap" in test_metrics:
                final_metrics["per_class_ap"] = test_metrics["per_class_ap"]
        write_json(final_metrics, self.artifacts.final_metrics_path)
        write_training_plots(
            history=self.state.history,
            loss_curve_path=self.artifacts.loss_curve_path,
            map_curve_path=self.artifacts.map_curve_path,
            vad_curve_path=self.artifacts.vad_curve_path,
            per_class_ap_path=self.artifacts.per_class_ap_path,
            lr_curve_path=self.artifacts.lr_curve_path,
            include_lr=self.scheduler is not None,
            per_class_ap=final_metrics.get("per_class_ap"),
        )
        self.write_summary(status="completed", reason="max_epochs_reached")
        return final_metrics

    def evaluate(
        self,
        *,
        split: str,
        checkpoint_path: str | Path | None = None,
        include_per_class_ap: bool = True,
    ) -> dict[str, object]:
        if self.dataloaders is None:
            self.prepare()
        assert self.dataloaders is not None
        if checkpoint_path is not None:
            checkpoint = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        loader = {
            "train": self.dataloaders.train_loader,
            "val": self.dataloaders.val_loader,
            "test": self.dataloaders.test_loader,
        }[split]
        return self._run_epoch(split=split, loader=loader, training=False, epoch=self.state.epoch or 0, include_per_class_ap=include_per_class_ap)
