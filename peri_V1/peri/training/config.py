"""Training configuration and reproducibility helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal
import random

import numpy as np
import torch

from peri.data.emotic_constants import PAPER_BODY_SIZE, PAPER_CONTEXT_SIZE

RunMode = Literal["paper_faithful", "experimental"]
DatasetBackend = Literal["npy", "jpg"]
PASFusionMode = Literal["cont_in", "late", "none"]
ContInVariant = Literal["paper", "residual"]
SchedulerName = Literal["none", "step", "onecycle", "cosine"]
OptimizerName = Literal["adamw"]
EmotionLossName = Literal["dynamic_mse", "focal"]


@dataclass
class TrainingConfig:
    paper_cont_in_stages: ClassVar[tuple[str, ...]] = ("layer1", "layer2", "layer3")
    mode: RunMode = "paper_faithful"
    model_name: str = "peri"
    experiment_name: str = "peri"
    run_name: str | None = None
    data_root: Path = Path(".")
    dataset_backend: DatasetBackend = "npy"
    annotations_root: Path | None = None
    images_root: Path | None = None
    annotations_mat_path: Path | None = None
    jpg_root: Path | None = None
    include_extra_train: bool = False
    context_size: int = PAPER_CONTEXT_SIZE
    body_size: int = PAPER_BODY_SIZE
    pretrained: bool = True
    pas_sigma: float = 3.0
    pas_rho: float | None = None
    pas_radius_scale: float = 2.0
    pas_binary: bool = True
    pas_fusion_mode: PASFusionMode = "cont_in"
    cont_in_stages: tuple[str, ...] = paper_cont_in_stages
    cont_in_variant: ContInVariant = "paper"
    pas_debug: bool = False
    pas_debug_max_samples: int = 5
    pas_debug_dir: Path | None = None
    mediapipe_asset_root: Path | None = None
    batch_size: int = 32
    num_workers: int = 0
    augment: bool = True
    vad_weight: float = 0.5
    landmark_cache_dir: Path | None = None
    precomputed_pas_root: Path | None = None
    npy_manifest_root: Path | None = None
    use_weighted_sampler: bool = False
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None
    epochs: int = 10
    label_smoothing: float = 0.0
    emotion_loss_name: EmotionLossName = "dynamic_mse"
    focal_gamma: float = 2.0
    optimizer_name: OptimizerName = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler_name: SchedulerName = "cosine"
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.1
    use_amp: bool = True
    device: str = "cpu"
    seed: int = 42
    output_root: Path = Path("outputs") / "runs"
    tensorboard_enabled: bool = True
    resume_from: Path | None = None
    allow_invalid_batches: bool = False
    primary_metric: str = "map"
    save_failed_batches: bool = False
    evaluate_test_after_train: bool = False
    grad_clip: float | None = 1.0

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).resolve()
        self.annotations_root = Path(self.annotations_root).resolve() if self.annotations_root is not None else None
        self.images_root = Path(self.images_root).resolve() if self.images_root is not None else None
        self.annotations_mat_path = Path(self.annotations_mat_path).resolve() if self.annotations_mat_path is not None else None
        self.jpg_root = Path(self.jpg_root).resolve() if self.jpg_root is not None else None
        self.output_root = Path(self.output_root).resolve()
        self.resume_from = Path(self.resume_from).resolve() if self.resume_from is not None else None
        self.pas_debug_dir = Path(self.pas_debug_dir).resolve() if self.pas_debug_dir is not None else None
        self.precomputed_pas_root = Path(self.precomputed_pas_root).resolve() if self.precomputed_pas_root is not None else None
        self.npy_manifest_root = Path(self.npy_manifest_root).resolve() if self.npy_manifest_root is not None else None
        if self.mediapipe_asset_root is None:
            self.mediapipe_asset_root = (self.data_root / "artifacts" / "mediapipe").resolve()
        else:
            self.mediapipe_asset_root = Path(self.mediapipe_asset_root).resolve()

        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.optimizer_name != "adamw":
            raise ValueError("Only optimizer_name='adamw' is supported.")
        if self.scheduler_name not in {"none", "step", "onecycle", "cosine"}:
            raise ValueError("scheduler_name must be 'none', 'step', 'onecycle' or 'cosine'.")
        if self.scheduler_name == "step" and self.scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be > 0 when using StepLR.")
        if self.scheduler_gamma <= 0.0:
            raise ValueError("scheduler_gamma must be > 0.")
        if self.emotion_loss_name not in {"dynamic_mse", "focal"}:
            raise ValueError("emotion_loss_name must be 'dynamic_mse' or 'focal'.")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be >= 0.")
        if self.pas_rho is not None and not (0.0 < float(self.pas_rho) < 1.0):
            raise ValueError("pas_rho must be in the open interval (0, 1).")
        if self.pas_radius_scale <= 0.0:
            raise ValueError("pas_radius_scale must be > 0.")
        if self.pas_debug_max_samples < 0:
            raise ValueError("pas_debug_max_samples must be >= 0.")
        if self.pas_sigma <= 0.0:
            raise ValueError("pas_sigma must be > 0.")
        if self.npy_manifest_root is not None and self.dataset_backend != "npy":
            raise ValueError("npy_manifest_root requires dataset_backend='npy'.")
        if self.precomputed_pas_root is not None:
            if self.dataset_backend != "npy":
                raise ValueError("precomputed_pas_root currently requires dataset_backend='npy'.")
            # augment=True is now allowed with precomputed PAS: the same random
            # transforms (horizontal flip, color jitter) are applied to both the
            # body crop and the precomputed PAS image in EMOTICPreprocessedDataset.
        if self.mode == "paper_faithful":
            if self.dataset_backend != "npy":
                raise ValueError("paper_faithful mode requires dataset_backend='npy'.")
            if not self.pretrained:
                raise ValueError("paper_faithful mode requires pretrained=True.")
            if self.context_size != PAPER_CONTEXT_SIZE or self.body_size != PAPER_BODY_SIZE:
                raise ValueError(
                    f"paper_faithful mode requires context_size={PAPER_CONTEXT_SIZE} and body_size={PAPER_BODY_SIZE}."
                )
            if self.pas_fusion_mode not in {"cont_in", "none"}:
                raise ValueError("paper_faithful mode allows only pas_fusion_mode='cont_in' or 'none'.")
            if self.cont_in_variant != "paper":
                raise ValueError("paper_faithful mode requires cont_in_variant='paper'.")
            if self.pas_fusion_mode == "cont_in" and tuple(self.cont_in_stages) != self.paper_cont_in_stages:
                raise ValueError(
                    "paper_faithful mode requires cont_in_stages=('layer1', 'layer2', 'layer3')."
                )
            if self.uses_pas and abs(self.pas_sigma - 3.0) > 1e-8:
                raise ValueError("paper_faithful mode requires pas_sigma=3.0.")
            if abs(float(self.label_smoothing)) > 1e-8:
                raise ValueError("paper_faithful mode requires label_smoothing=0.0.")
            if self.emotion_loss_name != "dynamic_mse":
                raise ValueError("paper_faithful mode requires emotion_loss_name='dynamic_mse'.")
            if self.use_weighted_sampler:
                raise ValueError("paper_faithful mode requires use_weighted_sampler=False.")
            if self.npy_manifest_root is not None:
                raise ValueError("paper_faithful mode requires npy_manifest_root=None so official EMOTIC splits are used.")
        if self.pas_fusion_mode != "none" and self.num_workers != 0 and self.precomputed_pas_root is None:
            raise ValueError("num_workers must be 0 when PAS generation is enabled.")

    @property
    def uses_pas(self) -> bool:
        return self.pas_fusion_mode != "none"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TrainingConfig":
        return cls(**payload)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
