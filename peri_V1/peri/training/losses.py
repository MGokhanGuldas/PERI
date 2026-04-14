"""Paper-faithful loss functions for PERI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class DynamicWeightedMSELoss(nn.Module):
    """Dynamic weighted MSE used by EMOTIC/PERI baselines.

    Loss is normalized by batch size to keep a consistent scale
    regardless of batch_size, preventing the emotion loss from
    dominating over the VAD L1 loss.
    """

    def __init__(self, *, c: float = 1.2, empty_class_weight: float = 1e-4, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.c = float(c)
        self.empty_class_weight = float(empty_class_weight)
        self.label_smoothing = float(label_smoothing)

    def _compute_batch_weights(self, target: torch.Tensor) -> torch.Tensor:
        class_probability = target.mean(dim=0, keepdim=True)
        weights = 1.0 / torch.log(class_probability + self.c)
        fallback = torch.full_like(weights, self.empty_class_weight)
        return torch.where(torch.isfinite(weights), weights, fallback)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
        if target.ndim != 2:
            raise ValueError(f"DynamicWeightedMSELoss expects [B, C] tensors, got target ndim={target.ndim}")

        batch_size = prediction.shape[0]

        target_float = target.float()
        if self.label_smoothing > 0:
            # Paper: y_smoothed = y*(1-alpha) + alpha/num_classes
            # For multi-label binary, it's usually y*(1-alpha) + alpha/2
            target_float = target_float * (1.0 - self.label_smoothing) + (self.label_smoothing / 2.0)

        if weights is None:
            weights = self._compute_batch_weights(target_float)
        elif weights.ndim == 1:
            weights = weights.unsqueeze(0)

        loss = ((prediction.float() - target_float) ** 2) * weights
        # Normalize by batch size so magnitude is independent of batch_size
        return loss.sum() / max(batch_size, 1)


class DynamicWeightedFocalLoss(nn.Module):
    """Multi-label focal loss with the same dynamic class weighting used by PERI."""

    def __init__(self, *, c: float = 1.2, empty_class_weight: float = 1e-4, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.c = float(c)
        self.empty_class_weight = float(empty_class_weight)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def _compute_batch_weights(self, target: torch.Tensor) -> torch.Tensor:
        class_probability = target.mean(dim=0, keepdim=True)
        weights = 1.0 / torch.log(class_probability + self.c)
        fallback = torch.full_like(weights, self.empty_class_weight)
        return torch.where(torch.isfinite(weights), weights, fallback)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
        if target.ndim != 2:
            raise ValueError(f"DynamicWeightedFocalLoss expects [B, C] tensors, got target ndim={target.ndim}")

        batch_size = prediction.shape[0]
        target_float = target.float()
        if self.label_smoothing > 0:
            target_float = target_float * (1.0 - self.label_smoothing) + (self.label_smoothing / 2.0)

        if weights is None:
            weights = self._compute_batch_weights(target_float)
        elif weights.ndim == 1:
            weights = weights.unsqueeze(0)

        prediction = prediction.float().clamp(min=1e-6, max=1.0 - 1e-6)
        target_float = target_float.float()
        ce_loss = -(
            target_float * torch.log(prediction)
            + (1.0 - target_float) * torch.log(1.0 - prediction)
        )
        p_t = prediction * target_float + (1.0 - prediction) * (1.0 - target_float)
        modulation = (1.0 - p_t) ** self.gamma
        loss = ce_loss * modulation * weights
        return loss.sum() / max(batch_size, 1)


@dataclass
class MultiTaskLossOutput:
    total_loss: torch.Tensor
    emotion_loss: torch.Tensor
    vad_loss: torch.Tensor


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        *,
        vad_weight: float = 0.5,
        label_smoothing: float = 0.0,
        emotion_loss_name: str = "dynamic_mse",
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if emotion_loss_name == "dynamic_mse":
            self.emotion_loss = DynamicWeightedMSELoss(label_smoothing=label_smoothing)
        elif emotion_loss_name == "focal":
            self.emotion_loss = DynamicWeightedFocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unsupported emotion_loss_name={emotion_loss_name!r}")
        self.vad_weight = float(vad_weight)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        emotion_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emotion_loss = self.emotion_loss(predictions["emotion_probs"], targets["emotion"], weights=emotion_weights)
        # Paper Eq. 5 averages across the three VAD dimensions.
        vad_loss = torch.abs(predictions["vad"] - targets["vad"]).mean()
        total_loss = emotion_loss + self.vad_weight * vad_loss
        return {
            "total_loss": total_loss,
            "emotion_loss": emotion_loss,
            "vad_loss": vad_loss,
        }


def build_loss_module(
    *,
    vad_weight: float = 0.5,
    label_smoothing: float = 0.0,
    emotion_loss_name: str = "dynamic_mse",
    focal_gamma: float = 2.0,
) -> MultiTaskLoss:
    return MultiTaskLoss(
        vad_weight=vad_weight,
        label_smoothing=label_smoothing,
        emotion_loss_name=emotion_loss_name,
        focal_gamma=focal_gamma,
    )
