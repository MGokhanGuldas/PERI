"""Explicit mAP and VAD error metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from peri.data.emotic_constants import EMOTION_COLUMNS


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sanitize_array(array: np.ndarray, *, name: str) -> np.ndarray:
    if array.ndim == 0:
        raise ValueError(f"{name} must not be scalar.")
    if np.isnan(array).any() or np.isinf(array).any():
        raise ValueError(f"{name} contains NaN or Inf values.")
    return array


def _average_precision_binary(target: np.ndarray, score: np.ndarray) -> float:
    positives = float(target.sum())
    if positives <= 0.0:
        return float("nan")

    order = np.argsort(-score)
    sorted_target = target[order]
    tp = np.cumsum(sorted_target)
    fp = np.cumsum(1.0 - sorted_target)
    precision = tp / np.maximum(tp + fp, 1e-8)
    recall = tp / positives

    ap = 0.0
    prev_recall = 0.0
    for prec, rec in zip(precision, recall):
        ap += float(prec) * max(float(rec - prev_recall), 0.0)
        prev_recall = float(rec)
    return ap


def compute_multilabel_metrics(
    probabilities: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    *,
    threshold: float = 0.5,
    include_per_class_ap: bool = False,
) -> dict[str, float]:
    probs_np = _sanitize_array(_to_numpy(probabilities).astype(np.float32), name="emotion_probs")
    targets_np = _sanitize_array(_to_numpy(targets).astype(np.float32), name="emotion_targets")
    if probs_np.ndim != 2 or targets_np.ndim != 2:
        raise ValueError("emotion probabilities and targets must be 2D [B, C].")
    if probs_np.shape != targets_np.shape:
        raise ValueError(f"probabilities and targets shape mismatch: {probs_np.shape} vs {targets_np.shape}")
    if probs_np.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "map": float("nan"), "valid_ap_classes": 0.0}

    probs_np = np.clip(probs_np, 0.0, 1.0)
    preds = (probs_np >= threshold).astype(np.float32)
    tp = float((preds * targets_np).sum())
    fp = float((preds * (1.0 - targets_np)).sum())
    fn = float(((1.0 - preds) * targets_np).sum())

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    ap_values = [_average_precision_binary(targets_np[:, index], probs_np[:, index]) for index in range(targets_np.shape[1])]
    valid_ap = [value for value in ap_values if not np.isnan(value)]
    map_score = float(np.mean(valid_ap)) if valid_ap else float("nan")
    result: dict[str, object] = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "map": map_score,
        "valid_ap_classes": float(len(valid_ap)),
    }
    if include_per_class_ap:
        result["per_class_ap"] = {
            EMOTION_COLUMNS[index]: (None if np.isnan(value) else float(value))
            for index, value in enumerate(ap_values)
        }
    return result  # type: ignore[return-value]


def compute_vad_metrics(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    pred_np = _sanitize_array(_to_numpy(predictions).astype(np.float32), name="vad_predictions")
    target_np = _sanitize_array(_to_numpy(targets).astype(np.float32), name="vad_targets")
    if pred_np.shape != target_np.shape:
        raise ValueError(f"VAD prediction/target shape mismatch: {pred_np.shape} vs {target_np.shape}")
    if pred_np.size == 0:
        return {
            "vad_valence_l1": 0.0,
            "vad_arousal_l1": 0.0,
            "vad_dominance_l1": 0.0,
            "vad_error": 0.0,
        }
    errors = np.abs(pred_np - target_np)
    valence = float(np.mean(errors[:, 0]))
    arousal = float(np.mean(errors[:, 1]))
    dominance = float(np.mean(errors[:, 2]))
    return {
        "vad_valence_l1": valence,
        "vad_arousal_l1": arousal,
        "vad_dominance_l1": dominance,
        "vad_error": float(np.mean([valence, arousal, dominance])),
    }


@dataclass
class BatchMetricAccumulator:
    emotion_probabilities: list[np.ndarray] = field(default_factory=list)
    emotion_targets: list[np.ndarray] = field(default_factory=list)
    vad_predictions: list[np.ndarray] = field(default_factory=list)
    vad_targets: list[np.ndarray] = field(default_factory=list)

    def update(
        self,
        *,
        emotion_probabilities: torch.Tensor,
        emotion_targets: torch.Tensor,
        vad_predictions: torch.Tensor,
        vad_targets: torch.Tensor,
    ) -> None:
        self.emotion_probabilities.append(_to_numpy(emotion_probabilities))
        self.emotion_targets.append(_to_numpy(emotion_targets))
        self.vad_predictions.append(_to_numpy(vad_predictions))
        self.vad_targets.append(_to_numpy(vad_targets))

    def compute(self, *, include_per_class_ap: bool = False) -> dict[str, float | dict[str, float | None]]:
        if not self.emotion_probabilities:
            return {}
        probabilities = np.concatenate(self.emotion_probabilities, axis=0)
        emotion_targets = np.concatenate(self.emotion_targets, axis=0)
        vad_predictions = np.concatenate(self.vad_predictions, axis=0)
        vad_targets = np.concatenate(self.vad_targets, axis=0)
        result = compute_multilabel_metrics(probabilities, emotion_targets, include_per_class_ap=include_per_class_ap)
        result.update(compute_vad_metrics(vad_predictions, vad_targets))
        return result
