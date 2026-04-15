"""Fusion modules for paper-faithful PERI and controlled ablations."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn
from torch.nn import functional as F


class FusionHead(nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int = 512, emotion_dim: int = 26, vad_dim: int = 3) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.emotion_head = nn.Sequential(nn.Linear(hidden_dim, emotion_dim), nn.Sigmoid())
        self.vad_head = nn.Linear(hidden_dim, vad_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self.fusion(features)
        return {
            "fused": fused,
            "emotion_probs": self.emotion_head(fused),
            "vad": self.vad_head(fused),
        }


class PaperPASStageEncoder(nn.Module):
    """Stage-specific PAS encoder used by each paper-faithful Cont-In block."""

    _stage_order = ("layer1", "layer2", "layer3", "layer4")
    _stage_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}

    def __init__(self, *, stage_name: str) -> None:
        super().__init__()
        if stage_name not in self._stage_order:
            raise ValueError(f"Unsupported PAS stage {stage_name!r}.")
        self.stage_name = stage_name
        blocks: list[nn.Module] = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ]
        current_channels = 64
        for next_stage in ("layer2", "layer3", "layer4"):
            if self._stage_order.index(next_stage) > self._stage_order.index(stage_name):
                break
            next_channels = self._stage_channels[next_stage]
            blocks.extend(
                [
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = next_channels
        self.encoder = nn.Sequential(*blocks)
        self.out_channels = current_channels

    def forward(self, pas_image: torch.Tensor) -> torch.Tensor:
        return self.encoder(pas_image)


class ContInBlock(nn.Module):
    """Context infusion block for the paper-faithful path and controlled ablations."""

    def __init__(
        self,
        *,
        channels: int,
        variant: str = "paper",
        stage_name: str | None = None,
        pas_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.channels = channels

        if variant == "paper":
            if stage_name is None:
                raise ValueError("stage_name is required for the paper Cont-In variant.")
            self.pas_encoder: nn.Module | None = PaperPASStageEncoder(stage_name=stage_name)
            self.modulation = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.output_activation = None
        elif variant == "residual":
            hidden = pas_hidden_channels or max(channels // 4, 32)
            self.pas_encoder = nn.Sequential(
                nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(hidden),
            )
            self.modulation = nn.Sequential(
                nn.Conv2d(channels + hidden, channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channels),
            )
            self.output_activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported Cont-In variant {variant!r}.")

    def forward(self, body_features: torch.Tensor, pas_signal: torch.Tensor) -> torch.Tensor:
        assert self.pas_encoder is not None
        pas_features = self.pas_encoder(pas_signal)
        if pas_features.shape[1] != self.channels:
            raise ValueError(
                f"Cont-In channel mismatch: expected PAS features with {self.channels} channels, "
                f"got {pas_features.shape[1]}."
            )
        if pas_features.shape[-2:] != body_features.shape[-2:]:
            pas_features = F.interpolate(pas_features, size=body_features.shape[-2:], mode="bilinear", align_corners=False)
        fused = torch.cat([body_features, pas_features], dim=1)
        update = self.modulation(fused)
        if self.variant == "paper":
            return update
        assert self.output_activation is not None
        return self.output_activation(body_features + update)


class LatePASFusion(nn.Module):
    """Lightweight late-fusion ablation path for PAS in experimental mode."""

    def __init__(self, *, out_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = out_dim

    def forward(self, pas_image: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.encoder(pas_image), 1)


def resolve_feature_concat(features: Dict[str, torch.Tensor], order: Iterable[str]) -> torch.Tensor:
    return torch.cat([features[name] for name in order], dim=1)
