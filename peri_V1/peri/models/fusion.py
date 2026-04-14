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


class SharedPASEncoder(nn.Module):
    """Encode PAS once and expose aligned multi-scale features for every Cont-In stage."""

    _stage_order = ("layer1", "layer2", "layer3", "layer4")

    def __init__(self, *, stage_names: Iterable[str] = _stage_order) -> None:
        super().__init__()
        self.stage_names = tuple(stage_names)
        unsupported = sorted(set(self.stage_names) - set(self._stage_order))
        if unsupported:
            raise ValueError(f"Unsupported PAS stages: {unsupported}")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1_proj = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleDict(
            {
                "layer2": self._make_down_block(64, 128),
                "layer3": self._make_down_block(128, 256),
                "layer4": self._make_down_block(256, 512),
            }
        )

    @staticmethod
    def _make_down_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, pas_image: torch.Tensor) -> dict[str, torch.Tensor]:
        stage_features: dict[str, torch.Tensor] = {}
        x = self.stem(pas_image)
        stage_features["layer1"] = self.layer1_proj(x)
        x = stage_features["layer1"]
        x = self.down_blocks["layer2"](x)
        stage_features["layer2"] = x
        x = self.down_blocks["layer3"](x)
        stage_features["layer3"] = x
        x = self.down_blocks["layer4"](x)
        stage_features["layer4"] = x
        return {stage_name: stage_features[stage_name] for stage_name in self.stage_names}


class ContInBlock(nn.Module):
    """Lightweight context infusion block using shared PAS features and body activations."""

    def __init__(
        self,
        *,
        channels: int,
        variant: str = "paper",
        pas_hidden_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.channels = channels

        if variant == "paper":
            self.modulation = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            # Start close to identity without hard-zeroing the whole residual path.
            self.residual_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.pas_encoder = None
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
            self.residual_scale = None
            self.output_activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported Cont-In variant {variant!r}.")

    def forward(self, body_features: torch.Tensor, pas_signal: torch.Tensor) -> torch.Tensor:
        pas_features = pas_signal if self.pas_encoder is None else self.pas_encoder(pas_signal)
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
            assert self.residual_scale is not None
            return body_features + (self.residual_scale * update)
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
