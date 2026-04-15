"""Paper-faithful PERI model and controlled ablations."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from peri.data.emotic_constants import IMAGENET_MEAN, IMAGENET_STD

from .backbones import ResNet18Backbone
from .fusion import ContInBlock, FusionHead, LatePASFusion, resolve_feature_concat


def _ensure_batched(name: str, tensor: torch.Tensor | None) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required.")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B, 3, H, W], got {tuple(tensor.shape)}.")
    return tensor


class PERIModel(nn.Module):
    paper_cont_in_stages = ("layer1", "layer2", "layer3")

    def __init__(
        self,
        *,
        pretrained: bool = True,
        emotion_dim: int = 26,
        vad_dim: int = 3,
        hidden_dim: int = 512,
        pas_fusion_mode: str = "cont_in",
        cont_in_stages: tuple[str, ...] = paper_cont_in_stages,
        cont_in_variant: str = "paper",
    ) -> None:
        super().__init__()
        self.pas_fusion_mode = pas_fusion_mode
        self.cont_in_stages = tuple(cont_in_stages)
        self.context_backbone = ResNet18Backbone(pretrained=pretrained)
        self.body_backbone = ResNet18Backbone(pretrained=pretrained)
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        if pas_fusion_mode == "cont_in":
            self.cont_in_blocks = nn.ModuleDict(
                {
                    stage: ContInBlock(
                        channels=ResNet18Backbone.stage_channels[stage],
                        variant=cont_in_variant,
                        stage_name=stage,
                    )
                    for stage in self.cont_in_stages
                }
            )
            self.late_pas_fusion = None
        elif pas_fusion_mode == "late":
            self.cont_in_blocks = nn.ModuleDict()
            self.late_pas_fusion = LatePASFusion(out_dim=128)
        elif pas_fusion_mode == "none":
            self.cont_in_blocks = nn.ModuleDict()
            self.late_pas_fusion = None
        else:
            raise ValueError(f"Unsupported PAS fusion mode {pas_fusion_mode!r}.")

        fusion_dim = self.context_backbone.feature_dim + self.body_backbone.feature_dim
        if self.late_pas_fusion is not None:
            fusion_dim += self.late_pas_fusion.out_dim
        self.head = FusionHead(in_dim=fusion_dim, hidden_dim=hidden_dim, emotion_dim=emotion_dim, vad_dim=vad_dim)

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.imagenet_mean.to(device=image.device)) / self.imagenet_std.to(device=image.device)

    def _forward_body_stream(self, body_image: torch.Tensor, pas_image: torch.Tensor | None) -> dict[str, torch.Tensor]:
        encoder = self.body_backbone
        x = encoder.relu(encoder.bn1(encoder.conv1(body_image)))
        x = encoder.maxpool(x)
        outputs: dict[str, torch.Tensor] = {}
        for stage_name in ("layer1", "layer2", "layer3", "layer4"):
            x = getattr(encoder, stage_name)(x)
            if stage_name in self.cont_in_blocks:
                if pas_image is None:
                    raise ValueError("pas_image is required when PAS fusion mode is 'cont_in'.")
                x = self.cont_in_blocks[stage_name](x, pas_image)
            outputs[stage_name] = x
        outputs["pooled"] = torch.flatten(encoder.avgpool(x), 1)
        return outputs

    def forward(
        self,
        batch: Mapping[str, Any] | None = None,
        *,
        full_image: torch.Tensor | None = None,
        person_crop: torch.Tensor | None = None,
        pas_image: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if batch is not None:
            full_image = batch["full_image"]
            person_crop = batch["person_crop"]
            pas_image = batch.get("pas_image")

        full_image = _ensure_batched("full_image", full_image)
        person_crop = _ensure_batched("person_crop", person_crop)
        normalized_context = self._normalize(full_image)
        normalized_body = self._normalize(person_crop)
        # PAS is consumed by the lightweight Cont-In branch rather than a pretrained
        # backbone stream, so keeping it in its native [0, 1] range is closer to the
        # paper's masked-body-crop formulation than forcing ImageNet normalization.
        normalized_pas = _ensure_batched("pas_image", pas_image) if pas_image is not None else None

        context_features = self.context_backbone(normalized_context)
        body_features = self._forward_body_stream(normalized_body, normalized_pas)
        fusion_features = {
            "context": context_features.pooled,
            "body": body_features["pooled"],
        }
        if self.late_pas_fusion is not None:
            if normalized_pas is None:
                raise ValueError("pas_image is required when PAS fusion mode is 'late'.")
            fusion_features["pas"] = self.late_pas_fusion(normalized_pas)

        ordered_keys = ("context", "body") if "pas" not in fusion_features else ("context", "body", "pas")
        outputs = self.head(resolve_feature_concat(fusion_features, ordered_keys))
        return {
            "emotion_probs": outputs["emotion_probs"],
            "vad": outputs["vad"],
            "features": {
                "context": context_features.pooled,
                "body": body_features["pooled"],
                "fused": outputs["fused"],
            },
        }
