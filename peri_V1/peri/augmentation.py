"""Shared augmentation helpers for EMOTIC images and PAS-aligned tensors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class SampledAugmentation:
    flip: bool
    angle: float
    translate_x: float
    translate_y: float
    brightness: float
    contrast: float
    saturation: float
    blur_sigma: float | None
    erase_box: tuple[float, float, float, float] | None


def _sample_erase_box(*, rng: random.Random) -> tuple[float, float, float, float] | None:
    for _ in range(10):
        area_fraction = rng.uniform(0.02, 0.12)
        aspect_ratio = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
        erase_height = math.sqrt(area_fraction * aspect_ratio)
        erase_width = math.sqrt(area_fraction / aspect_ratio)
        if erase_height >= 1.0 or erase_width >= 1.0:
            continue
        top = rng.uniform(0.0, 1.0 - erase_height)
        left = rng.uniform(0.0, 1.0 - erase_width)
        return (top, left, erase_height, erase_width)
    return None


def sample_strong_augmentation(*, rng: random.Random | None = None) -> SampledAugmentation:
    resolved_rng = rng or random
    blur_sigma = resolved_rng.uniform(0.1, 1.5) if resolved_rng.random() < 0.25 else None
    erase_box = _sample_erase_box(rng=resolved_rng) if resolved_rng.random() < 0.25 else None
    return SampledAugmentation(
        flip=resolved_rng.random() < 0.5,
        angle=resolved_rng.uniform(-10.0, 10.0),
        translate_x=resolved_rng.uniform(-0.1, 0.1),
        translate_y=resolved_rng.uniform(-0.1, 0.1),
        brightness=resolved_rng.uniform(0.75, 1.25),
        contrast=resolved_rng.uniform(0.75, 1.25),
        saturation=resolved_rng.uniform(0.75, 1.25),
        blur_sigma=blur_sigma,
        erase_box=erase_box,
    )


def _resolve_translate(params: SampledAugmentation, *, height: int, width: int) -> tuple[int, int]:
    return (
        int(round(params.translate_x * width)),
        int(round(params.translate_y * height)),
    )


def _apply_shared_geometry(
    image: torch.Tensor,
    *,
    params: SampledAugmentation,
    interpolation: InterpolationMode,
) -> torch.Tensor:
    output = image
    if params.flip:
        output = TF.hflip(output)
    output = TF.affine(
        output,
        angle=params.angle,
        translate=_resolve_translate(params, height=output.shape[-2], width=output.shape[-1]),
        scale=1.0,
        shear=[0.0, 0.0],
        interpolation=interpolation,
        fill=0.0,
    )
    return output


def apply_image_augmentation(
    image: torch.Tensor,
    params: SampledAugmentation,
    *,
    allow_erase: bool,
) -> torch.Tensor:
    output = _apply_shared_geometry(image, params=params, interpolation=InterpolationMode.BILINEAR)
    output = TF.adjust_brightness(output, params.brightness)
    output = TF.adjust_contrast(output, params.contrast)
    output = TF.adjust_saturation(output, params.saturation)
    if params.blur_sigma is not None:
        output = TF.gaussian_blur(output, kernel_size=5, sigma=[params.blur_sigma, params.blur_sigma])
    if allow_erase and params.erase_box is not None:
        top_frac, left_frac, height_frac, width_frac = params.erase_box
        height = output.shape[-2]
        width = output.shape[-1]
        erase_h = max(1, int(round(height_frac * height)))
        erase_w = max(1, int(round(width_frac * width)))
        top = min(max(0, int(round(top_frac * height))), max(height - erase_h, 0))
        left = min(max(0, int(round(left_frac * width))), max(width - erase_w, 0))
        output = TF.erase(output, i=top, j=left, h=erase_h, w=erase_w, v=0.0)
    return output.clamp(0.0, 1.0)


def apply_mask_augmentation(mask: torch.Tensor, params: SampledAugmentation) -> torch.Tensor:
    output = _apply_shared_geometry(mask.float(), params=params, interpolation=InterpolationMode.NEAREST)
    return (output > 0.5).to(dtype=mask.dtype)
