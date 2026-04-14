"""Deterministic PAS generation aligned to the body stream input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Mapping

import numpy as np
import torch
from PIL import Image


def _sanitize_sample_id(sample_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id.strip())
    return cleaned.strip("_") or "sample"


class PASDebugWriter:
    def __init__(self, output_dir: str | Path, *, max_samples: int = 5) -> None:
        self.output_dir = Path(output_dir)
        self.max_samples = int(max_samples)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._saved = 0

    def maybe_write(
        self,
        *,
        sample_id: str,
        image: np.ndarray,
        mask: np.ndarray,
        pas_image: np.ndarray,
    ) -> None:
        if self._saved >= self.max_samples:
            return
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HWC RGB, got {image.shape}")
        if mask.ndim != 2:
            raise ValueError(f"mask must be HxW, got {mask.shape}")
        if pas_image.shape != image.shape:
            raise ValueError(f"pas_image shape mismatch: {pas_image.shape} vs {image.shape}")
        mask_rgb = np.repeat((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None], 3, axis=2)
        strip = np.concatenate([image.astype(np.uint8), mask_rgb, pas_image.astype(np.uint8)], axis=1)
        filename = f"pas_debug_{self._saved:02d}_{_sanitize_sample_id(sample_id)}.png"
        Image.fromarray(strip, mode="RGB").save(self.output_dir / filename)
        self._saved += 1


@dataclass(frozen=True)
class PASGenerator:
    sigma: float = 3.0
    rho: float | None = None
    radius_scale: float = 2.0
    pose_weight: float = 1.0
    face_weight: float = 1.0
    binary_mask: bool = True

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0.")
        if self.radius_scale <= 0.0:
            raise ValueError("radius_scale must be > 0.")
        if self.rho is not None and not (0.0 < self.rho < 1.0):
            raise ValueError("rho must be in the open interval (0, 1).")

    @property
    def gaussian_radius(self) -> float:
        return float(self.radius_scale) * float(self.sigma)

    @property
    def resolved_rho(self) -> float:
        if self.rho is not None:
            return float(self.rho)
        radius = self.gaussian_radius
        sigma = float(self.sigma)
        # For a unit-normalized Gaussian, points at distance r have response exp(-(r^2)/(2*sigma^2)).
        # Using r = radius_scale * sigma gives a deterministic binary threshold derived from the Gaussian radius.
        return float(math.exp(-((radius ** 2) / (2.0 * (sigma ** 2)))))

    def _get_gaussian_kernel(self) -> np.ndarray:
        """Pre-calculate a single Gaussian kernel centered at (0,0)."""
        radius = int(math.ceil(self.gaussian_radius))
        side = 2 * radius + 1
        ys, xs = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        kernel = np.exp(-(xs**2 + ys**2) / (2.0 * (self.sigma**2)))
        return kernel.astype(np.float32)

    def generate(
        self,
        image: torch.Tensor | np.ndarray,
        landmarks: Mapping[str, Mapping[str, object]],
    ) -> dict[str, np.ndarray]:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
            image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"PAS expects an HWC RGB image, got {image.shape}.")

        height, width = image.shape[:2]
        response = np.zeros((height, width), dtype=np.float32)
        kernel = self._get_gaussian_kernel()
        k_radius = kernel.shape[0] // 2
        
        point_count = 0
        for kind in ("pose", "face"):
            block = landmarks.get(kind, {})
            keypoints = block.get("keypoints")
            if not isinstance(keypoints, np.ndarray) or keypoints.size == 0:
                continue
            
            weight = self.pose_weight if kind == "pose" else self.face_weight
            for point in keypoints:
                cx = int(round(float(point[0]) * (width - 1)))
                cy = int(round(float(point[1]) * (height - 1)))
                if 0 <= cx < width and 0 <= cy < height:
                    # Determine overlap region
                    x1, x2 = max(0, cx - k_radius), min(width, cx + k_radius + 1)
                    y1, y2 = max(0, cy - k_radius), min(height, cy + k_radius + 1)
                    
                    kx1, kx2 = x1 - (cx - k_radius), x2 - (cx - k_radius)
                    ky1, ky2 = y1 - (cy - k_radius), y2 - (cy - k_radius)
                    
                    # Update with MAXIMUM (Paper Eq. 2) instead of summation
                    weighted_kernel = kernel[ky1:ky2, kx1:kx2] * weight
                    response[y1:y2, x1:x2] = np.maximum(response[y1:y2, x1:x2], weighted_kernel)
                    point_count += 1

        if point_count == 0:
            zero_mask = np.zeros((height, width), dtype=np.float32)
            return {
                "mask": zero_mask,
                "pas_image": np.zeros_like(image, dtype=np.uint8),
                "response": zero_mask,
                "point_count": 0,
            }

        # Normalize response if weights were used
        max_val = float(response.max())
        if max_val > 1.0:
            response /= max_val
            
        mask = (response >= self.resolved_rho).astype(np.float32) if self.binary_mask else response
        pas_image = np.clip(image.astype(np.float32) * mask[..., None], 0.0, 255.0).astype(np.uint8)
        return {
            "mask": mask,
            "pas_image": pas_image,
            "response": response,
            "point_count": point_count,
            "rho": np.asarray(self.resolved_rho, dtype=np.float32),
            "radius": np.asarray(self.gaussian_radius, dtype=np.float32),
        }
