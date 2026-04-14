"""Landmark caching utilities for the PERI workflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_cache_path(cache_dir: Path, sample_id: str) -> Path:
    """Generate a safe, unique filename for a given sample_id."""
    # sample_id format: "split:full_image_name:crop_name:row_index"
    # Example: "train:train_4/399.jpg:train_4_399_0.jpg:0"
    # We replace colons and slashes with underscores to make it a valid filename.
    safe_name = sample_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_name}.json"


def load_landmarks_cache(cache_dir: Path, sample_id: str) -> dict[str, Any] | None:
    """Load landmarks from JSON cache if available."""
    path = get_cache_path(cache_dir, sample_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for kind in ("pose", "face"):
                keypoints = payload.get(kind, {}).get("keypoints")
                if isinstance(keypoints, list):
                    payload[kind]["keypoints"] = np.asarray(keypoints, dtype=np.float32)
            return payload
        except Exception as e:
            logger.warning(f"Failed to load landmark cache for {sample_id} at {path}: {e!r}")
    return None


def save_landmarks_cache(cache_dir: Path, sample_id: str, landmarks: dict[str, Any]) -> None:
    """Save landmarks to JSON cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(cache_dir, sample_id)
    try:
        serializable: dict[str, Any] = {}
        for kind, payload in landmarks.items():
            if not isinstance(payload, dict):
                serializable[kind] = payload
                continue
            serializable[kind] = {}
            for key, value in payload.items():
                if isinstance(value, np.ndarray):
                    serializable[kind][key] = value.tolist()
                else:
                    serializable[kind][key] = value
        # We don't use indentation to save space/speed (can be 30k+ files)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, separators=(",", ":"))
    except Exception as e:
        logger.error(f"Failed to save landmark cache for {sample_id} at {path}: {e!r}")
