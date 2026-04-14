"""Deterministic MediaPipe landmark extraction and PAS augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping
import os
import shutil

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from peri.augmentation import apply_image_augmentation, apply_mask_augmentation, sample_strong_augmentation

os.environ.setdefault("GLOG_minloglevel", "2")

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        HolisticLandmarker,
        HolisticLandmarkerOptions,
        PoseLandmarker,
        PoseLandmarkerOptions,
    )
except ImportError:
    mp = None
    BaseOptions = None
    FaceLandmarker = None
    FaceLandmarkerOptions = None
    HolisticLandmarker = None
    HolisticLandmarkerOptions = None
    PoseLandmarker = None
    PoseLandmarkerOptions = None

from peri.data.emotic_constants import MEDIAPIPE_ASSET_FILENAMES

from .pas import PASDebugWriter, PASGenerator
from .cache import load_landmarks_cache


@dataclass(frozen=True)
class LandmarkImageResult:
    pose: Dict[str, object]
    face: Dict[str, object]


def _empty_landmark_result(kind: str, source_image: str, message: str = "") -> Dict[str, object]:
    return {
        "kind": kind,
        "keypoints": np.empty((0, 3), dtype=np.float32),
        "detected": False,
        "count": 0,
        "source_image": source_image,
        "message": message,
    }


def _clone_landmark_result(result: Mapping[str, object]) -> Dict[str, object]:
    cloned: Dict[str, object] = {}
    for key, value in result.items():
        cloned[key] = value.copy() if isinstance(value, np.ndarray) else value
    return cloned


def _validate_hwc_image(image: np.ndarray, *, image_name: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{image_name} must be HWC RGB, got {image.shape}.")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        clipped = np.clip(image, 0.0, 1.0 if image.max(initial=0.0) <= 1.5 else 255.0)
        if clipped.max(initial=0.0) <= 1.5:
            clipped = clipped * 255.0
        return clipped.astype(np.uint8)
    raise ValueError(f"{image_name} has unsupported dtype {image.dtype}.")


def image_to_numpy_hwc(image: torch.Tensor | np.ndarray, *, image_name: str) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"{image_name} must be CHW RGB, got {tuple(image.shape)}.")
        return _validate_hwc_image(image.detach().cpu().permute(1, 2, 0).contiguous().numpy(), image_name=image_name)
    if isinstance(image, np.ndarray):
        return _validate_hwc_image(image, image_name=image_name)
    raise TypeError(f"{image_name} must be torch.Tensor or np.ndarray, got {type(image)!r}.")


def numpy_hwc_to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    image = _validate_hwc_image(image, image_name="pas_image").copy()
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().float() / 255.0


def numpy_mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"PAS mask must be HxW, got shape {mask.shape}.")
    return torch.from_numpy(mask.copy()).unsqueeze(0).contiguous().float()


def resize_rgb_image(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}.")
    target_h, target_w = target_shape
    if image.shape[:2] == (target_h, target_w):
        return image
    return np.asarray(
        Image.fromarray(image, mode="RGB").resize((target_w, target_h), resample=Image.BILINEAR),
        dtype=np.uint8,
    )


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected HxW mask, got {mask.shape}.")
    target_h, target_w = target_shape
    if mask.shape == (target_h, target_w):
        return mask.astype(np.float32, copy=False)
    resized = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").resize(
        (target_w, target_h),
        resample=Image.NEAREST,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def _bbox_key(bbox: object) -> tuple[int, int, int, int]:
    if isinstance(bbox, torch.Tensor):
        values = bbox.detach().cpu().flatten().tolist()
    elif isinstance(bbox, np.ndarray):
        values = bbox.reshape(-1).tolist()
    else:
        values = list(bbox)  # type: ignore[arg-type]
    if len(values) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(values)}")
    return tuple(int(round(float(v))) for v in values)


def _precomputed_pas_key(filename: str, bbox: object) -> tuple[str, tuple[int, int, int, int]]:
    return filename.replace("\\", "/"), _bbox_key(bbox)


def _precomputed_pas_lookup_key(split: str, filename: str, bbox: object) -> tuple[str, str, tuple[int, int, int, int]]:
    return split, filename.replace("\\", "/"), _bbox_key(bbox)


def _bbox_to_float_tuple(bbox: object) -> tuple[float, float, float, float]:
    if isinstance(bbox, torch.Tensor):
        values = bbox.detach().cpu().flatten().tolist()
    elif isinstance(bbox, np.ndarray):
        values = bbox.reshape(-1).tolist()
    else:
        values = list(bbox)  # type: ignore[arg-type]
    if len(values) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(values)}")
    return tuple(float(v) for v in values)


def _load_precomputed_pas_index_map(precomputed_pas_root: Path, split: str) -> dict[tuple[str, tuple[int, int, int, int]], int]:
    csv_path = Path(precomputed_pas_root).resolve() / f"{split}.csv"
    if not csv_path.exists():
        return {}
    index_map: dict[tuple[str, tuple[int, int, int, int]], int] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            key = (
                str(row["filename"]).replace("\\", "/"),
                (
                    int(round(float(row["x1"]))),
                    int(round(float(row["y1"]))),
                    int(round(float(row["x2"]))),
                    int(round(float(row["y2"]))),
                ),
            )
            index_map[key] = idx
    return index_map


def _load_precomputed_pas_image(
    *,
    precomputed_pas_root: Path,
    split: str,
    index: int | None,
    target_shape: tuple[int, int],
) -> np.ndarray:
    if index is None:
        raise FileNotFoundError("No precomputed PAS index available for this sample.")
    pas_path = Path(precomputed_pas_root).resolve() / f"pas_{split}" / f"{index:06d}.png"
    if not pas_path.exists():
        raise FileNotFoundError(f"Missing precomputed PAS image: {pas_path}")
    with Image.open(pas_path) as image:
        pas_image = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if pas_image.shape[:2] != target_shape:
        pas_image = np.asarray(
            Image.fromarray(pas_image, mode="RGB").resize((target_shape[1], target_shape[0]), resample=Image.BILINEAR),
            dtype=np.uint8,
        )
    return pas_image


def ensure_mediapipe_assets(asset_root: str | Path) -> Dict[str, Path]:
    root = Path(asset_root).resolve()
    source_paths = {name: root / filename for name, filename in MEDIAPIPE_ASSET_FILENAMES.items()}
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MediaPipe assets required for PAS generation:\n" + "\n".join(missing))
    cache_root = (Path.home() / ".peri_cache" / "mediapipe_assets").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved: Dict[str, Path] = {}
    for name, source_path in source_paths.items():
        cached_path = cache_root / source_path.name
        if not cached_path.exists() or cached_path.stat().st_size != source_path.stat().st_size:
            shutil.copy2(source_path, cached_path)
        resolved[name] = cached_path
    return resolved


def _landmarks_to_dict(
    landmarks: Iterable[object],
    *,
    kind: str,
    source_image: str,
    message: str = "",
) -> Dict[str, object]:
    coords = []
    for landmark in landmarks:
        coords.append(
            [
                float(getattr(landmark, "x", 0.0)),
                float(getattr(landmark, "y", 0.0)),
                float(getattr(landmark, "z", 0.0)),
            ]
        )
    keypoints = np.asarray(coords, dtype=np.float32)
    if keypoints.size == 0:
        return _empty_landmark_result(kind, source_image, message=message)
    return {
        "kind": kind,
        "keypoints": keypoints.reshape(-1, 3),
        "detected": True,
        "count": int(keypoints.shape[0]),
        "source_image": source_image,
        "message": message,
    }


class LandmarkExtractor:
    """Extract pose and face landmarks from the body crop."""

    def __init__(
        self,
        *,
        asset_root: str | Path = "artifacts/mediapipe",
        prefer_holistic: bool = True,
        use_full_image_fallback: bool = False,
        min_pose_detection_confidence: float = 0.5,
        min_face_detection_confidence: float = 0.5,
    ) -> None:
        if mp is None or BaseOptions is None:
            raise ImportError("MediaPipe is required for runtime landmark extraction but is not installed.")
        self.asset_root = Path(asset_root)
        self.prefer_holistic = prefer_holistic
        self.use_full_image_fallback = use_full_image_fallback
        self.min_pose_detection_confidence = 0.3
        self.min_face_detection_confidence = 0.3
        self._asset_paths = ensure_mediapipe_assets(self.asset_root)
        self._holistic: HolisticLandmarker | None = None
        self._pose: PoseLandmarker | None = None
        self._face: FaceLandmarker | None = None

    def close(self) -> None:
        for detector in (self._holistic, self._pose, self._face):
            if detector is not None:
                detector.close()
        self._holistic = None
        self._pose = None
        self._face = None

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_holistic(self) -> HolisticLandmarker:
        if self._holistic is None:
            # Note: HolisticLandmarker in Tasks API often has complexity defined by the .task file.
            # We set the confidence thresholds to 0.3 to match the high-detail requirement.
            options = HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._asset_paths["holistic"])),
                min_face_detection_confidence=0.3,
                min_pose_detection_confidence=0.3,
            )
            self._holistic = HolisticLandmarker.create_from_options(options)
        return self._holistic

    def _get_pose(self) -> PoseLandmarker:
        if self._pose is None:
            from mediapipe.tasks.python.vision import PoseLandmarkerOptions as PO
            options = PO(
                base_options=BaseOptions(model_asset_path=str(self._asset_paths["pose"])),
                num_poses=1,
                min_pose_detection_confidence=0.3,
                model_complexity=2,
            )
            self._pose = PoseLandmarker.create_from_options(options)
        return self._pose

    def _get_face(self) -> FaceLandmarker:
        if self._face is None:
            from mediapipe.tasks.python.vision import FaceLandmarkerOptions as FO
            options = FO(
                base_options=BaseOptions(model_asset_path=str(self._asset_paths["face"])),
                num_faces=1,
                min_face_detection_confidence=0.3,
            )
            self._face = FaceLandmarker.create_from_options(options)
        return self._face

    def _detect_on_image(self, image: np.ndarray, *, source_image: str) -> LandmarkImageResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        pose_result = _empty_landmark_result("pose", source_image)
        face_result = _empty_landmark_result("face", source_image)

        if self.prefer_holistic:
            holistic_result = self._get_holistic().detect(mp_image)
            pose_result = _landmarks_to_dict(
                getattr(holistic_result, "pose_landmarks", []),
                kind="pose",
                source_image=source_image,
                message="holistic",
            )
            face_result = _landmarks_to_dict(
                getattr(holistic_result, "face_landmarks", []),
                kind="face",
                source_image=source_image,
                message="holistic",
            )

        if not pose_result["detected"]:
            pose_only_result = self._get_pose().detect(mp_image)
            pose_landmarks = pose_only_result.pose_landmarks[0] if pose_only_result.pose_landmarks else []
            pose_result = _landmarks_to_dict(
                pose_landmarks,
                kind="pose",
                source_image=source_image,
                message="pose_fallback" if pose_landmarks else "pose_missing",
            )

        if not face_result["detected"]:
            face_only_result = self._get_face().detect(mp_image)
            face_landmarks = face_only_result.face_landmarks[0] if face_only_result.face_landmarks else []
            face_result = _landmarks_to_dict(
                face_landmarks,
                kind="face",
                source_image=source_image,
                message="face_fallback" if face_landmarks else "face_missing",
            )

        return LandmarkImageResult(pose=pose_result, face=face_result)

    # MediaPipe FaceMesh expressive landmark indices (subset of 468/478)
    FACE_PART_INDICES = {
        "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
        "left_eye": [33, 160, 158, 133, 153, 144],
        "right_eye": [263, 387, 385, 362, 380, 373],
        "eyebrows": [70, 63, 105, 66, 107, 336, 296, 334, 293, 300],
        "nose": [1, 2, 98, 327],
    }

    def _crop_from_full_image(
        self,
        full_image: torch.Tensor | np.ndarray,
        bbox: torch.Tensor | np.ndarray,
        meta: Mapping[str, object] | None,
    ) -> np.ndarray | None:
        full_hwc = image_to_numpy_hwc(full_image, image_name="full_image")
        image_h, image_w = full_hwc.shape[:2]
        original_width = float(meta.get("width", image_w)) if meta is not None else float(image_w)
        original_height = float(meta.get("height", image_h)) if meta is not None else float(image_h)
        if original_width <= 0.0 or original_height <= 0.0:
            return None

        x1, y1, x2, y2 = _bbox_to_float_tuple(bbox)
        scale_x = image_w / original_width
        scale_y = image_h / original_height

        left = max(int(np.floor(x1 * scale_x)), 0)
        top = max(int(np.floor(y1 * scale_y)), 0)
        right = min(int(np.ceil(x2 * scale_x)), image_w)
        bottom = min(int(np.ceil(y2 * scale_y)), image_h)
        if right <= left or bottom <= top:
            return None
        crop = full_hwc[top:bottom, left:right]
        return crop if crop.size > 0 else None

    def extract(
        self,
        person_crop: torch.Tensor | np.ndarray,
        full_image: torch.Tensor | np.ndarray | None = None,
        bbox: torch.Tensor | np.ndarray | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> Dict[str, Dict[str, object]]:
        crop_hwc = image_to_numpy_hwc(person_crop, image_name="person_crop")
        crop_result = self._detect_on_image(crop_hwc, source_image="person_crop")
        pose_result = _clone_landmark_result(crop_result.pose)
        face_result = _clone_landmark_result(crop_result.face)

        if self.use_full_image_fallback and full_image is not None and bbox is not None and (not pose_result["detected"] or not face_result["detected"]):
            full_crop = self._crop_from_full_image(full_image, bbox, meta)
            if full_crop is not None:
                full_result = self._detect_on_image(full_crop, source_image="full_image_bbox")
                if not pose_result["detected"] and full_result.pose["detected"]:
                    pose_result = _clone_landmark_result(full_result.pose)
                if not face_result["detected"] and full_result.face["detected"]:
                    face_result = _clone_landmark_result(full_result.face)

        if not pose_result["detected"]:
            pose_result["message"] = "pose_missing"
        if not face_result["detected"]:
            face_result["message"] = "face_missing"
        return {"pose": pose_result, "face": face_result}


def augment_sample_with_landmarks_and_pas(
    sample: Mapping[str, object],
    *,
    landmark_extractor: LandmarkExtractor | None,
    pas_generator: PASGenerator | None = None,
    pas_debug_writer: PASDebugWriter | None = None,
    landmark_cache_dir: Path | None = None,
    precomputed_pas_root: Path | None = None,
    precomputed_pas_index_map: Mapping[tuple[str, str, tuple[int, int, int, int]], int] | None = None,
) -> Dict[str, object]:
    person_crop = sample["person_crop"]
    augmented = dict(sample)
    meta = dict(sample.get("meta", {}))
    preprocess_notes = list(meta.get("preprocess_notes", []))
    pas_source_crop = sample.get("pas_source_person_crop", person_crop)
    pas_source_full_image = sample.get("pas_source_full_image", sample.get("full_image"))

    if precomputed_pas_root is not None:
        filename = str(meta.get("filename", ""))
        split = str(meta.get("pas_split") or meta.get("source_split") or meta.get("split") or "")
        bbox = sample.get("bbox")
        bbox_original = sample.get("bbox_original", bbox)
        if not filename or not split or bbox is None:
            raise ValueError(
                "precomputed_pas_root requires sample meta to include 'filename' and 'pas_split'/'source_split'/'split', plus 'bbox'."
            )
        index = None
        if precomputed_pas_index_map is not None:
            index = precomputed_pas_index_map.get(_precomputed_pas_lookup_key(split, filename, bbox_original))
        person_h, person_w = int(person_crop.shape[-2]), int(person_crop.shape[-1])
        if index is not None:
            pas_image = _load_precomputed_pas_image(
                precomputed_pas_root=Path(precomputed_pas_root),
                split=split,
                index=index,
                target_shape=(person_h, person_w),
            )
            preprocess_notes.append("precomputed_pas_hit")
        else:
            pas_image = np.zeros((person_h, person_w, 3), dtype=np.uint8)
            preprocess_notes.append("precomputed_pas_miss_zero")
        landmarks = {
            "pose": _empty_landmark_result("pose", "precomputed_pas", message="precomputed_pas"),
            "face": _empty_landmark_result("face", "precomputed_pas", message="precomputed_pas"),
        }
        augmented["landmarks"] = landmarks
        augmented["pas_mask"] = numpy_mask_to_tensor((pas_image.sum(axis=2) > 0).astype(np.float32))
        augmented["pas_image"] = numpy_hwc_to_chw_tensor(pas_image)
        augmented.pop("pas_source_person_crop", None)
        augmented.pop("pas_source_full_image", None)
        meta["preprocess_notes"] = preprocess_notes
        augmented["meta"] = meta
        return augmented

    landmarks = None
    if landmark_cache_dir is not None:
        landmarks = load_landmarks_cache(landmark_cache_dir, str(meta.get("sample_id", "sample")))
        if landmarks:
            preprocess_notes.append("landmark_cache_hit")

    if landmarks is None:
        if landmark_extractor is None:
            raise ValueError("landmark_extractor is required when precomputed_pas_root is not set.")
        try:
            landmarks = landmark_extractor.extract(
                person_crop=pas_source_crop,
                full_image=pas_source_full_image,
                bbox=sample.get("bbox"),
                meta=meta,
            )
            pose_source = landmarks.get("pose", {}).get("source_image")
            face_source = landmarks.get("face", {}).get("source_image")
            preprocess_notes.append(f"landmark_source:pose={pose_source},face={face_source}")
        except Exception as exc:
            preprocess_notes.append(f"landmark_fallback:{type(exc).__name__}")
            landmarks = {
                "pose": _empty_landmark_result("pose", "person_crop", message="landmark_exception"),
                "face": _empty_landmark_result("face", "person_crop", message="landmark_exception"),
            }

    augmented["landmarks"] = landmarks
    if pas_generator is None:
        augmented["pas_mask"] = torch.zeros((1, person_crop.shape[-2], person_crop.shape[-1]), dtype=person_crop.dtype)
        augmented["pas_image"] = torch.zeros_like(person_crop)
        meta["preprocess_notes"] = preprocess_notes
        augmented["meta"] = meta
        return augmented

    try:
        pas_output = pas_generator.generate(image=pas_source_crop, landmarks=landmarks)
    except Exception as exc:
        preprocess_notes.append(f"pas_fallback:{type(exc).__name__}")
        crop_hwc = image_to_numpy_hwc(pas_source_crop, image_name="person_crop")
        pas_output = {
            "mask": np.zeros(crop_hwc.shape[:2], dtype=np.float32),
            "pas_image": np.zeros_like(crop_hwc, dtype=np.uint8),
        }

    target_shape = (int(person_crop.shape[-2]), int(person_crop.shape[-1]))
    resized_pas_mask = resize_mask(pas_output["mask"], target_shape)
    resized_pas_image = resize_rgb_image(pas_output["pas_image"], target_shape)

    augmented["pas_mask"] = numpy_mask_to_tensor(resized_pas_mask)
    augmented["pas_image"] = numpy_hwc_to_chw_tensor(resized_pas_image)
    if pas_debug_writer is not None:
        pas_debug_writer.maybe_write(
            sample_id=str(meta.get("sample_id", "sample")),
            image=image_to_numpy_hwc(pas_source_crop, image_name="person_crop"),
            mask=pas_output["mask"],
            pas_image=pas_output["pas_image"],
        )
    augmented.pop("pas_source_person_crop", None)
    augmented.pop("pas_source_full_image", None)
    meta["preprocess_notes"] = preprocess_notes
    augmented["meta"] = meta
    return augmented


class EMOTICPreprocessedDataset(Dataset):
    """Wrapper that deterministically augments EMOTIC samples with PAS."""

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        landmark_extractor: LandmarkExtractor | None,
        pas_generator: PASGenerator | None = None,
        pas_debug_writer: PASDebugWriter | None = None,
        landmark_cache_dir: Path | None = None,
        precomputed_pas_root: Path | None = None,
        precomputed_pas_index_map: Mapping[tuple[str, str, tuple[int, int, int, int]], int] | None = None,
        augment: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.landmark_extractor = landmark_extractor
        self.pas_generator = pas_generator
        self.pas_debug_writer = pas_debug_writer
        self.landmark_cache_dir = landmark_cache_dir
        self.precomputed_pas_root = Path(precomputed_pas_root).resolve() if precomputed_pas_root is not None else None
        self.precomputed_pas_index_map = dict(precomputed_pas_index_map) if precomputed_pas_index_map is not None else None
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base_dataset)

    @staticmethod
    def _apply_synchronized_augmentation(sample: Dict[str, object]) -> Dict[str, object]:
        """Apply the same augmentation to full/body inputs and rebuild PAS from the transformed mask."""
        person_crop = sample["person_crop"]
        full_image = sample["full_image"]
        pas_mask = sample.get("pas_mask")
        params = sample_strong_augmentation()

        full_image = apply_image_augmentation(full_image, params, allow_erase=False)
        person_crop = apply_image_augmentation(person_crop, params, allow_erase=True)

        sample["person_crop"] = person_crop
        sample["full_image"] = full_image
        if isinstance(pas_mask, torch.Tensor):
            pas_mask = apply_mask_augmentation(pas_mask, params)
            sample["pas_mask"] = pas_mask
            sample["pas_image"] = (person_crop * pas_mask.to(dtype=person_crop.dtype)).clamp(0.0, 1.0)
        return sample

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = augment_sample_with_landmarks_and_pas(
            self.base_dataset[index],
            landmark_extractor=self.landmark_extractor,
            pas_generator=self.pas_generator,
            pas_debug_writer=self.pas_debug_writer,
            landmark_cache_dir=self.landmark_cache_dir,
            precomputed_pas_root=self.precomputed_pas_root,
            precomputed_pas_index_map=self.precomputed_pas_index_map,
        )
        if self.augment:
            sample = self._apply_synchronized_augmentation(sample)
        return sample
