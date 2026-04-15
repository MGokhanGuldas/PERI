"""
PAS (Part Aware Spatial) Image Generator
=========================================
Makale Sec. 3.3 + pose landmark koordinatları.

MediaPipe 3 farklı API varyantını destekler:
  1. solutions.holistic  — mediapipe 0.9.x
  2. solutions.pose + solutions.face_mesh — mediapipe 0.10.x (erken)
  3. tasks.vision.PoseLandmarker + FaceLandmarker — mediapipe 0.10.14+ (güncel)

Model dosyaları (Tasks API için) otomatik indirilir:
  ~/.mediapipe_models/pose_landmarker_{lite,full,heavy}.task
  ~/.mediapipe_models/face_landmarker.task
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple
import warnings

NUM_POSE_LANDMARKS = 33   # MediaPipe Pose landmark sayısı

# ── Model dosya URL'leri (Tasks API) ─────────────────────────────────
_MODEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".mediapipe_models")

_POSE_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_POSE_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}
_FACE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_FACE_MODEL_NAME = "face_landmarker.task"


# ── MediaPipe import ve versiyon tespiti ─────────────────────────────
try:
    import mediapipe as mp
    _has_solutions = hasattr(mp, "solutions")
    _has_holistic  = _has_solutions and hasattr(mp.solutions, "holistic")
    _has_pose      = _has_solutions and hasattr(mp.solutions, "pose")
    _has_facemesh  = _has_solutions and hasattr(mp.solutions, "face_mesh")
except ImportError:
    mp = None
    _has_solutions = _has_holistic = _has_pose = _has_facemesh = False

# Tasks API ayrı kontrol — mp.tasks.python attribute olarak erişilemeyebilir
# ama doğrudan import çalışır (mediapipe 0.10.14+)
_has_tasks_vision = False
try:
    from mediapipe.tasks.python import vision as _mp_vision
    _has_tasks_vision = hasattr(_mp_vision, "PoseLandmarker")
except (ImportError, AttributeError):
    pass

if mp is not None:
    MEDIAPIPE_AVAILABLE = _has_solutions or _has_tasks_vision
    if not MEDIAPIPE_AVAILABLE:
        warnings.warn(
            "MediaPipe kurulu ama ne solutions ne de Tasks API kullanılabilir.\n"
            "PAS imajları sıfır mask ile oluşturulacak."
        )
else:
    MEDIAPIPE_AVAILABLE = False
    warnings.warn(
        "MediaPipe bulunamadı. `pip install mediapipe` ile kurabilirsin.\n"
        "MediaPipe olmadan PAS imajları sıfır mask ile oluşturulacak."
    )


def _ensure_model(url: str, filename: str) -> str:
    """Model dosyasını indir (yoksa) ve yolunu döner."""
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    path = os.path.join(_MODEL_CACHE_DIR, filename)
    if not os.path.exists(path):
        import urllib.request
        print(f"MediaPipe model indiriliyor: {filename} ...")
        urllib.request.urlretrieve(url, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Kaydedildi: {path} ({size_mb:.1f} MB)")
    return path


class PASGenerator:
    """
    Part Aware Spatial (PAS) imaj + pose landmark üretici.

    Kullanım:
        gen = PASGenerator(sigma=3.0, threshold=0.05)
        pas_img, found, pose_landmarks = gen(body_crop_bgr)
        # pose_landmarks: (33, 2) float32, normalize [0,1]
    """

    def __init__(
        self,
        sigma: float = 3.0,
        threshold: float = 0.05,
        output_size: Tuple[int, int] = (128, 128),
        use_body: bool = True,
        use_face: bool = True,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.3,
    ):
        self.sigma       = sigma
        self.threshold   = threshold
        self.output_size = output_size
        self.use_body    = use_body
        self.use_face    = use_face

        # API handles
        self._holistic      = None   # solutions.holistic (0.9.x)
        self._pose          = None   # solutions.pose (early 0.10.x)
        self._face_mesh     = None   # solutions.face_mesh (early 0.10.x)
        self._pose_task     = None   # tasks.vision.PoseLandmarker (0.10.14+)
        self._face_task     = None   # tasks.vision.FaceLandmarker (0.10.14+)
        self._api_mode      = "none" # "holistic" | "solutions" | "tasks" | "none"

        if not MEDIAPIPE_AVAILABLE:
            return

        # ── API seçimi: eski → yeni sırasıyla dene ──────────────────
        if _has_holistic:
            self._api_mode = "holistic"
            self._holistic = mp.solutions.holistic.Holistic(
                static_image_mode=True,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )

        elif _has_pose or _has_facemesh:
            self._api_mode = "solutions"
            if use_body and _has_pose:
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                )
            if use_face and _has_facemesh:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    min_detection_confidence=min_detection_confidence,
                    refine_landmarks=True,
                )

        elif _has_tasks_vision:
            self._api_mode = "tasks"
            self._init_tasks_api(model_complexity, min_detection_confidence)

    def _init_tasks_api(
        self, model_complexity: int, min_detection_confidence: float,
    ) -> None:
        """MediaPipe Tasks API (0.10.14+) ile PoseLandmarker + FaceLandmarker başlat."""
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        if self.use_body:
            pose_model = _ensure_model(
                _POSE_MODEL_URLS[model_complexity],
                _POSE_MODEL_NAMES[model_complexity],
            )
            pose_options = vision.PoseLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=pose_model),
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
            self._pose_task = vision.PoseLandmarker.create_from_options(pose_options)

        if self.use_face:
            face_model = _ensure_model(_FACE_MODEL_URL, _FACE_MODEL_NAME)
            face_options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=face_model),
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
            )
            self._face_task = vision.FaceLandmarker.create_from_options(face_options)

    # ─────────────────────── Landmark Extraction ─────────────────────

    def _extract_all_landmarks(
        self, image_rgb: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        MediaPipe'ı bir kez çalıştırıp iki çıktı üretir:

        Returns:
            pixel_coords:    (N, 2) int32 — PAS mask için tüm landmark'lar (body+face).
                             None ise hiç bulunamadı.
            pose_normalized: (33, 2) float32 — body pose landmark'ları [0,1] normalize.
                             None ise pose bulunamadı.
        """
        if self._api_mode == "holistic":
            return self._extract_holistic(image_rgb)
        elif self._api_mode == "solutions":
            return self._extract_solutions(image_rgb)
        elif self._api_mode == "tasks":
            return self._extract_tasks(image_rgb)
        else:
            return None, None

    def _extract_holistic(
        self, image_rgb: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """solutions.holistic API (0.9.x)."""
        H, W = image_rgb.shape[:2]
        pixel_coords: list = []
        pose_normalized = None

        results = self._holistic.process(image_rgb)

        if self.use_body and results.pose_landmarks:
            pose_normalized = np.array(
                [[lm.x, lm.y] for lm in results.pose_landmarks.landmark],
                dtype=np.float32,
            )
            for lm in results.pose_landmarks.landmark:
                pixel_coords.append((
                    int(np.clip(lm.x * W, 0, W - 1)),
                    int(np.clip(lm.y * H, 0, H - 1)),
                ))

        if self.use_face and results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                pixel_coords.append((
                    int(np.clip(lm.x * W, 0, W - 1)),
                    int(np.clip(lm.y * H, 0, H - 1)),
                ))

        all_coords = (
            np.array(pixel_coords, dtype=np.int32) if pixel_coords else None
        )
        return all_coords, pose_normalized

    def _extract_solutions(
        self, image_rgb: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """solutions.pose + solutions.face_mesh API (early 0.10.x)."""
        H, W = image_rgb.shape[:2]
        pixel_coords: list = []
        pose_normalized = None

        if self.use_body and self._pose is not None:
            results = self._pose.process(image_rgb)
            if results.pose_landmarks:
                pose_normalized = np.array(
                    [[lm.x, lm.y] for lm in results.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                for lm in results.pose_landmarks.landmark:
                    pixel_coords.append((
                        int(np.clip(lm.x * W, 0, W - 1)),
                        int(np.clip(lm.y * H, 0, H - 1)),
                    ))

        if self.use_face and self._face_mesh is not None:
            results = self._face_mesh.process(image_rgb)
            if results.multi_face_landmarks:
                for lm in results.multi_face_landmarks[0].landmark:
                    pixel_coords.append((
                        int(np.clip(lm.x * W, 0, W - 1)),
                        int(np.clip(lm.y * H, 0, H - 1)),
                    ))

        all_coords = (
            np.array(pixel_coords, dtype=np.int32) if pixel_coords else None
        )
        return all_coords, pose_normalized

    def _extract_tasks(
        self, image_rgb: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """tasks.vision API (0.10.14+ / güncel)."""
        H, W = image_rgb.shape[:2]
        pixel_coords: list = []
        pose_normalized = None

        # RGB numpy → MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Pose
        if self.use_body and self._pose_task is not None:
            result = self._pose_task.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                lms = result.pose_landmarks[0]       # ilk kişi
                pose_normalized = np.array(
                    [[lm.x, lm.y] for lm in lms], dtype=np.float32,
                )
                for lm in lms:
                    pixel_coords.append((
                        int(np.clip(lm.x * W, 0, W - 1)),
                        int(np.clip(lm.y * H, 0, H - 1)),
                    ))

        # Face
        if self.use_face and self._face_task is not None:
            result = self._face_task.detect(mp_image)
            if result.face_landmarks and len(result.face_landmarks) > 0:
                for lm in result.face_landmarks[0]:
                    pixel_coords.append((
                        int(np.clip(lm.x * W, 0, W - 1)),
                        int(np.clip(lm.y * H, 0, H - 1)),
                    ))

        all_coords = (
            np.array(pixel_coords, dtype=np.int32) if pixel_coords else None
        )
        return all_coords, pose_normalized

    # ─────────────────────── PAS Image Generation ────────────────────

    def _gaussian_kernel(
        self, size: Tuple[int, int], cx: int, cy: int,
    ) -> np.ndarray:
        """Makale Eq.1: Gaussian kernel."""
        H, W = size
        ys, xs = np.mgrid[0:H, 0:W]
        kernel = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * self.sigma ** 2))
        kernel /= kernel.max() + 1e-8
        return kernel.astype(np.float32)

    def _build_mask(
        self, image_shape: Tuple[int, int], landmarks: np.ndarray,
    ) -> np.ndarray:
        """Makale Eq.2: Binary part-aware mask B'."""
        H, W = image_shape
        acc = np.zeros((H, W), dtype=np.float32)
        for cx, cy in landmarks:
            acc = np.maximum(acc, self._gaussian_kernel((H, W), cx, cy))
        return (acc >= self.threshold).astype(np.float32)

    def generate(
        self, body_crop_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        """
        Body crop (BGR) → PAS imajı + pose landmarks.

        Returns:
            pas_image:       (H, W, 3) float32 [0,1]
            landmark_found:  bool
            pose_landmarks:  (33, 2) float32, normalize [0,1].
                             Landmark bulunamadıysa sıfır array.
        """
        H, W = body_crop_bgr.shape[:2]
        image_rgb = cv2.cvtColor(body_crop_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe'ı bir kez çalıştır
        pixel_coords, pose_normalized = self._extract_all_landmarks(image_rgb)
        found = pixel_coords is not None

        # PAS imajı oluştur
        if found:
            mask = self._build_mask((H, W), pixel_coords)
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            pas = (image_rgb.astype(np.float32) / 255.0) * mask_3ch
        else:
            pas = np.zeros((H, W, 3), dtype=np.float32)

        if (H, W) != self.output_size:
            pas = cv2.resize(
                pas,
                (self.output_size[1], self.output_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Pose landmarks — bulunamadıysa sıfır
        if pose_normalized is None:
            pose_normalized = np.zeros(
                (NUM_POSE_LANDMARKS, 2), dtype=np.float32,
            )

        return pas, found, pose_normalized

    def __call__(
        self, body_crop_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, bool, np.ndarray]:
        return self.generate(body_crop_bgr)

    def close(self):
        """Tüm MediaPipe kaynaklarını serbest bırak."""
        for handle in [self._holistic, self._pose, self._face_mesh]:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

        for handle in [self._pose_task, self._face_task]:
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass

        self._holistic = self._pose = self._face_mesh = None
        self._pose_task = self._face_task = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ── Precompute fonksiyonu ────────────────────────────────────────────

def precompute_pas_images(
    image_paths: list,
    bbox_list: list,
    output_dir: str,
    sigma: float = 3.0,
    threshold: float = 0.05,
    output_size: Tuple[int, int] = (128, 128),
) -> None:
    """PAS imajlarını ve pose landmark'larını toplu hesaplayıp diske kaydeder."""
    import os
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)
    gen = PASGenerator(sigma=sigma, threshold=threshold, output_size=output_size)
    found_count = 0

    for i, (img_path, bbox) in enumerate(tqdm(
        zip(image_paths, bbox_list), total=len(image_paths), desc="PAS üretiliyor"
    )):
        save_path = os.path.join(output_dir, f"{i:06d}.png")
        lm_path   = os.path.join(output_dir, f"{i:06d}_landmarks.npy")
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Okunamadı: {img_path}")

            x1, y1, x2, y2 = [int(v) for v in bbox]
            H_i, W_i = image.shape[:2]
            body_crop = image[max(0, y1):min(H_i, y2), max(0, x1):min(W_i, x2)]
            if body_crop.size == 0:
                body_crop = image

            pas, found, pose_lm = gen(body_crop)
            if found:
                found_count += 1

            pas_uint8 = (pas * 255).astype(np.uint8)
            cv2.imwrite(save_path, cv2.cvtColor(pas_uint8, cv2.COLOR_RGB2BGR))
            np.save(lm_path, pose_lm)

        except Exception:
            cv2.imwrite(save_path, np.zeros((*output_size, 3), dtype=np.uint8))
            np.save(lm_path, np.zeros((NUM_POSE_LANDMARKS, 2), dtype=np.float32))

    gen.close()
    print(f"Tamamlandı: {found_count}/{len(image_paths)} imajda landmark bulundu.")
