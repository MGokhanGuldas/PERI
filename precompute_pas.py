"""
PAS + Landmarks Paralel Precompute Script
==========================================
Her body crop için MediaPipe ile PAS imajı + pose landmark koordinatları üretir.
Multiprocessing ile paralel çalışarak CPU core'larını kullanır.

Kullanım:
    python precompute_pas.py --split train
    python precompute_pas.py --split train --workers 8
    python precompute_pas.py --split train --model_complexity 2   # daha doğru ama yavaş
    python precompute_pas.py --split train --workers 8 --model_complexity 2

Çıktı:
    data/emotic/pas_train/000000.png              ← PAS imajı
    data/emotic/pas_train/000000_landmarks.npy    ← Pose landmarks (33,2)
    ...

Eğitimde:
    python train_v2.py --pas_dir_train data/emotic/pas_train --pas_dir_val data/emotic/pas_val
"""

import os
import sys
import ast
import csv
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

try:
    import config as cfg  # type: ignore[import-not-found]
except ImportError:
    class _FallbackConfig:
        DATA_ROOT = SCRIPT_ROOT / "emotic"
        ARCHIVE_ROOT = DATA_ROOT / "archive"
        IMAGES_DIR = str(ARCHIVE_ROOT / "img_arrs")
        JPG_ROOT = str(DATA_ROOT / "jpg" / "emotic" / "emotic")
        TRAIN_CSV = str(ARCHIVE_ROOT / "annots_arrs" / "annot_arrs_train.csv")
        VAL_CSV = str(ARCHIVE_ROOT / "annots_arrs" / "annot_arrs_val.csv")
        TEST_CSV = str(ARCHIVE_ROOT / "annots_arrs" / "annot_arrs_test.csv")
        BODY_INPUT_SIZE = 128
        GAUSSIAN_SIGMA = 3.0
        GAUSSIAN_THRESHOLD = 0.05

    cfg = _FallbackConfig()

from pas_generator import NUM_POSE_LANDMARKS


# ─────────────────────── Argument Parser ─────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="PAS + Landmarks paralel precompute",
    )
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--sigma", type=float, default=cfg.GAUSSIAN_SIGMA)
    parser.add_argument("--threshold", type=float, default=cfg.GAUSSIAN_THRESHOLD)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Varsayılan: data/emotic/pas_{split}")
    parser.add_argument("--source_mode", type=str, default="auto",
                        choices=["auto", "npy_crop", "npy_full", "jpg_bbox"],
                        help="Body crop kaynağı: auto, npy_crop, npy_full veya jpg_bbox")
    parser.add_argument("--images_root", type=str, default=cfg.IMAGES_DIR,
                        help="NPY archive kökü (Arr_name/Crop_name için)")
    parser.add_argument("--jpg_root", type=str, default=getattr(cfg, "JPG_ROOT", None),
                        help="Orijinal JPG kökü (jpg_bbox için)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Paralel worker sayısı. 0=otomatik (cpu_count-1)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="CSV içinde hangi satır indeksinden başlanacağı. "
                             "Örn: 12001 verilirse ilk çıktı 12001.png olur.")
    parser.add_argument("--end_index", type=int, default=None,
                        help="CSV içinde hangi satır indeksinde durulacağı "
                             "(dahil). Örn: 12000 verilirse son çıktı "
                             "0012000.png olur.")
    parser.add_argument("--model_complexity", type=int, default=1,
                        choices=[0, 1, 2],
                        help="MediaPipe model complexity: 0=hızlı, 1=orta, 2=yüksek kalite")
    parser.add_argument("--min_detection_confidence", type=float, default=0.3,
                        help="MediaPipe min detection confidence")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Mevcut dosyaları atla (varsayılan: True)")
    parser.add_argument("--no_skip", action="store_true",
                        help="Mevcut dosyaları üzerine yaz")
    return parser.parse_args()


def format_output_stem(idx: int) -> str:
    """Dosya adında 7 haneli zero-padded satır indeksini kullan."""
    return f"{int(idx):07d}"


def write_precomputed_pas_manifest(
    *,
    manifest_root: str | Path,
    split: str,
    df: pd.DataFrame,
) -> Path:
    """
    peri_V1 precomputed PAS loader'ının beklediği split manifestini yaz.

    Not:
      - Satır sırası PAS index'i olarak kullanılır.
      - Bu yüzden manifest, resmi split CSV'sinin tamamını orijinal sırada içermelidir.
    """
    root = Path(manifest_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"{split}.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        for _, row in df.iterrows():
            bbox = get_bbox(row)
            if bbox is None:
                raise ValueError(f"Manifest yazılırken bbox bulunamadı: split={split}, filename={row.get('filename', '')}")
            x1, y1, x2, y2 = bbox
            writer.writerow(
                {
                    "filename": str(row.get("filename", "")).replace("\\", "/"),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
            )
    return manifest_path


# ─────────────────────── BBox / Path Yardımcıları ────────────────────

def get_bbox(row) -> tuple:
    """Satırdaki bbox koordinatlarını döner."""
    cols = [c.lower() for c in row.index]

    if all(c in cols for c in ["x1", "y1", "x2", "y2"]):
        return int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
    elif all(c in cols for c in ["x_min", "y_min", "x_max", "y_max"]):
        return (
            int(row["x_min"]),
            int(row["y_min"]),
            int(row["x_max"]),
            int(row["y_max"]),
        )
    elif all(c in cols for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]):
        x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
        return x, y, x + w, y + h
    elif all(c in cols for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]):
        return int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"])
    elif "body_bbox" in cols:
        bbox = ast.literal_eval(str(row["body_bbox"])) if isinstance(row["body_bbox"], str) else row["body_bbox"]
        return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    else:
        return None


def get_image_path(row, images_root: str) -> str:
    """Satırdan imaj yolunu oluştur."""
    filename = str(row.get("filename", row.get("image_name", ""))).strip()
    folder = str(row.get("folder", row.get("subfolder", ""))).strip()

    if folder and folder.lower() != "nan":
        return os.path.join(images_root, folder, filename)
    return os.path.join(images_root, filename)


def _normalize_rgb_array(image: np.ndarray, *, source: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Beklenen np.ndarray, gelen: {type(image)!r} ({source})")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB imaj bekleniyordu, gelen shape={image.shape} ({source})")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def load_body_crop(
    row,
    images_root: str,
    *,
    source_mode: str = "auto",
    jpg_root: str | None = None,
) -> np.ndarray:
    """
    Satırdan body crop yükle.

    source_mode:
      - auto: npy_crop -> npy_full -> jpg_bbox
      - npy_crop: sadece `Crop_name`
      - npy_full: sadece `Arr_name` + bbox
      - jpg_bbox: sadece orijinal JPG + bbox
    """
    def _load_npy_crop() -> np.ndarray | None:
        crop_name = str(row.get("crop_name", "")).strip()
        if crop_name and crop_name.lower() != "nan":
            crop_path = os.path.join(images_root, crop_name.replace("\\", "/"))
            if os.path.exists(crop_path):
                crop_rgb = _normalize_rgb_array(np.load(crop_path, allow_pickle=False), source=crop_path)
                return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        return None

    def _load_npy_full() -> np.ndarray | None:
        arr_name = str(row.get("arr_name", "")).strip()
        if arr_name and arr_name.lower() != "nan":
            full_path = os.path.join(images_root, arr_name.replace("\\", "/"))
            if os.path.exists(full_path):
                full_rgb = _normalize_rgb_array(np.load(full_path, allow_pickle=False), source=full_path)
                bbox = get_bbox(row)
                if bbox is not None:
                    h, w = full_rgb.shape[:2]
                    x1, y1, x2, y2 = bbox
                    orig_w = float(row.get("width", w) or w)
                    orig_h = float(row.get("height", h) or h)
                    if orig_w > 0 and orig_h > 0:
                        scale_x = w / orig_w
                        scale_y = h / orig_h
                        x1 = int(round(x1 * scale_x))
                        x2 = int(round(x2 * scale_x))
                        y1 = int(round(y1 * scale_y))
                        y2 = int(round(y2 * scale_y))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    body_rgb = full_rgb[y1:y2, x1:x2]
                    if body_rgb.size != 0:
                        return cv2.cvtColor(np.ascontiguousarray(body_rgb), cv2.COLOR_RGB2BGR)
                return cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)
        return None

    def _load_jpg_bbox() -> np.ndarray:
        resolved_jpg_root = jpg_root or images_root
        img_path = get_image_path(row, resolved_jpg_root)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"İmaj okunamadı: {img_path}")

        bbox = get_bbox(row)
        if bbox is not None:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            body_crop = image[y1:y2, x1:x2]
            if body_crop.size != 0:
                return body_crop
        return image

    if source_mode == "npy_crop":
        crop = _load_npy_crop()
        if crop is None:
            raise FileNotFoundError("Crop_name NPY bulunamadı.")
        return crop

    if source_mode == "npy_full":
        crop = _load_npy_full()
        if crop is None:
            raise FileNotFoundError("Arr_name NPY bulunamadı.")
        return crop

    if source_mode == "jpg_bbox":
        return _load_jpg_bbox()

    crop = _load_npy_crop()
    if crop is not None:
        return crop

    crop = _load_npy_full()
    if crop is not None:
        return crop

    return _load_jpg_bbox()


# ─────────────────────── Tek Satır İşleme ────────────────────────────

def process_single_row(task: dict) -> dict:
    """
    Tek bir satırı işleyip PAS + landmarks üretir.

    Worker process'lerde çalışır — her worker kendi PASGenerator'ını
    global olarak tutar (process başına bir kez init).

    Args:
        task: {idx, row_dict, images_root, output_dir, sigma, threshold,
               output_size, skip_existing, model_complexity, min_det_conf}
    Returns:
        {idx, found, error}
    """
    idx        = task["idx"]
    row_dict   = task["row_dict"]
    images_root = task["images_root"]
    jpg_root   = task.get("jpg_root")
    source_mode = task.get("source_mode", "auto")
    output_dir = task["output_dir"]
    output_size = task["output_size"]
    skip       = task["skip_existing"]

    stem = format_output_stem(idx)
    save_path = os.path.join(output_dir, f"{stem}.png")
    lm_path   = os.path.join(output_dir, f"{stem}_landmarks.npy")

    # Skip check
    if skip and os.path.exists(save_path) and os.path.exists(lm_path):
        return {"idx": idx, "found": None, "error": None, "skipped": True}

    try:
        # İmaj yükle
        row = pd.Series(row_dict)
        body_crop = load_body_crop(row, images_root, source_mode=source_mode, jpg_root=jpg_root)

        # PAS + landmarks üret (worker-local generator)
        gen = _get_worker_generator(task)
        pas_float, found, pose_lm = gen(body_crop)

        # Kaydet
        pas_uint8 = (pas_float * 255).astype(np.uint8)
        pas_bgr = cv2.cvtColor(pas_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, pas_bgr)
        np.save(lm_path, pose_lm)

        return {"idx": idx, "found": found, "error": None, "skipped": False}

    except Exception as e:
        # Sıfır fallback
        zeros = np.zeros((*output_size, 3), dtype=np.uint8)
        cv2.imwrite(save_path, zeros)
        np.save(lm_path, np.zeros((NUM_POSE_LANDMARKS, 2), dtype=np.float32))
        return {"idx": idx, "found": False, "error": str(e), "skipped": False}


# ── Worker-local PASGenerator ───────────────────────────────────────
# Her worker process'in kendi MediaPipe instance'ı olacak.
# Global değişkenle tutulur (process başına bir init).
_worker_gen = None


def _init_worker(sigma, threshold, output_size, model_complexity, min_det_conf):
    """Worker process başlatılırken çağrılır."""
    global _worker_gen
    from pas_generator import PASGenerator
    _worker_gen = PASGenerator(
        sigma=sigma,
        threshold=threshold,
        output_size=output_size,
        model_complexity=model_complexity,
        min_detection_confidence=min_det_conf,
    )


def _get_worker_generator(task):
    """Worker'ın PASGenerator'ını döner. Tek process modunda lazy init."""
    global _worker_gen
    if _worker_gen is None:
        from pas_generator import PASGenerator
        _worker_gen = PASGenerator(
            sigma=task["sigma"],
            threshold=task["threshold"],
            output_size=task["output_size"],
            model_complexity=task.get("model_complexity", 1),
            min_detection_confidence=task.get("min_det_conf", 0.3),
        )
    return _worker_gen


# ─────────────────────── Ana Fonksiyon ───────────────────────────────

def main():
    args = parse_args()
    skip_existing = not args.no_skip

    # CSV seç
    csv_map = {"train": cfg.TRAIN_CSV, "val": cfg.VAL_CSV, "test": cfg.TEST_CSV}
    csv_path = csv_map[args.split]

    # Output klasörü
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.images_root), f"pas_{args.split}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # CSV oku
    print(f"CSV okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Toplam {len(df)} örnek")

    manifest_path = write_precomputed_pas_manifest(
        manifest_root=Path(output_dir).resolve().parent,
        split=args.split,
        df=df,
    )
    print(f"Manifest yazıldı: {manifest_path}")

    if args.start_index < 0:
        raise ValueError("--start_index negatif olamaz.")
    if args.start_index >= len(df):
        raise ValueError(f"--start_index={args.start_index} geçersiz. CSV uzunluğu {len(df)}.")
    if args.end_index is not None:
        if args.end_index < 0:
            raise ValueError("--end_index negatif olamaz.")
        if args.end_index >= len(df):
            raise ValueError(f"--end_index={args.end_index} geçersiz. CSV uzunluğu {len(df)}.")
        if args.end_index < args.start_index:
            raise ValueError("--end_index, --start_index'ten küçük olamaz.")

    slice_end = None if args.end_index is None else args.end_index + 1
    df = df.iloc[args.start_index:slice_end]
    if df.empty:
        raise ValueError("Seçilen indeks aralığı boş.")

    if args.end_index is None:
        print(f"{args.start_index}. satırdan başlanıyor. İlk çıktı: {format_output_stem(df.index[0])}.png")
    else:
        print(
            f"{args.start_index}..{args.end_index} aralığı işlenecek. "
            f"İlk çıktı: {format_output_stem(df.index[0])}.png, "
            f"son çıktı: {format_output_stem(df.index[-1])}.png"
        )

    output_size = (cfg.BODY_INPUT_SIZE, cfg.BODY_INPUT_SIZE)

    # Task listesi oluştur
    tasks = []
    for idx, row in df.iterrows():
        tasks.append({
            "idx":            idx,
            "row_dict":       row.to_dict(),
            "images_root":    args.images_root,
            "jpg_root":       args.jpg_root,
            "source_mode":    args.source_mode,
            "output_dir":     output_dir,
            "sigma":          args.sigma,
            "threshold":      args.threshold,
            "output_size":    output_size,
            "skip_existing":  skip_existing,
            "model_complexity": args.model_complexity,
            "min_det_conf":   args.min_detection_confidence,
        })

    # Worker sayısı
    num_workers = args.workers if args.workers > 0 else max(1, cpu_count() - 1)

    print(f"Kaynak modu: {args.source_mode}")
    print(f"NPY root:    {args.images_root}")
    if args.jpg_root:
        print(f"JPG root:    {args.jpg_root}")

    found_count = 0
    error_count = 0
    skip_count  = 0
    errors_log  = []

    if num_workers == 1:
        # ── Tek process modu ─────────────────────────────────────────
        print(f"Tek process modu (model_complexity={args.model_complexity})")
        for result in tqdm(
            map(process_single_row, tasks),
            total=len(tasks),
            desc=f"PAS+LM ({args.split})",
        ):
            if result["skipped"]:
                skip_count += 1
            elif result["error"]:
                error_count += 1
                if len(errors_log) < 5:
                    errors_log.append(f"  [{result['idx']}]: {result['error']}")
            elif result["found"]:
                found_count += 1
    else:
        # ── Paralel modu ─────────────────────────────────────────────
        print(f"{num_workers} worker ile paralel çalışılıyor "
              f"(model_complexity={args.model_complexity})")
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(
                args.sigma, args.threshold, output_size,
                args.model_complexity, args.min_detection_confidence,
            ),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(process_single_row, tasks, chunksize=16),
                total=len(tasks),
                desc=f"PAS+LM ({args.split})",
            ):
                if result["skipped"]:
                    skip_count += 1
                elif result["error"]:
                    error_count += 1
                    if len(errors_log) < 5:
                        errors_log.append(f"  [{result['idx']}]: {result['error']}")
                elif result["found"]:
                    found_count += 1

    # Özet
    processed = len(tasks) - skip_count
    print(f"\n{'='*60}")
    print(f"  Tamamlandı: {args.split}")
    print(f"{'='*60}")
    print(f"  Toplam örnek:    {len(tasks)}")
    print(f"  Atlanan (mevcut):{skip_count}")
    print(f"  İşlenen:        {processed}")
    print(f"  Landmark bulundu:{found_count}/{processed}")
    print(f"  Hatalı:          {error_count}/{processed}")
    print(f"  Klasör:          {output_dir}")

    if errors_log:
        print(f"\n  İlk hatalar:")
        for e in errors_log:
            print(e)

    print(f"\nEğitimde kullanmak için:")
    print(f"  python train_v2.py --pas_dir_{args.split} {output_dir}")


if __name__ == "__main__":
    main()
