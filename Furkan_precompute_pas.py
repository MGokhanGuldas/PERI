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
import argparse
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg
from models.pas_generator import NUM_POSE_LANDMARKS


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
    parser.add_argument("--workers", type=int, default=0,
                        help="Paralel worker sayısı. 0=otomatik (cpu_count-1)")
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
    output_dir = task["output_dir"]
    output_size = task["output_size"]
    skip       = task["skip_existing"]

    save_path = os.path.join(output_dir, f"{idx:06d}.png")
    lm_path   = os.path.join(output_dir, f"{idx:06d}_landmarks.npy")

    # Skip check
    if skip and os.path.exists(save_path) and os.path.exists(lm_path):
        return {"idx": idx, "found": None, "error": None, "skipped": True}

    try:
        # İmaj yükle
        row = pd.Series(row_dict)
        img_path = get_image_path(row, images_root)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"İmaj okunamadı: {img_path}")

        H, W = image.shape[:2]

        # Body crop
        bbox = get_bbox(row)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            body_crop = image[y1:y2, x1:x2]
        else:
            body_crop = image

        if body_crop.size == 0:
            body_crop = image

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
    from models.pas_generator import PASGenerator
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
        from models.pas_generator import PASGenerator
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
        os.path.dirname(cfg.IMAGES_DIR), f"pas_{args.split}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # CSV oku
    print(f"CSV okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"Toplam {len(df)} örnek")

    output_size = (cfg.BODY_INPUT_SIZE, cfg.BODY_INPUT_SIZE)

    # Task listesi oluştur
    tasks = []
    for idx, row in df.iterrows():
        tasks.append({
            "idx":            idx,
            "row_dict":       row.to_dict(),
            "images_root":    cfg.IMAGES_DIR,
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
