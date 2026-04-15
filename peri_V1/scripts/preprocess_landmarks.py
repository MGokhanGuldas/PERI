"""Script to pre-calculate landmarks for the EMOTIC dataset for faster PERI training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.data.factory import create_emotic_dataset
from peri.preprocess.cache import save_landmarks_cache, load_landmarks_cache
from peri.preprocess.landmarks import LandmarkExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-calculate landmarks for EMOTIC dataset.")
    parser.add_argument("--root", type=str, required=True, help="Path to EMOTIC dataset root")
    parser.add_argument("--cache-dir", type=str, required=True, help="Path to store landmark cache")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to process")
    parser.add_argument("--backend", type=str, default="npy", choices=["npy", "jpg"])
    parser.add_argument("--limit", type=int, default=None, help="Stop after N samples (for testing)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for MediaPipe (cpu/cuda)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = LandmarkExtractor(asset_root=root / "artifacts" / "mediapipe")

    for split in args.splits:
        logger.info(f"Processing split: {split}")
        dataset = create_emotic_dataset(
            data_root=root,
            split=split,
            backend=args.backend,
            include_pas_source=True,
        )

        num_samples = len(dataset)
        if args.limit:
            num_samples = min(num_samples, args.limit)

        pbar = tqdm(total=num_samples, desc=f"{split}")
        for i in range(num_samples):
            sample = dataset[i]
            sample_id = sample["meta"]["sample_id"]
            
            # Skip if already cached
            if not args.overwrite and load_landmarks_cache(cache_dir, sample_id):
                pbar.update(1)
                continue

            try:
                landmarks = extractor.extract(
                    person_crop=sample.get("pas_source_person_crop", sample["person_crop"]),
                    full_image=sample.get("pas_source_full_image", sample.get("full_image")),
                    bbox=sample.get("bbox"),
                    meta=sample.get("meta"),
                )
                save_landmarks_cache(cache_dir, sample_id, landmarks)
            except Exception as e:
                logger.error(f"Failed for {sample_id}: {e!r}")

            pbar.update(1)
            if args.limit and (i + 1) >= args.limit:
                break
        
        pbar.close()

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
