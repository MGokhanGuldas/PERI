from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.analysis import write_json
from peri.data import DatasetValidationError, assert_dataset_summary_ok, build_dataset_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep EMOTIC dataset validation with full archive scan.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--backend", choices=("npy", "jpg"), default="npy")
    parser.add_argument("--annotations-root", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--annotations-mat-path", type=Path, default=None)
    parser.add_argument("--jpg-root", type=Path, default=None)
    parser.add_argument("--include-extra-train", action="store_true")
    parser.add_argument("--asset-root", type=Path, default=PROJECT_ROOT / "artifacts" / "mediapipe")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "validation" / "dataset_summary_deep.json")
    args = parser.parse_args()

    try:
        summary = build_dataset_summary(
            data_root=args.root,
            backend=args.backend,
            annotations_root=args.annotations_root,
            images_root=args.images_root,
            annotations_mat_path=args.annotations_mat_path,
            jpg_root=args.jpg_root,
            include_extra_train=args.include_extra_train,
            validate_images=True,
            deep_scan=True,
            sample_head=0,
            sample_random=0,
            mediapipe_asset_root=args.asset_root,
        )
        write_json(summary, args.output)
        print(json.dumps(summary, indent=2, ensure_ascii=True, default=str))
        assert_dataset_summary_ok(summary)
    except DatasetValidationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
