from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.analysis import compute_multilabel_metrics, compute_vad_metrics, write_json
from peri.data import DatasetValidationError, assert_dataset_summary_ok, build_dataset_summary
from peri.models import PERIModel
from peri.training import TrainingConfig, build_dataloaders, build_loss_module


def main() -> None:
    parser = argparse.ArgumentParser(description="One-batch dry run for the PERI pipeline.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mode", choices=("paper_faithful", "experimental"), default="paper_faithful")
    parser.add_argument("--backend", choices=("npy", "jpg"), default="npy")
    parser.add_argument("--pas-fusion-mode", choices=("cont_in", "late", "none"), default="cont_in")
    parser.add_argument("--cont-in-variant", choices=("paper", "residual"), default="paper")
    parser.add_argument("--pas-sigma", type=float, default=3.0)
    parser.add_argument("--pas-rho", type=float, default=None)
    parser.add_argument("--pas-radius-scale", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--emotion-loss", choices=("dynamic_mse", "focal"), default="dynamic_mse")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "validation" / "dry_run_summary.json")
    args = parser.parse_args()

    config = TrainingConfig(
        mode=args.mode,
        data_root=args.root,
        dataset_backend=args.backend,
        pas_fusion_mode=args.pas_fusion_mode,
        cont_in_variant=args.cont_in_variant,
        pas_sigma=args.pas_sigma,
        pas_rho=args.pas_rho,
        pas_radius_scale=args.pas_radius_scale,
        batch_size=args.batch_size,
        emotion_loss_name=args.emotion_loss,
        focal_gamma=args.focal_gamma,
        device=args.device,
        max_train_samples=args.batch_size,
        max_val_samples=args.batch_size,
        max_test_samples=args.batch_size,
    )
    try:
        summary = build_dataset_summary(
            data_root=config.data_root,
            backend=config.dataset_backend,
            annotations_root=config.annotations_root,
            images_root=config.images_root,
            annotations_mat_path=config.annotations_mat_path,
            jpg_root=config.jpg_root,
            include_extra_train=config.include_extra_train,
            validate_images=True,
            mediapipe_asset_root=config.mediapipe_asset_root,
        )
        assert_dataset_summary_ok(summary)
    except DatasetValidationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    dataloaders = build_dataloaders(config)
    try:
        batch = next(iter(dataloaders.train_loader))
        device = torch.device(config.device)
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        for nested_key in ("meta", "landmarks"):
            if nested_key in batch and isinstance(batch[nested_key], dict):
                batch[nested_key] = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch[nested_key].items()
                }
        model = PERIModel(
            pretrained=config.pretrained,
            pas_fusion_mode=config.pas_fusion_mode,
            cont_in_stages=config.cont_in_stages,
            cont_in_variant=config.cont_in_variant,
        ).to(device)
        loss_module = build_loss_module(
            vad_weight=config.vad_weight,
            label_smoothing=config.label_smoothing,
            emotion_loss_name=config.emotion_loss_name,
            focal_gamma=config.focal_gamma,
        ).to(device)
        outputs = model(batch)
        losses = loss_module(outputs, {"emotion": batch["emotion"], "vad": batch["vad"]})
        result = {
            "dataset_summary_valid": summary["is_valid"],
            "emotion_shape": list(outputs["emotion_probs"].shape),
            "vad_shape": list(outputs["vad"].shape),
            "losses": {key: float(value.detach().item()) for key, value in losses.items()},
            "metrics": {
                **compute_multilabel_metrics(outputs["emotion_probs"], batch["emotion"]),
                **compute_vad_metrics(outputs["vad"], batch["vad"]),
            },
        }
    finally:
        dataloaders.close()

    write_json(result, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
