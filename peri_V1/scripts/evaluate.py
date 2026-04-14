from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.analysis import BatchMetricAccumulator, write_json
from peri.models import PERIModel
from peri.training import TrainingConfig, build_dataloaders, build_loss_module


def _move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            nested: dict[str, object] = {}
            for nested_key, nested_value in value.items():
                nested[nested_key] = nested_value.to(device) if isinstance(nested_value, torch.Tensor) else nested_value
            moved[key] = nested
        else:
            moved[key] = value
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved PERI checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TrainingConfig.from_dict(dict(checkpoint["config"]))
    config.device = args.device
    config.resume_from = None

    dataloaders = build_dataloaders(config)
    model = PERIModel(
        pretrained=config.pretrained,
        pas_fusion_mode=config.pas_fusion_mode,
        cont_in_stages=config.cont_in_stages,
        cont_in_variant=config.cont_in_variant,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loss_module = build_loss_module(
        vad_weight=config.vad_weight,
        label_smoothing=config.label_smoothing,
        emotion_loss_name=config.emotion_loss_name,
        focal_gamma=config.focal_gamma,
    ).to(config.device)
    accumulator = BatchMetricAccumulator()
    loss_sums = {"total_loss": 0.0, "emotion_loss": 0.0, "vad_loss": 0.0}
    processed = 0
    loader = {"train": dataloaders.train_loader, "val": dataloaders.val_loader, "test": dataloaders.test_loader}[args.split]

    try:
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch_to_device(batch, torch.device(config.device))
                outputs = model(batch)
                losses = loss_module(outputs, {"emotion": batch["emotion"], "vad": batch["vad"]})
                for key in loss_sums:
                    loss_sums[key] += float(losses[key].detach().item())
                accumulator.update(
                    emotion_probabilities=outputs["emotion_probs"],
                    emotion_targets=batch["emotion"],
                    vad_predictions=outputs["vad"],
                    vad_targets=batch["vad"],
                )
                processed += 1
    finally:
        dataloaders.close()

    metrics = accumulator.compute(include_per_class_ap=True)
    metrics.update(
        {
            "split": args.split,
            "processed_batches": processed,
            "total_loss": loss_sums["total_loss"] / max(processed, 1),
            "emotion_loss": loss_sums["emotion_loss"] / max(processed, 1),
            "vad_loss": loss_sums["vad_loss"] / max(processed, 1),
        }
    )
    if args.output is not None:
        write_json(metrics, args.output)
    print(json.dumps(metrics, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
