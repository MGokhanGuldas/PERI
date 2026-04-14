from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.analysis import write_json
from peri.training import Trainer, TrainingConfig


PRESETS: dict[str, dict[str, object]] = {
    "pas_off": {
        "mode": "paper_faithful",
        "model_name": "baseline_twostream",
        "pas_fusion_mode": "none",
        "cont_in_stages": ("layer1", "layer2", "layer3", "layer4"),
        "pas_sigma": 3.0,
    },
    "pas_on_cont_in_off_sigma3": {
        "mode": "experimental",
        "model_name": "peri_pas_only",
        "pas_fusion_mode": "cont_in",
        "cont_in_stages": (),
        "pas_sigma": 3.0,
    },
    "pas_on_cont_in_on_sigma1": {
        "mode": "experimental",
        "model_name": "peri_sigma1",
        "pas_fusion_mode": "cont_in",
        "cont_in_stages": ("layer1", "layer2", "layer3", "layer4"),
        "pas_sigma": 1.0,
    },
    "pas_on_cont_in_on_sigma3": {
        "mode": "paper_faithful",
        "model_name": "peri",
        "pas_fusion_mode": "cont_in",
        "cont_in_stages": ("layer1", "layer2", "layer3", "layer4"),
        "pas_sigma": 3.0,
    },
    "pas_on_cont_in_on_sigma5": {
        "mode": "experimental",
        "model_name": "peri_sigma5",
        "pas_fusion_mode": "cont_in",
        "cont_in_stages": ("layer1", "layer2", "layer3", "layer4"),
        "pas_sigma": 5.0,
    },
}


def _run_training(config: TrainingConfig) -> dict[str, object]:
    trainer = Trainer(config)
    try:
        return trainer.fit()
    finally:
        trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled PERI ablation experiments.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ablations" / "verified")
    parser.add_argument("--experiments", nargs="+", default=["pas_off", "pas_on_cont_in_on_sigma3"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--max-test-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    common = dict(
        data_root=args.root,
        dataset_backend="npy",
        experiment_name="ablations",
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        device=args.device,
        seed=args.seed,
        output_root=PROJECT_ROOT / "outputs" / "runs",
        evaluate_test_after_train=True,
    )

    for name in args.experiments:
        if name not in PRESETS:
            raise SystemExit(f"Unknown ablation preset: {name}")
        preset = PRESETS[name]
        config = TrainingConfig(
            run_name=name,
            **common,
            **preset,
        )
        metrics = _run_training(config)
        results.append(
            {
                "name": name,
                "mode": config.mode,
                "model_name": config.model_name,
                "uses_pas": config.uses_pas,
                "cont_in_enabled": len(config.cont_in_stages) > 0 and config.pas_fusion_mode == "cont_in",
                "sigma": config.pas_sigma,
                "map": metrics["test_metrics"]["map"],
                "vad_error": metrics["test_metrics"]["vad_error"],
                "run_root": str((PROJECT_ROOT / "outputs" / "runs" / config.mode / "ablations" / name).resolve()),
            }
        )

    payload = {"available_presets": PRESETS, "results": results}
    write_json(payload, output_dir / "ablation_results.json")
    print(json.dumps(payload, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
