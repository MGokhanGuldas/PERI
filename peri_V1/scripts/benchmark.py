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


def _run_training(config: TrainingConfig) -> dict[str, object]:
    trainer = Trainer(config)
    try:
        return trainer.fit()
    finally:
        trainer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs PERI comparison on a controlled subset.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "benchmarks" / "baseline_vs_peri")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=16)
    parser.add_argument("--max-val-samples", type=int, default=8)
    parser.add_argument("--max-test-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        data_root=args.root,
        dataset_backend="npy",
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

    baseline_metrics = _run_training(
        TrainingConfig(
            mode="paper_faithful",
            model_name="baseline_twostream",
            experiment_name="benchmark",
            run_name="baseline_twostream",
            pas_fusion_mode="none",
            **common,
        )
    )
    peri_metrics = _run_training(
        TrainingConfig(
            mode="paper_faithful",
            model_name="peri",
            experiment_name="benchmark",
            run_name="peri",
            pas_fusion_mode="cont_in",
            pas_sigma=3.0,
            **common,
        )
    )

    comparison = {
        "baseline_map": baseline_metrics["test_metrics"]["map"],
        "peri_map": peri_metrics["test_metrics"]["map"],
        "baseline_vad": baseline_metrics["test_metrics"]["vad_error"],
        "peri_vad": peri_metrics["test_metrics"]["vad_error"],
        "delta_map": float(peri_metrics["test_metrics"]["map"]) - float(baseline_metrics["test_metrics"]["map"]),
        "delta_vad": float(peri_metrics["test_metrics"]["vad_error"]) - float(baseline_metrics["test_metrics"]["vad_error"]),
        "baseline_run": str((PROJECT_ROOT / "outputs" / "runs" / "paper_faithful" / "benchmark" / "baseline_twostream").resolve()),
        "peri_run": str((PROJECT_ROOT / "outputs" / "runs" / "paper_faithful" / "benchmark" / "peri").resolve()),
    }
    write_json(comparison, output_dir / "comparison.json")
    print(json.dumps(comparison, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
