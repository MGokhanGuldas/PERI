from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.training import Trainer, TrainingConfig


def _parse_cont_in_stages(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"none", ""}:
        return ()
    return tuple(stage.strip() for stage in value.split(",") if stage.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PERI with standardized run tracking.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mode", choices=("paper_faithful", "experimental"), default="paper_faithful")
    parser.add_argument("--backend", choices=("npy", "jpg"), default="npy")
    parser.add_argument("--model-name", type=str, default="peri")
    parser.add_argument("--experiment-name", type=str, default="peri")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--pas-fusion-mode", choices=("cont_in", "late", "none"), default="cont_in")
    parser.add_argument("--cont-in-stages", type=str, default="layer1,layer2,layer3,layer4")
    parser.add_argument("--cont-in-variant", choices=("paper", "residual"), default="paper")
    parser.add_argument("--pas-sigma", type=float, default=3.0)
    parser.add_argument("--pas-rho", type=float, default=None)
    parser.add_argument("--pas-radius-scale", type=float, default=2.0)
    parser.add_argument("--pas-debug", action="store_true")
    parser.add_argument("--pas-debug-max-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--optimizer", choices=("adamw",), default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "step", "onecycle", "cosine"), default="cosine")
    parser.add_argument("--scheduler-step-size", type=int, default=7)
    parser.add_argument("--scheduler-gamma", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "runs")
    parser.add_argument("--save-failed-batches", action="store_true")
    parser.add_argument("--allow-invalid-batches", action="store_true")
    parser.add_argument("--evaluate-test-after-train", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vad-weight", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--emotion-loss", choices=("dynamic_mse", "focal"), default="dynamic_mse")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--no-augment", action="store_true", help="Disable training data augmentation")
    parser.add_argument("--landmark-cache-dir", type=str, default=None, help="Path to pre-calculated landmarks")
    parser.add_argument(
        "--precomputed-pas-root",
        type=str,
        default=None,
        help="Path to precomputed PAS folders (pas_train/pas_val/pas_test)",
    )
    parser.add_argument(
        "--npy-manifest-root",
        type=str,
        default=None,
        help="Path to split manifest CSVs (train.csv/val.csv/test.csv) used to align NPY records to paper/PAS splits",
    )
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Use WeightedRandomSampler to balance classes")
    args = parser.parse_args()

    config = TrainingConfig(
        mode=args.mode,
        model_name=args.model_name,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        data_root=args.root,
        dataset_backend=args.backend,
        pas_fusion_mode=args.pas_fusion_mode,
        cont_in_stages=_parse_cont_in_stages(args.cont_in_stages),
        cont_in_variant=args.cont_in_variant,
        pas_sigma=args.pas_sigma,
        pas_rho=args.pas_rho,
        pas_radius_scale=args.pas_radius_scale,
        pas_debug=args.pas_debug,
        pas_debug_max_samples=args.pas_debug_max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
        device=args.device,
        seed=args.seed,
        output_root=args.output_root,
        resume_from=args.resume_from,
        save_failed_batches=args.save_failed_batches,
        allow_invalid_batches=args.allow_invalid_batches,
        evaluate_test_after_train=args.evaluate_test_after_train,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        vad_weight=args.vad_weight,
        label_smoothing=args.label_smoothing,
        emotion_loss_name=args.emotion_loss,
        focal_gamma=args.focal_gamma,
        augment=not args.no_augment,
        landmark_cache_dir=Path(args.landmark_cache_dir) if args.landmark_cache_dir else None,
        precomputed_pas_root=Path(args.precomputed_pas_root) if args.precomputed_pas_root else None,
        npy_manifest_root=Path(args.npy_manifest_root) if args.npy_manifest_root else None,
        use_weighted_sampler=args.use_weighted_sampler,
    )

    trainer = Trainer(config)
    try:
        final_metrics = trainer.fit()
        print(json.dumps(final_metrics, indent=2, ensure_ascii=True, default=str))
    except KeyboardInterrupt:
        trainer.write_summary(status="interrupted", reason="keyboard_interrupt")
        raise SystemExit(130) from None
    except Exception as exc:
        trainer.write_summary(status="failed", reason=f"{type(exc).__name__}: {exc}")
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
