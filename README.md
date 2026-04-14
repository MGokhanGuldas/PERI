# PERI

Research-grade PERI reproduction focused on a strict paper-faithful default path and a clearly isolated experimental path.

## Modes

- `paper_faithful`
  - default mode
  - requires the NPY EMOTIC archive layout
  - uses `224x224` context images and `128x128` body/PAS inputs
  - uses two ImageNet-pretrained ResNet-18 streams
  - uses dynamic weighted MSE for the 26 emotion labels and L1 for VAD
  - uses PAS only through Cont-In blocks on the body stream
- `experimental`
  - opt-in mode for non-paper changes
  - allows JPG-backed loading and late PAS fusion ablations

## Expected Dataset Layout

Paper-faithful mode expects:

```text
<repo-or-data-root>/
  archive/
    annots_arrs/
      annot_arrs_train.csv
      annot_arrs_val.csv
      annot_arrs_test.csv
      annot_arrs_extra_train.csv        # optional
    img_arrs/
      *.npy
  artifacts/
    mediapipe/
      holistic_landmarker.task
      pose_landmarker_full.task
      face_landmarker.task
```

The NPY CSV files must contain the canonical EMOTIC columns and must map:

- full image arrays through `Arr_name`
- body crop arrays through `Crop_name`
- 26 emotion categories in canonical EMOTIC order
- VAD targets
- body bounding boxes

Experimental JPG mode expects:

```text
<repo-or-data-root>/
  jpg/
    Annotations/
      Annotations/
        Annotations.mat
    emotic/
      emotic/
        ...
```

## Workflow

1. Run fast sampled dataset validation first.
2. Run a dry run on one batch.
3. Train.
4. Evaluate a saved checkpoint.

## Run Artifacts

Each training run saves only:

- `run_config.json`
- `dataset_summary.json`
- `training_history.json`
- `final_metrics.json`
- `summary.json`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `tensorboard/`
- `plots/loss_curve.png`
- `plots/map_curve.png`
- `plots/vad_curve.png`
- `plots/per_class_ap.png` when per-class AP is available
- `plots/lr_curve.png` when a scheduler is enabled

## Commands

Fast dataset validation:

```bash
python scripts/validate_dataset.py --root .
```

Deep full-archive validation:

```bash
python scripts/validate_dataset_deep.py --root .
```

Dry run:

```bash
python scripts/dry_run.py --root . --mode paper_faithful --backend npy --pas-fusion-mode cont_in --batch-size 2
```

Paper-faithful training:

```bash
python scripts/train.py --root . --mode paper_faithful --backend npy --model-name peri --experiment-name peri --run-name paper_faithful_run --pas-fusion-mode cont_in --batch-size 16 --epochs 10 --device cuda --evaluate-test-after-train
```

TensorBoard is enabled for training runs by default. Logs are stored in:

```text
outputs/runs/<mode>/<experiment>/<run_name>/tensorboard/
```

Open TensorBoard:

```bash
python -m tensorboard.main --logdir outputs/runs
```

Baseline two-stream ablation without PAS:

```bash
python scripts/train.py --root . --mode paper_faithful --backend npy --model-name baseline_twostream --experiment-name baseline --run-name baseline_run --pas-fusion-mode none --batch-size 16 --epochs 10 --device cuda --evaluate-test-after-train
```

Experimental late-fusion PAS ablation:

```bash
python scripts/train.py --root . --mode experimental --backend npy --model-name peri_late_pas --experiment-name experimental --run-name late_pas_run --pas-fusion-mode late --pas-sigma 3 --batch-size 16 --epochs 10 --device cuda --evaluate-test-after-train
```

Checkpoint evaluation:

```bash
python scripts/evaluate.py --checkpoint outputs/runs/paper_faithful/peri/<run_name>/checkpoints/best.pt --split test --device cuda --output outputs/validation/eval_metrics.json
```

Supported scripts:

- `scripts/validate_dataset.py`
- `scripts/validate_dataset_deep.py`
- `scripts/dry_run.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/benchmark.py`
- `scripts/run_ablations.py`

## Notes

- `scripts/validate_dataset.py` is the fast default validator. It checks folder structure, annotation/index consistency, split overlap, and sampled NPY loading.
- `scripts/validate_dataset_deep.py` is the optional full-archive scan. It is intentionally separated so normal iteration stays fast.
- The validator fails loudly if the NPY archive layout is missing or inconsistent.
- Each run writes its own `dataset_summary.json` using the fast sampled validation path for reproducibility.
- Each run also writes TensorBoard event files to `tensorboard/` alongside the saved JSON artifacts and PNG plots.
- PAS generation is deterministic and uses the local MediaPipe assets; assets are not auto-downloaded by the default path.
- In the paper-faithful path, JPG loading and late PAS fusion are intentionally blocked.
