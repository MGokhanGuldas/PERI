from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.analysis import ensure_dir, write_json


JSON_FILENAMES = (
    "run_config.json",
    "summary.json",
    "final_metrics.json",
    "dataset_summary.json",
    "training_history.json",
)


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}__{key}" if prefix else str(key)
            _flatten(child_prefix, child, out)
        return
    if isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=True)
        return
    out[prefix] = value


def _safe_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _summarize_training_history(history: Any) -> dict[str, Any]:
    if not isinstance(history, list) or not history:
        return {
            "training_history__epochs_recorded": 0,
            "training_history__last_epoch": "",
            "training_history__last_train_map": "",
            "training_history__last_val_map": "",
            "training_history__last_train_loss": "",
            "training_history__last_val_loss": "",
        }
    last = history[-1]
    return {
        "training_history__epochs_recorded": len(history),
        "training_history__last_epoch": last.get("epoch", ""),
        "training_history__last_train_map": last.get("train_map", ""),
        "training_history__last_val_map": last.get("val_map", ""),
        "training_history__last_train_loss": last.get("train_loss", ""),
        "training_history__last_val_loss": last.get("val_loss", ""),
    }


def _summarize_dataset_summary(dataset_summary: Any) -> dict[str, Any]:
    if not isinstance(dataset_summary, dict):
        return {}
    summary: dict[str, Any] = {}
    split_counts = dataset_summary.get("split_counts")
    if isinstance(split_counts, dict):
        for split_name, count in split_counts.items():
            summary[f"dataset_summary__split_counts__{split_name}"] = count
    split_overlap = dataset_summary.get("split_overlap_counts")
    if isinstance(split_overlap, dict):
        for overlap_name, count in split_overlap.items():
            summary[f"dataset_summary__split_overlap_counts__{overlap_name}"] = count
    for key in ("is_valid", "invalid_bbox_count", "missing_image_count", "missing_annotation_count"):
        if key in dataset_summary:
            summary[f"dataset_summary__{key}"] = dataset_summary.get(key)
    return summary


def _best_worst_ap(flat_row: dict[str, Any]) -> dict[str, Any]:
    ap_items: list[tuple[str, float]] = []
    for key, value in flat_row.items():
        if not key.startswith("final_metrics__per_class_ap__"):
            continue
        if value == "" or value is None:
            continue
        try:
            ap_items.append((key.split("__")[-1], float(value)))
        except (TypeError, ValueError):
            continue
    if not ap_items:
        return {}
    ap_items.sort(key=lambda item: item[1])
    worst_label, worst_value = ap_items[0]
    best_label, best_value = ap_items[-1]
    return {
        "derived__best_ap_label": best_label,
        "derived__best_ap_value": best_value,
        "derived__worst_ap_label": worst_label,
        "derived__worst_ap_value": worst_value,
    }


def build_row(run_config_path: Path, runs_root: Path) -> dict[str, Any]:
    run_root = run_config_path.parent
    row: dict[str, Any] = {
        "run_root": str(run_root),
        "run_root_relative": str(run_root.relative_to(runs_root)),
    }

    documents = {name: _load_json(run_root / name) for name in JSON_FILENAMES}
    row["has_run_config"] = documents["run_config.json"] is not None
    row["has_summary"] = documents["summary.json"] is not None
    row["has_final_metrics"] = documents["final_metrics.json"] is not None
    row["has_dataset_summary"] = documents["dataset_summary.json"] is not None
    row["has_training_history"] = documents["training_history.json"] is not None

    if isinstance(documents["run_config.json"], dict):
        _flatten("run_config", documents["run_config.json"], row)
    if isinstance(documents["summary.json"], dict):
        _flatten("summary", documents["summary.json"], row)
    if isinstance(documents["final_metrics.json"], dict):
        _flatten("final_metrics", documents["final_metrics.json"], row)

    row.update(_summarize_dataset_summary(documents["dataset_summary.json"]))
    row.update(_summarize_training_history(documents["training_history.json"]))
    row.update(_best_worst_ap(row))

    # Normalize a few handy aliases for faster filtering in spreadsheets.
    row["mode"] = row.get("run_config__mode", "")
    row["experiment_name"] = row.get("run_config__experiment_name", "")
    row["run_name"] = row.get("run_config__run_name", "")
    row["model_name"] = row.get("run_config__model_name", "")
    row["status"] = row.get("summary__status", "")
    row["reason_training_ended"] = row.get("summary__reason_training_ended", "")
    row["best_val_map"] = row.get("final_metrics__best_map", "")
    row["best_val_epoch"] = row.get("final_metrics__best_epoch", "")
    row["best_val_vad_error"] = row.get("final_metrics__best_vad_error", "")
    row["test_map"] = row.get("final_metrics__test_metrics__map", "")
    row["test_precision"] = row.get("final_metrics__test_metrics__precision", "")
    row["test_recall"] = row.get("final_metrics__test_metrics__recall", "")
    row["test_f1"] = row.get("final_metrics__test_metrics__f1", "")
    row["test_vad_error"] = row.get("final_metrics__test_metrics__vad_error", "")

    return {key: _safe_value(value) for key, value in row.items()}


def _ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "run_root_relative",
        "mode",
        "experiment_name",
        "run_name",
        "model_name",
        "status",
        "reason_training_ended",
        "best_val_map",
        "best_val_epoch",
        "best_val_vad_error",
        "test_map",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_vad_error",
        "derived__best_ap_label",
        "derived__best_ap_value",
        "derived__worst_ap_label",
        "derived__worst_ap_value",
        "run_root",
        "has_run_config",
        "has_summary",
        "has_final_metrics",
        "has_dataset_summary",
        "has_training_history",
    ]
    all_columns = set()
    for row in rows:
        all_columns.update(row.keys())
    ordered = [column for column in preferred if column in all_columns]
    remaining = sorted(column for column in all_columns if column not in set(ordered))
    return ordered + remaining


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all training runs into flat CSV tables.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "run_inventory",
    )
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    output_dir = ensure_dir(args.output_dir)

    run_config_paths = sorted(runs_root.rglob("run_config.json"))
    rows = [build_row(path, runs_root) for path in run_config_paths]
    rows.sort(key=lambda row: row.get("run_root_relative", ""))

    full_columns = _ordered_columns(rows)
    _write_csv(output_dir / "all_runs_full.csv", rows, full_columns)

    summary_columns = [
        column
        for column in [
            "run_root_relative",
            "mode",
            "experiment_name",
            "run_name",
            "model_name",
            "status",
            "reason_training_ended",
            "run_config__dataset_backend",
            "run_config__include_extra_train",
            "run_config__pas_fusion_mode",
            "run_config__precomputed_pas_root",
            "run_config__batch_size",
            "run_config__num_workers",
            "run_config__augment",
            "run_config__optimizer_name",
            "run_config__learning_rate",
            "run_config__scheduler_name",
            "run_config__label_smoothing",
            "run_config__use_weighted_sampler",
            "run_config__device",
            "run_config__seed",
            "run_config__epochs",
            "summary__resume_used",
            "summary__resumed_from",
            "summary__final_epoch",
            "training_history__epochs_recorded",
            "best_val_map",
            "best_val_epoch",
            "best_val_vad_error",
            "test_map",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_vad_error",
            "derived__best_ap_label",
            "derived__best_ap_value",
            "derived__worst_ap_label",
            "derived__worst_ap_value",
        ]
        if any(column in row for row in rows)
    ]
    _write_csv(output_dir / "all_runs_summary.csv", rows, summary_columns)

    write_json(
        {
            "runs_root": str(runs_root),
            "output_dir": str(output_dir.resolve()),
            "run_count": len(rows),
            "files": [
                str((output_dir / "all_runs_full.csv").resolve()),
                str((output_dir / "all_runs_summary.csv").resolve()),
            ],
        },
        output_dir / "manifest.json",
    )
    print(json.dumps({"run_count": len(rows), "output_dir": str(output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
