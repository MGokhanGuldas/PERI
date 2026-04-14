from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peri.analysis import ensure_dir, write_json, write_text
from peri.data.emotic_constants import EMOTION_COLUMNS


SPLIT_FILES = {
    "train": "annot_arrs_train.csv",
    "val": "annot_arrs_val.csv",
    "test": "annot_arrs_test.csv",
}


def _safe_binary_value(value: str | None) -> float:
    if value is None:
        return 0.0
    text = value.strip()
    if not text:
        return 0.0
    return float(text)


def load_labels(csv_path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append([_safe_binary_value(row.get(column)) for column in EMOTION_COLUMNS])
    if not rows:
        return np.zeros((0, len(EMOTION_COLUMNS)), dtype=np.float32)
    labels = np.asarray(rows, dtype=np.float32)
    labels[labels != 0.0] = 1.0
    return labels


def _corrcoef_binary(labels: np.ndarray) -> np.ndarray:
    if labels.shape[0] < 2:
        return np.zeros((labels.shape[1], labels.shape[1]), dtype=np.float32)
    corr = np.corrcoef(labels, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


def _conditional_matrix(pair_counts: np.ndarray, label_counts: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = pair_counts / label_counts[:, None]
    conditional = np.nan_to_num(conditional, nan=0.0, posinf=0.0, neginf=0.0)
    return conditional.astype(np.float32)


def _jaccard_matrix(pair_counts: np.ndarray, label_counts: np.ndarray) -> np.ndarray:
    unions = label_counts[:, None] + label_counts[None, :] - pair_counts
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = pair_counts / unions
    jaccard = np.nan_to_num(jaccard, nan=0.0, posinf=0.0, neginf=0.0)
    return jaccard.astype(np.float32)


def _top_pairs(pair_counts: np.ndarray, sample_count: int, limit: int = 20) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    label_count = len(EMOTION_COLUMNS)
    for left in range(label_count):
        for right in range(left + 1, label_count):
            count = int(pair_counts[left, right])
            if count == 0:
                continue
            pairs.append(
                {
                    "label_a": EMOTION_COLUMNS[left],
                    "label_b": EMOTION_COLUMNS[right],
                    "count": count,
                    "joint_probability": count / sample_count if sample_count else 0.0,
                }
            )
    pairs.sort(key=lambda item: (-item["joint_probability"], -item["count"], item["label_a"], item["label_b"]))
    return pairs[:limit]


def _top_companions(
    conditional: np.ndarray,
    pair_counts: np.ndarray,
    joint: np.ndarray,
    limit_per_label: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    companions: dict[str, list[dict[str, Any]]] = {}
    for row_index, label in enumerate(EMOTION_COLUMNS):
        entries: list[dict[str, Any]] = []
        for col_index, other_label in enumerate(EMOTION_COLUMNS):
            if row_index == col_index:
                continue
            count = int(pair_counts[row_index, col_index])
            probability = float(conditional[row_index, col_index])
            if count == 0 or probability <= 0.0:
                continue
            entries.append(
                {
                    "label": other_label,
                    "count": count,
                    "conditional_probability": probability,
                    "joint_probability": float(joint[row_index, col_index]),
                }
            )
        entries.sort(
            key=lambda item: (
                -item["conditional_probability"],
                -item["count"],
                item["label"],
            )
        )
        companions[label] = entries[:limit_per_label]
    return companions


def summarize_split(split_name: str, labels: np.ndarray) -> dict[str, Any]:
    sample_count = int(labels.shape[0])
    label_counts = labels.sum(axis=0).astype(np.int64)
    label_prevalence = (label_counts / sample_count) if sample_count else np.zeros(len(EMOTION_COLUMNS), dtype=np.float32)
    positives_total = int(label_counts.sum())
    share_of_positive_labels = (
        label_counts / positives_total if positives_total else np.zeros(len(EMOTION_COLUMNS), dtype=np.float32)
    )
    label_cardinality = labels.sum(axis=1) if sample_count else np.zeros(0, dtype=np.float32)
    pair_counts = (labels.T @ labels).astype(np.int64)
    joint = pair_counts / sample_count if sample_count else np.zeros_like(pair_counts, dtype=np.float32)
    conditional = _conditional_matrix(pair_counts, label_counts.astype(np.float32))
    jaccard = _jaccard_matrix(pair_counts.astype(np.float32), label_counts.astype(np.float32))
    correlation = _corrcoef_binary(labels)

    return {
        "split": split_name,
        "sample_count": sample_count,
        "label_count": len(EMOTION_COLUMNS),
        "positive_label_total": positives_total,
        "mean_labels_per_sample": float(label_cardinality.mean()) if sample_count else 0.0,
        "median_labels_per_sample": float(np.median(label_cardinality)) if sample_count else 0.0,
        "max_labels_per_sample": int(label_cardinality.max()) if sample_count else 0,
        "zero_label_samples": int((label_cardinality == 0).sum()) if sample_count else 0,
        "single_label_samples": int((label_cardinality == 1).sum()) if sample_count else 0,
        "multi_label_samples": int((label_cardinality > 1).sum()) if sample_count else 0,
        "label_density": float(label_cardinality.mean() / len(EMOTION_COLUMNS)) if sample_count else 0.0,
        "label_counts": {label: int(label_counts[index]) for index, label in enumerate(EMOTION_COLUMNS)},
        "label_prevalence": {label: float(label_prevalence[index]) for index, label in enumerate(EMOTION_COLUMNS)},
        "positive_share": {label: float(share_of_positive_labels[index]) for index, label in enumerate(EMOTION_COLUMNS)},
        "top_joint_pairs": _top_pairs(pair_counts, sample_count, limit=20),
        "top_companions": _top_companions(conditional, pair_counts, joint, limit_per_label=5),
        "conditional_probability": conditional.tolist(),
        "joint_probability": joint.tolist(),
        "jaccard_similarity": jaccard.tolist(),
        "correlation": correlation.tolist(),
    }


def compare_prevalence(reference_name: str, reference: dict[str, Any], other_name: str, other: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for label in EMOTION_COLUMNS:
        ref_value = float(reference["label_prevalence"][label])
        other_value = float(other["label_prevalence"][label])
        rows.append(
            {
                "label": label,
                "reference_split": reference_name,
                "other_split": other_name,
                "reference_prevalence": ref_value,
                "other_prevalence": other_value,
                "absolute_delta": abs(ref_value - other_value),
                "signed_delta": other_value - ref_value,
            }
        )
    rows.sort(key=lambda item: (-item["absolute_delta"], item["label"]))
    absolute_deltas = [row["absolute_delta"] for row in rows]
    return {
        "reference_split": reference_name,
        "other_split": other_name,
        "mean_absolute_delta": float(np.mean(absolute_deltas)) if absolute_deltas else 0.0,
        "max_absolute_delta": float(max(absolute_deltas)) if absolute_deltas else 0.0,
        "top_shifted_labels": rows[:10],
        "all_labels": rows,
    }


def _write_table(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def export_prevalence_csv(output_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows: list[list[Any]] = []
    for label in EMOTION_COLUMNS:
        row = [label]
        for split_name in ("train", "val", "test"):
            summary = summaries[split_name]
            row.extend(
                [
                    summary["label_counts"][label],
                    round(summary["label_prevalence"][label], 6),
                    round(summary["positive_share"][label], 6),
                ]
            )
        rows.append(row)
    _write_table(
        output_dir / "label_prevalence_by_split.csv",
        [
            "label",
            "train_count",
            "train_prevalence",
            "train_positive_share",
            "val_count",
            "val_prevalence",
            "val_positive_share",
            "test_count",
            "test_prevalence",
            "test_positive_share",
        ],
        rows,
    )


def export_summary_csv(output_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows: list[list[Any]] = []
    for split_name in ("train", "val", "test"):
        summary = summaries[split_name]
        rows.append(
            [
                split_name,
                summary["sample_count"],
                summary["positive_label_total"],
                round(summary["mean_labels_per_sample"], 6),
                round(summary["median_labels_per_sample"], 6),
                summary["max_labels_per_sample"],
                summary["zero_label_samples"],
                summary["single_label_samples"],
                summary["multi_label_samples"],
                round(summary["label_density"], 6),
            ]
        )
    _write_table(
        output_dir / "split_summary.csv",
        [
            "split",
            "sample_count",
            "positive_label_total",
            "mean_labels_per_sample",
            "median_labels_per_sample",
            "max_labels_per_sample",
            "zero_label_samples",
            "single_label_samples",
            "multi_label_samples",
            "label_density",
        ],
        rows,
    )


def export_matrix_csv(output_dir: Path, split_name: str, stem: str, matrix: list[list[float]]) -> None:
    rows: list[list[Any]] = []
    for label, values in zip(EMOTION_COLUMNS, matrix):
        rows.append([label, *[round(float(value), 6) for value in values]])
    _write_table(output_dir / f"{split_name}_{stem}.csv", ["label", *EMOTION_COLUMNS], rows)


def export_shift_csv(output_dir: Path, stem: str, comparison: dict[str, Any]) -> None:
    rows = [
        [
            row["label"],
            round(row["reference_prevalence"], 6),
            round(row["other_prevalence"], 6),
            round(row["signed_delta"], 6),
            round(row["absolute_delta"], 6),
        ]
        for row in comparison["all_labels"]
    ]
    _write_table(
        output_dir / f"{stem}.csv",
        [
            "label",
            f"{comparison['reference_split']}_prevalence",
            f"{comparison['other_split']}_prevalence",
            "signed_delta",
            "absolute_delta",
        ],
        rows,
    )


def _render_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    vmin: float,
    vmax: float,
    cmap: str,
    fmt: str,
) -> None:
    if plt is None:
        return
    size = max(12.0, len(EMOTION_COLUMNS) * 0.45)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(EMOTION_COLUMNS)))
    ax.set_yticks(range(len(EMOTION_COLUMNS)))
    ax.set_xticklabels(EMOTION_COLUMNS, rotation=90, fontsize=8)
    ax.set_yticklabels(EMOTION_COLUMNS, fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(fmt, rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_prevalence_comparison(output_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    if plt is None:
        return
    x = np.arange(len(EMOTION_COLUMNS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(18, 7))
    for offset, split_name in zip((-width, 0.0, width), ("train", "val", "test")):
        values = [summaries[split_name]["label_prevalence"][label] for label in EMOTION_COLUMNS]
        ax.bar(x + offset, values, width=width, label=split_name)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_COLUMNS, rotation=90, fontsize=8)
    ax.set_ylabel("Prevalence P(label=1)")
    ax.set_title("EMOTIC Split Label Prevalence")
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "label_prevalence_by_split.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_split_heatmaps(output_dir: Path, split_name: str, summary: dict[str, Any]) -> None:
    conditional = np.asarray(summary["conditional_probability"], dtype=np.float32)
    correlation = np.asarray(summary["correlation"], dtype=np.float32)
    _render_heatmap(
        conditional,
        f"{split_name.upper()} Conditional Co-occurrence P(column | row)",
        output_dir / f"{split_name}_conditional_probability.png",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        fmt="P(column | row)",
    )
    max_abs = float(np.max(np.abs(correlation))) if correlation.size else 1.0
    _render_heatmap(
        correlation,
        f"{split_name.upper()} Binary Label Correlation",
        output_dir / f"{split_name}_correlation.png",
        vmin=-max(0.2, max_abs),
        vmax=max(0.2, max_abs),
        cmap="coolwarm",
        fmt="corr",
    )


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_markdown_report(
    summaries: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# EMOTIC Split Distribution Analysis")
    lines.append("")
    lines.append(f"Output directory: `{output_dir}`")
    lines.append("")
    lines.append("## Split summary")
    lines.append("")
    lines.append("| Split | Samples | Avg labels/sample | Median | Max | Multi-label rate | Zero-label |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for split_name in ("train", "val", "test"):
        summary = summaries[split_name]
        multi_rate = summary["multi_label_samples"] / summary["sample_count"] if summary["sample_count"] else 0.0
        lines.append(
            "| "
            + " | ".join(
                [
                    split_name,
                    str(summary["sample_count"]),
                    f"{summary['mean_labels_per_sample']:.3f}",
                    f"{summary['median_labels_per_sample']:.3f}",
                    str(summary["max_labels_per_sample"]),
                    _fmt_percent(multi_rate),
                    str(summary["zero_label_samples"]),
                ]
            )
            + " |"
        )
    lines.append("")

    for split_name in ("train", "val", "test"):
        summary = summaries[split_name]
        top_labels = sorted(
            EMOTION_COLUMNS,
            key=lambda label: (
                -summary["label_prevalence"][label],
                label,
            ),
        )[:10]
        lines.append(f"## {split_name.upper()} top labels")
        lines.append("")
        lines.append("| Label | Count | Prevalence |")
        lines.append("| --- | ---: | ---: |")
        for label in top_labels:
            lines.append(
                f"| {label} | {summary['label_counts'][label]} | {_fmt_percent(summary['label_prevalence'][label])} |"
            )
        lines.append("")
        lines.append(f"### {split_name.upper()} strongest joint pairs")
        lines.append("")
        lines.append("| Pair | Count | Joint P(A and B) |")
        lines.append("| --- | ---: | ---: |")
        for pair in summary["top_joint_pairs"][:10]:
            lines.append(
                f"| {pair['label_a']} + {pair['label_b']} | {pair['count']} | {_fmt_percent(pair['joint_probability'])} |"
            )
        lines.append("")

    lines.append("## Largest prevalence shifts")
    lines.append("")
    for comparison in comparisons:
        lines.append(
            f"### {comparison['reference_split']} vs {comparison['other_split']} "
            f"(mean abs delta: {_fmt_percent(comparison['mean_absolute_delta'])})"
        )
        lines.append("")
        lines.append("| Label | Ref | Other | Delta | Abs delta |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in comparison["top_shifted_labels"]:
            lines.append(
                f"| {row['label']} | {_fmt_percent(row['reference_prevalence'])} | "
                f"{_fmt_percent(row['other_prevalence'])} | {_fmt_percent(row['signed_delta'])} | "
                f"{_fmt_percent(row['absolute_delta'])} |"
            )
        lines.append("")

    lines.append("## Example companions")
    lines.append("")
    for anchor in ("Happiness", "Pleasure", "Engagement", "Sadness", "Anger", "Fear"):
        lines.append(f"### {anchor}")
        lines.append("")
        lines.append("| Split | Companion | P(companion | anchor) | Joint P | Count |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for split_name in ("train", "val", "test"):
            companions = summaries[split_name]["top_companions"].get(anchor, [])
            if not companions:
                lines.append(f"| {split_name} | - | 0.00% | 0.00% | 0 |")
                continue
            best = companions[:3]
            for index, item in enumerate(best):
                split_cell = split_name if index == 0 else ""
                lines.append(
                    f"| {split_cell} | {item['label']} | {_fmt_percent(item['conditional_probability'])} | "
                    f"{_fmt_percent(item['joint_probability'])} | {item['count']} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze EMOTIC train/val/test label distributions and co-occurrence.")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=PROJECT_ROOT.parent / "emotic" / "archive" / "annots_arrs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "emotic_split_distribution",
    )
    args = parser.parse_args()

    annotations_root = args.annotations_root.resolve()
    output_dir = ensure_dir(args.output_dir)

    summaries: dict[str, dict[str, Any]] = {}
    for split_name, filename in SPLIT_FILES.items():
        csv_path = annotations_root / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing annotation CSV for split '{split_name}': {csv_path}")
        labels = load_labels(csv_path)
        summaries[split_name] = summarize_split(split_name, labels)

    comparisons = [
        compare_prevalence("train", summaries["train"], "val", summaries["val"]),
        compare_prevalence("train", summaries["train"], "test", summaries["test"]),
        compare_prevalence("val", summaries["val"], "test", summaries["test"]),
    ]

    write_json(
        {
            "annotations_root": str(annotations_root),
            "emotion_columns": list(EMOTION_COLUMNS),
            "plots_available": plt is not None,
            "split_summaries": summaries,
            "prevalence_comparisons": comparisons,
        },
        output_dir / "emotic_split_distribution.json",
    )

    export_summary_csv(output_dir, summaries)
    export_prevalence_csv(output_dir, summaries)

    for split_name, summary in summaries.items():
        export_matrix_csv(output_dir, split_name, "conditional_probability", summary["conditional_probability"])
        export_matrix_csv(output_dir, split_name, "correlation", summary["correlation"])
        export_matrix_csv(output_dir, split_name, "joint_probability", summary["joint_probability"])
        export_matrix_csv(output_dir, split_name, "jaccard_similarity", summary["jaccard_similarity"])
        plot_split_heatmaps(output_dir, split_name, summary)

    export_shift_csv(output_dir, "train_vs_val_prevalence_delta", comparisons[0])
    export_shift_csv(output_dir, "train_vs_test_prevalence_delta", comparisons[1])
    export_shift_csv(output_dir, "val_vs_test_prevalence_delta", comparisons[2])
    plot_prevalence_comparison(output_dir, summaries)

    report = build_markdown_report(summaries, comparisons, output_dir)
    write_text(report, output_dir / "report.md")
    print(json.dumps({"output_dir": str(output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
