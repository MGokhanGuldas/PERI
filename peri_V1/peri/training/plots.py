"""Simple training-curve generation for standardized run folders."""

from __future__ import annotations
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False


def _epochs(history: list[dict[str, object]]) -> list[int]:
    return [int(row["epoch"]) for row in history]


def _values(history: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in history:
        value = row.get(key)
        if value is None:
            values.append(float("nan"))
        else:
            values.append(float(value))
    return values


def _save_plot(path: Path, *, title: str, ylabel: str, series: list[tuple[str, list[float]]], epochs: list[int]) -> None:
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    for label, values in series:
        axis.plot(epochs, values, marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    if len(series) > 1:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_per_class_ap_plot(path: Path, per_class_ap: dict[str, float | None]) -> None:
    if not MATPLOTLIB_AVAILABLE or plt is None:
        return
    filtered = [(label, float(value)) for label, value in per_class_ap.items() if value is not None]
    if not filtered:
        return
    filtered.sort(key=lambda item: item[1], reverse=True)
    labels = [label for label, _ in filtered]
    values = [value for _, value in filtered]

    figure_height = max(5.0, 0.3 * len(labels))
    figure, axis = plt.subplots(figsize=(9, figure_height))
    axis.barh(labels, values, color="#4C78A8")
    axis.invert_yaxis()
    axis.set_title("Per-Class AP")
    axis.set_xlabel("Average Precision")
    axis.set_ylabel("Emotion Class")
    axis.set_xlim(0.0, max(1.0, max(values) * 1.05))
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_training_plots(
    *,
    history: list[dict[str, object]],
    loss_curve_path: Path,
    map_curve_path: Path,
    vad_curve_path: Path,
    per_class_ap_path: Path,
    lr_curve_path: Path,
    include_lr: bool,
    per_class_ap: dict[str, float | None] | None = None,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    if not history:
        return
    epochs = _epochs(history)
    _save_plot(
        loss_curve_path,
        title="Loss",
        ylabel="Loss",
        series=[
            ("Train", _values(history, "train_loss")),
            ("Val", _values(history, "val_loss")),
        ],
        epochs=epochs,
    )
    _save_plot(
        map_curve_path,
        title="mAP",
        ylabel="mAP",
        series=[
            ("Train", _values(history, "train_map")),
            ("Val", _values(history, "val_map")),
        ],
        epochs=epochs,
    )
    _save_plot(
        vad_curve_path,
        title="VAD Error",
        ylabel="L1 Error",
        series=[
            ("Train", _values(history, "train_vad_error")),
            ("Val", _values(history, "val_vad_error")),
        ],
        epochs=epochs,
    )
    if include_lr:
        _save_plot(
            lr_curve_path,
            title="Learning Rate",
            ylabel="LR",
            series=[("LR", _values(history, "learning_rate"))],
            epochs=epochs,
        )
    if per_class_ap:
        write_per_class_ap_plot(per_class_ap_path, per_class_ap)
