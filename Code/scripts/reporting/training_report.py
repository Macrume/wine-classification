from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_network.metrics import ClassificationMetrics
from training.trainer import EpochStats


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_loss_history(
    history: list[EpochStats],
    output_path: Path,
) -> None:
    """Save train/test loss history plot."""
    _ensure_directory(output_path.parent)

    epochs = [stats.epoch for stats in history]
    train_loss = [stats.train_metrics.loss for stats in history]

    test_loss = [
        stats.test_metrics.loss if stats.test_metrics is not None else np.nan
        for stats in history
    ]

    plt.figure()
    plt.plot(epochs, train_loss, label="train")

    if not np.all(np.isnan(test_loss)):
        plt.plot(epochs, test_loss, label="test")

    plt.xlabel("Epoka")
    plt.ylabel("Wartość funkcji kosztu")
    plt.title("Przebieg funkcji kosztu w procesie uczenia")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    class_labels: list[int],
    output_path: Path,
) -> None:
    """Save confusion matrix plot."""
    _ensure_directory(output_path.parent)

    matrix = metrics.confusion_matrix

    plt.figure()
    plt.imshow(matrix)
    plt.title("Macierz pomyłek")
    plt.xlabel("Klasa przewidziana")
    plt.ylabel("Klasa rzeczywista")

    positions = np.arange(len(class_labels))
    
    label_names = [str(label) for label in class_labels]

    plt.xticks(positions, label_names)
    plt.yticks(positions, label_names)
    

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_final_test_metrics_tex(
    metrics: ClassificationMetrics,
    output_path: Path,
) -> None:
    """Save final test metrics as a LaTeX table."""
    _ensure_directory(output_path.parent)

    lines = [
        r"\begin{tabular}{lr}",
        r"\hline",
        r"Metryka & Wartość \\",
        r"\hline",
        f"Loss & {metrics.loss:.4f} \\\\",
        f"Accuracy & {metrics.accuracy:.4f} \\\\",
        f"Macro sensitivity & {metrics.macro_sensitivity:.4f} \\\\",
        f"Macro specificity & {metrics.macro_specificity:.4f} \\\\",
        r"\hline",
        r"\end{tabular}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_confusion_matrix_tex(
    metrics: ClassificationMetrics,
    class_labels: list[int],
    output_path: Path,
) -> None:
    """Save confusion matrix as a LaTeX table."""
    _ensure_directory(output_path.parent)

    matrix = metrics.confusion_matrix
    column_spec = "l" + "r" * len(class_labels)

    lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        "Klasa rzeczywista / predykcja & "
        + " & ".join(str(label) for label in class_labels)
        + r" \\",
        r"\hline",
    ]

    for label, row in zip(class_labels, matrix):
        row_values = " & ".join(str(value) for value in row.tolist())
        lines.append(f"{label} & {row_values} \\\\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_sensitivity_specificity_tex(
    metrics: ClassificationMetrics,
    class_labels: list[int],
    output_path: Path,
) -> None:
    """Save sensitivity and specificity values as a LaTeX table."""
    _ensure_directory(output_path.parent)

    lines = [
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Klasa & Sensitivity & Specificity \\",
        r"\hline",
    ]

    for label, sensitivity, specificity in zip(
        class_labels,
        metrics.sensitivity_per_class,
        metrics.specificity_per_class,
    ):
        lines.append(
            f"{label} & {float(sensitivity):.4f} & {float(specificity):.4f} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            (
                "Macro avg"
                f" & {metrics.macro_sensitivity:.4f}"
                f" & {metrics.macro_specificity:.4f}"
                r" \\"
            ),
            r"\hline",
            r"\end{tabular}",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_training_report_artifacts(
    history: list[EpochStats],
    output_dir: Path,
    class_labels: list[int],
) -> None:
    """Generate only artifacts required for the report."""
    if not history:
        raise ValueError("history cannot be empty")

    last_epoch = history[-1]

    if last_epoch.test_metrics is None:
        raise ValueError("test_metrics are required to generate report artifacts")

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    test_metrics = last_epoch.test_metrics

    plot_loss_history(
        history=history,
        output_path=figures_dir / "loss_history.png",
    )

    plot_confusion_matrix(
        metrics=test_metrics,
        class_labels=class_labels,
        output_path=figures_dir / "test_confusion_matrix.png",
    )

    save_final_test_metrics_tex(
        metrics=test_metrics,
        output_path=tables_dir / "final_test_metrics.tex",
    )

    save_confusion_matrix_tex(
        metrics=test_metrics,
        class_labels=class_labels,
        output_path=tables_dir / "test_confusion_matrix.tex",
    )

    save_sensitivity_specificity_tex(
        metrics=test_metrics,
        class_labels=class_labels,
        output_path=tables_dir / "test_sensitivity_specificity.tex",
    )