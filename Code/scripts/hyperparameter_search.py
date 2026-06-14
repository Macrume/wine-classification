from __future__ import annotations

import csv
import os
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from data import BatchLoader, load_wine_dataset
from neural_network import MultiLayerPerceptron
from neural_network.losses import CrossEntropyLoss
from neural_network.optimizers import SGD
from training.trainer import EpochStats, Trainer


load_dotenv()

ROOT_PATH = pathlib.Path(os.environ["ROOT_PATH"])
DATASET_PATH = pathlib.Path(os.environ["DATASET_PATH"])
ARTIFACTS_PATH = pathlib.Path(os.environ["ARTIFACTS_PATH"])
FEATURE_NAMES_PATH = pathlib.Path(os.environ["FEATURE_NAMES_PATH"])

os.chdir(ROOT_PATH)


# ----------------------------- Experiment setup -----------------------------

SEEDS = list(range(10))

LEARNING_RATES = [
    0.001,
    0.01,
    0.1,
]

EPOCHS = 1000
EVAL_EVERY = 100
PRINT_EVERY = 50
BATCH_SIZE = 16


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    learning_rate: float
    layer_sizes: list[int]


def build_network_structures(
    num_features: int,
    num_classes: int,
) -> list[tuple[str, list[int]]]:
    return [
        ("small_1_hidden_4", [num_features, 4, num_classes]),
        ("small_1_hidden_8", [num_features, 8, num_classes]),
        ("medium_1_hidden_13", [num_features, 13, num_classes]),
        ("medium_2_hidden_13_8", [num_features, 13, 8, num_classes]),
        ("large_2_hidden_32_16", [num_features, 32, 16, num_classes]),
        ("large_3_hidden_64_32_16", [num_features, 64, 32, 16, num_classes]),
    ]


def make_configs(
    num_features: int,
    num_classes: int,
) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []

    for structure_name, layer_sizes in build_network_structures(
        num_features=num_features,
        num_classes=num_classes,
    ):
        for learning_rate in LEARNING_RATES:
            lr_name = str(learning_rate).replace(".", "_")

            configs.append(
                ExperimentConfig(
                    name=f"{structure_name}_lr_{lr_name}",
                    learning_rate=learning_rate,
                    layer_sizes=layer_sizes,
                )
            )

    return configs


# ----------------------------- Helpers -----------------------------


def make_loaders(dataset, seed: int):
    if dataset.x_val is None or dataset.y_val is None:
        raise RuntimeError("Dataset must contain validation split")

    train_loader = BatchLoader(
        x=dataset.x_train,
        y=dataset.y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        random_state=seed,
    )

    train_eval_loader = BatchLoader(
        x=dataset.x_train,
        y=dataset.y_train,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    val_loader = BatchLoader(
        x=dataset.x_val,
        y=dataset.y_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = BatchLoader(
        x=dataset.x_test,
        y=dataset.y_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, train_eval_loader, val_loader, test_loader


def run_single_seed(
    config: ExperimentConfig,
    seed: int,
) -> list[EpochStats]:
    dataset = load_wine_dataset(
        data_path=DATASET_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=seed,
    )

    train_loader, train_eval_loader, val_loader, test_loader = make_loaders(
        dataset=dataset,
        seed=seed,
    )

    model = MultiLayerPerceptron(
        layer_sizes=config.layer_sizes,
        random_state=seed,
    )

    trainer = Trainer(
        model=model,
        loss_fn=CrossEntropyLoss(),
        optimizer=SGD(learning_rate=config.learning_rate),
    )

    history = trainer.fit(
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        eval_every=EVAL_EVERY,
        print_every=PRINT_EVERY,
    )

    return history


def get_metrics_for_split(stats: EpochStats, split: str):
    if split == "train":
        return stats.train_metrics

    if split == "val":
        return stats.val_metrics

    if split == "test":
        return stats.test_metrics

    raise ValueError(f"Unknown split: {split}")


def metric_series(
    history: list[EpochStats],
    split: str,
    metric_name: str,
) -> np.ndarray:
    values: list[float] = []

    for stats in history:
        metrics = get_metrics_for_split(stats, split)

        if metrics is None:
            raise RuntimeError(f"Missing {split} metrics")

        values.append(float(getattr(metrics, metric_name)))

    return np.array(values, dtype=np.float64)


def save_seed_history_csv(
    history: list[EpochStats],
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "test_loss",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "train_macro_sensitivity",
            "val_macro_sensitivity",
            "test_macro_sensitivity",
            "train_macro_specificity",
            "val_macro_specificity",
            "test_macro_specificity",
        ])

        for stats in history:
            train = stats.train_metrics
            val = stats.val_metrics
            test = stats.test_metrics

            if val is None or test is None:
                raise RuntimeError("Expected validation and test metrics")

            writer.writerow([
                stats.epoch,
                train.loss,
                val.loss,
                test.loss,
                train.accuracy,
                val.accuracy,
                test.accuracy,
                train.macro_sensitivity,
                val.macro_sensitivity,
                test.macro_sensitivity,
                train.macro_specificity,
                val.macro_specificity,
                test.macro_specificity,
            ])


def final_metrics_row(
    config: ExperimentConfig,
    seed: int,
    history: list[EpochStats],
) -> dict[str, float | int | str]:
    final_stats = history[-1]

    row: dict[str, float | int | str] = {
        "config": config.name,
        "seed": seed,
        "learning_rate": config.learning_rate,
        "layer_sizes": " -> ".join(str(size) for size in config.layer_sizes),
    }

    for split in ["train", "val", "test"]:
        metrics = get_metrics_for_split(final_stats, split)

        if metrics is None:
            raise RuntimeError(f"Missing {split} metrics")

        row[f"final_{split}_loss"] = metrics.loss
        row[f"final_{split}_accuracy"] = metrics.accuracy
        row[f"final_{split}_macro_sensitivity"] = metrics.macro_sensitivity
        row[f"final_{split}_macro_specificity"] = metrics.macro_specificity

        for class_id, value in enumerate(metrics.sensitivity_per_class):
            row[f"final_{split}_sensitivity_class_{class_id}"] = float(value)

        for class_id, value in enumerate(metrics.specificity_per_class):
            row[f"final_{split}_specificity_class_{class_id}"] = float(value)

    return row


def save_rows_csv(
    rows: list[dict[str, float | int | str]],
    output_path: pathlib.Path,
) -> None:
    if not rows:
        raise ValueError("No rows to save")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    run_rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    configs = sorted(set(str(row["config"]) for row in run_rows))

    metric_columns = [
        column
        for column in run_rows[0].keys()
        if column.startswith("final_")
    ]

    aggregate: list[dict[str, float | int | str]] = []

    for config_name in configs:
        config_rows = [
            row for row in run_rows
            if row["config"] == config_name
        ]

        first = config_rows[0]

        aggregate_row: dict[str, float | int | str] = {
            "config": config_name,
            "learning_rate": first["learning_rate"],
            "layer_sizes": first["layer_sizes"],
            "num_seeds": len(config_rows),
        }

        for column in metric_columns:
            values = np.array(
                [float(row[column]) for row in config_rows],
                dtype=np.float64,
            )

            aggregate_row[f"{column}_mean"] = float(np.mean(values))
            aggregate_row[f"{column}_std"] = float(np.std(values, ddof=1))

        aggregate.append(aggregate_row)

    aggregate.sort(
        key=lambda row: float(row["final_val_macro_sensitivity_mean"]),
        reverse=True,
    )

    return aggregate


def plot_average_loss_curve(
    histories: list[list[EpochStats]],
    output_path: pathlib.Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([stats.epoch for stats in histories[0]], dtype=np.int64)

    plt.figure(figsize=(10, 6))

    for split in ["train", "val", "test"]:
        all_values = np.vstack([
            metric_series(history, split=split, metric_name="loss")
            for history in histories
        ])

        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0, ddof=1)

        plt.plot(epochs, mean_values, label=f"{split} mean")
        plt.fill_between(
            epochs,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
        )

    plt.xlabel("Epoka")
    plt.ylabel("Funkcja kosztu")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ----------------------------- Main -----------------------------


def main() -> None:
    output_dir = ARTIFACTS_PATH / "grid_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_dataset = load_wine_dataset(
        data_path=DATASET_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        random_state=SEEDS[0],
    )

    configs = make_configs(
        num_features=reference_dataset.num_features,
        num_classes=reference_dataset.num_classes,
    )

    print(f"Number of configurations: {len(configs)}")
    print(f"Number of seeds per configuration: {len(SEEDS)}")
    print(f"Total runs: {len(configs) * len(SEEDS)}")

    all_run_rows: list[dict[str, float | int | str]] = []
    histories_by_config: dict[str, list[list[EpochStats]]] = {}

    for config in configs:
        print()
        print("=" * 80)
        print(f"Config: {config.name}")
        print(f"Layer sizes: {config.layer_sizes}")
        print(f"Learning rate: {config.learning_rate}")

        config_histories: list[list[EpochStats]] = []

        for seed in SEEDS:
            print()
            print(f"Running seed={seed}")

            history = run_single_seed(
                config=config,
                seed=seed,
            )

            config_histories.append(history)

            seed_dir = output_dir / config.name / f"seed_{seed}"
            save_seed_history_csv(
                history=history,
                output_path=seed_dir / "history.csv",
            )

            all_run_rows.append(
                final_metrics_row(
                    config=config,
                    seed=seed,
                    history=history,
                )
            )

        histories_by_config[config.name] = config_histories

        plot_average_loss_curve(
            histories=config_histories,
            output_path=output_dir / config.name / "loss_mean_over_seeds.png",
            title=(
                f"{config.name}\n"
                f"lr={config.learning_rate}, seeds={len(SEEDS)}"
            ),
        )

    save_rows_csv(
        rows=all_run_rows,
        output_path=output_dir / "all_seed_results.csv",
    )

    summary_rows = aggregate_rows(all_run_rows)

    save_rows_csv(
        rows=summary_rows,
        output_path=output_dir / "summary_mean_std.csv",
    )

    best = summary_rows[0]

    print()
    print("=" * 80)
    print("Best configuration by mean validation macro sensitivity:")
    print(f"config: {best['config']}")
    print(f"learning_rate: {best['learning_rate']}")
    print(f"layer_sizes: {best['layer_sizes']}")
    print(
        "val_macro_sensitivity: "
        f"{best['final_val_macro_sensitivity_mean']:.6f} "
        f"+/- {best['final_val_macro_sensitivity_std']:.6f}"
    )
    print(
        "test_macro_sensitivity: "
        f"{best['final_test_macro_sensitivity_mean']:.6f} "
        f"+/- {best['final_test_macro_sensitivity_std']:.6f}"
    )
    print(
        "test_macro_specificity: "
        f"{best['final_test_macro_specificity_mean']:.6f} "
        f"+/- {best['final_test_macro_specificity_std']:.6f}"
    )

    print()
    print("Artifacts saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()