from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from data.batch_loader import BatchLoader
from neural_network.metrics import ClassificationMetrics, classification_metrics_from_logits
from neural_network.mlp import MultiLayerPerceptron
from utils.types import FloatArray, IntArray


class LossFunction(Protocol):
    def loss(self, logits: FloatArray, y_true: IntArray) -> float:
        ...

    def derivative(self, logits: FloatArray, y_true: IntArray) -> FloatArray:
        ...


class Optimizer(Protocol):
    def step(self, model: MultiLayerPerceptron) -> None:
        ...


@dataclass
class EpochStats:
    epoch: int

    train_metrics: ClassificationMetrics
    val_metrics: ClassificationMetrics | None = None
    test_metrics: ClassificationMetrics | None = None


class Trainer:
    def __init__(
        self,
        model: MultiLayerPerceptron,
        loss_fn: LossFunction,
        optimizer: Optimizer,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.history: list[EpochStats] = []

    def fit(
        self,
        train_loader: BatchLoader,
        epochs: int,
        val_loader: BatchLoader | None = None,
        test_loader: BatchLoader | None = None,
        print_every: int = 1,
    ) -> list[EpochStats]:
        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(train_loader)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)

            test_metrics = None
            if test_loader is not None:
                test_metrics = self.evaluate(test_loader)

            stats = EpochStats(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )

            self.history.append(stats)

            if epoch % print_every == 0:
                self._print_stats(stats)

        return self.history

    def _train_one_epoch(self, train_loader: BatchLoader) -> ClassificationMetrics:
        total_loss = 0.0
        total_samples = 0

        all_logits: list[FloatArray] = []
        all_y_true: list[IntArray] = []

        for batch_x, batch_y in train_loader:
            logits = self.model(batch_x)

            batch_loss = self.loss_fn.loss(logits, batch_y)
            loss_gradient = self.loss_fn.derivative(logits, batch_y)

            self.model.backward(loss_gradient)
            self.optimizer.step(self.model)

            batch_size = batch_x.shape[0]

            total_loss += batch_loss * batch_size
            total_samples += batch_size

            all_logits.append(logits)
            all_y_true.append(batch_y)

        average_loss = total_loss / total_samples

        logits_all = np.vstack(all_logits)
        y_true_all = np.concatenate(all_y_true)

        return classification_metrics_from_logits(
            logits=logits_all,
            y_true=y_true_all,
            loss=average_loss,
            num_classes=self.model.layer_sizes[-1],
        )

    def evaluate(self, loader: BatchLoader) -> ClassificationMetrics:
        total_loss = 0.0
        total_samples = 0

        all_logits: list[FloatArray] = []
        all_y_true: list[IntArray] = []

        for batch_x, batch_y in loader:
            logits = self.model(batch_x)

            batch_loss = self.loss_fn.loss(logits, batch_y)
            batch_size = batch_x.shape[0]

            total_loss += batch_loss * batch_size
            total_samples += batch_size

            all_logits.append(logits)
            all_y_true.append(batch_y)

        average_loss = total_loss / total_samples

        logits_all = np.vstack(all_logits)
        y_true_all = np.concatenate(all_y_true)

        return classification_metrics_from_logits(
            logits=logits_all,
            y_true=y_true_all,
            loss=average_loss,
            num_classes=self.model.layer_sizes[-1],
        )

    @staticmethod
    def _format_metrics(name: str, metrics: ClassificationMetrics) -> str:
        return (
            f"{name}_loss={metrics.loss:.4f} | "
            f"{name}_acc={metrics.accuracy:.4f} | "
            f"{name}_sens={metrics.macro_sensitivity:.4f} | "
            f"{name}_spec={metrics.macro_specificity:.4f}"
        )

    def _print_stats(self, stats: EpochStats) -> None:
        parts = [
            f"epoch={stats.epoch:04d}",
            self._format_metrics("train", stats.train_metrics),
        ]

        if stats.val_metrics is not None:
            parts.append(self._format_metrics("val", stats.val_metrics))

        if stats.test_metrics is not None:
            parts.append(self._format_metrics("test", stats.test_metrics))

        print(" | ".join(parts))