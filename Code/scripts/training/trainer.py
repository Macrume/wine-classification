from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from data.batch_loader import BatchLoader
from neural_network.metrics import ClassificationMetrics, classification_metrics_from_logits
from neural_network.mlp import MultiLayerPerceptron
from utils.types import FloatArray, IntArray


class LossFunction(Protocol):
    """Protocol for loss functions used during model training.

    A compatible loss function must provide methods for computing the scalar
    loss value and the gradient of the loss with respect to model outputs.
    """
    def loss(self, logits: FloatArray, y_true: IntArray) -> float:
        ...

    def derivative(self, logits: FloatArray, y_true: IntArray) -> FloatArray:
        ...


class Optimizer(Protocol):
    """Protocol for optimizers used to update model parameters."""
    def step(self, model: MultiLayerPerceptron) -> None:
        ...


@dataclass
class EpochStats:
    """Store metrics collected after one training epoch.

    Attributes:
        epoch: Epoch number.
        train_metrics: Metrics computed on the training set.
        val_metrics: Metrics computed on the validation set, if provided.
        test_metrics: Metrics computed on the test set, if provided.
    """
    epoch: int

    train_metrics: ClassificationMetrics
    val_metrics: ClassificationMetrics | None = None
    test_metrics: ClassificationMetrics | None = None


class Trainer:
    """Train and evaluate a multilayer perceptron model.

    The trainer combines a model, loss function, optimizer and data loaders
    into a complete training loop. It also stores training history.

    Attributes:
        model: Model trained by the trainer.
        loss_fn: Loss function used during training and evaluation.
        optimizer: Optimizer used to update model parameters.
        history: List of epoch-level training statistics.
    """
    def __init__(
        self,
        model: MultiLayerPerceptron,
        loss_fn: LossFunction,
        optimizer: Optimizer,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train.
            loss_fn: Loss function used to compute loss and gradients.
            optimizer: Optimizer used to update model parameters.
        """
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
        train_eval_loader: BatchLoader | None = None,
        print_every: int = 1,
        eval_every: int = 1,
    ) -> list[EpochStats]:
        """Train the model for a given number of epochs.

        Args:
            train_loader: Mini-batch loader for the training set.
            epochs: Number of training epochs.
            val_loader: Optional mini-batch loader for the validation set.
            test_loader: Optional mini-batch loader for the test set.
            print_every: Print training statistics every given number of epochs.
            eval_every: Evaluate model every given number of epochs.

        Returns:
            List of ``EpochStats`` objects collected during training.

        Raises:
            ValueError: If ``epochs`` is not positive.
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        if eval_every <= 0:
            raise ValueError(f"eval_every must be positive, got {eval_every}")

        self.history = []

        for epoch in range(1, epochs + 1):
            self._train_one_epoch(train_loader)

            should_evaluate = (
                epoch == 1
                or epoch % eval_every == 0
                or epoch == epochs
            )

            if not should_evaluate:
                continue

            eval_train_loader = train_eval_loader
            if eval_train_loader is None:
                eval_train_loader = train_loader

            train_metrics = self.evaluate(eval_train_loader)

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

            if print_every > 0 and epoch % print_every == 0:
                self._print_stats(stats)

        return self.history

    def _train_one_epoch(self, train_loader: BatchLoader) -> ClassificationMetrics:
        """Run one training epoch.

        For each mini-batch, the method performs a forward pass, computes loss,
        runs backpropagation and updates model parameters. Metrics are computed
        once at the end of the epoch using predictions collected from all
        mini-batches.

        Args:
            train_loader: Mini-batch loader for the training set.

        Returns:
            Classification metrics computed over the whole training epoch.
        """
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
        """Evaluate the model on a dataset.

        The method performs only forward passes. It does not compute gradients
        and does not update model parameters.

        Args:
            loader: Mini-batch loader for the evaluated dataset.

        Returns:
            Classification metrics computed over the whole dataset.
        """
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
        """Format classification metrics for printing.

        Args:
            name: Prefix used to identify the dataset split, for example
                ``"train"``, ``"val"`` or ``"test"``.
            metrics: Metrics to format.

        Returns:
            Formatted string containing loss, accuracy, sensitivity and
            specificity.
        """
        return (
            f"{name}_loss={metrics.loss:.4f} | "
            f"{name}_acc={metrics.accuracy:.4f} | "
            f"{name}_sens={metrics.macro_sensitivity:.4f} | "
            f"{name}_spec={metrics.macro_specificity:.4f}"
        )

    def _print_stats(self, stats: EpochStats) -> None:
        """Print epoch statistics.

        Args:
            stats: Epoch statistics to print.
        """
        parts = [
            f"epoch={stats.epoch:04d}",
            self._format_metrics("train", stats.train_metrics),
        ]

        if stats.val_metrics is not None:
            parts.append(self._format_metrics("val", stats.val_metrics))

        if stats.test_metrics is not None:
            parts.append(self._format_metrics("test", stats.test_metrics))

        print(" | ".join(parts))