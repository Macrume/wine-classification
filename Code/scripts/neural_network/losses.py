from __future__ import annotations

import numpy as np

from utils.types import FloatArray, IntArray

def softmax(logits: FloatArray) -> FloatArray:
    """Compute row-wise softmax probabilities from logits.

    Args:
        logits: Model outputs, with shape ``(batch_size, num_classes)``.

    Returns:
        Probability matrix with shape ``(batch_size, num_classes)``. Each row sums to 1.
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


class MeanSquaredError:
    """Compute mean squared error loss and its derivative.

    Methods:
        loss: Compute the mean squared error.
        derivative: Compute the derivative of the loss with respect to
            predictions.
    """
    def loss(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Compute mean squared error.

        Args:
            y_pred: Predicted values.
            y_true: Target values with the same shape as ``y_pred``.

        Returns:
            Mean squared error value.
        """
        return float(np.mean((y_pred - y_true) ** 2))

    def derivative(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        """Compute the derivative of mean squared error.

        Args:
            y_pred: Predicted values.
            y_true: Target values with the same shape as ``y_pred``.

        Returns:
            Gradient of the loss with respect to ``y_pred``.
        """
        return 2 * (y_pred - y_true) / y_true.size
    
class CrossEntropyLoss:
    """Compute softmax cross-entropy loss and its derivative.

    The input ``logits`` should be raw model outputs, not probabilities.
    Softmax is applied internally.

    Methods:
        loss: Compute the average cross-entropy loss.
        derivative: Compute the derivative with respect to logits.
    """
    def loss(self, logits: FloatArray, y_true: IntArray) -> float:
        """Compute average softmax cross-entropy loss.

        Args:
            logits: Raw model outputs with shape ``(batch_size, num_classes)``.
            y_true: Integer class labels with shape ``(batch_size)``.

        Returns:
            Average cross-entropy loss over the batch.
        """
        probabilities = softmax(logits)

        batch_size = logits.shape[0]

        correct_class_probabilities = probabilities[
            np.arange(batch_size),
            y_true,
        ]

        epsilon = 1e-15
        correct_class_probabilities = np.clip(
            correct_class_probabilities,
            epsilon,
            1.0,
        )

        return float(-np.mean(np.log(correct_class_probabilities)))

    def derivative(self, logits: FloatArray, y_true: IntArray) -> FloatArray:
        """Compute the derivative of softmax cross-entropy.

        The returned gradient is averaged over the batch.

        Args:
            logits: Raw model outputs with shape ``(batch_size, num_classes)``.
            y_true: Integer class labels with shape ``(batch_size)``.

        Returns:
            Gradient of the loss with respect to ``logits``, with shape ``(batch_size, num_classes)``.
        """
        probabilities = softmax(logits)

        batch_size = logits.shape[0]

        probabilities[np.arange(batch_size), y_true] -= 1
        probabilities /= batch_size

        return probabilities