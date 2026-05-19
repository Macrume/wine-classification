from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.types import FloatArray, IntArray


@dataclass
class ClassificationMetrics:
    """Store classification metrics for one evaluation pass.

    Attributes:
        loss: Average loss value.
        accuracy: Classification accuracy.
        sensitivity_per_class: Sensitivity values computed separately for each
            class using the one-vs-rest approach.
        specificity_per_class: Specificity values computed separately for each
            class using the one-vs-rest approach.
        macro_sensitivity: Mean sensitivity across all classes.
        macro_specificity: Mean specificity across all classes.
        confusion_matrix: Confusion matrix with shape
            ``(num_classes, num_classes)``. Rows represent true classes and
            columns represent predicted classes.
    """

    loss: float
    accuracy: float

    sensitivity_per_class: FloatArray
    specificity_per_class: FloatArray

    macro_sensitivity: float
    macro_specificity: float

    confusion_matrix: IntArray


def predict_classes_from_logits(logits: FloatArray) -> IntArray:
    """Return predicted class indices from raw model outputs."""
    return np.argmax(logits, axis=1).astype(np.int64)


def accuracy(y_pred: IntArray, y_true: IntArray) -> float:
    """Compute classification accuracy."""
    return float(np.mean(y_pred == y_true))


def accuracy_from_logits(logits: FloatArray, y_true: IntArray) -> float:
    """Compute classification accuracy from logits. """
    y_pred = predict_classes_from_logits(logits)
    return accuracy(y_pred, y_true)


def count_correct_from_logits(logits: FloatArray, y_true: IntArray) -> int:
    """Count correctly classified samples from logits."""
    y_pred = predict_classes_from_logits(logits)
    return int(np.sum(y_pred == y_true))


def confusion_matrix(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> IntArray:
    """Compute a multiclass confusion matrix.

    Rows represent true classes and columns represent predicted classes.

    Args:
        y_pred: Predicted class labels with shape ``(num_samples)``.
        y_true: True class labels with shape ``(num_samples)``.
        num_classes: Number of classes.

    Returns:
        Confusion matrix with shape ``(num_classes, num_classes)``.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_class, predicted_class in zip(y_true, y_pred):
        matrix[true_class, predicted_class] += 1

    return matrix


def sensitivity_per_class(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> FloatArray:
    """Compute one-vs-rest sensitivity for each class.

    ``sensitivity = TP / (TP + FN)``

    Args:
        y_pred: Predicted class labels with shape ``(num_samples)``.
        y_true: True class labels with shape ``(num_samples)``.
        num_classes: Number of classes.

    Returns:
        Sensitivity vector with shape ``(num_classes)``.
    """
    sensitivities = np.zeros(num_classes, dtype=np.float64)

    for class_id in range(num_classes):
        true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
        false_negative = np.sum((y_true == class_id) & (y_pred != class_id))

        denominator = true_positive + false_negative

        if denominator == 0:
            sensitivities[class_id] = 0.0
        else:
            sensitivities[class_id] = true_positive / denominator

    return sensitivities


def specificity_per_class(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> FloatArray:
    """Compute one-vs-rest specificity for each class.

    ``specificity = TN / (TN + FP)``

    Args:
        y_pred: Predicted class labels with shape ``(num_samples)``.
        y_true: True class labels with shape ``(num_samples)``.
        num_classes: Number of classes.

    Returns:
        Specificity vector with shape ``(num_classes)``.
    """
    specificities = np.zeros(num_classes, dtype=np.float64)

    for class_id in range(num_classes):
        true_negative = np.sum((y_true != class_id) & (y_pred != class_id))
        false_positive = np.sum((y_true != class_id) & (y_pred == class_id))

        denominator = true_negative + false_positive

        if denominator == 0:
            specificities[class_id] = 0.0
        else:
            specificities[class_id] = true_negative / denominator

    return specificities


def macro_sensitivity(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> float:
    """Compute mean one-vs-rest sensitivity across all classes.

    Args:
        y_pred: Predicted class labels with shape ``(num_samples)``.
        y_true: True class labels with shape ``(num_samples)``.
        num_classes: Number of classes.

    Returns:
        Mean sensitivity across classes.
    """
    values = sensitivity_per_class(y_pred, y_true, num_classes)
    return float(np.mean(values))


def macro_specificity(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> float:
    """Compute mean one-vs-rest specificity across all classes.

    Args:
        y_pred: Predicted class labels with shape ``(num_samples)``.
        y_true: True class labels with shape ``(num_samples)``.
        num_classes: Number of classes.

    Returns:
        Mean specificity across classes.
    """
    values = specificity_per_class(y_pred, y_true, num_classes)
    return float(np.mean(values))


def classification_metrics_from_logits(
    logits: FloatArray,
    y_true: IntArray,
    loss: float,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute classification metrics from logits and true labels.

    Args:
        logits: Raw model outputs with shape ``(num_samples, num_classes)``.
        y_true: True class labels with shape ``(num_samples)``.
        loss: Average loss value associated with the provided predictions.
        num_classes: Number of classes.

    Returns:
        A ``ClassificationMetrics`` instance containing loss, accuracy,
        sensitivity, specificity, and confusion matrix.
    """
    y_pred = predict_classes_from_logits(logits)

    sensitivities = sensitivity_per_class(
        y_pred=y_pred,
        y_true=y_true,
        num_classes=num_classes,
    )

    specificities = specificity_per_class(
        y_pred=y_pred,
        y_true=y_true,
        num_classes=num_classes,
    )

    matrix = confusion_matrix(
        y_pred=y_pred,
        y_true=y_true,
        num_classes=num_classes,
    )

    return ClassificationMetrics(
        loss=loss,
        accuracy=accuracy(y_pred, y_true),
        sensitivity_per_class=sensitivities,
        specificity_per_class=specificities,
        macro_sensitivity=float(np.mean(sensitivities)),
        macro_specificity=float(np.mean(specificities)),
        confusion_matrix=matrix,
    )