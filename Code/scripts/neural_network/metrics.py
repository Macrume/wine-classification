from __future__ import annotations
from dataclasses import dataclass


import numpy as np

from utils.types import FloatArray, IntArray

@dataclass
class ClassificationMetrics:
    loss: float
    accuracy: float

    sensitivity_per_class: FloatArray
    specificity_per_class: FloatArray

    macro_sensitivity: float
    macro_specificity: float

    confusion_matrix: IntArray

def predict_classes_from_logits(logits: FloatArray) -> IntArray:
    return np.argmax(logits, axis=1).astype(np.int64)

def accuracy(y_pred: IntArray, y_true: IntArray) -> float:
    return float(np.mean(y_pred == y_true))

def accuracy_from_logits(logits: FloatArray, y_true: IntArray) -> float:
    y_pred = predict_classes_from_logits(logits)
    return accuracy(y_pred, y_true)

def count_correct_from_logits(logits: FloatArray, y_true: IntArray) -> int:
    y_pred = np.argmax(logits, axis=1)
    return int(np.sum(y_pred == y_true))

def confusion_matrix(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> IntArray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_class, predicted_class in zip(y_true, y_pred):
        matrix[true_class, predicted_class] += 1

    return matrix

def sensitivity_per_class(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> FloatArray:
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
    values = sensitivity_per_class(y_pred, y_true, num_classes)
    return float(np.mean(values))


def macro_specificity(
    y_pred: IntArray,
    y_true: IntArray,
    num_classes: int,
) -> float:
    values = specificity_per_class(y_pred, y_true, num_classes)
    return float(np.mean(values))

def classification_metrics_from_logits(
    logits: FloatArray,
    y_true: IntArray,
    loss: float,
    num_classes: int,
) -> ClassificationMetrics:
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


