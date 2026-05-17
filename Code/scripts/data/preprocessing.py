from __future__ import annotations

import numpy as np

from utils.types import FloatArray, IntArray


def map_labels_to_zero_based(y: IntArray) -> tuple[IntArray, list[int]]:
    """
    Mapuje etykiety klas do zakresu 0, 1, ..., n_classes - 1.

    Przykład:
        [1, 2, 3] -> [0, 1, 2]
        [2, 4, 7] -> [0, 1, 2]
    """
    class_labels = sorted(np.unique(y).tolist())
    mapping = {label: index for index, label in enumerate(class_labels)}

    mapped = np.array(
        [mapping[label] for label in y],
        dtype=np.int64,
    )

    return mapped, class_labels


def standardize(
    x_train: FloatArray,
    x_test: FloatArray,
    x_val: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray, FloatArray]:
    """
    Standaryzuje dane według:
        x' = (x - mean) / std

    Mean i std są liczone tylko na zbiorze treningowym.
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    std[std == 0.0] = 1.0

    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    x_val_std = None
    if x_val is not None:
        x_val_std = (x_val - mean) / std

    return x_train_std, x_test_std, x_val_std, mean, std