from __future__ import annotations

import numpy as np

from utils.types import FloatArray, IntArray


def map_labels_to_zero_based(y: IntArray) -> tuple[IntArray, list[int]]:
    """ Maps labels to range ``[0, ..., num_classes - 1]``.

    Args:
        y: Label vector  with shape ``(num_samples)``.
        
    Returns:
        A tuple containing:
            - Mapped label vector with shape ``(num_samples)``.
            - Sorted list of original class labels.
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
    """Standardize input features using training-set statistics.
        
    The transformation is:
        ``x_standardized = (x - mean) / std``
        
    Features with zero standard deviation are left unchanged by replacing their
    standard deviation with ``1.0``.
    
    Args:
        x_train: Training feature matrix with shape ``(num_train_samples, num_features)``.
        x_test: Test feature matrix with shape ``(num_test_samples, num_features)``.
        x_val: Validation feature matrix with shape ``(num_val_samples, num_features)``.
        
    Returns:
        A tuple containing:
            - Standardized training feature matrix.
            - Standardized test feature matrix.
            - Standardized validation feature matrix.
            - Feature-wise means computed from ``x_train``.
            - Feature-wise standard deviations computed from ``x_train``.
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