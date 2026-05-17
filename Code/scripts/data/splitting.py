from __future__ import annotations

import numpy as np

from utils.types import IntArray


def stratified_split_indices(
    y: IntArray,
    first_ratio: float,
    second_ratio: float,
    random_state: int,
) -> tuple[IntArray, IntArray]:
    """
    Dzieli indeksy na dwie części, zachowując proporcje klas.
    """
    if first_ratio <= 0 or second_ratio <= 0:
        raise ValueError("Both ratios must be greater than 0")

    total = first_ratio + second_ratio
    first_share = first_ratio / total

    rng = np.random.default_rng(random_state)

    first_indices: list[int] = []
    second_indices: list[int] = []

    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]

        if len(class_indices) < 2:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} sample(s), "
                "cannot perform stratified split into two non-empty parts"
            )

        rng.shuffle(class_indices)

        first_count = int(round(len(class_indices) * first_share))
        first_count = max(1, first_count)
        first_count = min(len(class_indices) - 1, first_count)

        first_indices.extend(class_indices[:first_count].tolist())
        second_indices.extend(class_indices[first_count:].tolist())

    first_indices_array = np.array(first_indices, dtype=np.int64)
    second_indices_array = np.array(second_indices, dtype=np.int64)

    rng.shuffle(first_indices_array)
    rng.shuffle(second_indices_array)

    return first_indices_array, second_indices_array