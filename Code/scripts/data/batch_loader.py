from __future__ import annotations

from typing import Iterator

import numpy as np

from utils.types import FloatArray, IntArray


class BatchLoader:
    """Iterate over mini-batches of features and labels.
    
    The loader stores input samples and labels, optionaly shuffles samples
    indeces at the begining of each iteration, and yields subsequent batches
    of size ``batch_size``.
    
    ## Attributes:
        x: input feature matrix with shape ``(num_samples, num_features)``.
        y: Label vector  with shape ``(num_samples)``.
        batch_size: Number of samples returned in each batch.
        shuffle: Whethere to shuffle samples befor each itteration or nit.
        random_state: Seed used to generate random number generator.
    
    ## Raises:
        ValueError if ``batch_size`` is smaller than 1, if ``x`` is not 2D,
        if ``y`` is not 1D, or ``x`` and ``y`` contains different numbers
        of samples.
    """
    
    def __init__(
        self,
        x: FloatArray,
        y: IntArray,
        batch_size: int = 16,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have the same number of samples, "
                f"got {x.shape[0]} and {y.shape[0]}"
            )

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(random_state)

    def __iter__(self) -> Iterator[tuple[FloatArray, IntArray]]:
        """Yields mini-batches of features and labels."""
        indices = np.arange(self.x.shape[0])

        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield self.x[batch_indices], self.y[batch_indices]

    def __len__(self) -> int:
        """Return the number of mini-batches per full iteration."""
        return int(np.ceil(self.x.shape[0] / self.batch_size))