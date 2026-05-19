from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from data.dataset import Dataset, DatasetSplit
from data.preprocessing import map_labels_to_zero_based, standardize
from data.splitting import stratified_split_indices
from utils.types import FloatArray, IntArray


class TabularDataLoader:
    """Load, store, split, and preprocess tabular classification data.
    
    The loader stores a feature matrix and a label vector. It can be created
    directly from NumPy arrays or from a CSV file using ``from_csv``. The class
    supports stratified train/test or train/validation/test splitting and
    optional feature standardization based on the training split.
    
    Args:
        x: Input feature matrix with shape ``(num_samples, num_features)``.
        y: Label vector with shape ``(num_samples)``.
        feature_names: Names of input features.
        class_labels: Original class labels.

    Raises:
        ValueError: If ``x`` is not 2D, if ``y`` is not 1D, if ``x`` and ``y``
            contain different numbers of samples, or if the number of feature
            names does not match the number of features."""

    def __init__(
        self,
        x: FloatArray,
        y: IntArray,
        feature_names: list[str] | None = None,
        class_labels: list[int] | None = None,
    ) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {x.shape}")

        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have the same number of samples, "
                f"got {x.shape[0]} and {y.shape[0]}"
            )

        if feature_names is not None and len(feature_names) != x.shape[1]:
            raise ValueError(
                f"Expected {x.shape[1]} feature names, got {len(feature_names)}"
            )

        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.class_labels = class_labels

    @classmethod
    def from_csv(
        cls,
        data_path: str | Path,
        target_column: int | str = 0,
        feature_columns: Sequence[int | str] | None = None,
        header: int | None = None,
        feature_names: list[str] | None = None,
        map_labels: bool = True,
    ) -> TabularDataLoader:
        """Create a loader from a CSV file.
        
        The target column is extracted as the label vector, and selected feature
        columns are extracted as the input matrix. If ``feature_columns`` is not
        provided, all columns except the target column are used as features.
        
        Args:
            data_path: Path to the CSV file.
            target_column: Column containing class labels.
            feature_columns: Columns used as input features. If None, all
                columns except ``target_column`` are used.
            header: Row number used as the CSV header. Use None for files
                without a header.
            feature_names: Optional feature names.
            map_labels: Whether to map labels to the range``[0, num_classes - 1]``.
        
        Returns:
            A ``TabularDataLoader`` instance containing features, labels, and
            optional metadata.
                """
        df = pd.read_csv(data_path, header=header)

        y_raw = df[target_column].to_numpy(dtype=np.int64)

        if feature_columns is None:
            feature_columns = [
                column for column in df.columns
                if column != target_column
            ]

        x = df[list(feature_columns)].to_numpy(dtype=np.float64)

        if feature_names is None and header is not None:
            feature_names = [str(column) for column in feature_columns]

        if map_labels:
            y, class_labels = map_labels_to_zero_based(y_raw)
        else:
            y = y_raw
            class_labels = sorted(np.unique(y).tolist())

        return cls(
            x=x,
            y=y,
            feature_names=feature_names,
            class_labels=class_labels,
        )

    def split(
        self,
        train_size: float = 0.8,
        test_size: float = 0.2,
        val_size: float = 0.0,
        standardize_features: bool = True,
        random_state: int = 42,
    ) -> DatasetSplit:
        """Split the dataset into train, test, and optional validation sets.
        
        The split is stratified, so class proportions are approximately
        preserved in each subset. If ``standardize_features`` is enabled, the
        mean and standard deviation are computed only from the training set and
        then applied to the remaining subsets.
        
        Args:
            train_size: Proportion of samples assigned to the training set.
            test_size: Proportion of samples assigned to the test set.
            val_size: Proportion of samples assigned to the validation set.
                Use 0.0 to disable validation split.
            standardize_features: Whether to standardize input features.
            random_state: Seed used by the random number generator.

        Returns:
            A ``DatasetSplit`` object containing train, test, and optional
            validation datasets.

        Raises:
            ValueError: If split sizes are outside valid ranges or do not sum
                to 1.0."""
        self._validate_split_sizes(
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
        )

        if np.isclose(val_size, 0.0):
            return self._split_train_test(
                train_size=train_size,
                test_size=test_size,
                standardize_features=standardize_features,
                random_state=random_state,
            )

        return self._split_train_val_test(
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            standardize_features=standardize_features,
            random_state=random_state,
        )

    def _split_train_test(
        self,
        train_size: float,
        test_size: float,
        standardize_features: bool,
        random_state: int,
    ) -> DatasetSplit:
        """Create a stratified train/test split.
        
        Args:
            train_size: Proportion of samples assigned to the training set.
            test_size: Proportion of samples assigned to the test set.
            standardize_features: Whether to standardize input features.
            random_state: Seed used by the random number generator.

        Returns:
            A ``DatasetSplit`` object containing training and test datasets."""
        train_indices, test_indices = stratified_split_indices(
            y=self.y,
            first_ratio=train_size,
            second_ratio=test_size,
            random_state=random_state,
        )

        x_train = self.x[train_indices]
        y_train = self.y[train_indices]

        x_test = self.x[test_indices]
        y_test = self.y[test_indices]

        train_mean = None
        train_std = None

        if standardize_features:
            x_train, x_test, _, train_mean, train_std = standardize(
                x_train=x_train,
                x_test=x_test,
                x_val=None,
            )

        return DatasetSplit(
            train=Dataset(
                x=x_train,
                y=y_train,
                feature_names=self.feature_names,
                class_labels=self.class_labels,
            ),
            test=Dataset(
                x=x_test,
                y=y_test,
                feature_names=self.feature_names,
                class_labels=self.class_labels,
            ),
            val=None,
            train_mean=train_mean,
            train_std=train_std,
        )

    def _split_train_val_test(
        self,
        train_size: float,
        test_size: float,
        val_size: float,
        standardize_features: bool,
        random_state: int,
    ) -> DatasetSplit:
        """Create a stratified train/validation/test split.

        The training split is created first. The remaining samples are then
        split into validation and test subsets.

        Args:
            train_size: Proportion of samples assigned to the training set.
            test_size: Proportion of samples assigned to the test set.
            val_size: Proportion of samples assigned to the validation set.
            standardize_features: Whether to standardize input features.
            random_state: Seed used by the random number generator.

        Returns:
            A ``DatasetSplit`` object containing training, validation, and test
            datasets.

        Raises:
            RuntimeError: If standardized validation data is unexpectedly None.
        """
        train_indices, temp_indices = stratified_split_indices(
            y=self.y,
            first_ratio=train_size,
            second_ratio=val_size + test_size,
            random_state=random_state,
        )

        x_train = self.x[train_indices]
        y_train = self.y[train_indices]

        x_temp = self.x[temp_indices]
        y_temp = self.y[temp_indices]

        val_indices, test_indices = stratified_split_indices(
            y=y_temp,
            first_ratio=val_size,
            second_ratio=test_size,
            random_state=random_state + 1,
        )

        x_val = x_temp[val_indices]
        y_val = y_temp[val_indices]

        x_test = x_temp[test_indices]
        y_test = y_temp[test_indices]

        train_mean = None
        train_std = None

        if standardize_features:
            x_train, x_test, x_val_std, train_mean, train_std = standardize(
            x_train=x_train,
            x_test=x_test,
            x_val=x_val,
            )

            if x_val_std is None:
                raise RuntimeError("x_val_std should not be None when x_val is provided")

            x_val = x_val_std

        return DatasetSplit(
            train=Dataset(
                x=x_train,
                y=y_train,
                feature_names=self.feature_names,
                class_labels=self.class_labels,
            ),
            val=Dataset(
                x=x_val,
                y=y_val,
                feature_names=self.feature_names,
                class_labels=self.class_labels,
            ),
            test=Dataset(
                x=x_test,
                y=y_test,
                feature_names=self.feature_names,
                class_labels=self.class_labels,
            ),
            train_mean=train_mean,
            train_std=train_std,
        )

    @staticmethod
    def _validate_split_sizes(
        train_size: float,
        test_size: float,
        val_size: float,
    ) -> None:
        """Validate train, test, and validation split proportions.

        Args:
            train_size: Proportion of samples assigned to the training set.
            test_size: Proportion of samples assigned to the test set.
            val_size: Proportion of samples assigned to the validation set.

        Raises:
            ValueError: If any split size is outside its valid range or if all
                split sizes do not sum to 1.0.
        """
        if not (0 < train_size < 1):
            raise ValueError(
                f"train_size must be in range (0, 1), got {train_size}"
            )

        if not (0 < test_size < 1):
            raise ValueError(
                f"test_size must be in range (0, 1), got {test_size}"
            )

        if not (0 <= val_size < 1):
            raise ValueError(
                f"val_size must be in range [0, 1), got {val_size}"
            )

        total = train_size + test_size + val_size

        if not np.isclose(total, 1.0):
            raise ValueError(
                f"train_size + test_size + val_size must equal 1.0, got {total}"
            )