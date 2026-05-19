from __future__ import annotations
from dataclasses import dataclass
from utils.types import FloatArray, IntArray


@dataclass
class Dataset:
    """Stores features, labels and optional dataset metadata.
    
    Attributes:
        x: input feature matrix with shape ``(num_samples, num_features)``.
        y: Label vector  with shape ``(num_samples)``.
        feature_names: Names of input features.
        class_labels: Orginal class labeles befor optional remaping.
    """
    x: FloatArray
    y: IntArray
    feature_names: list[str] | None = None
    class_labels: list[int] | None = None


@dataclass
class DatasetSplit:
    """Store train, test and optional validation dataset split.
    
    The class stores optional standarization parameters computed from
    training split. Propeties expose comonly ussed arrays firectly, 
    such as ``x_train`` and ``y_train``.
    
    Attributes:
        train: Training dataset split.
        test: Test dataset split.
        val: Validation dataset split.
        train_mean: Features means computed on training split.
        train_std: Features standard deviations computed on training split.
    """
    train: Dataset
    test: Dataset
    val: Dataset | None = None

    train_mean: FloatArray | None = None
    train_std: FloatArray | None = None

    @property
    def x_train(self) -> FloatArray:
        """Return training features"""
        return self.train.x

    @property
    def y_train(self) -> IntArray:
        """Return training labels"""
        return self.train.y

    @property
    def x_test(self) -> FloatArray:
        """Return test features"""
        return self.test.x

    @property
    def y_test(self) -> IntArray:
        """Return test labels"""
        return self.test.y

    @property
    def x_val(self) -> FloatArray | None:
        """Return validation features"""
        if self.val is None:
            return None

        return self.val.x

    @property
    def y_val(self) -> IntArray | None:
        """Return validation labels"""
        if self.val is None:
            return None

        return self.val.y

    @property
    def num_features(self) -> int:
        """Return number of input features."""
        return self.train.x.shape[1]

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        if self.train.class_labels is None:
            return int(len(set(self.train.y.tolist())))

        return len(self.train.class_labels)