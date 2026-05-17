from __future__ import annotations

from dataclasses import dataclass

from utils.types import FloatArray, IntArray


@dataclass
class Dataset:
    x: FloatArray
    y: IntArray
    feature_names: list[str] | None = None
    class_labels: list[int] | None = None


@dataclass
class DatasetSplit:
    train: Dataset
    test: Dataset
    val: Dataset | None = None

    train_mean: FloatArray | None = None
    train_std: FloatArray | None = None

    @property
    def x_train(self) -> FloatArray:
        return self.train.x

    @property
    def y_train(self) -> IntArray:
        return self.train.y

    @property
    def x_test(self) -> FloatArray:
        return self.test.x

    @property
    def y_test(self) -> IntArray:
        return self.test.y

    @property
    def x_val(self) -> FloatArray | None:
        if self.val is None:
            return None

        return self.val.x

    @property
    def y_val(self) -> IntArray | None:
        if self.val is None:
            return None

        return self.val.y

    @property
    def num_features(self) -> int:
        return self.train.x.shape[1]

    @property
    def num_classes(self) -> int:
        if self.train.class_labels is None:
            return int(len(set(self.train.y.tolist())))

        return len(self.train.class_labels)