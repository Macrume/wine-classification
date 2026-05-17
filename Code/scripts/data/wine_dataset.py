from __future__ import annotations

from pathlib import Path

from data.dataset import DatasetSplit
from data.tabular_data_loader import TabularDataLoader
from utils.utils import load_feature_names


WINE_FEATURE_NAMES = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]


def load_wine_dataset(
    data_path: str | Path,
    train_size: float = 0.8,
    test_size: float = 0.2,
    val_size: float = 0.0,
    feature_names_path: str | Path | None = None,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Cienka nakładka konfiguracyjna dla Wine Dataset.

    Zakładamy format CSV:
    - kolumna 0: etykieta klasy,
    - kolumny 1-13: cechy wejściowe,
    - brak nagłówka.
    """
    feature_names = WINE_FEATURE_NAMES

    if feature_names_path is not None:
        feature_names = load_feature_names(Path(feature_names_path))
        feature_names.pop(0)

    loader = TabularDataLoader.from_csv(
        data_path=data_path,
        target_column=0,
        feature_columns=range(1, 14),
        header=None,
        feature_names=feature_names,
        map_labels=True,
    )

    return loader.split(
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        standardize_features=True,
        random_state=random_state,
    )