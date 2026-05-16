from __future__ import annotations

"""
Moduł odpowiedzialny za wczytywanie i przygotowanie danych dla projektu
klasyfikacji wina z wykorzystaniem sieci MLP
Zakres:
- wczytanie zbioru Wine z pliku CSV
- wydzielenie cech wejściowych i etykiet klas
- mapowanie klas z zakresu {1, 2, 3} do {0, 1, 2}
- stratyfikowany podział na zbióry train/test/validation
- standaryzacja cech na podstawie danych treningowych
- generowanie mini-batchy
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Nazwy cech zgodne ze zbiorem Wine
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


@dataclass
class DatasetSplit:
    """
    Struktura przechowująca dane po podziale i preprocessingu
    Attributes:
        x_train: Zestandaryzowane cechy zbioru treningowego
        y_train: Etykiety klas dla zbioru treningowego
        x_val: Zestandaryzowane cechy zbioru walidacyjnego
        y_val: Etykiety klas dla zbioru walidacyjnego
        x_test: Zestandaryzowane cechy zbioru testowego
        y_test: Etykiety klas dla zbioru testowego
        feature_names: Lista nazw cech wejściowych
        class_labels: Oryginalne etykiety klas występujące w zbiorze
        train_mean: Średnie cech wyznaczone na zbiorze treningowym
        train_std: Odchylenia standardowe cech wyznaczone na zbiorze treningowym
    """
    x_train: FloatArray
    y_train: IntArray
    x_val: FloatArray | None
    y_val: IntArray | None
    x_test: FloatArray
    y_test: IntArray
    feature_names: list[str]
    class_labels: list[int]
    train_mean: FloatArray
    train_std: FloatArray

def _map_labels_to_zero_based(y: IntArray) -> tuple[IntArray, list[int]]:
    """
    Mapuje etykiety klas do numeracji rozpoczynającej się od zera
    Dla zbioru Wine oznacza to zamianę klas:
    1 -> 0
    2 -> 1
    3 -> 2
    Args:
        y: Tablica z oryginalnymi etykietami klas
    Returns:
        Krotka zawierająca:
        - tablicę z przemapowanymi etykietami
        - listę oryginalnych etykiet klas
    """
    class_labels = sorted(np.unique(y).tolist())
    mapping = {label: index for index, label in enumerate(class_labels)}
    mapped = np.array([mapping[label] for label in y], dtype=np.int64)
    return mapped, class_labels


def _stratified_split_indices(
        y: IntArray,
        first_ratio: float,
        second_ratio: float,
        random_state: int,
) -> tuple[IntArray, IntArray]:
    """
    Dzieli indeksy na dwie części w sposób stratyfikowany
    Funkcja zachowuje proporcje klas w obu częściach podziału
    Args:
        y: Wektor etykiet klas
        first_ratio: Udział pierwszej części
        second_ratio: Udział drugiej części
        random_state: Ziarno generatora liczb losowych
    Returns:
        Krotka (first_indices, second_indices)
    """
    if first_ratio <= 0 or second_ratio <= 0:
        raise ValueError("Both ratios must be > 0 for stratified split")

    total = first_ratio + second_ratio
    first_share = first_ratio / total

    rng = np.random.default_rng(random_state)

    first_indices: list[int] = []
    second_indices: list[int] = []

    # Kazda klase dzielimy osobno
    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)
        first_count = int(round(len(class_indices) * first_share))

        # Zabezpieczenie, żeby obie części nie były puste dla danej klasy
        if first_count <= 0:
            first_count = 1
        if first_count >= len(class_indices):
            first_count = len(class_indices) - 1

        first_indices.extend(class_indices[:first_count].tolist())
        second_indices.extend(class_indices[first_count:].tolist())

    first_indices = np.array(first_indices, dtype=np.int64)
    second_indices = np.array(second_indices, dtype=np.int64)

    rng.shuffle(first_indices)
    rng.shuffle(second_indices)

    return first_indices, second_indices


def _standardize(
    x_train: FloatArray,
    x_test: FloatArray,
    x_val: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray | None, FloatArray, FloatArray]:
    """
    Standaryzuje dane wejściowe na podstawie zbioru treningowego
    Standaryzacja według wzoru:
        x' = (x - mean) / std
    Parametry mean i std są wyznaczane wyłącznie na zbiorze treningowym
    stosowane również do zbioru testowego i opcjonalnie walidacyjnego
    Args:
        x_train: Cechy zbioru treningowego
        x_test: Cechy zbioru testowego
        x_val: Cechy zbioru walidacyjnego
    Returns:
        Krotka zawierająca:
        - zestandaryzowany zbiór treningowy
        - zestandaryzowany zbiór testowy
        - zestandaryzowany zbiór walidacyjny
        - średnie cech dla train
        - odchylenia standardowe cech dla train
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    # Zabezpieczenie przed dzieleniem przez zero
    std[std == 0.0] = 1.0

    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    x_val_std = None
    if x_val is not None:
        x_val_std = (x_val - mean) / std

    return x_train_std, x_test_std, x_val_std, mean, std

def load_wine_dataset(
    data_path: str | Path,
    train_size: float = 0.8,
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Wczytuje zbiór i przygotowuje go do dalszego użycia
    Domyślnie podzial train/test = 80/20
    jeżeli val_size > 0, wykonywane podzial train/test/val

    Args:
        data_path: Ścieżka do pliku data.csv
        train_size: Finalna propocja zbioru treningowego
        test_size: Finalna propocja zbioru testowego
        val_size: Finalna propocja zbioru walidacyjnego
            wartość 0.0 oznacza brak zbioru
        random_state: Ziarno generatora liczb losowych
    Returns:
        Obiekt DatasetSplit zawierający dane po preprocessingu
    """
    # Sprawdzanie poprawności zakresów
    if not (0 < train_size < 1):
        raise ValueError(f"train_size must be in range (0, 1), got {train_size}")

    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be in range (0, 1), got {test_size}")

    if not (0 <= val_size < 1):
        raise ValueError(f"val_size must be in range [0, 1), got {val_size}")

    # Suma proporcji musi dawać 1
    total = train_size + test_size + val_size
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_size + test_size + val_size must equal 1.0, got {total}"
        )

    # Wczytanie danych bez nagłówków
    df = pd.read_csv(data_path, header=None)

    # Pierwsza kolumna to etykieta klasy, pozostałe to cechy
    y_raw = df.iloc[:, 0].to_numpy(dtype=np.int64)
    x_raw = df.iloc[:, 1:].to_numpy(dtype=np.float64)

    # Mapowanie klas z {1,2,3} do {0,1,2}
    y, class_labels = _map_labels_to_zero_based(y_raw)

    if np.isclose(val_size, 0.0):
        train_indices, test_indices = _stratified_split_indices(
            y=y,
            first_ratio=train_size,
            second_ratio=test_size,
            random_state=random_state,
        )

        x_train = x_raw[train_indices]
        y_train = y[train_indices]

        x_test = x_raw[test_indices]
        y_test = y[test_indices]

        # Standaryzacja liczona tylko na train
        x_train, x_test, _, train_mean, train_std = _standardize(
            x_train=x_train,
            x_test=x_test,
            x_val=None,
        )

        return DatasetSplit(
            x_train=x_train,
            y_train=y_train,
            x_val=None,
            y_val=None,
            x_test=x_test,
            y_test=y_test,
            feature_names=WINE_FEATURE_NAMES,
            class_labels=class_labels,
            train_mean=train_mean,
            train_std=train_std,
        )

    # Jeśli val_size > 0, najpierw wydzielamy train,
    # z pozostałej puli wydzielamy validation i test
    train_indices, temp_indices = _stratified_split_indices(
        y=y,
        first_ratio=train_size,
        second_ratio=val_size + test_size,
        random_state=random_state,
    )

    x_train = x_raw[train_indices]
    y_train = y[train_indices]

    x_temp = x_raw[temp_indices]
    y_temp = y[temp_indices]

    # Z puli tymczasowej wydzielamy validation i test
    val_indices_local, test_indices_local = _stratified_split_indices(
        y=y_temp,
        first_ratio=val_size,
        second_ratio=test_size,
        random_state=random_state + 1,
    )

    x_val = x_temp[val_indices_local]
    y_val = y_temp[val_indices_local]

    x_test = x_temp[test_indices_local]
    y_test = y_temp[test_indices_local]

    # Standaryzacja wyznaczana wyłącznie na train
    x_train, x_test, x_val, train_mean, train_std = _standardize(
        x_train = x_train,
        x_test = x_test,
        x_val = x_val,
    )

    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=WINE_FEATURE_NAMES,
        class_labels=class_labels,
        train_mean=train_mean,
        train_std=train_std,
    )


class BatchLoader:
    """
    Iterator zwracający mini-batche danych
    Klasa służy do dzielenia danych treningowych na mniejsze partie
    wykorzystywane podczas uczenia
    Attributes:
        x: Macierz cech wejściowych
        y: Wektor etykiet klas
        batch_size: Rozmiar pojedynczego batcha
        shuffle: Czy przetasować dane przed iteracją
        random_state: Ziarno generatora liczb losowych
    """
    def __init__(
        self,
        x: FloatArray,
        y: IntArray,
        batch_size: int = 16,
        shuffle: bool = True,
        random_state: int | None = None,
    ) -> None:
        """
        Inicjalizacja loader'a mini-batchy
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have the same number of samples, got {x.shape[0]} and {y.shape[0]}"
            )

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(random_state)

    def __iter__(self) -> Iterator[tuple[FloatArray, IntArray]]:
        """
        Zwraca kolejne mini-batche danych.
        Yields:
            Krotki postaci (batch_x, batch_y)
        """
        indices = np.arange(self.x.shape[0])

        # Opcjonalne tasowanie danych przed rozpoczęciem epoki
        if self.shuffle:
            self._rng.shuffle(indices)
        # Zwracanie kolejnych batchy
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            yield self.x[batch_indices], self.y[batch_indices]

    def __len__(self) -> int:
        """
        Zwraca liczbę batchy w jednej epoce
        Returns:
            Liczba partii danych
        """
        return int(np.ceil(self.x.shape[0] / self.batch_size))