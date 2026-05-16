from __future__ import annotations

"""
Moduł odpowiedzialny za wczytywanie i przygotowanie danych dla projektu
klasyfikacji wina z wykorzystaniem sieci MLP
Zakres:
- wczytanie zbioru Wine z pliku CSV
- wydzielenie cech wejściowych i etykiet klas
- mapowanie klas z zakresu {1, 2, 3} do {0, 1, 2}
- stratyfikowany podział na zbiór treningowy i testowy
- standaryzacja cech na podstawie danych treningowych
- generowanie mini-batchy do procesu uczenia
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
        x_test: Zestandaryzowane cechy zbioru testowego
        y_test: Etykiety klas dla zbioru testowego
        feature_names: Lista nazw cech wejściowych
        class_labels: Oryginalne etykiety klas występujące w zbiorze
        train_mean: Średnie cech wyznaczone na zbiorze treningowym
        train_std: Odchylenia standardowe cech wyznaczone na zbiorze treningowym
    """
    x_train: FloatArray
    y_train: IntArray
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


def _stratified_split(
    x: FloatArray,
    y: IntArray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    """
    Wykonuje stratyfikowany podział danych na zbiór treningowy i testowy
    Stratyfikacja - zachowanie proporcji klas w obu częściach zbioru
    Args:
        x: Macierz cech wejściowych
        y: Wektor etykiet klas
        test_size: Udział danych testowych w całym zbiorze
        random_state: Ziarno generatora liczb losowych
    Returns:
        Krotka zawierająca:
        - cechy treningowe
        - etykiety treningowe
        - cechy testowe
        - etykiety testowe
    """
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    test_indices: list[int] = []

    # Dla każdej klasy losujemy osobno próbki do train i test
    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        rng.shuffle(class_indices)
        test_count = max(1, int(round(len(class_indices) * test_size)))
        test_indices.extend(class_indices[:test_count].tolist())
        train_indices.extend(class_indices[test_count:].tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def _standardize(
    x_train: FloatArray,
    x_test: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """
    Standaryzuje dane wejściowe na podstawie zbioru treningowego
    Standaryzacja według wzoru:
        x' = (x - mean) / std
    Parametry mean i std są wyznaczane wyłącznie na zbiorze treningowym
    stosowane również do zbioru testowego
    Args:
        x_train: Cechy zbioru treningowego
        x_test: Cechy zbioru testowego
    Returns:
        Krotka zawierająca:
        - zestandaryzowany zbiór treningowy
        - zestandaryzowany zbiór testowy
        - średnie cech dla train
        - odchylenia standardowe cech dla train
    """
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    # Zabezpieczenie przed dzieleniem przez zero
    std[std == 0.0] = 1.0
    x_train_std = (x_train - mean) / std
    x_test_std = (x_test - mean) / std

    return x_train_std, x_test_std, mean, std

def load_wine_dataset(
    data_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Wczytuje zbiór i przygotowuje go do dalszego użycia
    Funkcja:
    - wczytuje dane z pliku CSV
    - rozdziela cechy i etykiety klas
    - mapuje klasy do zakresu {0, 1, 2}
    - wykonuje stratyfikowany podział train/test
    - standaryzuje dane wejściowe
    Args:
        data_path: Ścieżka do pliku data.csv
        test_size: Udział danych testowych w całym zbiorze
        random_state: Ziarno generatora liczb losowych
    Returns:
        Obiekt DatasetSplit zawierający dane po preprocessingu
    """
    # Wczytanie danych bez nagłówków
    df = pd.read_csv(data_path, header=None)

    # Pierwsza kolumna to etykieta klasy, pozostałe to cechy
    y_raw = df.iloc[:, 0].to_numpy(dtype=np.int64)
    x_raw = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    # Mapowanie klas z {1,2,3} do {0,1,2}
    y, class_labels = _map_labels_to_zero_based(y_raw)
    # Podział danych zgodny z założeniami
    x_train, y_train, x_test, y_test = _stratified_split(
        x_raw,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    # Standaryzacja wyznaczana wyłącznie na train
    x_train, x_test, train_mean, train_std = _standardize(x_train, x_test)
    return DatasetSplit(
        x_train=x_train,
        y_train=y_train,
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
    Klasa służy do dzielenia danych treningowych na mniejsze partie,
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

        # Opcjonalne tasowanie danych przed rozpoczęciem epoki !
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