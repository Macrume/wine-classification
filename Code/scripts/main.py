from __future__ import annotations

import pathlib
import os
from dotenv import load_dotenv; load_dotenv()

import numpy as np

from data import BatchLoader, load_wine_dataset
from neural_network import MultiLayerPerceptron


# ---------------------- Enviomental varibles ----------------------
ENV_VARIBLES        : os._Environ[str]  = os.environ

# ----------------------------- Paths ------------------------------
ROOT_PATH           : pathlib.Path = pathlib.Path(ENV_VARIBLES['ROOT_PATH'])
DATASET_PATH        : pathlib.Path = pathlib.Path(ENV_VARIBLES['DATASET_PATH'])
ARTIFACTS_PATH      : pathlib.Path = pathlib.Path(ENV_VARIBLES['ARTIFACTS_PATH'])
FEATURE_NAMES_PATH  : pathlib.Path = pathlib.Path(ENV_VARIBLES['FEATURE_NAMES_PATH'])

# Change current directory to project root
os.chdir(ROOT_PATH)

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y]

def print_dataset_info(dataset) -> None:
    print("=== Dataset info ===")
    print(f"x_train shape: {dataset.x_train.shape}")
    print(f"y_train shape: {dataset.y_train.shape}")

    if dataset.x_val is not None:
        print(f"x_val shape:   {dataset.x_val.shape}")
        print(f"y_val shape:   {dataset.y_val.shape}")

    print(f"x_test shape:  {dataset.x_test.shape}")
    print(f"y_test shape:  {dataset.y_test.shape}")

    print()
    print(f"num_features: {dataset.num_features}")
    print(f"num_classes:  {dataset.num_classes}")
    print(f"class_labels: {dataset.train.class_labels}")

    print()
    print("Train feature mean after standardization:")
    print(np.round(np.mean(dataset.x_train, axis=0), 4))

    print()
    print("Train feature std after standardization:")
    print(np.round(np.std(dataset.x_train, axis=0), 4))


def print_class_distribution(name: str, y: np.ndarray) -> None:
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))

    print(f"{name} class distribution: {distribution}")


def main() -> None:
    dataset = load_wine_dataset(
        data_path=DATASET_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    print_dataset_info(dataset)

    print()
    print("=== Class distributions ===")
    print_class_distribution("train", dataset.y_train)

    if dataset.y_val is not None:
        print_class_distribution("val", dataset.y_val)

    print_class_distribution("test", dataset.y_test)

    train_loader = BatchLoader(
        x=dataset.x_train,
        y=dataset.y_train,
        batch_size=16,
        shuffle=True,
        random_state=42,
    )

    print()
    print("=== BatchLoader test ===")
    print(f"number of batches: {len(train_loader)}")

    batch_x, batch_y = next(iter(train_loader))

    print(f"batch_x shape: {batch_x.shape}")
    print(f"batch_y shape: {batch_y.shape}")
    print(f"batch_y: {batch_y}")

    model = MultiLayerPerceptron(
        layer_sizes=[
            dataset.num_features,
            16,
            8,
            dataset.num_classes,
        ],
    )

    print()
    print("=== Model forward test ===")

    y_pred = model(batch_x)

    print(f"input shape:  {batch_x.shape}")
    print(f"output shape: {y_pred.shape}")

    print()
    print("First 5 predictions:")
    print(np.round(y_pred[:5], 4))

    print()
    print("Expected output shape:")
    print(f"({batch_x.shape[0]}, {dataset.num_classes})")


if __name__ == "__main__":
    main()