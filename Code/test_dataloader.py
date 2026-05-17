import os
from pathlib import Path

from dotenv import load_dotenv

from Code.scripts.data.dataloader import BatchLoader, load_wine_dataset


def main() -> None:
    # Ścieżka do głównego katalogu projektu
    project_root = Path(__file__).resolve().parents[1]

    load_dotenv(project_root / ".env")

    data_path_str = os.getenv("WINE_DATA_PATH")
    if data_path_str is None:
        raise ValueError("Environment variable WINE_DATA_PATH is not set.")

    data_path = Path(data_path_str)

    # Wczytanie i przygotowanie danych
    dataset = load_wine_dataset(
        data_path=data_path,
        train_size=0.7,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    # Podstawowe informacje
    print("=== Dataset info ===")
    print("x_train shape:", dataset.x_train.shape)
    print("y_train shape:", dataset.y_train.shape)
    if dataset.x_val is not None and dataset.y_val is not None:
        print("x_val shape:", dataset.x_val.shape)
        print("y_val shape:", dataset.y_val.shape)
    print("x_test shape:", dataset.x_test.shape)
    print("y_test shape:", dataset.y_test.shape)
    print("feature names:", dataset.feature_names)
    print("class labels:", dataset.class_labels)
    print()

    # Test działania loadera
    train_loader = BatchLoader(
        dataset.x_train,
        dataset.y_train,
        batch_size=16,
        shuffle=True,
        random_state=42,
    )

    print("=== First batch ===")
    for batch_x, batch_y in train_loader:
        print("batch_x shape:", batch_x.shape)
        print("batch_y shape:", batch_y.shape)
        print("batch_y:", batch_y)
        break

    # Loader walidacyjny, jeśli istnieje
    if dataset.x_val is not None and dataset.y_val is not None:
        val_loader = BatchLoader(
            dataset.x_val,
            dataset.y_val,
            batch_size=16,
            shuffle=False,
            random_state=42,
        )
        print()
        print("=== First validation batch ===")
        for batch_x, batch_y in val_loader:
            print("val batch_x shape:", batch_x.shape)
            print("val batch_y shape:", batch_y.shape)
            print("val batch_y:", batch_y)
            break

if __name__ == "__main__":
    main()