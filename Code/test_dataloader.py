from pathlib import Path

from scripts.dataloader import BatchLoader, load_wine_dataset


def main() -> None:
    # Ścieżka do głównego katalogu projektu
    project_root = Path(__file__).resolve().parents[1]

    # Ścieżka do pliku danych
    data_path = project_root / "wine" / "data.csv"

    # Wczytanie i przygotowanie danych
    dataset = load_wine_dataset(
        data_path=data_path,
        test_size=0.2,
        random_state=42,
    )

    # Podstawowe informacje o danych
    print("=== Dataset info ===")
    print("x_train shape:", dataset.x_train.shape)
    print("y_train shape:", dataset.y_train.shape)
    print("x_test shape:", dataset.x_test.shape)
    print("y_test shape:", dataset.y_test.shape)
    print("feature names:", dataset.feature_names)
    print("class labels:", dataset.class_labels)
    print()

    # Test działania loadera batchy
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


if __name__ == "__main__":
    main()