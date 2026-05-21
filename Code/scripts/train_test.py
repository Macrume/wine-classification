from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv

from data import BatchLoader, load_wine_dataset
from neural_network import MultiLayerPerceptron
from neural_network.losses import CrossEntropyLoss
from neural_network.optimizers import SGD
from reporting import generate_training_report_artifacts
from training.trainer import Trainer


load_dotenv()

# ---------------------- Environmental variables ----------------------
ENV_VARIABLES: os._Environ[str] = os.environ

# ----------------------------- Paths ------------------------------
ROOT_PATH: pathlib.Path = pathlib.Path(ENV_VARIABLES["ROOT_PATH"])
DATASET_PATH: pathlib.Path = pathlib.Path(ENV_VARIABLES["DATASET_PATH"])
ARTIFACTS_PATH: pathlib.Path = pathlib.Path(ENV_VARIABLES["ARTIFACTS_PATH"])
FEATURE_NAMES_PATH: pathlib.Path = pathlib.Path(ENV_VARIABLES["FEATURE_NAMES_PATH"])

# Change current directory to project root
os.chdir(ROOT_PATH)


def main() -> None:
    dataset = load_wine_dataset(
        data_path=DATASET_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
    )

    train_loader = BatchLoader(
        x=dataset.x_train,
        y=dataset.y_train,
        batch_size=16,
        shuffle=True,
        random_state=42,
    )

    test_loader = BatchLoader(
        x=dataset.x_test,
        y=dataset.y_test,
        batch_size=16,
        shuffle=False,
    )

    model = MultiLayerPerceptron(
        layer_sizes=[
            dataset.num_features,
            13,
            8,
            dataset.num_classes,
        ],
        random_state=42,
    )

    loss_fn = CrossEntropyLoss()

    optimizer = SGD(
        learning_rate=0.01,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )

    history = trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=1000,
        print_every=50,
    )

    class_labels = dataset.train.class_labels

    if class_labels is None:
        class_labels = list(range(dataset.num_classes))

    output_dir = ARTIFACTS_PATH / "training_results"

    generate_training_report_artifacts(
        history=history,
        output_dir=output_dir,
        class_labels=class_labels,
    )

    print()
    print("Artifacts saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()