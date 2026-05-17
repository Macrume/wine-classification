from __future__ import annotations

import pathlib
import os
from dotenv import load_dotenv; load_dotenv()

import numpy as np

from data import BatchLoader, load_wine_dataset
from neural_network import MultiLayerPerceptron
from neural_network.losses import MeanSquaredError, CrossEntropyLoss, softmax
from training.trainer import Trainer
from neural_network.optimizers import SGD


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
        shuffle=True,
        random_state=42,
    ) 

model = MultiLayerPerceptron(
    layer_sizes=[dataset.num_features, 13, 8, dataset.num_classes],
    random_state=42
)

learning_rate = 0.01
epochs = 1000

loss_fn = CrossEntropyLoss()
optimizer = SGD(
        learning_rate=0.01,
    )

trainer = Trainer(model, loss_fn, optimizer)

trainer.fit(
    train_loader= train_loader,
    test_loader=test_loader,
    epochs=epochs,
    print_every=1
    )





       