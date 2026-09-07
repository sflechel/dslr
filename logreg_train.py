from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.typing import NDArray


def load_csv_to_numpy(
    file_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.integer], pd.Index]:
    data: pd.DataFrame = pd.read_csv(file_path)
    data = data.drop(
        labels=["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
    )
    raw_labels = data["Hogwarts House"].factorize()
    labels: NDArray[np.integer] = raw_labels[0]
    label_code: pd.Index = raw_labels[1]
    data.apply(pd.to_numeric, errors="coerce").fillna(0)
    features: NDArray[np.float64] = data.drop(labels=["Hogwarts House"]).values
    return features, labels, label_code


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(
    binary_labels: NDArray[np.integer], predictions: NDArray[np.float64]
) -> float:
    loss = np.mean(
        binary_labels * np.log(predictions)
        + (1 - binary_labels) * np.log(1 - predictions)
    )
    return float(loss)


def training(
    features: NDArray[np.float64],
    labels: NDArray[np.integer],
    classes: pd.Index,
    learning_rate: float = 0.01,
    max_iter: int = 10000,
    tolerance: float = 1e-6,
    patience: int = 100,
) -> dict[Any, NDArray[np.float64]]:
    all_weights: dict[Any, NDArray[np.float64]] = {}
    nb_datum, nb_features = features.shape

    intercept_column = np.ones([nb_datum, 1], dtype=np.float64)
    features_biased = np.hstack([intercept_column, features])

    for class_id, target_class in enumerate(classes):
        print(f"Training binary model for class: {target_class}")
        binary_labels = np.where(labels == class_id, 1, 0)

        weights: NDArray[np.float64] = np.zeros(nb_features + 1, dtype=np.float64)
        best_loss = float("inf")
        patience_counter: int = 0

        for i in range(max_iter):
            z = np.matmul(features_biased, weights)
            predictions = sigmoid(z)
            loss: float = compute_loss(binary_labels, predictions)
            if best_loss - loss > tolerance:
                loss = best_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

            gradient: NDArray[np.float64] = (
                np.matmul(features_biased, predictions - binary_labels) / nb_datum
            )
            weights -= gradient * learning_rate

        all_weights[target_class] = weights
    return all_weights
