from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.typing import NDArray
import logging
import matplotlib.pyplot as plt
import joblib
import sys

from utils import load_test_dataset, normalize_features, sigmoid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def compute_loss(
    binary_labels: NDArray[np.integer], predictions: NDArray[np.float64]
) -> float:
    eps: float = 1e-15
    predictions_clipped: NDArray[np.float64] = np.clip(predictions, eps, 1 - eps)
    loss = -np.mean(
        binary_labels * np.log(predictions_clipped)
        + (1 - binary_labels) * np.log(1 - predictions_clipped)
    )
    return float(loss)


def training(
    features: NDArray[np.float64],
    labels: NDArray[np.integer],
    classes: pd.Index,
    learning_rate: float = 0.01,
    max_iter: int = 10000,
    tolerance: float = 1e-4,
    patience: int = 100,
) -> tuple[dict[Any, NDArray[np.float64]], dict[Any, list[float]]]:

    all_weights: dict[Any, NDArray[np.float64]] = {}
    all_histories: dict[Any, list[float]] = {}
    nb_datum, nb_features = features.shape
    intercept_column = np.ones([nb_datum, 1], dtype=np.float64)
    features_biased = np.hstack([intercept_column, features])

    for class_id, target_class in enumerate(classes):
        logging.info(f"Training binary model for class: {target_class}")
        binary_labels = np.where(labels == class_id, 1, 0)

        weights: NDArray[np.float64] = np.zeros(nb_features + 1, dtype=np.float64)
        history: list[float] = []
        best_loss = float("inf")
        patience_counter: int = 0

        for i in range(max_iter):
            z: NDArray[np.float64] = np.matmul(features_biased, weights)
            predictions: NDArray[np.float64] = sigmoid(z)
            loss: float = compute_loss(binary_labels, predictions)
            history.append(loss)
            if best_loss - loss > tolerance:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                logging.info(
                    f"Quitting as no more progress is being made, loss is {loss}"
                )
                break

            gradient: NDArray[np.float64] = (
                np.matmul(features_biased.T, predictions - binary_labels) / nb_datum
            )
            weights -= gradient * learning_rate
        else:
            logging.info("Stopped training after max number of iterations")
        all_histories[target_class] = history
        all_weights[target_class] = weights
    return all_weights, all_histories


def plot_training_loss(loss_histories: dict[Any, list[float]]) -> None:
    plt.figure(figsize=(10, 6))

    for target_class, history in loss_histories.items():
        plt.plot(history, label=f"Class: {target_class}")

    plt.title("Training Loss Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def main() -> None:
    if len(sys.argv) == 2:
        dataset_path = Path(sys.argv[1])
    else:
        logging.error("Pass in training data csv as first and only argument")
        exit(1)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    features: NDArray[np.float64]
    labels: NDArray[np.integer]
    label_code: pd.Index

    logging.info("Loading dataset...")
    features, labels, label_code = load_test_dataset(Path(dataset_path))

    normalized_features: NDArray[np.float64] = normalize_features(features)

    all_weights: dict[Any, NDArray[np.float64]]
    all_histories: dict[Any, list[float]]
    logging.info("Starting training")
    all_weights, all_histories = training(normalized_features, labels, label_code)

    export_path = "logreg_weights.joblib"
    joblib.dump({"weights": all_weights, "label_code": label_code}, export_path)
    logging.info(f"Weights exported to {export_path}")

    plot_training_loss(all_histories)


if __name__ == "__main__":
    main()
