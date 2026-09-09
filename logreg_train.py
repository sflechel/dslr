from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.typing import NDArray
import logging
import matplotlib.pyplot as plt
from typing import cast
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def load_csv_to_numpy(
    file_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.integer], pd.Index]:
    data: pd.DataFrame = pd.read_csv(file_path)
    data = data.drop(
        labels=["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1
    )
    raw_labels = data["Hogwarts House"].factorize()
    labels: NDArray[np.integer] = raw_labels[0]
    label_code: pd.Index = raw_labels[1]
    data = cast(pd.DataFrame, data.apply(pd.to_numeric, errors="coerce").fillna(0))
    features: NDArray[np.float64] = data.drop(labels=["Hogwarts House"], axis=1).values
    return features, labels, label_code


def normalize_features(features: NDArray[np.float64]) -> NDArray[np.float64]:
    mean: NDArray[np.float64] = np.mean(features, axis=0)
    std: NDArray[np.float64] = np.std(features, axis=0)

    std[std == 0.0] = 1e-15

    return (features - mean) / std


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z_clipped: NDArray[np.float64] = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


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

    plt.title("Training Loss Over Iterations (One-vs-Rest)")
    plt.xlabel("Iteration")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def main() -> None:
    features: NDArray[np.float64]
    labels: NDArray[np.integer]
    label_code: pd.Index

    logging.info("Loading dataset...")
    features, labels, label_code = load_csv_to_numpy(Path("data/dataset_train.csv"))

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
