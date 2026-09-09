from pathlib import Path
import joblib
import logging
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Any
from typing import cast


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z_clipped: NDArray[np.float64] = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def predict(
    features: NDArray[np.float64],
    weights: dict[Any, NDArray[np.float64]],
    label_code: pd.Index,
) -> pd.Index:
    nb_datum, _ = features.shape
    intercept_column = np.ones([nb_datum, 1], dtype=np.float64)
    features_biased = np.hstack(
        [intercept_column, features]
    )  # data in rows, features in columns
    weights_matrix: NDArray[np.float64] = np.array(
        [weights[target_class] for target_class in label_code]
    ).T  # the list comprehension puts each class as a row, so we need to transpose

    print(weights_matrix.shape, features_biased.shape)

    logits: NDArray[np.float64] = np.matmul(features_biased, weights_matrix)
    probabilities: NDArray[np.float64] = sigmoid(logits)
    predicted_classes: pd.Index = label_code[np.argmax(probabilities, axis=1)]

    return predicted_classes


def normalize_features(features: NDArray[np.float64]) -> NDArray[np.float64]:
    mean: NDArray[np.float64] = np.mean(features, axis=0)
    std: NDArray[np.float64] = np.std(features, axis=0)

    std[std == 0.0] = 1e-15

    return (features - mean) / std


def load_csv_to_numpy(
    file_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.integer]]:
    data: pd.DataFrame = pd.read_csv(file_path)
    data = data.drop(
        labels=["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand"],
        axis=1,
    )
    data = cast(pd.DataFrame, data.apply(pd.to_numeric, errors="coerce").fillna(0))
    features: NDArray[np.float64] = data.drop(labels=["Index"], axis=1).values
    indices: NDArray[np.integer] = data["Index"].values.astype(np.int16)
    return features, indices


def main() -> None:
    weights_path = Path("logreg_weights.joblib")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Run logreg_train first"
        )

    logging.info("Loading weights...")
    model_weights = joblib.load(weights_path)
    weights: dict[Any, NDArray[np.float64]] = model_weights["weights"]
    label_code: pd.Index = model_weights["label_code"]
    logging.info(f"Weights loaded successfully, using classes: {label_code.tolist()}")

    dataset_path = Path("data/dataset_test.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    features: NDArray[np.float64]
    indices: NDArray[np.integer]
    features, indices = load_csv_to_numpy(dataset_path)
    logging.info("Dataset loaded successfully")

    features = normalize_features(features)
    predictions: pd.Index = predict(features, weights, label_code)

    output_df = pd.DataFrame({"Index": indices, "Hogwarts House": predictions})
    output_df.to_csv("houses.csv", index=False)
    print("Predictions successfully exported to houses.csv!")


if __name__ == "__main__":
    main()
