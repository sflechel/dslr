from pathlib import Path
import joblib
import logging
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Any
from typing import cast
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


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
) -> NDArray[np.float64]:
    try:
        data: pd.DataFrame = pd.read_csv(file_path)
    except Exception as _:
        raise ValueError(f"Failed to parse csv file at {file_path}")

    try:
        data = data.drop(
            labels=[
                "Hogwarts House",
                "First Name",
                "Last Name",
                "Birthday",
                "Best Hand",
            ],
            axis=1,
        )
    except KeyError as _:
        raise KeyError("Missing expected columns")

    data = cast(pd.DataFrame, data.apply(pd.to_numeric, errors="coerce").dropna())
    if data.empty:
        raise ValueError("Dataset contains no values after cleaning")

    try:
        features: NDArray[np.float64] = data.drop(labels=["Index"], axis=1).values
    except KeyError as _:
        raise KeyError("Missing expected columns")

    return features


def load_model_parameters(
    weights_path: Path,
) -> tuple[dict[Any, NDArray[np.float64]], pd.Index]:
    model_weights = joblib.load(weights_path)
    try:
        model_weights = joblib.load(weights_path)
    except Exception as e:
        raise ValueError(f"Failed to parse model artifact at {weights_path}: {e}")
    if not isinstance(model_weights, dict):
        raise TypeError(
            f"Invalid model artifact format. Expected a dictionary, got {type(model_weights).__name__}."
        )

    required_keys = {"weights", "label_code"}
    missing_keys = required_keys - model_weights.keys()
    if missing_keys:
        raise KeyError(
            f"Model artifact schema error: Missing expected keys -> {missing_keys}"
        )

    return model_weights["weights"], model_weights["label_code"]


def main() -> None:
    if len(sys.argv) == 3:
        dataset_path = Path(sys.argv[1])
        weights_path = Path(sys.argv[2])
    else:
        logging.error("Pass in dataset_train.csv as first and only argument")
        exit(1)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. Run logreg_train first"
        )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    logging.info("Loading weights...")
    weights: dict[Any, NDArray[np.float64]]
    label_code: pd.Index
    weights, label_code = load_model_parameters(weights_path)
    logging.info(f"Weights loaded successfully, using classes: {label_code.tolist()}")

    features: NDArray[np.float64] = load_csv_to_numpy(dataset_path)
    indices: NDArray[np.int16] = np.arange(len(features), dtype=np.int16)
    logging.info("Dataset loaded successfully")

    features = normalize_features(features)
    predictions: pd.Index = predict(features, weights, label_code)

    output_df = pd.DataFrame({"Index": indices, "Hogwarts House": predictions})
    output_df.to_csv("houses.csv", index=False)
    logging.info("Predictions successfully exported to houses.csv")


if __name__ == "__main__":
    main()
