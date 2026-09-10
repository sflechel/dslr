from numpy.typing import NDArray
import numpy as np
import pandas as pd
from pathlib import Path
from typing import cast


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z_clipped: NDArray[np.float64] = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def normalize_features(features: NDArray[np.float64]) -> NDArray[np.float64]:
    mean: NDArray[np.float64] = np.mean(features, axis=0)
    std: NDArray[np.float64] = np.std(features, axis=0)

    std[std == 0.0] = 1e-15

    return (features - mean) / std


def load_predict_dataset(
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


def load_test_dataset(
    file_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.integer], pd.Index]:
    try:
        data: pd.DataFrame = pd.read_csv(file_path)
    except Exception as _:
        raise ValueError(f"Failed to parse csv file at {file_path}")

    try:
        data = data.drop(
            labels=["Index", "First Name", "Last Name", "Birthday", "Best Hand"],
            axis=1,
        )
        target = data["Hogwarts House"]
        features_df = data.drop(labels=["Hogwarts House"], axis=1)
    except KeyError as _:
        raise KeyError("Missing expected columns")

    features_numeric = features_df.apply(pd.to_numeric, errors="coerce")
    clean_data = pd.concat([features_numeric, target], axis=1).dropna()
    if clean_data.empty:
        raise ValueError("Dataset contains no values after cleaning")

    features_clean = clean_data.drop(labels=["Hogwarts House"], axis=1)
    target_clean = clean_data["Hogwarts House"]

    raw_labels = target_clean.factorize()
    labels: NDArray[np.integer] = raw_labels[0]
    label_code: pd.Index = raw_labels[1]

    features: NDArray[np.float64] = features_clean.values
    return features, labels, label_code
