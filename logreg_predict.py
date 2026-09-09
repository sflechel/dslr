from pathlib import Path
import joblib
import logging
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Any
from typing import cast


def predict(features: NDArray[np.float64], weights: dict[Any, NDArray[np.float64]]):

def load_csv_to_numpy(
    file_path: Path,
) -> NDArray[np.float64]:
    data: pd.DataFrame = pd.read_csv(file_path)
    data = data.drop(
        labels=["First Name", "Last Name", "Birthday", "Best Hand"], axis=1
    )
    data = cast(pd.DataFrame, data.apply(pd.to_numeric, errors="coerce").fillna(0))
    features: NDArray[np.float64] = data.drop(labels=["Index"], axis=1).values
    return features

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
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}"
        )
    features: NDArray[np.float64] = load_csv_to_numpy(dataset_path)
    logging.info("Dataset loaded successfully")
    predict(features, weights)


if __name__ == "__main__":
    main()
