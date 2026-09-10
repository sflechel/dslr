from pathlib import Path
import logging
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from utils import load_predict_dataset, load_test_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    houses_path = Path("houses.csv")
    if not houses_path.exists():
        raise FileNotFoundError("houses.csv not found. Run logreg_predict.py first")

    train_path = Path("data/dataset_train.csv")
    test_path = Path("data/dataset_test.csv")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Training or testing dataset could not be found")

    y_train: NDArray[np.integer]
    label_code: pd.Index
    X_train: NDArray[np.float64]
    X_train, y_train, label_code = load_test_dataset(train_path)
    X_test: NDArray[np.float64] = load_predict_dataset(test_path)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    logging.info("Training scikit-learn OvR logistic regression")
    sk_model = OneVsRestClassifier(LogisticRegression()).fit(X_train_scaled, y_train)
    sk_model.fit(X_train_scaled, y_train)

    sk_predictions = sk_model.predict(X_test_scaled)
    sk_predictions = label_code[sk_predictions]

    logging.info("Loading predictions from houses.csv")
    custom_df = pd.read_csv(houses_path)
    custom_predictions = custom_df["Hogwarts House"].values

    if len(custom_predictions) != len(sk_predictions):
        logging.warning(
            f"Length mismatch, houses.csv has {len(custom_predictions)} rows, "
            f"scikit-learn has {len(sk_predictions)} rows"
        )
        exit(1)

    match_count = np.sum(np.array([custom_predictions == sk_predictions]))
    match_percentage = (match_count / len(custom_predictions)) * 100

    logging.info(f"Agreement Rate: {match_percentage:.2f}%")


if __name__ == "__main__":
    main()
