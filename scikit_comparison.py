from pathlib import Path
import logging
import numpy as np
import pandas as pd
from typing import cast
from numpy.typing import NDArray

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    houses_path = Path("houses.csv")
    if not houses_path.exists():
        raise FileNotFoundError("houses.csv not found. Run logreg_predict.py first")

    train_path = Path("data/dataset_train.csv")
    test_path = Path("data/dataset_test.csv")
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Training or testing dataset could not be found")

    logging.info("Loading training and testing data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = train_df.drop(
        labels=["Index", "First Name", "Last Name", "Birthday", "Best Hand"], axis=1
    )
    train_features = train_df.drop(["Hogwarts House"], axis=1)
    train_targets = train_df["Hogwarts House"]

    train_features_clean = train_features.apply(pd.to_numeric, errors="coerce")
    train_clean = pd.concat([train_features_clean, train_targets], axis=1).dropna()

    raw_labels = train_clean["Hogwarts House"].factorize()
    y_train: NDArray[np.integer] = raw_labels[0]
    label_code: pd.Index = raw_labels[1]
    X_train: NDArray[np.float64] = train_clean.drop(
        labels=["Hogwarts House"], axis=1
    ).values

    test_df = test_df.drop(
        labels=[
            "Hogwarts House",
            "First Name",
            "Last Name",
            "Birthday",
            "Best Hand",
            "Index",
        ],
        axis=1,
    )
    test_clean = cast(
        pd.DataFrame, test_df.apply(pd.to_numeric, errors="coerce").dropna()
    )
    X_test: NDArray[np.float64] = test_clean.values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    logging.info("Training scikit-learn reference LogisticRegression (OvR)...")
    sk_model = OneVsRestClassifier(LogisticRegression()).fit(X_train_scaled, y_train)
    sk_model.fit(X_train_scaled, y_train)

    sk_predictions = sk_model.predict(X_test_scaled)
    sk_predictions = label_code[sk_predictions]
    print(sk_predictions)

    logging.info("Loading predictions from houses.csv...")
    custom_df = pd.read_csv(houses_path)
    if "Hogwarts House" not in custom_df.columns:
        raise ValueError(
            "houses.csv does not contain the expected 'Hogwarts House' column."
        )
    custom_predictions = custom_df["Hogwarts House"].values
    print(custom_predictions)

    if len(custom_predictions) != len(sk_predictions):
        logging.warning(
            f"Length mismatch! houses.csv has {len(custom_predictions)} rows, "
            f"while scikit-learn evaluated {len(sk_predictions)} rows. Check your row-dropping logic."
        )
        exit(1)

    match_count = np.sum(np.array([custom_predictions == sk_predictions]))
    match_percentage = (match_count / len(custom_predictions)) * 100

    logging.info("=" * 50)
    logging.info(f"Total evaluated samples: {len(custom_predictions)}")
    logging.info(
        f"Exact matches with Scikit-Learn: {match_count} / {len(custom_predictions)}"
    )
    logging.info(f"Agreement Rate: {match_percentage:.2f}%")
    logging.info("=" * 50)

    # Optional: Save a diagnostic difference dataframe for inspection
    comparison_df = pd.DataFrame(
        {
            "Index": custom_df["Index"][: len(custom_predictions)],
            "Custom House": custom_predictions,
            "Sklearn House": sk_predictions,
            "Match": custom_predictions == sk_predictions,
        }
    )
    comparison_df.to_csv("model_comparison_diff.csv", index=False)
    logging.info("Detailed comparison saved to model_comparison_diff.csv")


if __name__ == "__main__":
    main()
