import numpy as np
import pandas as pd
from pathlib import Path
from numpy.typing import NDArray


def load_csv_to_numpy(file_path: Path):
    data: pd.DataFrame = pd.read_csv(file_path)
    data = data.drop(
        labels=["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
    )
    raw = data["Hogwarts House"].factorize()
    labels: NDArray[np.integer] = raw[0]
    label_code: pd.Index = raw[1]
    data.apply(pd.to_numeric, errors="coerce").fillna(0)
    features: NDArray[np.float64] = data.drop(labels=["Hogwarts House"]).values
