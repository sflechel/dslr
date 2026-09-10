# dslr

# Hogwarts House Classifier: OvR Logistic Regression from Scratch

A robust, fully vectorized **One-vs-Rest (OvR) Logistic Regression** implementation built from scratch using pure `NumPy` and `Pandas`. This project replicates core machine learning mechanics without relying on external training black boxes, matching the performance and output schema of industry-standard libraries like `scikit-learn`.

---

## 🚀 Key Features

* **From-Scratch Implementation:** Custom gradient descent optimization and sigmoid probability estimation.
* **Vectorized Inference:** Eliminates slow Python loops during prediction using optimized matrix multiplication (`features_biased @ weights_matrix.T`).
* **Hardened Data Pipeline:** Strict schema validation with descriptive error handling for missing files, corrupted columns, and malformed rows.
* **Scikit-Learn Parity:** Includes benchmarking tools to measure prediction agreement rates against standard library implementations.

---

## 📁 Project Structure

├── data/
│   ├── dataset_train.csv      # Training dataset
│   └── dataset_test.csv       # Testing dataset
├── utils.py                   # Shared data loading and validation logic
├── logreg_train.py            # Training script (serializes model weights)
├── logreg_predict.py          # Prediction script (generates houses.csv)
├── scikit_comparison.py       # Benchmarking script against scikit-learn
├── requirements.txt           # Project dependencies
└── .gitignore

```

---

## 🛠️ Prerequisites & Installation

Ensure you have Python 3.10+ installed. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Train the Model

Run the training script on your training dataset to optimize the weights for each Hogwarts House and serialize the artifact:

```bash
python logreg_train.py data/dataset_train
```

*This generates the `logreg_weights.joblib` artifact containing your trained weights and class label mappings.*

### 2. Generate Predictions (`houses.csv`)

Run the prediction script on your test dataset. It will automatically clean malformed inputs, align row indices, and output a strict submission file:

```bash
python logreg_predict.py data/dataset_test logreg_weights.joblib
```

*This outputs `houses.csv` structured precisely for evaluation:*

```csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
...
```

### 3. Benchmark Against Scikit-Learn

To verify the mathematical accuracy and see how closely your custom NumPy matrix math matches `scikit-learn`, run the evaluation benchmark:

```bash
python scikit_comparison.py
```

*This outputs an explicit agreement percentage rate and generates a difference report (`model_comparison_diff.csv`) for auditing.*

---

## 🛡️ Robust Error Handling

The data pipeline features built-in guardrails to handle messy datasets:

* **Schema Validation:** Aborts immediately if mandatory columns are missing from input files.
* **Type Coercion:** Gracefully converts dirty or malformed feature cells into `NaN` values using `pd.to_numeric(..., errors="coerce")`.
* **Synchronized Cleaning:** Drops incomplete rows across both features and targets simultaneously to preserve data alignment.
