## Project: LEAF metrics for LIME/SHAP (Breast Cancer Wisconsin)

### Dataset

We use the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset (also known as **WDBC**).
For reproducibility and to avoid manual file downloads, we load it via:

- `sklearn.datasets.load_breast_cancer`

This dataset contains **569 samples**, **30 numeric features**, and a **binary target**
(`malignant` vs `benign`), matching the project proposal in `LEAF.pdf`.

### What is in this repo

- **Dataset preparation**: `prepare_dataset.py` writes reproducible `train/val/test` splits to `data_processed/`.
- **MLP training**: `mlp_training.ipynb` trains `MLPClassifier` and saves artifacts to `artifacts/mlp/`.
- **LIME explanations**: `lime_explanations.ipynb` generates explanations and saves to `artifacts/lime/`.

## Project: LEAF metrics for LIME/SHAP (Breast Cancer Wisconsin)

### Dataset

We use the **UCI Breast Cancer Wisconsin (Diagnostic)** dataset (also known as **WDBC**).
For reproducibility and to avoid manual file downloads, we load it via:

- `sklearn.datasets.load_breast_cancer`

This dataset contains **569 samples**, **30 numeric features**, and a **binary target**
(`malignant` vs `benign`), matching the project proposal in `LEAF.pdf`.

### Pipeline status

- **1) Dataset preparation**: `prepare_dataset.py` writes reproducible `train/val/test` splits to `data_processed/`.
- **2) MLP training**: `mlp_training.ipynb` trains `MLPClassifier` and saves artifacts to `artifacts/mlp/`.
- **3) LIME explanations**: `lime_explanations.ipynb` generates explanations and saves to `artifacts/lime/`.

