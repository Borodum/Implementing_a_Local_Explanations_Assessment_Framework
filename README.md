# LEAF Project: Evaluating Local Explanations

This project compares two popular explainability methods, **LIME** and **SHAP**, using the **LEAF** framework.

The goal is simple:  
we train a black-box model for breast cancer diagnosis, generate local explanations, and measure explanation quality with LEAF metrics.

## What This Project Does

- Uses the **Wisconsin Breast Cancer** dataset (`sklearn.datasets.load_breast_cancer`)
- Trains a binary classifier (MLP in PyTorch)
- Builds local explanations with:
  - **LIME**
  - **SHAP** (KernelExplainer)
- Evaluates explanations with LEAF metrics:
  - Conciseness
  - Local Fidelity
  - Local Concordance
  - Reiteration Similarity
  - Prescriptivity

## Project Structure

- `data_preprocessing.ipynb` - data loading, EDA, split, scaling, saving processed data
- `mpl_training.ipynb` - training and evaluating the MLP model
- `lime_explanations.ipynb` - generating LIME explanations and saving results
- `shap_explanations.ipynb` - generating SHAP explanations and saving results
- `leaf_explanations.ipynb` - LEAF metric calculation and comparison
- `final_report.ipynb` - figures and summary visuals
- `blog_post.md` - detailed write-up of the full project story

## Main Output Folders

- `artifacts/mlp/` - trained model config and weights
- `artifacts/lime/` - LIME explanations (`.html`) and summary JSON
- `artifacts/shap/` - SHAP summaries and expected value JSON
- `artifacts/leaf/` - LEAF metrics JSON and comparison plots
- `artifacts/report/` - final figures for analysis (created by `final_report.ipynb`)

## Requirements

The notebooks install needed packages inside cells, but it is better to install once in your environment:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib torch lime shap
```

## How To Run (Step by Step)

Run notebooks in this exact order:

1. `data_preprocessing.ipynb`
2. `mpl_training.ipynb`
3. `lime_explanations.ipynb`
4. `shap_explanations.ipynb`
5. `leaf_explanations.ipynb`
6. `final_report.ipynb` (optional, for final figures)

This order is important because each next notebook reads files created by the previous one.

## Data and Model Details (Short Version)

- Dataset: 569 samples, 30 numeric features, 2 classes (malignant / benign)
- Split strategy: train / validation / test
- Scaling: `StandardScaler`
- Model: MLP (input 30 -> hidden 100 -> hidden 50 -> output 1)
- Loss: `BCEWithLogitsLoss`
- Optimizer: `Adam`

## Saved Result Files (Important)

After running all notebooks, you should have at least:

- `artifacts/mlp/model_config.json`
- `artifacts/mlp/model_state.pt`
- `artifacts/lime/lime_summary.json`
- `artifacts/lime/chosen_test_indices.json`
- `artifacts/shap/summary.json`
- `artifacts/shap/expected_value.json`
- `artifacts/leaf/leaf_metrics.json`

## Notes

- Some result files are already in the repository (`artifacts/...`), so you can inspect outputs without re-running everything.
- The `data/` folder may be generated after preprocessing if it is not present yet.
- For reproducibility, notebooks use fixed random seeds (mainly `seed = 42`).

## Quick Summary

In this repository, we do not only explain model predictions, but also **evaluate explanation quality** in a structured way.  
This makes the project useful for students who want to learn practical XAI and compare methods with clear metrics, not only visual plots.
