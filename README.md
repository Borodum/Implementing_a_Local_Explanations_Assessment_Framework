# Implementing_a_Local_Explanations_Assessment_Framework
We implement LEAF, a framework for evaluating local linear explanations

## 1. Data Preparation

We use the `data_preprocessing.ipynb` notebook to prepare the dataset. The following steps are performed:

- Load the Breast Cancer dataset using `load_breast_cancer` from sklearn
- Perform a brief exploratory data analysis (EDA)
- Split the data into training and test sets
- Apply feature scaling using StandardScaler
- Save the processed data to a directory for дальнейшего использования

## 2. Simple mlp 

`mpl_training.ipynb` notebook implements a baseline feed-forward neural network (MLP) for binary classification.

__Training setup__
- Loss function: BCEWithLogitsLoss
- Optimizer: Adam
- Input data: standardized features (already scaled in preprocessing step)
- Output: single logit for binary classification

__Evaluation__

Model performance is evaluated on the validation and test sets using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix


Save model to `artifacts/mlp/`
