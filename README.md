# Implementing_a_Local_Explanations_Assessment_Framework
We implement LEAF, a framework for evaluating local linear explanations

## Data Preparation

We use the `data_preprocessing.ipynb` notebook to prepare the dataset. The following steps are performed:

- Load the Breast Cancer dataset using `load_breast_cancer` from sklearn
- Perform a brief exploratory data analysis (EDA)
- Split the data into training and test sets
- Apply feature scaling using StandardScaler
- Save the processed data to a directory for дальнейшего использования