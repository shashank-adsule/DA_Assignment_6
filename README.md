# Assignment 6 (Imputation via Regression for Missing Data)

## Student Info:
Name: Shashank Satish Adsule\
Roll no.: DA25M005

## Dataset Used
- [**UCI Credit Card Default Clients Dataset**](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- this dataset contains : 
    - `UCI_Credit_Card.csv`
- The dataset contains 30000 smaples entries with 24 column feature, and 1 binary target variable `default.payment.next.month`:
    - ID: ID of each client
    - LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
    - SEX: Gender (1=male, 2=female)
    - EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
    - MARRIAGE: Marital status (1=married, 2=single, 3=others)
    - AGE: Age in years
    - PAY_0 to PAY_6: repayment status of each month
    - BILL_AMT1 to  BILL_AMT6: Amount of bill statement
    - PAY_AMT1 to PAY_AMT6: Amount of previous payment


## Objective

The goal of this assignment is to **handle missing data using regression-based imputation** techniques.  
The focus is on demonstrating how to:
1. Introduce artificial missing values in selected columns (`AGE`, `BILL_AMT`, and `MARRIAGE`).
2. Apply **median-based imputation** for baseline comparison.
3. Use **regression models** (Linear Regression, Decision Tree Regressor, and KNN Regressor) to **predict and impute missing values** more accurately.
4. Evaluate the quality of imputation through regression metrics and visualize the effect on downstream model performance.

## Methodology

1. **Preprocessing & Exploration**
   - Loaded dataset and inspected duplicates, null counts, and class distribution.
   - Visualized distributions and correlations using `seaborn` and `matplotlib`.

2. **Simulating Missing Data**
   - Artificially removed a percentage (e.g., 20%) of values in selected columns.
   - Created multiple versions of the dataset for different missing ratios.

3. **Imputation Approaches**
   - **Baseline:** Filled missing values using column **median**.
   - **Regression-based:**
     - Built regression models (Linear, KNN, Decision Tree) to predict missing entries.
     - Compared model-based imputation with the median fill.

4. **Evaluation**
   - Evaluated reconstructed data consistency.
   - Used R², RMSE, and correlation plots to measure imputation quality.
   - Compared imputed data effect on logistic regression classifier accuracy.

1. **Preprocessing & Exploration**
   - Loaded dataset and inspected duplicates, null counts, and class distribution.
   - Visualized distributions and correlations using `seaborn` and `matplotlib`.

## Observations

- Median filling is **simple and fast**, but less accurate when the variable relationships are nonlinear.
- Regression-based imputation produced **more realistic estimates**, especially for variables like `BILL_AMT` that correlate with `PAY_AMT` features.
- Decision Tree and KNN regressors handled **nonlinear dependencies** better than Linear Regression.
- Increasing missing ratio decreased overall accuracy and correlation stability.
- Imputation quality directly impacted the **classification accuracy** of the downstream model predicting default payment.


## Python Dependencies
The following libraries were used in the analysis:

```bash
os                      # file path handling and system operations
pandas                  # data manipulation and analysis
numpy                   # numerical operations and array handling
matplotlib              # data visualization (plots, charts)
seaborn                 # advanced visualization and statistical plotting
copy                    # object copying for safe data handling

scikit-learn            # machine learning library (modeling, evaluation, preprocessing)
    ├── model_selection (train_test_split, cross_val_score, KFold)
    │       # data splitting, cross-validation, and performance evaluation
    ├── preprocessing (StandardScaler)
    │       # feature standardization and normalization
    ├── metrics (confusion_matrix, accuracy_score, classification_report, auc, 
    │             average_precision_score, roc_curve, precision_recall_curve,
    │             precision_recall_fscore_support)
    │       # model evaluation metrics for classification performance
    ├── linear_model (LinearRegression, LogisticRegression)
    │       # linear and logistic regression models
    ├── neighbors (KNeighborsRegressor)
    │       # non-linear regression based on nearest neighbors
    └── tree (DecisionTreeRegressor)

```
<!-- # ├── and └── -->

## Conclusion

This assignment demonstrates that:
- **Regression-based imputation** outperforms simple statistical methods (mean/median) by leveraging relationships between features.
- Among tested models, **Decision Tree Regressor** and **KNN Regressor** offered the best trade-off between accuracy and robustness to nonlinearity.
- Handling missing data properly is critical for maintaining the reliability of machine learning pipelines.
- The approach can be extended to other domains where missing data is frequent, such as finance or healthcare analytics.
