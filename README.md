# DA5401 A6: Imputation via Regression for Missing Data

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

## Observations



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



link: https://docs.google.com/forms/d/e/1FAIpQLSe7hCeqn1irfsm4E-0EyRRIOBKyJlyS6jHz4uypirhMX0kpVQ/viewform
