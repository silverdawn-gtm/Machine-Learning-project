# Machine-Learning-project

## Car Price Prediction Project

## Overview

This project aims to predict car prices based on various features using different machine learning regression models. The dataset consists of multiple car attributes that influence pricing. The objective is to identify the most significant factors affecting car prices and develop a robust predictive model.

## Dataset

- The dataset is stored in `car_price_dataset.csv`.
- It contains categorical and numerical variables.
- The target variable is `price`.

## Steps Implemented

### 1. Data Loading and Preprocessing

- Load the dataset using Pandas.
- Check for missing values and handle them.
- Identify categorical and numerical columns.
- Apply preprocessing: StandardScaler for numerical data and OneHotEncoder for categorical data.

### 2. Model Implementation

The following regression models are implemented:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor

Each model is trained using a pipeline that integrates preprocessing and training.

### 3. Model Evaluation

- Metrics used: R-squared (R2), Mean Squared Error (MSE), Mean Absolute Error (MAE).
- The performance of all models is compared to select the best one.

### 4. Hyperparameter Tuning

- The best-performing model undergoes hyperparameter tuning using `GridSearchCV`.
- The tuned model is evaluated again to check for performance improvements.

### 5. Feature Importance Analysis

- The best model’s feature importances are analyzed.
- A bar chart is plotted to visualize the most influential factors in determining car prices.

## Results

- Model performances are stored in a Pandas DataFrame.
- The best model is identified based on R2 score and other evaluation metrics.
- Feature importance analysis highlights the key predictors of car price.

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Place the dataset (`car_price_dataset.csv`) in the project directory.
2. Run the script in a Jupyter Notebook or as a standalone Python script.
3. Review the model evaluation results and feature importance analysis.

##

