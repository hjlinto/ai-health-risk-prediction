# Health Risk Prediction AI Model

This project presents a machine learning solution designed to predict urban health risk scores based on environmental and air quality metrics. Built using Python, this AI model implements and optimizes multiple regression techniques to address a real-world health forecasting challenge.  

The project was developed with feature selection, regularization, ensemble learning, and comprehensive model evaluation.

---

## Problem Statement

Urban environments face fluctuating air quality that can impact population health. This project uses historical environmental data to train and optimize models that predict a composite `healthRiskScore`. The primary goal was to improve predictive accuracy through AI-based optimization.

---

## Features

- Linear Regression baseline modeling  
- Feature selection via correlation filtering  
- Outlier detection and removal using Z-scores  
- Regularization using **Ridge** and **Lasso**  
- Ensemble techniques: **Bagging** and **Gradient Boosting**  
- Performance metrics: RMSE and R² Score  
- Visualizations for data distribution and prediction comparison

---

## Model Evaluation Summary

| Model                       | RMSE | R² Score |
|-----------------------------|------|----------|
| Original Linear Regression  | 0.12 | 0.97     |
| Optimized Linear Regression | 0.13 | 0.96     |
| Ridge Regression            | 0.13 | 0.96     |
| Lasso Regression            | 0.14 | 0.95     |
| Bagging Regressor           | 0.13 | 0.96     |
| Gradient Boosting Regressor | 0.09 | 0.98     |

> The **Gradient Boosting Regressor** delivered the best overall performance, demonstrating strong generalization and lower prediction error.

---

## Dataset

- **Source**: `DQN1_Dataset.csv` (provided by WGU)
- Includes weather, air quality, and calculated health severity indicators
- Preprocessing steps:
  - Feature engineering
  - Correlation-based feature selection (`R > 0.2`)
  - Z-score outlier removal (`±3 SD`, ~3.8% removed)

---

## Technologies Used

- Python 3.10+
- Pandas, NumPy, Matplotlib
- scikit-learn
- SciPy

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/hjlinto/ai-health-risk-prediction.git
```
2. Run the program:
```bash
python linear_regression_model.py
```

## Author
Created by: Hunter J Linton

