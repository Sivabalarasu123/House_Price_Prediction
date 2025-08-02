# House Price Prediction
A Machine Learning project that predicts house prices using regression models (Linear Regression, Random Forest, XGBoost). The goal is to demonstrate end-to-end ML model building, evaluation, and optimization techniques.


---

## 📊 Dataset

- **Name:** King County House Sales Dataset  
- **Source:** [Kaggle - House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction)
- **Features:** 21 features including bedrooms, bathrooms, sqft_living, condition, location, etc.  
- **Target:** `price` (house sale price)

---

## 🚀 Project Objectives

1. Load and explore the dataset.
2. Perform data cleaning and preprocessing.
3. Implement three ML models:
   - Model 1: Linear Regression (baseline)
   - Model 2: Random Forest Regressor (improved)
   - Model 3: XGBoost Regressor with hyperparameter tuning (optimized)
4. Evaluate each model using key metrics.
5. Visualize performance with prediction vs actual plots.

---

## 🛠️ Models and Techniques Used

| Model        | Technique                          | Tools & Libraries                      |
|--------------|-------------------------------------|----------------------------------------|
| Model 1      | Linear Regression                   | scikit-learn                           |
| Model 2      | Random Forest Regressor             | scikit-learn                           |
| Model 3      | XGBoost + GridSearchCV Optimization | XGBoost, scikit-learn, seaborn         |

### ⚙️ Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Visual analysis using seaborn and matplotlib

---

## ✅ Results Summary

| Model     | MAE        | RMSE       | R² Score  |
|-----------|------------|------------|-----------|
| Model 1   | ~174,000   | ~272,000   | ~0.51     |
| Model 2   | ~72,000    | ~148,000   | ~0.85     |
| Model 3   | ~70,000    | ~142,000   | ~0.87     |

_Model 3 with XGBoost and GridSearchCV provided the best performance._

---

## 🖼️ Visualizations

- Distribution of features
- Correlation heatmap
- Actual vs Predicted price scatterplots
- Residual plots

---
