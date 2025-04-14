# 📊 Decision Tree Regression – Salary Prediction

This project implements a **Decision Tree Regressor** to predict salaries based on **Years of Experience** and **Age**. It uses brute-force parameter tuning to find the best combination of `random_state` and `max_depth` for optimal model performance.

---

## 📁 Dataset Overview

The dataset contains the following features:
- `YearsExperience`: Number of years of experience
- `Age`: Age of the individual
- `Salary`: Corresponding annual salary in INR

---

## ⚙️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 Key Features

- Data preprocessing and scaling using `StandardScaler`
- Exhaustive tuning of `random_state` (0–100) and `max_depth` (1–20)
- Visualization of regression curves and decision tree
- Final selection of model with:
  - Best Test Accuracy: **99.36%**
  - Optimal Parameters: `random_state=84`, `max_depth=6`
  - Minimum Train-Test Accuracy Difference: **0.0033**

---

## 📊 Visualizations

### ✅ Decision Tree Structure  
Shows how the model splits data based on feature values.  
📍 `assets/DecisionTreeRegressor.jpg`

### 📈 YearsExperience vs Salary (Prediction Curve)  
📍 `assets/DecisionTreeRegressionCurve1.jpg`

### 📉 Age vs Salary (Prediction Curve)  
📍 `assets/DecisionTreeRegressionCurve2.jpg`

---

## 📦 Project Structure

