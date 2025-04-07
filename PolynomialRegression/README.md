# 📈 Polynomial Salary Predictor

A complete machine learning project using **Polynomial Regression** to predict salary based on employee job level. Built with Python, Jupyter, and Scikit-learn. Visualized beautifully with Seaborn and Matplotlib.

---

## 🧠 What is Polynomial Regression?

Polynomial Regression is an advanced version of Linear Regression where the relationship between the independent variable \( x \) and dependent variable \( y \) is modeled as an nth degree polynomial.

### 📐 Formula:
For degree `n`, the equation is:

\[
y = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_n x^n
\]

Where:
- \( y \): predicted salary
- \( x \): job level
- \( \theta \): coefficients learned by the model

---

## 🧾 Problem Statement

Predict the salary of an employee based on their job level using **polynomial regression**, and visualize the data to understand the model’s performance.

---

## ✅ Workflow

1. **Import Dataset**
2. **Outlier Removal** using IQR method
3. **Feature Transformation**: Convert to polynomial features
4. **Model Training** with `LinearRegression`
5. **Prediction**
6. **Visualization** of data, model fit, and outliers

---

## 🔍 Dataset

- `polynomial.csv` – contains two columns:
  - `Level`: job level (1–10)
  - `Salary`: corresponding salary

---

## 🔧 Technologies Used

- **Python**
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
- **Jupyter Notebook**

---

## 🔬 Exploratory Data Analysis

- Checked for outliers using the **IQR method**:
```python
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
filtered_df = df[~((df['Salary'] < (Q1 - 1.5 * IQR)) | (df['Salary'] > (Q3 + 1.5 * IQR)))]
