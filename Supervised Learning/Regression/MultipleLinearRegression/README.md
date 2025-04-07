# 📈 Multiple Linear Regression (MLR) - Salary Prediction

This project demonstrates how to perform **Multiple Linear Regression** using Python to predict **Salary** based on two independent variables: `YearsExperience` and `Age`.

---

## 🔧 Tools & Libraries Used

- Python
- Pandas
- Numpy
- Seaborn
- Matplotlib
- Scikit-Learn

---

## 📁 Dataset

**File Used:** `MultipleLinearRegression.csv`  
**Columns:**
- `YearsExperience`: Total years of experience
- `Age`: Current age of the employee
- `Salary`: Salary in ₹

---

## 📌 Objective

To build a **Multiple Linear Regression** model to predict the salary based on **experience and age**.

---

## 🧮 Multiple Linear Regression Formula

The general form of the **Multiple Linear Regression** equation is:

\[
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2
\]

Where:
- \(\hat{y}\) = Predicted Salary  
- \(\beta_0\) = Intercept  
- \(\beta_1\) = Coefficient for `YearsExperience`  
- \(\beta_2\) = Coefficient for `Age`  
- \(x_1 =\) YearsExperience, \(x_2 =\) Age

---

## 🧾 Steps Followed

### 1. **Data Loading & Preprocessing**

```python
data = pd.read_csv('MultipleLinearRegression.csv')
print(data.head())
print(data.isnull().sum())
