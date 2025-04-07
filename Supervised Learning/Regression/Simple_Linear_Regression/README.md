# 📊 Simple Linear Regression – CGPA vs Salary Package

This project demonstrates a **Simple Linear Regression** model to predict the **Salary Package** of a student based on their **CGPA**.

---

## 🧠 Concept

### What is Simple Linear Regression?

Simple Linear Regression establishes a **linear relationship** between:
- 📈 **Independent variable (X)**: CGPA
- 📉 **Dependent variable (y)**: Salary Package (in LPA)

**Formula**:
\[
y = mx + c
\]
- \( y \) → predicted package  
- \( m \) → slope (coefficient)  
- \( x \) → input CGPA  
- \( c \) → y-intercept  

---

## 🛠️ Tools Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🔍 Dataset Summary

- 📁 File: `SimpleLinearRegression.csv`
- ✅ Columns: `cgpa`, `package`

---

## 📂 Sections in the Notebook

### 1️⃣ Load and Inspect Data

```python
import pandas as pd

data = pd.read_csv('SimpleLinearRegression.csv')
print(data.isnull().sum())
print(data.info())
print(data.describe())
