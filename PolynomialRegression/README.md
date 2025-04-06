# Polynomial-Salary-Predictor
Polynomial Regression model to predict salary based on experience level using Python, Scikit-learn, and data visualization.

# 📈 Polynomial Regression - Salary Predictor

This project demonstrates how to use **Polynomial Regression** to predict salary based on job level. It includes:

- Outlier removal using IQR method
- Polynomial features (degree=5)
- Model training & testing
- Data visualizations using seaborn & matplotlib

---

## 🔍 Problem Statement

Given a dataset of employee salaries and levels, predict the salary using advanced regression techniques and visualize the results.

---

## 📊 Technologies Used

- Python (Pandas, NumPy)
- Seaborn & Matplotlib
- Scikit-learn (PolynomialFeatures, LinearRegression)
- Jupyter Notebook

---

## 📁 Files

- `polynomial.csv` - Dataset
- `Salary_Prediction.ipynb` - Jupyter notebook with full analysis
- `README.md` - Project overview and instructions

---

## 📈 Visuals

![Scatter](images/fit.png)  
![Boxplot](images/boxplot.png)

> *(Add plots after taking screenshots and saving them as `fit.png`, `boxplot.png` in an `/images` folder)*

---

## 🚀 How to Run

1. Clone this repo  
2. Open `Salary_Prediction.ipynb` using Jupyter Notebook  
3. Run all cells and observe the results  

---

## ✨ Prediction Example

```python
example = pd.DataFrame([[6]], columns=["Level"])
predicted_salary = model.predict(pf.transform(example.values))[0]
