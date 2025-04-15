# K-Nearest Neighbors (KNN) - Customer Purchase Prediction

This project implements a K-Nearest Neighbors classifier to predict whether a user will purchase a product based on features like Age and Estimated Salary. It includes hyperparameter tuning, visualization of decision boundaries, and performance evaluation using cross-validation.

## 📊 Dataset
- A CSV file with the following columns:
  - `User ID`
  - `Gender`
  - `Age`
  - `EstimatedSalary`
  - `Purchased`

## 🚀 Workflow
1. Data preprocessing (dropping irrelevant columns, scaling features)
2. Visualizing data with Seaborn
3. Hyperparameter tuning (looping over `k` and random states)
4. Model training and evaluation
5. Visualizing decision boundaries
6. Generating metrics: confusion matrix, classification report
7. Using cross-validation for performance evaluation
8. Elbow method to find the optimal `k`

## 🔧 Tech Stack
- Python
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- NumPy
- Mlxtend

## 📈 Accuracy
- Best k: *varies per seed*
- Train Accuracy: ~**92%**
- Test Accuracy: ~**92%**
- Cross-validation Accuracy: ~**89%**

## 📂 Outputs
- Decision boundary plot
- Elbow method plot for k optimization

## 📁 Project Structure
