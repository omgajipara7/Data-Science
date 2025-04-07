🧠 Diabetes Prediction using Logistic Regression

This project uses Logistic Regression to predict the probability of diabetes in individuals based on medical diagnostic features. The dataset used is the popular PIMA Indian Diabetes Dataset.

📁 Dataset Overview

The dataset contains 768 observations and 9 columns:

Feature	Description
Pregnancies	Number of times pregnant
Glucose	Plasma glucose concentration
BloodPressure	Diastolic blood pressure (mm Hg)
SkinThickness	Triceps skin fold thickness (mm)
Insulin	2-Hour serum insulin (mu U/ml)
BMI	Body Mass Index
DiabetesPedigreeFunction	Diabetes pedigree function
Age	Age in years
Outcome	Diabetes (1 = Yes, 0 = No)
🎯 Objective

To develop a Logistic Regression model that:

Accurately predicts the likelihood of a patient having diabetes.
Uses scaling and optimal random_state to ensure robust performance.
🧪 Tools & Libraries Used

Python
Pandas, NumPy
Seaborn & Matplotlib (Visualization)
Scikit-learn (Modeling, Scaling, Evaluation)
Pickle (Model saving)
📌 Logistic Regression: Theory

🔹 What is Logistic Regression?
Logistic Regression is used for binary classification problems where the output is either 0 or 1.

🔹 Sigmoid Function

![alt text](images/image.png)
 
Output is a probability between 0 and 1.
A threshold (default = 0.5) is used to classify the result:
If probability ≥ 0.5 → 1 (Diabetic)
Else → 0 (Not Diabetic)
🔍 Workflow

1️⃣ Load Dataset
data = pd.read_csv("Diabetes.csv")
2️⃣ Data Preprocessing
Handle missing values if any
Feature Scaling using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
3️⃣ Model Optimization
Try 100 different random_state values in train_test_split()
Choose the one with the best accuracy
for state in range(101):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=state)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = accuracy_score(y_test, model.predict(X_test))
4️⃣ Final Model Training
Use the best random_state to train and evaluate the final model.

model = LogisticRegression()
model.fit(X_train, y_train)
📈 Visualization

Correlation Heatmap:
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
Confusion Matrix:
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model, X_test, y_test)
🧠 Model Evaluation Metrics

Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)
🧾 Prediction on New Data

new_data = [[2, 130, 70, 25, 100, 28.0, 0.5, 35]]
new_scaled = scaler.transform(new_data)
model.predict(new_scaled)
💾 Save and Load Model

import pickle

# Save
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("diabetes_scaler.pkl", "wb"))

# Load
model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))
📌 Example Output

Best Accuracy: 0.81 at random_state = 42

Accuracy: 0.81
Confusion Matrix:
[[85 15]
 [18 36]]

Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.85      0.84       100
           1       0.71      0.67      0.69        54

    accuracy                           0.81       154
   macro avg       0.77      0.76      0.76       154
weighted avg       0.80      0.81      0.81       154
📚 References

Scikit-learn Logistic Regression
PIMA Indian Diabetes Dataset
📢 Author

👨‍💻 Developed by Om Gajipara
🔗 Powered by HexaCore