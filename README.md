# 🏥 Predicting Patient Survival — Healthcare Analytics Project

This project simulates a healthcare setting where we use demographic and clinical data to predict **patient survival outcomes**. Using machine learning models like **Logistic Regression** and **Random Forests**, we explore which features most influence the likelihood of survival — empowering healthcare providers to make smarter, data-driven decisions.

---

## 🎯 Project Objective

- Predict whether a patient survives based on key characteristics
- Use ML models to compare performance and interpret feature importance
- Simulate how smart feature engineering can improve real-world healthcare predictions

---

## 📦 Dataset Overview

> This project repurposes the Titanic dataset to represent a healthcare survival problem.

| Feature        | Description                               |
|----------------|-------------------------------------------|
| `Age`          | Age of the patient                        |
| `Sex`          | Gender                                    |
| `Pclass`       | Simulated risk tier / socio-economic class |
| `SibSp`        | Number of siblings/spouses (simulated family support) |
| `Parch`        | Number of parents/children (simulated dependent burden) |
| `Fare`         | Simulated treatment cost proxy            |
| `Embarked`     | Simulated hospital region code            |
| `Survived`     | Target: 1 = Survived, 0 = Did not survive |

---

## 🧰 Tools & Tech Stack

| Purpose           | Stack                                 |
|-------------------|----------------------------------------|
| Data Handling     | `pandas`, `numpy`                      |
| Visualization     | `matplotlib`, `seaborn`                |
| Modeling          | `scikit-learn`                         |
| Model Tuning      | `GridSearchCV`, `RandomForestClassifier` |
| Feature Engineering | Custom new columns: `FamilySize`, `IsAlone` |

---

## 🧠 ML Pipeline Steps

### ✅ Step 1: Data Cleaning
- Imputed missing values (`Age`, `Fare`, `Embarked`)
- Converted categorical variables using one-hot encoding

### ✅ Step 2: Feature Engineering
- Created:
  - `FamilySize` = `SibSp` + `Parch` + 1
  - `IsAlone` = binary flag if patient has no family

### ✅ Step 3: Modeling & Evaluation
- Trained two models:
  - **Logistic Regression**
  - **Random Forest with GridSearchCV**
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - Feature Importance

---

## 🔍 Key Findings

- Gender, class (risk tier), and `IsAlone` were strong predictors of survival
- Patients with family support (`FamilySize` > 1) had a higher survival chance
- Random Forest outperformed Logistic Regression with a tuned `max_depth`

---

## 📊 Visualizations

- 📈 Confusion Matrix
- 📊 Feature Importance Bar Charts
- 📌 Fare vs. Class survival boxplots
- 🚹 Gender breakdowns by risk tier

---

## 📁 Project Structure

