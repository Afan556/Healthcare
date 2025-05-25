import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
import shap

# --- 1. Data Loading and Initial Exploration ---
try:
    data = pd.read_csv(r'E:\Data_Quest\Projects\healthcare-dataset-stroke-data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found at the specified path.")
    exit()

print("\nInitial Data Head:")
print(data.head())
print("\nData Info:")
print(data.info())
print("\nMissing Values Before Imputation:")
print(data.isnull().sum())

# --- 2. Data Preprocessing ---

def preprocess_data(df):
    """
    Preprocesses the healthcare dataset for stroke prediction.
    Handles missing values and encodes categorical features.
    """
    df_processed = df.copy()

    # Impute missing BMI values using median (as in original code)
    df_processed['bmi'].fillna(df_processed['bmi'].median(), inplace=True)
    print("\nMissing Values After BMI Imputation:")
    print(df_processed.isnull().sum())

    # Encode categorical features consistently using One-Hot Encoding
    categorical_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    # drop_first=True to avoid multicollinearity

    # Drop the 'id' column as it's not a predictive feature
    df_processed.drop('id', axis=1, inplace=True, errors='ignore')

    print("\nProcessed Data Head:")
    print(df_processed.head())
    print("\nProcessed Data Info:")
    print(df_processed.info())

    return df_processed

processed_data = preprocess_data(data)

# --- 3. Feature and Target Separation, Data Splitting ---

X = processed_data.drop('stroke', axis=1)
y = processed_data['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y to maintain the class proportions in the splits

print("\nTraining Data Shape:", X_train.shape, y_train.shape)
print("Testing Data Shape:", X_test.shape, y_test.shape)

# --- 4. Feature Scaling ---

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("\nScaled Training Data Head:")
print(X_train_scaled_df.head())
print("\nScaled Testing Data Head:")
print(X_test_scaled_df.head())

# --- 5. Model Training and Evaluation ---

def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """
    Trains and evaluates a given classification model.
    """
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n{model_name} Evaluation:")
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{confusion}')
    print(f'Classification Report:\n{report}')

    return model

# Train and evaluate Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42, solver='liblinear') # Added solver for stability
trained_lr_model = train_evaluate_model(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression")

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
trained_rf_model = train_evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, "Random Forest")

# --- 6. Feature Importance using SHAP (for Random Forest) ---

print("\n--- SHAP Analysis (Random Forest) ---")
explainer_rf = shap.TreeExplainer(trained_rf_model)
shap_values_rf = explainer_rf.shap_values(X_test_scaled_df)

shap.initjs()

# Debugging Prints
print("Shape of shap_values_rf[1]:", shap_values_rf[1].shape)
print("Shape of X_test_scaled_df:", X_test_scaled_df.shape)
print("Columns of X_test_scaled_df:", X_test_scaled_df.columns.tolist())
print("Number of features in RF model:", trained_rf_model.n_features_in_)

# Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_rf[1], X_test_scaled_df, feature_names=X_test_scaled_df.columns)
plt.title("SHAP Summary Plot (Random Forest - Class 1: Stroke)")
plt.show()

# Force Plot (for a single instance)
instance_index = 0
shap.force_plot(explainer_rf.expected_value[1], shap_values_rf[1][instance_index], X_test_scaled_df.iloc[instance_index,:], link="logit")

# Dependence Plots (for key features)
plt.figure(figsize=(8, 6))
shap.dependence_plot('bmi', shap_values_rf[1], X_test_scaled_df)
plt.title("SHAP Dependence Plot for BMI")
plt.show()

plt.figure(figsize=(8, 6))
shap.dependence_plot('avg_glucose_level', shap_values_rf[1], X_test_scaled_df)
plt.title("SHAP Dependence Plot for Average Glucose Level")
plt.show()

print("\nSHAP analysis completed.")