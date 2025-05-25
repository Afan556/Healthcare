import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap
data = pd.read_csv(r'E:\Data_Quest\Projects\healthcare-dataset-stroke-data.csv')
#Filling missing
data['bmi'].fillna(data['bmi'].median(),inplace=True)
print(data['bmi'].head())
#LabelEncoder
encode=LabelEncoder()
#Gender
data['gender_encode']=encode.fit_transform(data['gender'])
print(data[['gender','gender_encode']].head())
#evermarried
data['married_encode']=encode.fit_transform(data['ever_married'])
print(data[['ever_married','married_encode']].head())
#residancetype
data['residence_encode']=encode.fit_transform(data['Residence_type'])
print(data[['Residence_type','residence_encode']].head())
#onehotencoder
onehot=OneHotEncoder()
#worktype
worktype_encoded = onehot.fit_transform(data[['work_type']]).toarray()
worktype_df = pd.DataFrame(worktype_encoded, columns=onehot.get_feature_names_out(['work_type']))
data = pd.concat([data, worktype_df], axis=1)
data.drop('work_type', axis=1, inplace=True)
print(worktype_df.head())
#smokingstatus
smokingstatus_encoded = onehot.fit_transform(data[['smoking_status']]).toarray()
smokingstatus_df = pd.DataFrame(smokingstatus_encoded, columns=onehot.get_feature_names_out(['smoking_status']))
data = pd.concat([data, smokingstatus_df], axis=1)
data.drop('smoking_status', axis=1, inplace=True)
print(smokingstatus_df.head())
#countplot age vs other
plt.figure(figsize=(10,5))
sns.countplot(x='age',data=data)
plt.title('Count of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
#distribution plot by stroke
plt.figure(figsize=(10,5))
sns.histplot(data=data,x='age',hue='stroke',multiple='stack')
plt.title('Distribution of Age by Stroke')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
#correlation heatmap
numerical_data = data.select_dtypes(include=np.number) # Select only numerical columns
plt.figure(figsize=(12, 10))
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
#feature comparison between stroke and non-stroke
plt.figure(figsize=(10, 5))
sns.countplot(x='stroke', data=data)
plt.title('Count of Stroke vs Non-Stroke')
plt.xlabel('Stroke')
plt.ylabel('Count')
plt.show()
#logistic regression
train_data = data.drop(['stroke'], axis=1)
# Splitting the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(train_data, data['stroke'], test_size=0.2, random_state=42)
LogisticRegression_model = LogisticRegression(max_iter=1000)
X = data.drop(['stroke', 'gender', 'ever_married', 'Residence_type'], axis=1) # Drop original categorical columns
y = data['stroke']
LogisticRegression_model.fit(X, y)
# Predicting the target variable
y_pred = LogisticRegression_model.predict(X)
# Evaluating the model

accuracy = accuracy_score(y, y_pred)
confusion = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion}')
print(f'Classification Report:\n{report}')
#random forest train
#random forest train
X_train, X_test, y_train, y_test = train_test_split(train_data, data['stroke'], test_size=0.2, random_state=42)
X_train_processed = X_train.drop(['gender', 'ever_married', 'Residence_type'], axis=1)
X_test_processed = X_test.drop(['gender', 'ever_married', 'Residence_type'], axis=1)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_processed)
#prediction for model availability using shap
shap.initjs()
#summary plot
shap.summary_plot(shap_values, X_test_processed, feature_names=X_test_processed.columns)
#force plot
instance_index = 1 # Choose the index of the instance you want to plot
print("Shape of X_test_processed for instance:", X_test_processed.iloc[[instance_index], :].shape)
print("Columns of X_test_processed:", X_test_processed.columns.tolist())
print("Shape of shap_values[1] for instance:", shap_values[1][instance_index].shape)
shap.force_plot(explainer.expected_value[1], shap_values[1][instance_index], X_test_processed.iloc[instance_index,:], link="logit")
#highlighting the most important features glucose and bmi
shap.dependence_plot('bmi', shap_values, X_test_processed)
shap.dependence_plot('glucose', shap_values, X_test_processed)