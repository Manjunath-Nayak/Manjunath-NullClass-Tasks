import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score
import joblib

# Load the data
data = pd.read_csv("C:\\Users\\jatot\\OneDrive\\Desktop\\Null Class Tasks\\Activity Recognition\\Dataset\\train.csv")
data_test = pd.read_csv("C:\\Users\\jatot\\OneDrive\\Desktop\\Null Class Tasks\\Activity Recognition\\Dataset\\test.csv")

# Display the top 5 rows
data.head()

# Display the bottom rows
data.tail()

# Display the number of rows and columns
print(data.shape)

# Check for duplicate values
data.duplicated().any()

# Remove duplicate columns
duplicated_columns = data.columns[data.T.duplicated()].tolist()
print(len(duplicated_columns))
data = data.drop(duplicated_columns, axis=1)
print(data.shape)

# Check for missing values
data.isnull().sum()

# Create a countplot for the 'Activity' column
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Activity')
plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees

# Show the plot
plt.show()

# Split the data
X = data.drop('Activity', axis=1)
y = data['Activity']
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Logistic Regression
log = LogisticRegression(max_iter=1000)  # You can adjust max_iter as needed
log.fit(X_train, y_train)
y_pred1 = log.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred1)
print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")

# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred2 = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred2)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

# Feature Selection with SelectKBest
k = 200
selector = SelectKBest(f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Feature Selection with RFE
estimator = RandomForestClassifier()
k = 100
rfe_selector = RFE(estimator, n_features_to_select=k)
X_train_selected_rfe = rfe_selector.fit_transform(X_train_selected, y_train)
X_test_selected_rfe = rfe_selector.transform(X_test_selected)

# Random Forest with Selected Features
rf_selected = RandomForestClassifier()
rf_selected.fit(X_train_selected_rfe, y_train)
y_pred_rf = rf_selected.predict(X_test_selected_rfe)
accuracy_rf_selected = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest with Feature Selection Accuracy: {accuracy_rf_selected:.2f}")

# Save models and selectors
joblib.dump(rf_selected, "model_rfe.pkl")
joblib.dump(selector, "k_best_selector.pkl")
joblib.dump(rfe_selector, "rfe_selector.pkl")

# Load the test data and apply the models and selectors
data_test = data_test.drop("Activity", axis=1)
duplicated_columns_test = data_test.columns[data_test.T.duplicated()].tolist()
data_test = data_test.drop(duplicated_columns_test, axis=1)

model_loaded = joblib.load('model_rfe.pkl')
selector_loaded = joblib.load('k_best_selector.pkl')
rfe_selector_loaded = joblib.load('rfe_selector.pkl')

selector_test = selector_loaded.transform(data_test)
X_test_selected_rfe_loaded = rfe_selector_loaded.transform(selector_test)

predictions = model_loaded.predict(X_test_selected_rfe_loaded)
print(predictions)
