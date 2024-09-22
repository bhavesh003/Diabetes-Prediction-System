import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Define features and target variable
X = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

# Handle class imbalance using SMOTE Synthetic Minority Over Sampling Technique
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model 1: Support Vector Machine with GridSearchCV
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'probability': [True]  # Ensure probability is enabled
}

svm_grid = GridSearchCV(SVC(probability=True), param_grid_svm, refit=True, verbose=3, cv=5)
svm_grid.fit(X_train, y_train)


# Model 2: Random Forest Classifier with GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(RandomForestClassifier(), param_grid_rf, refit=True, verbose=3, cv=5)
rf_grid.fit(X_train, y_train)

# Model 3: Logistic Regression with GridSearchCV
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100]
}

lr_grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, refit=True, verbose=3, cv=5)
lr_grid.fit(X_train, y_train)

# Stacking: Combine SVM, Random Forest, and Logistic Regression
estimators = [
    ('svm', svm_grid.best_estimator_),
    ('rf', rf_grid.best_estimator_),
    ('lr', lr_grid.best_estimator_)
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_stacked = stacking_model.predict(X_test)

# Evaluate the model
accuracy_stacked = accuracy_score(y_test, y_pred_stacked)
print(f"Accuracy of Stacking Model: {accuracy_stacked}")
print("\nClassification Report for Stacking Model:")
print(classification_report(y_test, y_pred_stacked))

# Save the best model (stacking model)
with open('model/trained_model.pkl', 'wb') as file:
    pickle.dump(stacking_model, file)

# Inspect the saved model
def inspect_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model type: {type(model)}\n")
    print(f"Model Parameters:\n {model.get_params()}\n")
    sample_data = X_test[0].reshape(1, -1)
    print(f"Prediction for the sample data: {model.predict(sample_data)}")

# Inspect the saved model
inspect_model('model/trained_model.pkl')
