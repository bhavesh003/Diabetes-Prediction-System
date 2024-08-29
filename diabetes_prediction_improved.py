import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Define features and target variable
X = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}

grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=5)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Save the trained model
import pickle
with open('model/trained_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Accuracy score of the testing data is: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Inspect the model
def inspect_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model type: {type(model)}\n")
    print(f"Model Parameters:\n {model.get_params()}\n")
    sample_data = X_test[0].reshape(1, -1)
    print(f"Prediction for the sample data: {model.predict(sample_data)}")

# Inspect the saved model
inspect_model('model/trained_model.pkl')
