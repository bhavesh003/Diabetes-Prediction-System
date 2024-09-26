import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle

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

# Model 1: Support Vector Machine with RandomizedSearchCV
param_dist_svm = {
    'C': np.logspace(-3, 3, 7),
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'probability': [True]  # Ensure probability is enabled
}

svm_random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist_svm, n_iter=50, refit=True, verbose=3, cv=5)
svm_random_search.fit(X_train, y_train)

# Model 2: Random Forest Classifier with RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist_rf, n_iter=50, refit=True, verbose=3, cv=5)
rf_random_search.fit(X_train, y_train)

# Model 3: Gradient Boosting Classifier with RandomizedSearchCV
param_dist_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 5],
    'subsample': [0.8, 0.9, 1.0]
}

gb_random_search = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=param_dist_gb, n_iter=50, refit=True, verbose=3, cv=5)
gb_random_search.fit(X_train, y_train)

# Stacking: Combine SVM, Random Forest, and Gradient Boosting
estimators = [
    ('svm', svm_random_search.best_estimator_),
    ('rf', rf_random_search.best_estimator_),
    ('gb', gb_random_search.best_estimator_)
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
