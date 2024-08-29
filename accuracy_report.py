# accuracy_report.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv("diabetes.csv")

# Split the data into features and target variable
X = dataset.drop(columns='Outcome', axis=1)
Y = dataset['Outcome']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Load the saved model
with open('model/trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict on the test data
Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy score of the testing data is: {accuracy}")

# Generate confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(11,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Generate classification report
report = classification_report(Y_test, Y_pred)
print("Classification Report:\n", report)

# Save the classification report to a file
with open('accuracy_report.txt', 'w') as file:
    file.write(f"Accuracy score of the testing data is: {accuracy}\n\n")
    file.write("Classification Report:\n")
    file.write(report)

print("Accuracy report saved to 'accuracy_report.txt'")
