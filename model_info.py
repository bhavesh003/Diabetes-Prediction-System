import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the data (replace 'diabetes.csv' with your actual dataset path)
data = pd.read_csv('diabetes.csv')  # Adjust the path as necessary

# Features and target variable
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

# Step 2: Preprocess the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train the model
model = SVC(
    C=1.0, 
    kernel='linear', 
    degree=3, 
    gamma='scale', 
    coef0=0.0, 
    shrinking=True, 
    probability=False, 
    tol=0.001, 
    cache_size=200, 
    class_weight=None, 
    verbose=False, 
    max_iter=-1, 
    decision_function_shape='ovr', 
    break_ties=False, 
    random_state=None
)

model.fit(X_train, y_train)

# Step 4: Save the trained model as a .pkl file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Step 5: Load the model from the .pkl file to verify
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Step 6: Evaluate the model to get accuracy and classification report
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model type: {type(loaded_model)}\n")
print(f"Model Parameters:\n {loaded_model.get_params()}\n")
print(f"Accuracy score of the testing data is: {accuracy}\n")
print(f"Classification Report:\n{report}")
