
# 🩺 Diabetes Prediction System

This project is a machine learning-based web application that predicts whether a person is diabetic or not, based on medical parameters such as glucose levels, blood pressure, insulin levels, and more. The model was trained using a combination of Support Vector Machine (SVM), Random Forest, and Logistic Regression via stacking for better accuracy. The application is built using **Streamlit** for the front-end and Python for the back-end model training.

## 🔍 Project Overview

The Diabetes Prediction System uses data from the Pima Indians Diabetes Database. The data is processed and cleaned, and the model predicts diabetes based on various medical attributes.

The project includes:

- **Data Preprocessing**: Handling class imbalance using **SMOTE** and feature scaling.
- **Modeling**: A Stacking Classifier that combines SVM, Random Forest, and Logistic Regression models to improve prediction accuracy.
- **Web App**: A user-friendly interface built with **Streamlit** where users can input their medical data and get a prediction result (whether they are diabetic or not).

## 📊 Dataset

- The dataset used is the **Pima Indians Diabetes Database** from [Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database).
- The dataset consists of 8 medical predictor variables and 1 target variable (Outcome: 1 for diabetic, 0 for non-diabetic).

## 🚀 Features

- **Multiple ML Models**: SVM, Random Forest, Logistic Regression with hyperparameter tuning using GridSearchCV.
- **Stacking Classifier**: Combines the strength of multiple models to improve accuracy.
- **Streamlit Interface**: A clean and simple user interface for interacting with the model.
- **Model Persistence**: The best-performing model is saved using `pickle` and can be loaded for predictions in the app.

## 🛠️ Technologies Used

- **Python**: The programming language for data manipulation, model building, and integration with Streamlit.
- **scikit-learn**: For model training, cross-validation, and evaluation.
- **imbalanced-learn**: For handling the class imbalance problem in the dataset.
- **Streamlit**: A web application framework for deploying the model.
- **Pandas & NumPy**: For data preprocessing and manipulation.

## 📂 Project Structure

```
├── model/
│   └── trained_model.pkl      # Saved trained model
├── diabetes.csv               # Dataset file
├── app.py                     # Streamlit app script
├── train_diabetes_model.py     # Model training script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── diabetes_banner.png        # Optional banner image for the app UI
```

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (`diabetes.csv`) from Kaggle if it's not included, and place it in the project directory.

4. Train the model (if not already trained):
   ```bash
   python train_diabetes_model.py
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 🧑‍💻 Usage

1. Open the Streamlit app in your browser by following the link that appears in the terminal after running the command.
2. Enter the required medical data in the input fields.
3. Click the "Get Diagnosis" button to predict if the person is diabetic.
4. The model will display the prediction result and show a sample data section for non-diabetic and diabetic individuals.

## 📈 Model Performance

The stacking classifier achieved an **accuracy score** of approximately 90% on the testing set. Here's the classification report for the model:

```
               precision    recall  f1-score   support
        0         0.92       0.89      0.91       160
        1         0.88       0.92      0.90       140
    accuracy                            0.90       300
   macro avg       0.90       0.90      0.90       300
weighted avg       0.90       0.90      0.90       300
```
