# Diabetes Prediction Application

This project provides a machine learning-based web application for predicting the likelihood of diabetes using user-provided health metrics. The project involves building a prediction model using various machine learning algorithms, which is then integrated into a Streamlit app for easy accessibility and user interaction. The model also includes interpretability features using LIME for explaining predictions.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Explanation and Visualization](#explanation-and-visualization)
- [Accuracy and Evaluation](#accuracy-and-evaluation)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to predict whether a person has diabetes based on diagnostic measurements. It uses a machine learning model trained on the Pima Indians Diabetes Database. The project also incorporates a web interface using Streamlit, allowing users to input their health information and get instant predictions.

The machine learning model is built using a Stacking Classifier that combines Support Vector Machine (SVM), Random Forest, and Logistic Regression models. The app also provides personalized health suggestions based on glucose, BMI, and age inputs.

## Features

- **Diabetes Prediction**: Predicts the likelihood of diabetes based on user input.
- **LIME Explanation**: Provides an interpretable explanation of the model's decision using LIME (Local Interpretable Model-Agnostic Explanations).
- **Risk Classification**: Classifies the user's risk level (Low, Medium, High) based on specific health metrics (Glucose, BMI, Age).
- **Personalized Suggestions**: Offers personalized health suggestions to manage potential risk.
- **Interactive Visualizations**: Uses Plotly to display risk level as a gauge chart and LIME explanations as bar charts.

## Installation

### Prerequisites
- Python 3.8 or later
- [Streamlit](https://streamlit.io/)
- Required Python libraries (listed in `requirements.txt`)

### Steps to Install
1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhavesh003/diabetes-prediction-app.git
   cd diabetes-prediction-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Once the app is running, you can access it in your browser at `http://localhost:8501`.
2. Fill in the health information in the input fields (e.g., Pregnancies, Glucose, BMI).
3. Click the **Get Diagnosis** button to receive the prediction.
4. The result will display whether the user is predicted to have diabetes, along with risk classification and personalized health suggestions.

## Dataset

The dataset used for training the model is the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) and personal data collection. It contains approx 2000 observations of patients, with 8 diagnostic features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The target variable is `Outcome`, which is either 0 (non-diabetic) or 1 (diabetic).

## Model Training

### Training Process

1. **Data Preprocessing**: The data was preprocessed to handle missing values and was scaled using `StandardScaler`.
2. **SMOTE**: The class imbalance in the dataset was handled using the SMOTE (Synthetic Minority Over-sampling Technique) algorithm.
3. **Models**: Three machine learning models were trained:
   - Support Vector Machine (SVM)
   - Random Forest
   - Logistic Regression
4. **Stacking Classifier**: The final model combines the three models using a stacking classifier, with Logistic Regression as the meta-model.
5. **Hyperparameter Tuning**: GridSearchCV was used to find the optimal hyperparameters for each model.

### Training Script

The model was trained using the script `train_diabetes_model.py`, which includes the following:
- Hyperparameter tuning via GridSearchCV
- Model evaluation using accuracy and classification report
- Saving the trained stacking model to `trained_model.pkl`

## Explanation and Visualization

The app uses **LIME (Local Interpretable Model-Agnostic Explanations)** to explain the predictions. LIME helps visualize the impact of each feature on the prediction, making the model's decision transparent and understandable.

### Risk Classification

A risk level is calculated based on Glucose, BMI, and Age inputs, and visualized using a Plotly gauge chart.

### Personalized Suggestions

Based on the user’s health information, personalized health suggestions are provided to guide the user on managing their risk.

## Accuracy and Evaluation

The model was evaluated on a hold-out test set using accuracy score and classification report. Detailed evaluation metrics and hyperparameters can be found in `accuracy_report.py`.

To check the model's accuracy, you can run the script:
```bash
python accuracy_report.py
```

## Disclaimer

The predictions made by this app are based on a machine learning model trained on a specific dataset. These predictions are not meant to replace professional medical advice, diagnosis, or treatment. Always consult a healthcare provider for advice specific to your health condition.

## Contributing

Contributions are welcome! If you find any issues or would like to add improvements, please feel free to open a pull request or raise an issue.

1. Fork the repository.
2. Create your feature branch: `git checkout -b feature/feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature/feature-name`
5. Open a pull request.
