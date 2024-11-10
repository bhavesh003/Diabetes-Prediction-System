import streamlit as st
import pickle
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
import plotly.graph_objects as go

# Load the trained model
def load_model():
    with open('model/trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Mock training data for LIME (can be replaced with actual data)
def load_training_data():
    X_train = np.random.rand(100, 8)  # Mock data with 100 samples and 8 features
    return X_train

# Create LIME explainer
def get_lime_explainer(X_train):
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        mode='classification'
    )
    return explainer

# Explain the prediction with LIME
def explain_with_lime(model, explainer, instance):
    try:
        # Try using predict_proba for LIME explanation
        explanation = explainer.explain_instance(instance[0], model.predict_proba)
    except AttributeError:
        # If predict_proba is not available, fallback to predict
        st.warning("The model does not support probability prediction. Falling back to 'predict'.")
        explanation = explainer.explain_instance(instance[0], model.predict)
    return explanation

# Predict diabetes based on user input and get LIME explanation
def diabetes_prediction(inputs):
    model = load_model()
    inputs_array = np.array(inputs).reshape(1, -1)
    
    # Make prediction
    try:
        prediction_proba = model.predict_proba(inputs_array)
        prediction = np.argmax(prediction_proba, axis=1)  # Class with highest probability
    except AttributeError:
        prediction = model.predict(inputs_array)
    
    # Load training data and get explainer
    X_train = load_training_data()  
    explainer = get_lime_explainer(X_train)
    
    # Get LIME explanation for the prediction
    explanation = explain_with_lime(model, explainer, inputs_array)
    
    return prediction, explanation

# Function to classify risk level based on user inputs
def classify_risk_level(Glucose, BMI, Age):
    risk_score = 0  # Default risk score

    # Simple risk classification based on thresholds
    if Glucose > 180:
        risk_score += 40
    elif Glucose > 140:
        risk_score += 20

    if BMI > 30:
        risk_score += 30
    elif BMI > 25:
        risk_score += 15

    if Age > 60:
        risk_score += 30
    elif Age > 45:
        risk_score += 15

    risk_score = min(risk_score, 100)  # Ensure the score does not exceed 100

    if risk_score < 34:
        risk_level = 'Low Risk'
    elif risk_score < 67:
        risk_level = 'Medium Risk'
    else:
        risk_level = 'High Risk'

    return risk_level, risk_score

# Function to create a gauge chart for risk level
def create_risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "green"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': score}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=40, r=40, t=40, b=40))
    return fig

# Function to provide personalized suggestions based on user inputs
def get_personalized_suggestions(Glucose, BMI, Age):
    suggestions = ""

    if Glucose > 180:
        suggestions += "- **Glucose levels are high**: Consider a low-carb diet and regular glucose monitoring.\n"
    elif Glucose > 140:
        suggestions += "- **Glucose levels are slightly elevated**: Moderate your sugar intake and exercise regularly.\n"
    
    if BMI > 30:
        suggestions += "- **High BMI (Obesity)**: Aim for regular physical activity and a balanced, calorie-controlled diet.\n"
    elif BMI > 25:
        suggestions += "- **Elevated BMI (Overweight)**: Consider exercise and healthy eating to reduce weight.\n"
    
    if Age > 60:
        suggestions += "- **Age factor**: As age increases, it's important to maintain regular health check-ups and stay active.\n"
    elif Age > 45:
        suggestions += "- **Middle age**: Monitor health regularly and maintain a balanced lifestyle to reduce risk of diabetes.\n"

    return suggestions

def disclaimer():
    with st.expander("📜 Disclaimer", expanded=False):  # The user can expand/collapse this section
        st.markdown(""" 
        The predictions made by this app are based on a machine learning model trained on specific datasets and may not fully reflect your individual health status. The results are not guaranteed to be accurate and should not be solely relied upon for making health decisions. 

        While we strive to provide useful information, the app may not always be correct due to the limitations of the dataset and model used. Therefore, this application is not intended to replace professional medical advice, diagnosis, or treatment. Always consult your physician or other qualified health provider with any questions you may have regarding a medical condition. 

        Please do not make health decisions based solely on the results from this app. We encourage you to seek professional medical advice and care. 
        """)

# Modify main function to include risk classification and personalized suggestions
def main():
    # Set the page configuration (title, icon)
    st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

    # Title and header
    st.title('🩺 Diabetes Prediction Application')

    # Add banner or image for visual appeal
    st.image("diabetes_banner.png", use_column_width=True)
    st.markdown("---")

    st.markdown("<h3 style='text-align: center;'>Provide the following details to predict diabetes</h3>", unsafe_allow_html=True)

    st.markdown("---")

    # Organizing input fields into columns (keep the current input field setup)
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.slider("Select the number of pregnancies", min_value=0, max_value=20, step=1, key="pregnancies", help="Total number of pregnancies the patient has had.")
        Glucose = st.slider("Select glucose level", min_value=0, max_value=300, step=1, key="glucose", help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test.")
        BloodPressure = st.slider("Select blood pressure", min_value=0, max_value=200, step=1, key="blood_pressure", help="Diastolic blood pressure (mm Hg).")
        SkinThickness = st.slider("Select skin thickness", min_value=0, max_value=100, step=1, key="skin_thickness", help="Triceps skin fold thickness (mm).")

    with col2:
        Insulin = st.slider("Select insulin level", min_value=0, max_value=900, step=1, key="insulin", help="2-Hour serum insulin (mu U/ml).")
        BMI = st.slider("Select BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1, key="bmi", help="Body mass index (weight in kg/(height in m)^2).")
        DiabetesPedigreeFunction = st.slider("Select diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.0, step=0.01, key="diabetes_pedigree", help="Diabetes pedigree function (a function which scores the likelihood of diabetes based on family history).")
        Age = st.slider("Select age", min_value=0, max_value=120, step=1, key="age", help="Age (years).")


    # Prediction button and result display
    if st.button('🔍 Get Diagnosis'):
        with st.spinner('Predicting...'):
            # Call the function for prediction and display result
            prediction, explanation = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            st.markdown(f"### Diagnosis: {'Diabetic ⚠️' if prediction[0] == 1 else 'Not Diabetic ✅'}")

            st.markdown("---")
            
            # Compute risk score and classify
            risk_level, risk_score = classify_risk_level(Glucose, BMI, Age)
            st.markdown(f"### Risk Level:")

            # Display risk level gauge chart
            fig = create_risk_gauge(risk_score)
            st.plotly_chart(fig)
            st.markdown(f"<h4 style='text-align: center;'>{risk_level}</h4>", unsafe_allow_html=True)

            st.markdown("---")

            # Provide personalized suggestions based on the user inputs
            suggestions = get_personalized_suggestions(Glucose, BMI, Age)
            if suggestions:
                # Only display if there are suggestions
                st.markdown("### Suggestions:")
                st.markdown(suggestions)
            
            # Display LIME explanation as a pie chart
            feature_importance = explanation.as_list()
            feature_names, feature_values = zip(*feature_importance)

            # Normalize the values to be positive (since pie charts only show positive values)
            normalized_values = [abs(val) for val in feature_values]

            st.markdown("---")
    
            # Create pie chart for feature importance
            fig_exp = go.Figure(data=[go.Pie(labels=feature_names, values=normalized_values, hole=0.3)])
            
            st.markdown(f"### Factors Impacting Your Diagnosis:")
            # Show pie chart
            st.plotly_chart(fig_exp)

    disclaimer()

if __name__ == "__main__":
    main()
