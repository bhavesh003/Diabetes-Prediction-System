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
        training_data=np.array(X_train),
        feature_names=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        mode='classification'
    )
    return explainer

# Explain the prediction with LIME
def explain_with_lime(model, explainer, instance):
    instance = np.array(instance).reshape(1, -1)  # Ensure it's 2D
    explanation = explainer.explain_instance(instance[0], model.predict_proba)
    return explanation

# Predict diabetes based on user input and get LIME explanation
def diabetes_prediction(inputs):
    model = load_model()
    inputs_array = np.array(inputs).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(inputs_array)
    
    # Load training data and get explainer
    X_train = load_training_data()  
    explainer = get_lime_explainer(X_train)
    
    # Get LIME explanation for the prediction
    explanation = explain_with_lime(model, explainer, inputs)
    
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
        suggestions += "- **Age factor**: As age increases, its important to maintain regular health check-ups and stay active.\n"
    elif Age > 45:
        suggestions += "- **Middle age**: Monitor health regularly and maintain a balanced lifestyle to reduce risk of diabetes.\n"

    return suggestions


def disclaimer():
    st.markdown("""
    **Disclaimer:**
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
        Pregnancies = st.slider("Select the number of pregnancies", min_value=0, max_value=20, step=1, key="pregnancies")
        Glucose = st.slider("Select glucose level", min_value=0, max_value=300, step=1, key="glucose")
        BloodPressure = st.slider("Select blood pressure", min_value=0, max_value=200, step=1, key="blood_pressure")
        SkinThickness = st.slider("Select skin thickness", min_value=0, max_value=100, step=1, key="skin_thickness")

    with col2:
        Insulin = st.slider("Select insulin level", min_value=0, max_value=900, step=1, key="insulin")
        BMI = st.slider("Select BMI", min_value=0.0, max_value=70.0, value=0.0, step=0.1, key="bmi")
        DiabetesPedigreeFunction = st.slider("Select diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.0, step=0.01, key="diabetes_pedigree")
        Age = st.slider("Select age", min_value=0, max_value=120, step=1, key="age")

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
            if suggestions:  # Only display if there are suggestions
                st.markdown("### Personalized Health Suggestions:")
                st.markdown(suggestions)

            st.markdown("---")

            st.markdown("### Analysis of Prediction:")
            exp_list = explanation.as_list()

            # Create a Plotly bar chart for the explanation
            features = [x[0] for x in exp_list]
            weights = [x[1] for x in exp_list]
            total_weight = sum([abs(w) for w in weights])

            percentages = [(abs(w) / total_weight) * 100 for w in weights]

            fig = go.Figure([go.Bar(x=features, y=percentages, text=[f'{p:.2f}%' for p in percentages],hoverinfo='text', marker=dict(color='rgb(26, 118, 255)'))])

            fig.update_layout(
                xaxis_title='Features',
                yaxis_title='Contribution Percentage',
                template='plotly_white',
                hovermode='closest',
                height=480,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig)

            st.markdown("---")
            
            disclaimer()

    # Custom CSS to improve design
    st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 8px 16px;
            margin-top: 20px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stMarkdown h4 {
            color: #333;
            margin-bottom: 10px;
        }
        .stSlider {
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
