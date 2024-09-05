import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Load the saved model
loaded_model = pickle.load(open("model/trained_model.pkl", 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_array = np.asarray(input_data, dtype=np.float64).reshape(1, -1)
    result = loaded_model.predict(input_data_array)
    
    if result[0] == 1:
        return "The person is Diabetic", "⚠️", "danger"
    else:
        return "The person is not Diabetic", "✅", "success"

# Main function for Streamlit app
def main():
    # Set the page configuration (title, icon)
    st.set_page_config(page_title="Diabetes Prediction Application", page_icon="🩺", layout="centered")
    
    # Title and header
    st.title('🩺 Diabetes Prediction Application')
    
    # Add banner or image for visual appeal
    img = Image.open("diabetes_banner.png")
    st.image(img, use_column_width=True)
    
    st.markdown("### Please provide the following details to predict diabetes:")
    
    # Organizing input fields into two columns
    col1, col2 = st.columns(2)
    
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies (Range: 0-20)", min_value=0, max_value=20)
        Glucose = st.number_input("Glucose Level (mg/dL) (Range: 0-300)", min_value=0, max_value=300)
        BloodPressure = st.number_input("Blood Pressure (mm Hg) (Range: 0-200)", min_value=0, max_value=200, step=1)
        SkinThickness = st.number_input("Skin Thickness (mm) (Range: 0-100)", min_value=0, max_value=100, step=1)
    
    with col2:
        Insulin = st.number_input("Insulin Level (mcU/mL) (Range: 0-900)", min_value=0, max_value=900, step=1)
        BMI = st.number_input("BMI (Range: 0.0-70.0)", min_value=0.0, max_value=70.0, step=0.1)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function (Range: 0.0-3.0)", min_value=0.0, max_value=3.0, step=0.01)
        Age = st.number_input("Age (years) (Range: 0-120)", min_value=0, max_value=120, step=1)
    
    # Prediction button and result display
    if st.button('🔍 Get Diagnosis'):
        if any(field == '' for field in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please fill in all the fields correctly!")
        else:
            diagnosis, icon, alert_type = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            st.markdown(f"### {icon} {diagnosis}")
    
    # Sample data section
    st.markdown("***")
    st.markdown("""
    ### Sample Data:
    - **Non-Diabetic Person**: 4, 110, 92, 32, 88, 31.0, 0.248, 26
    - **Diabetic Person**: 6, 150, 80, 35, 150, 32.0, 0.6, 50
    """)
    
    # Explanation of fields
    with st.expander("ℹ️ Explanation of Input Fields"):
        st.markdown("""
        - **Pregnancies**: Number of times pregnant (0 to 20).
        - **Glucose**: Plasma glucose concentration (0 to 300 mg/dL).
        - **Blood Pressure**: Diastolic blood pressure (0 to 200 mm Hg).
        - **Skin Thickness**: Triceps skin fold thickness (0 to 100 mm).
        - **Insulin**: 2-hour serum insulin (0 to 900 mcU/mL).
        - **BMI**: Body Mass Index (0.0 to 70.0).
        - **Diabetes Pedigree Function**: Likelihood of diabetes based on family history (0.0 to 3.0).
        - **Age**: Age in years (0 to 120).
        """)
    
    # Custom CSS to improve design
    st.markdown("""
    <style>
        body {
            background-color: #f4f4f9;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            margin-top: 20px;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }
        .stAlert {
            background-color: #f8d7da;
            color: #721c24;
            border-color: #f5c6cb;
        }
        .stAlert-success {
            background-color: #d4edda;
            color: #155724;
            border-color: #c3e6cb;
        }
        .stAlert-warning {
            background-color: #fff3cd;
            color: #856404;
            border-color: #ffeeba;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
