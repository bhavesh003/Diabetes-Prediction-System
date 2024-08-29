import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open("model/trained_model.pkl", 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    # Convert input data to numpy array and reshape it
    input_data_array = np.asarray(input_data, dtype=np.float64).reshape(1, -1)
    
    # Make prediction
    result = loaded_model.predict(input_data_array)
    
    if result[0] == 1:
        return "The Person is Diabetic"
    else:
        return "The Person is not Diabetic"

# Main function for Streamlit app
def main():
    # Set title
    st.title('Diabetes Prediction Application')
    
    st.markdown("""
    ### Please provide the following details to predict diabetes:
    """)
    
    # Input fields with validation
    Pregnancies = st.number_input("Number of Pregnancies:", min_value=0, max_value=20, step=1, format="%d")
    Glucose = st.number_input("Glucose Level (mg/dL):", min_value=0, max_value=300, step=1, format="%d")
    BloodPressure = st.number_input("Blood Pressure (mm Hg):", min_value=0, max_value=200, step=1, format="%d")
    SkinThickness = st.number_input("Skin Thickness (mm):", min_value=0, max_value=100, step=1, format="%d")
    Insulin = st.number_input("Insulin Level (mcU/mL):", min_value=0, max_value=900, step=1, format="%d")
    BMI = st.number_input("BMI:", min_value=0.0, max_value=70.0, step=0.1, format="%.1f")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function:", min_value=0.0, max_value=3.0, step=0.01, format="%.2f")
    Age = st.number_input("Age (years):", min_value=0, max_value=120, step=1, format="%d")
    
    # Prediction button and result display
    diagnosis = ''
    
    if st.button('Diagnosis Test Result'):
        if any(field == '' for field in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            st.warning("Please fill in all the fields correctly!")
        else:
            diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
            st.success(diagnosis)
    
    # Display sample input for users
    st.markdown("***")
    st.markdown("""
    ### Sample Data:
    - **Non-Diabetic Person**: 4, 110, 92, 32, 88, 31.0, 0.248, 26
    - **Diabetic Person**: 6, 150, 80, 35, 150, 32.0, 0.6, 50
    """)
    
    st.markdown("***")
    
    # Explanation of data fields
    st.markdown("""
    ### Explanation of Input Fields:
    - **Pregnancies**: Number of times pregnant (0 to 20).
    - **Glucose**: Plasma glucose concentration (0 to 300 mg/dL).
    - **Blood Pressure**: Diastolic blood pressure (0 to 200 mm Hg).
    - **Skin Thickness**: Triceps skin fold thickness (0 to 100 mm).
    - **Insulin**: 2-hour serum insulin (0 to 900 mcU/mL).
    - **BMI**: Body Mass Index (0.0 to 70.0).
    - **Diabetes Pedigree Function**: Likelihood of diabetes based on family history (0.0 to 3.0).
    - **Age**: Age in years (0 to 120).
    """)
    
if __name__ == '__main__':
    main()
