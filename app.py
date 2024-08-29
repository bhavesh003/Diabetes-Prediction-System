import numpy as np
import pickle
import streamlit as st

# loading the saved model 
loaded_model = pickle.load(open("model/trained_model.pkl",'rb'))

# creating a function for prediction 

def diabetes_prediction(input_data):
    
    # changing data to numpy array 
    input_data_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped =  input_data_array.reshape(1,-1)
    
    result = loaded_model.predict(input_data_reshaped)
    print("The prediction is : ",result)
    
    if (result[0] == 1):
        return "The Person is Diabetic"        
    else:
        return "The Person is not Diabetic"

def main():
    # giving a title 
    st.title('Diabetes Predictior')
    
    # getting the input data from input user
    
    Pregnancies = st.text_input("Number of Pregnancies : ")
    Glucose = st.text_input("Glucose level : ")
    BloodPressure = st.text_input("Blood Pressure value: ")
    SkinThickness = st.text_input("Measure of Skin Thickness : ")
    Insulin = st.text_input("Insulin level : ")
    BMI = st.text_input("BMI value : ")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value : ")
    Age= st.text_input("Age of person : ")
    
    
    # code for prediction 
    diagnosis = '' # null string 
    
    # creating a button for prediction 
    
    if st.button('Diagonasis Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age ])
        
    st.success(diagnosis)
    
    st.markdown("***")
    
    st.markdown("""
    
    ## Sample Data to fill: 
    
    - **Non Diabetic Person** : 4   110 92  32  88  31  0.248  26
    - **Diabetic Person** :   6      150     80     35    150    32    0.6    50
    
    """)
    
    st.markdown("""
    
    ## About the Data to be Filled

    - **Pregnancies**: This feature represents the number of times the individual has been pregnant.
      - **Range**: 0 to any positive integer (0+).
    
    - **Glucose**: Plasma glucose concentration measured after 2 hours in an oral glucose tolerance test.
      - **Range**: 0 to 300.
    
    - **BloodPressure**: Diastolic blood pressure in millimeters of mercury (mm Hg).
      - **Range**: 0 to 200.
    
    - **SkinThickness**: Triceps skin fold thickness in millimeters.
      - **Range**: 0 to any positive value (0+).
    
    - **Insulin**: 2-hour serum insulin level in micro units per milliliter (mu U/ml).
      - **Range**: 0 to any positive value.
    
    - **BMI (Body Mass Index)**: Calculated as weight in kilograms divided by the square of height in meters.
      - **Range**: 9 to 72.
    
    - **Diabetes Pedigree Function**: A score that indicates the likelihood of diabetes based on family history.
      - **Range**: 0.0 to 3.0.
    
    - **Age**: The age of the individual in years.
      - **Range**: 0 to any positive value (0+).
    
    """)
    
if __name__ == '__main__':
    main()
    
    
