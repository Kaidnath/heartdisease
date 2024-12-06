import streamlit as st
import pickle
import pandas as pd
import os

# Global variable for model path
MODEL_PATH = "heartdisease.pkl"

@st.cache_resource
def load_model():
    """Load the prediction model and cache it."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Error: The model file '{MODEL_PATH}' was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

def predict_heart_disease(model, age, sex, cp, trestbps, chol, fbs, restecg, 
                         thalach, exang, oldpeak, slope, ca, thal):
    """Make prediction using the loaded model."""
    if model is None:
        st.error("Model not loaded. Please ensure the model file exists and is valid.")
        return None
    
    try:
        prediction = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]])
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    # Page configuration
    st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
    
    # Title and image
    st.title("Heart Disease Prediction App")
    if os.path.exists("image11.png"):
        st.image("image11.png", width=600)
    
    # Load model at startup
    model = load_model()
    if model is None:
        st.warning("Please ensure the model file 'heartdisease1.pkl' is in the same directory as this script.")
        return
    
    st.write("Enter patient information to predict heart disease risk")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["0 (Female)", "1 (Male)"])
        cp = st.selectbox("Chest Pain Type", 
                         ["1 (Typical Angina)", "2 (Atypical Angina)", 
                          "3 (Non-anginal Pain)", "4 (Asymptomatic)"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["0 (No)", "1 (Yes)"])
        restecg = st.selectbox("Resting ECG", 
                              ["0 (Normal)", "1 (ST-T Wave Abnormality)", 
                               "2 (Left Ventricular Hypertrophy)"])
    
    with col2:
        thalach = st.number_input("Maximum Heart Rate", 70, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["0 (No)", "1 (Yes)"])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0, 0.1)
        slope = st.selectbox("ST Segment Slope", ["1 (Up)", "2 (Flat)", "3 (Down)"])
        ca = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3"])
        thal = st.selectbox("Thalassemia", ["1", "2", "3"])
    
    # Process inputs
    sex = int(sex[0])
    cp = int(cp[0])
    fbs = int(fbs[0])
    restecg = int(restecg[0])
    exang = int(exang[0])
    slope = int(slope[0])
    ca = int(ca[0])
    thal = int(thal[0])
    
    if st.button("Predict"):
        result = predict_heart_disease(model, age, sex, cp, trestbps, chol, fbs, 
                                     restecg, thalach, exang, oldpeak, 
                                     slope, ca, thal)
        if result is not None:
            if result == 1:
                st.error("Warning: High Risk of Heart Disease")
                st.write("Please consult with a healthcare professional for further evaluation.")
            else:
                st.success("Low Risk of Heart Disease")
                st.write("Maintain a healthy lifestyle and regular check-ups.")
                
        # Display input summary
        st.subheader("Input Summary")
        data = {
            'Parameter': ['Age', 'Sex', 'Chest Pain Type', 'Blood Pressure', 
                         'Cholesterol', 'High Blood Sugar', 'ECG Results',
                         'Max Heart Rate', 'Exercise Angina', 'ST Depression',
                         'ST Slope', 'Major Vessels', 'Thalassemia'],
            'Value': [age, sex, cp, trestbps, chol, fbs, restecg, 
                     thalach, exang, oldpeak, slope, ca, thal]
        }
        st.table(pd.DataFrame(data))

if __name__ == '__main__':
    main()
