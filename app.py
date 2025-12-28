import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
from src.data_preprocessing import load_data # To get medians
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

# Title and Header
st.title("🏥 Diabetes Risk Prediction System")
st.markdown("Enter patient details below to estimate the risk of diabetes.")

# Load Model
@st.cache_resource
def get_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model.pkl')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = get_model()

if model is None:
    st.error("Model not found! Please train the model first by running `src/train_models.py`.")
else:
    # Sidebar or Main Input
    st.subheader("Patient Attributes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 140, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        
    with col2:
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 30)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age (years)", 0, 120, 30)

    # Preprocessing Logic (Zero handling)
    # We need medians. Loading data...
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data', 'diabetes.csv')
        df_orig = pd.read_csv(data_path)
        # Calc medians for 0-replacement
        medians = {}
        for col in ['Glucose', 'BloodPressure', 'BMI', 'Insulin']:
            medians[col] = df_orig[df_orig[col] != 0][col].median()
    except Exception as e:
        st.warning(f"Could not load data for medians: {e}. Using defaults.")
        medians = {'Glucose': 117, 'BloodPressure': 72, 'BMI': 32, 'Insulin': 100}

    # Input Dict
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    # Handle Zeros
    for col in ['Glucose', 'BloodPressure', 'BMI', 'Insulin']:
        if input_data[col] == 0:
            input_data[col] = medians[col]
            
    # Create DataFrame for model
    input_df = pd.DataFrame([input_data])
    
    # Sensitivity Threshold Slider
    st.divider()
    st.subheader("Model Configuration")
    threshold = st.slider("Risk Sensitivity Threshold", 0.0, 1.0, 0.5, 0.05, help="Lower threshold = Higher sensitivity (catches more cases but may increase false alarms).")
    
    # Predict
    if st.button("Predict Risk"):
        # Get probability
        prob = model.predict_proba(input_df)[0][1]
        
        # Apply Threshold
        prediction = 1 if prob >= threshold else 0
        
        st.divider()
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"**High Risk Detectd!** (Positive)")
            st.markdown(f"**Probability:** {prob*100:.1f}% (Threshold: {threshold})")
        else:
            st.success(f"**Low Risk** (Negative)")
            st.markdown(f"**Probability:** {prob*100:.1f}% (Threshold: {threshold})")
            
        st.info("Note: This is an AI-assisted prediction. Please consult a doctor for clinical diagnosis.")
