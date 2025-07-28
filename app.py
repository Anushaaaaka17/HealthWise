# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and expected feature columns
model = joblib.load("models/heart_disease_model.pkl")
expected_features = joblib.load(
    "models/feature_names.pkl"
)  # Column names used during training

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")

st.markdown("Provide the following details to check your heart health risk:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["M", "F"])
cp = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120
)
chol = st.number_input(
    "Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200
)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
thalach = st.number_input(
    "Max Heart Rate Achieved", min_value=60, max_value=250, value=150
)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input(
    "Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0
)
slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Up", "Flat", "Down"])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", ["Normal", "Fixed", "Reversible"])

# Mapping user inputs to match training data
user_input = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": cp,
    "RestingBP": trestbps,
    "Cholesterol": chol,
    "FastingBS": 1 if fbs == "Yes" else 0,
    "RestingECG": restecg,
    "MaxHR": thalach,
    "ExerciseAngina": "Y" if exang == "Yes" else "N",
    "Oldpeak": oldpeak,
    "ST_Slope": slope,
    "CA": ca,
    "Thal": thal,
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# One-hot encoding (must match training)
input_df = pd.get_dummies(input_df)

# Reindex to match training feature columns
input_df = input_df.reindex(columns=expected_features, fill_value=0)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ You are at risk of heart disease. (Probability: {probability:.2f})"
        )
    else:
        st.success(
            f"✅ You are not at risk of heart disease. (Probability: {probability:.2f})"
        )

st.markdown("---")
st.caption("Model trained using RandomForestClassifier and health indicators dataset.")
