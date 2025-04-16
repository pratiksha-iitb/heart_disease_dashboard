import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")  # Load saved MinMaxScaler

# Set dashboard title
st.title("❤️ Heart Disease Prediction Dashboard")

# Sidebar for user input
st.sidebar.header("Enter Patient Details")

# Collect user inputs
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=500, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=5.0, value=1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Prediction function
if st.sidebar.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Apply MinMaxScaler transformation
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[0][1]  # Probability of heart disease

    # Display result
    if prediction == 1:
        st.error(f"⚠️ High Risk: {prediction_proba:.2f} probability of heart disease.")
    else:
        st.success(f"✅ Low Risk: {prediction_proba:.2f} probability of heart disease.")
