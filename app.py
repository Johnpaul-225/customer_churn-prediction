import streamlit as st
import pandas as pd
import joblib

# Load model & scaler
model = joblib.load("customer_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Churn Prediction")

st.title("Customer Churn Prediction")

# Sidebar inputs
age = st.slider("Age", 18, 80, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
location = st.selectbox("Location", ["New York", "Los Angeles", "Miami", "Chicago", "Houston"])
sub_len = st.slider("Subscription Length (Months)", 1, 36, 12)
bill = st.slider("Monthly Bill", 20.0, 150.0, 50.0)
usage = st.slider("Total Usage (GB)", 50, 1000, 300)

# Encoding (same as training)
gender_map = {"Male": 1, "Female": 0}
location_map = {
    "New York": 0,
    "Los Angeles": 1,
    "Miami": 2,
    "Chicago": 3,
    "Houston": 4
}

input_df = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_map[gender]],
    "Location": [location_map[location]],
    "Subscription_Length_Months": [sub_len],
    "Monthly_Bill": [bill],
    "Total_Usage_GB": [usage]
})

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"Customer will churn (Probability: {prob:.2f})")
    else:
        st.success(f"Customer will stay (Probability: {1-prob:.2f})")
