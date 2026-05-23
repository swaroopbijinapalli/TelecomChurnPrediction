import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load("telecom_churn_model.pkl")
scaler = joblib.load("scaler.pkl")  # This is critical!

# Feature label mapping
feature_labels = {
    'total_og_mou_8': 'Total Outgoing Minutes (August)',
    'total_ic_mou_8': 'Total Incoming Minutes (August)',
    'roam_ic_mou_8': 'Roaming Incoming Minutes (August)',
    'loc_og_mou_8': 'Local Outgoing Minutes (August)',
    'total_rech_amt_8': 'Total Recharge Amount (August)',
    'roam_og_mou_8': 'Roaming Outgoing Minutes (August)',
    'last_day_rch_amt_8': 'Last Day Recharge Amount (August)',
    'arpu_8': 'Average Revenue Per User (August)',
    'max_rech_amt_8': 'Maximum Recharge Amount (August)',
    'loc_ic_t2m_mou_8': 'Local Incoming T2M Minutes (August)',
    'offnet_mou_8': 'Off-network Minutes of Usage (August)',
}

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #dbe6f6);
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Telecom Churn Prediction App")
st.markdown("Use this tool to predict whether a telecom customer is likely to churn based on usage patterns.")

# Collect input from user
st.header("📊 Customer Usage Inputs")

user_input = {}
cols = st.columns(2)
for i, (key, label) in enumerate(feature_labels.items()):
    with cols[i % 2]:
        user_input[key] = st.number_input(label, min_value=0.0, format="%.2f", step=0.1, key=key)

# Prediction button
if st.button("🔍 Predict Churn"):
    # Convert dict to DataFrame
    input_df = pd.DataFrame([user_input])

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of churn

    # Display result
    st.markdown("### 🧾 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ The customer is likely to **CHURN**.")
    else:
        st.success("✅ The customer is likely to **STAY**.")
    
    st.info(f"📈 Churn Probability: **{probability * 100:.2f}%**")

st.markdown("---")
st.markdown("Made with ❤️ by [Chinni Anjaneyulu](https://github.com/AnjaneyuluChinni)")
