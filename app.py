import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="River Nutrient Predictor", page_icon="🌊")

st.title("🌊 River Nutrient Level Prediction")
st.write("Predict Dissolved Inorganic Nitrogen")

# -----------------------------
# Load Model & Encoder
# -----------------------------
if not os.path.exists("nutrient_model.pkl"):
    st.error("❌ Model not found. Train the model first.")
    st.stop()

model = pickle.load(open("nutrient_model.pkl", "rb"))
encoder = pickle.load(open("region_encoder.pkl", "rb"))

# -----------------------------
# User Inputs
# -----------------------------
region = st.selectbox(
    "Region",
    encoder.classes_
)

conductivity = st.number_input(
    "Water Electrical Conductivity", min_value=0.0
)

ph = st.number_input(
    "Water pH", min_value=0.0, max_value=14.0
)

phosphorus = st.number_input(
    "Dissolved Inorganic Phosphorus", min_value=0.0
)

# -----------------------------
# Encode Inputs
# -----------------------------
region_encoded = encoder.transform([region])[0]

X = np.array([[
    region_encoded,
    conductivity,
    ph,
    phosphorus
]])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Nitrogen Level"):
    prediction = model.predict(X)[0]
    st.success(f"🌱 Predicted Nitrogen Level: {prediction:.2f}")
