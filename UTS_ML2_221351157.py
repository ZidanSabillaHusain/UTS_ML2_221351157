import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model dan encoder
model = load_model("traffic_model.h5")
scaler = joblib.load("scaler.save")
label_encoder = joblib.load("label_encoder.save")

st.title("Prediksi Kondisi Lalu Lintas")

# Input fitur
temperature = st.slider("Temperature (°C)", 0, 50, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 10)
visibility = st.slider("Visibility (m)", 0, 10000, 2000)
dew_point = st.slider("Dew Point (°C)", -20, 40, 10)
solar_radiation = st.slider("Solar Radiation (MJ/m²)", 0, 5, 1)
rainfall = st.slider("Rainfall (mm)", 0, 50, 0)
snowfall = st.slider("Snowfall (cm)", 0, 20, 0)

# Pilih hari
day = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# One-hot encoding manual untuk 'Day of the week'
days = ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_encoded = [1 if d == day else 0 for d in days]  # drop_first=True diasumsikan Monday = base case

# Gabungkan semua input
input_data = np.array([[temperature, humidity, wind_speed, visibility,
                        dew_point, solar_radiation, rainfall, snowfall] + day_encoded])

# Normalisasi
input_scaled = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi Traffic"):
    pred = model.predict(input_scaled)
    pred_label = label_encoder.inverse_transform([np.argmax(pred)])
    st.success(f"Prediksi Kondisi Lalu Lintas: {pred_label[0]}")