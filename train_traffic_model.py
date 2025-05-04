import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Judul aplikasi Streamlit
st.title("Traffic Prediction Model")

# Upload file CSV
uploaded_file = st.file_uploader("Upload Traffic Data CSV", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded:")
    st.dataframe(df.head())

    # Preprocessing
    if "DateTime" in df.columns:
        df = df.drop(columns=["DateTime"])

    # Encode 'Day of the week'
    df = pd.get_dummies(df, columns=["Day of the week"], drop_first=True)

    # Encode target 'Traffic Situation'
    le = LabelEncoder()
    df["Traffic Situation"] = le.fit_transform(df["Traffic Situation"])

    # Split fitur dan label
    X = df.drop(columns=["Traffic Situation"])
    y = df["Traffic Situation"]

    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Bangun model ANN
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation="relu"),
        Dense(32, activation="relu"),
        Dense(len(np.unique(y)), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Training model
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=1)

    # Simpan model dan preprocessing
    model.save("traffic_model.h5")
    joblib.dump(scaler, "scaler.save")
    joblib.dump(le, "label_encoder.save")

    st.success("Model training selesai!")

    # Tampilkan hasil
    st.write("Model telah disimpan sebagai 'traffic_model.h5', scaler dan label encoder juga disimpan.")
    
    # Prediksi contoh jika ada input
    st.subheader("Predict Traffic Situation")
    input_data = []
    for column in X.columns:
        value = st.number_input(f"Enter value for {column}", min_value=0, max_value=100)
        input_data.append(value)

    if st.button("Predict"):
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)
        predicted_class = le.inverse_transform(np.argmax(prediction, axis=1))
        st.write(f"Predicted Traffic Situation: {predicted_class[0]}")
