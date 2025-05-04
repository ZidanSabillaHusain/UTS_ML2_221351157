import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

st.title("Traffic Prediction")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "DateTime" in df.columns:
        df = df.drop(columns=["DateTime"])

    df = pd.get_dummies(df, columns=["Day of the week"], drop_first=True)

    le = LabelEncoder()
    df["Traffic Situation"] = le.fit_transform(df["Traffic Situation"])

    X = df.drop(columns=["Traffic Situation"])
    y = df["Traffic Situation"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation="relu"),
        Dense(32, activation="relu"),
        Dense(len(np.unique(y)), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"Akurasi Model: {acc:.2f}")

    # Simpan model
    model.save("traffic_model.h5")
    joblib.dump(scaler, "scaler.save")
    joblib.dump(le, "label_encoder.save")

    st.write("Model dan encoder berhasil disimpan.")
