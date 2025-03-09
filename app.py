import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import requests

# Load the trained model
model_url = "https://raw.githubusercontent.com/aufabi/2024-Bali-Accomodations-Prediction/main/xgb_model.pkl"
model_path = "xgb_model.pkl"
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
model = joblib.load(model_path)

# Load the scaler
scaler_url = "https://raw.githubusercontent.com/aufabi/2024-Bali-Accomodations-Prediction/main/scaler.pkl"
scaler_path = "scaler.pkl"
if not os.path.exists(scaler_path):
    response = requests.get(scaler_url)
    with open(scaler_path, "wb") as f:
        f.write(response.content)
scaler = joblib.load(scaler_path)

# Streamlit UI
st.title("Accommodation Price Prediction App")
st.write("Enter the property details below:")

# User Inputs
travel_points = st.number_input("Travel Points", min_value=0.0, max_value=10.0, step=0.1)
stars = st.number_input("Stars", min_value=0.0, max_value=5.0, step=0.1)
users = st.number_input("Users", min_value=0, step=1)
num_of_features = st.number_input("Number of Features", min_value=0, step=1)

# Feature Lists
numerical_features = ['travel_points', 'stars', 'users', 'num_of_features']
one_hot_columns = ['beach', 'bar', 'massage', 'child_care', 'restaurant_show', 'bike_rent', 'car_rent', 
                   'rooftop', 'fitness', 'spa', 'inclusive', 'billyard', 'swimming_pool', 'kitchen', 'fishing']

# Collect feature values from user
feature_values = [int(st.checkbox(f)) for f in one_hot_columns]  # Convert checkbox to 0/1

# Create input dataframe
input_data = pd.DataFrame([[travel_points, stars, users, num_of_features] + feature_values], 
                          columns=numerical_features + one_hot_columns)

if st.button("Predict Price"):
    # Pisahkan fitur numerik dan kategorikal
    numerical_data = input_data[numerical_features].astype(float)  # Pastikan numerik
    categorical_data = input_data[one_hot_columns]  # One-hot tetap 0/1

    # Scale hanya fitur numerik
    numerical_scaled = scaler.transform(numerical_data.values.reshape(1, -1))
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_features)

    # Gabungkan fitur numerik yang sudah di-scale dengan fitur kategorikal
    input_scaled = pd.concat([numerical_scaled_df, categorical_data], axis=1)

    # **🔹 Pastikan fitur sesuai dengan urutan model**
    expected_features = model.feature_names_in_  # Fitur yang diharapkan model
    input_scaled = input_scaled[expected_features]  # Urutkan sesuai dengan model

    # **🔹 Cek apakah jumlah fitur sesuai**
    if input_scaled.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch: Expected {model.n_features_in_}, but got {input_scaled.shape[1]}")
    else:
        # **🔹 Prediksi harga**
        prediction = model.predict(input_scaled)
        st.write(f"Estimated Price: Rp {prediction[0]:,.2f}")
