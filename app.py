import streamlit as st
import joblib
import numpy as np
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

numerical_features = ['travel_points', 'stars', 'users', 'num_of_features']
one_hot_columns = ['beach', 'bar', 'massage', 'child_care',	'restaurant_show', 'bike_rent', 'car_rent',	
                   'rooftop', 'fitness', 'spa', 'inclusive', 'billyard', 'swimming_pool', 'kitchen', 'fishing']
feature_values = [st.checkbox(f) for f in one_hot_columns]

input_data = pd.DataFrame([[travel_points, stars, users, num_of_features] + feature_values], 
                          columns=numerical_features + one_hot_columns)

if st.button("Predict Price"):
    numerical_data = input_data[numerical_features]
    other_features = input_data.drop(columns=numerical_features)
    
    numerical_scaled = scaler.transform(numerical_data)
    numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_features)

    input_scaled = pd.concat([numerical_scaled_df, other_features], axis=1)
    prediction = model.predict(input_scaled)

    st.write(f"Estimated Price: Rp {prediction[0]:,.2f}")
