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

# Binary features (0 or 1)
features = {
    "Beach": st.checkbox("Beach"),
    "Bar": st.checkbox("Bar"),
    "Massage": st.checkbox("Massage"),
    "Child Care": st.checkbox("Child Care"),
    "Restaurant Show": st.checkbox("Restaurant Show"),
    "Bike Rent": st.checkbox("Bike Rent"),
    "Car Rent": st.checkbox("Car Rent"),
    "Rooftop": st.checkbox("Rooftop"),
    "Fitness": st.checkbox("Fitness"),
    "Spa": st.checkbox("Spa"),
    "Inclusive": st.checkbox("Inclusive"),
    "Billyard": st.checkbox("Billyard"),
    "Swimming Pool": st.checkbox("Swimming Pool"),
    "Kitchen": st.checkbox("Kitchen"),
    "Fishing": st.checkbox("Fishing"),
}

# Convert checkboxes to 0 or 1
feature_values = [1 if features[key] else 0 for key in features]

numerical_features = ['travel_points', 'stars', 'users', 'num_of_features']
one_hot_columns = ['beach', 'bar', 'massage', 'child_care',	'restaurant_show', 'bike_rent', 'car_rent',	
                   'rooftop', 'fitness', 'spa', 'inclusive', 'billyard', 'swimming_pool', 'kitchen', 'fishing']

input_df = pd.DataFrame([[travel_points, stars, users, num_of_features] + feature_values], 
                          columns=numerical_features + one_hot_columns)

# Make Prediction
if st.button("Predict Price"):
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])  # Apply standard scaling
    prediction = model.predict(input_df)
    st.write(f"Estimated Price: Rp {prediction[0]:,.2f}")
