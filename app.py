import streamlit as st
import joblib
import numpy as np
import os
import requests

# Load the trained model
model_url = "https://raw.githubusercontent.com/your-username/your-repo/main/linear_regression_model.pkl"
model_path = "linear_regression_model.pkl"
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
model = joblib.load(model_path)

# Streamlit UI
st.title("Accomodation Price Prediction App")

st.write("Enter the property details below:")

# User Inputs
bedroom = st.number_input("Number of Bedrooms", min_value=1.0, max_value=10.0, step=1.0)
bathroom = st.number_input("Number of Bathrooms", min_value=1.0, max_value=5.0, step=1.0)

# Binary features (0 or 1)
features = {
    "Pool_View": st.checkbox("Pool View"),
    "Villa": st.checkbox("Villa"),
    "Amazing_Pool": st.checkbox("Amazing Pool"),
    "Golfing": st.checkbox("Golfing"),
    "Surfing": st.checkbox("Surfing"),
    "Tropical": st.checkbox("Tropical"),
    "Guest_House": st.checkbox("Guest House"),
    "Ocean_View": st.checkbox("Ocean View"),
    "Beachfront": st.checkbox("Beachfront"),
    "View": st.checkbox("View"),
    "Residential": st.checkbox("Residential"),
    "Jungle_View": st.checkbox("Jungle View"),
    "Amazing_View": st.checkbox("Amazing View"),
    "Island_Life": st.checkbox("Island Life"),
    "Rice_Paddy_View": st.checkbox("Rice Paddy View"),
    "Style": st.checkbox("Style"),
}

# Convert checkboxes to 0 or 1
feature_values = [1 if features[key] else 0 for key in features]

# Make Prediction
if st.button("Predict Price"):
    input_data = np.array([[bedroom, bathroom] + feature_values])
    prediction = model.predict(input_data)
    st.write(f"Estimated Price: $ {prediction[0]:,.2f}")
