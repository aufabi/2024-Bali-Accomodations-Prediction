import streamlit as st
import joblib
import numpy as np
import os
import requests
import pandas as pd

# Load the trained model
model_url = "https://raw.githubusercontent.com/aufabi/2024-Bali-Accomodations-Prediction/main/random_forest_model.pkl"
model_path = "random_forest_model.pkl"
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
model = joblib.load(model_path)

# Define all possible locations (from training data)
locations = ['Nusa Dua Beach', 'Seminyak', 'Kuta', 'Denpasar Selatan', 'Tuban',
       'Legian', 'Denpasar Utara', 'Denpasar Barat', 'Sanur', 'Canggu',
       'Nusa Dua', 'Ubud', 'Jimbaran', 'Pecatu', 'Kintamani', 'Klungkung',
       'Denpasar Timur', 'Ketewel', 'Monkey Forest', 'Singaraja',
       'Payangan', 'Renon -  AT', 'Kedewatan', 'Lovina', 'Tanjung Benoa',
       'Candidasa', 'Bedugul', 'Poppies', 'Umalas', 'Kerobokan',
       'Tanah Lot', 'Tampaksiring', 'Sidakarya - AT', 'Tegallalang',
       'Negara', 'Baturiti', 'Sayan', 'Blahbatuh', 'Amed', 'Nusa Penida',
       'Ungasan', 'Mendoyo', 'Sanur Kaja - AT', 'Dauh Puri - AT',
       'Menjangan', 'Sanur Kauh - AT', 'Jatiluwih', 'Sebatu',
       'Padangsambian Kaja - AT', 'Amlapura', 'Mengwi', 'Kemenuh',
       'Marga', 'Nusa Lembongan', 'Gianyar Kota', 'Kutuh', 'Selemadeg',
       'Semarapura', 'Sukawati', 'Seririt', 'Tulamben', 'Pekutatan',
       'Tembok', 'Sukasada', 'Sidemen', 'Tabanan Kota', 'Kerambitan',
       'Taro', 'Keramas', 'Kelating', 'Pemuteran', 'Gerokgak',
       'Gilimanuk', 'Batubulan', 'Padangbai', 'Saba', 'Nusa Ceningan',
       'Plaga', 'Batuan', 'Selat', 'Munduk', 'Sawan', 'Cemagi', 'Cepaka',
       'Pejeng Kangin', 'Jembrana', 'Abiansemal', 'Melaya', 'Abang',
       'Banjar', 'Umeanyar', 'Bangli', 'Tejakula', 'Soka',
       'Dauh Puri Kauh - AT', 'Panjer - AT', 'Penebel', 'Besakih',
       'Tembuku', 'Pupuan', 'Karangasem', 'Kubutambahan', 'Pemogan - AT',
       'Mambal', 'Busung Biu', 'Medewi', 'Manggis', 'Tirta Gangga',
       'Balian', 'Belimbing', 'Sesetan - AT', 'Kerta', 'Bali', 'Uluwatu',
       'Jayagiri', 'Susut', 'Badung', 'Celuk']

# Streamlit UI
st.title("Accommodation Price Prediction App")

st.write("Enter the property details below:")

# Numerical Inputs
travel_points = st.number_input("Travel Points (0-5)", min_value=0.0, max_value=5.0, step=0.1)
stars = st.number_input("Stars (0-5)", min_value=0.0, max_value=5.0, step=0.1)
users = st.number_input("Number of Users", min_value=0, step=1)
num_of_features = st.number_input("Number of Features", min_value=0, step=1)

# Location Dropdown
selected_location = st.selectbox("Select Location", locations)

# Binary Features (Checkboxes)
binary_features = {
    "beach": st.checkbox("Beach"),
    "bar": st.checkbox("Bar"),
    "massage": st.checkbox("Massage"),
    "child_care": st.checkbox("Child Care"),
    "restaurant_show": st.checkbox("Restaurant Show"),
    "bike_rent": st.checkbox("Bike Rent"),
    "car_rent": st.checkbox("Car Rent"),
    "rooftop": st.checkbox("Rooftop"),
    "fitness": st.checkbox("Fitness"),
    "spa": st.checkbox("Spa"),
    "inclusive": st.checkbox("Inclusive"),
    "billyard": st.checkbox("Billiard"),
    "swimming_pool": st.checkbox("Swimming Pool"),
    "kitchen": st.checkbox("Kitchen"),
    "fishing": st.checkbox("Fishing")
}

# Create one-hot encoded location features
location_features = {loc: 1 if loc == selected_location else 0 for loc in locations}

# Combine all features into a single input array
feature_values = [travel_points, stars, users, num_of_features] + \
                 list(binary_features.values()) + list(location_features.values())

# Load StandardScaler
scaler_url = "https://raw.githubusercontent.com/aufabi/2024-Bali-Accomodations-Prediction/main/scaler.pkl"
scaler_path = "scaler.pkl"
if not os.path.exists(scaler_path):
    response = requests.get(scaler_url)
    with open(scaler_path, "wb") as f:
        f.write(response.content)
scaler = joblib.load(scaler_path)

# Apply StandardScaler
scaled_input = scaler.transform([feature_values])

# Make Prediction
if st.button("Predict Price"):
    prediction = model.predict(scaled_input)
    st.write(f"Estimated Price: Rp {prediction[0]:,.2f}")
