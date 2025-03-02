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

# Load the scaler
scaler_url = "https://raw.githubusercontent.com/aufabi/2024-Bali-Accomodations-Prediction/main/scaler.pkl"
scaler_path = "scaler.pkl"
if not os.path.exists(scaler_path):
    response = requests.get(scaler_url)
    with open(scaler_path, "wb") as f:
        f.write(response.content)
scaler = joblib.load(scaler_path)

# List of possible locations (from your one-hot encoding)
location_options = np.array(['Nusa Dua Beach', 'Seminyak', 'Kuta', 'Denpasar Selatan', 'Tuban',
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
       'Jayagiri', 'Susut', 'Badung', 'Celuk'])

# Streamlit UI
st.title("Accommodation Price Prediction App")
st.write("Enter the accommodation details below:")

# User Inputs
travel_points = st.number_input("Travel Points", min_value=0.0, max_value=5.0, step=0.1)
stars = st.number_input("Stars", min_value=0.0, max_value=5.0, step=0.1)
users = st.number_input("Number of Users", min_value=1, step=1)
num_of_features = st.number_input("Number of Features", min_value=1, step=1)

# Dropdown for location
selected_location = st.selectbox("Select Location", location_options)

# One-hot encode the location
location_encoded = [1 if loc == selected_location else 0 for loc in location_options]

# Binary features
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
    "billyard": st.checkbox("Billyard"),
    "swimming_pool": st.checkbox("Swimming Pool"),
    "kitchen": st.checkbox("Kitchen"),
    "fishing": st.checkbox("Fishing")
}

# Convert binary features to 0 or 1
binary_values = [1 if binary_features[key] else 0 for key in binary_features]

# Combine all inputs
input_data = np.array([[travel_points, stars, users, num_of_features] + binary_values + location_encoded])

# Scale the input data
scaled_input = scaler.transform(input_data)

# Make Prediction
if st.button("Predict Price"):
    prediction = model.predict(scaled_input)
    st.write(f"Estimated Price: Rp {prediction[0]:,.2f}")
