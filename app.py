# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Function to apply custom CSS for background image
def set_bg_img(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Use this function at the top of your main Streamlit script
set_bg_img("https://your-image-url.jpg")  # Replace with your actual image URL

# Function to load data
def load_data():
    data = pd.read_csv('heart.csv')  # Ensure the correct path to your CSV file
    return data

# Function to save the model
def save_model(model):
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model, file)

# Function to train the model
def train_model(data):
    X = data.drop('target', axis=1)  # Assuming 'target' is the label column
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Check if the model file exists and load or train model
if os.path.exists('trained_model.pkl'):
    model = pickle.load(open('trained_model.pkl', 'rb'))
else:
    data = load_data()
    model = train_model(data)
    save_model(model)

# Streamlit application
st.title('Program Deteksi Dini Penyakit Jantung')

with st.form("my_form"):
    st.write("Masukan Beberapa Gejala-Gejala Umum Pada Prediksi Penyakit Jantung:")
    age = st.number_input('Umur', min_value=1, max_value=120, value=30)
    sex = st.selectbox('Gender', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Nyeri Dada', options=[0, 1, 2, 3])
    trestbps = st.number_input('Tekanan Darah (in mm Hg)', value=120)
    chol = st.number_input('Kolesterol in mg/dl', value=200)
    fbs = st.selectbox('Kadar Gula Darah > 120 mg/dl', options=[0, 1])
    restecg = st.selectbox('Abnormal Jantung', options=[0, 1, 2])
    thalach = st.number_input('Denyut Jantung', value=120)
    exang = st.selectbox('Angina Koroner', options=[0, 1])
    oldpeak = st.number_input('Depresi Jantung', value=1.0)
    slope = st.selectbox('Respon Jantung Pada ST Segment', options=[0, 1, 2])
    ca = st.selectbox('Penyumbatan Arteri ', options=[0, 1, 2, 3])
    thal = st.selectbox('Genetik / Faktor Keturunan', options=[1, 2, 3])

    submitted = st.form_submit_button("Deteksi Sekarang")
    if submitted:
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = model.predict(input_data)
        st.write("Kemungkinan Anda Menderita Gejala Penyakit Jantung :", "Yes" if prediction[0] == 1 else "No")

st.write("Mohon Diisi sesuai dengan fakta agar hasilnya akurat")
