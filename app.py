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
set_bg_img("data:image/jpeg;base64,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")  # Replace with your actual image URL

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
