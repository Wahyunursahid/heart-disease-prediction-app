# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

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
st.title('Heart Disease Prediction Application')

with st.form("my_form"):
    st.write("Enter the following details to predict heart disease:")
    age = st.number_input('Age', min_value=1, max_value=120, value=30)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', value=120)
    chol = st.number_input('Serum Cholesterol in mg/dl', value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', value=120)
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('ST depression induced by exercise relative to rest', value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-3) Colored by Fluoroscopy', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', options=[1, 2, 3])

    submitted = st.form_submit_button("Submit")
    if submitted:
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = model.predict(input_data)
        st.write("The predicted probability of heart disease is:", "Yes" if prediction[0] == 1 else "No")

st.write("Please fill out the information accurately to predict heart disease.")
