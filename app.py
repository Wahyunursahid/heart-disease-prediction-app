import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your trained model (make sure to save it locally first using pickle)
model = pickle.load(open('/heart.csv', 'rb'))

# Title of your application
st.title('Heart Disease Prediction Application')

# Collecting user input features into dataframe
with st.form("my_form"):
    st.write("Enter the following details to predict heart disease:")
    age = st.number_input('Age', min_value=1, max_value=120, value=30)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', value=120)
    chol = st.number_input('Serum Cholestrol in mg/dl', value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', value=120)
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('ST depression induced by exercise relative to rest', value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-3) Colored by Fluoroscopy', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', options=[1, 2, 3])

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        prediction = model.predict(input_data)
        st.write("The predicted probability of heart disease is:", "Yes" if prediction[0] == 1 else "No")

# Optional: Add some explanations about the input fields or the method
st.write("Please fill out the information accurately to predict heart disease.")
