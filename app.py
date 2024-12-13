import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the pre-trained model (ensure model.pkl is in the same folder as app.py)
model = joblib.load('model.pkl')

# Function to process and clean input data
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Set up the Streamlit app title and description
st.title('LinkedIn Usage Prediction')
st.write('This app predicts whether a person uses LinkedIn based on their demographic information.')

# User input for features
income = st.selectbox("Select Income Level", range(1, 10))
education = st.selectbox("Select Education Level", range(1, 9))
parent = st.radio("Are you a parent?", ["Yes", "No"])
married = st.radio("Are you married?", ["Yes", "No"])
female = st.radio("Are you female?", ["Yes", "No"])
age = st.number_input("Enter your age", min_value=0, max_value=100)

# Convert categorical inputs to binary (0 or 1) for machine learning
parent = 1 if parent == "Yes" else 0
married = 1 if married == "Yes" else 0
female = 1 if female == "Yes" else 0

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent],
    'married': [married],
    'female': [female],
    'age': [age]
})

# Use the model to make a prediction
prediction = model.predict(input_data)

# Display the prediction and probability
if prediction == 1:
    st.write("This person is predicted to use LinkedIn.")
else:
    st.write("This person is predicted not to use LinkedIn.")

# Show the probability of using LinkedIn
probability = model.predict_proba(input_data)[0][1]
st.write(f"The probability of using LinkedIn is: {probability:.2f}")