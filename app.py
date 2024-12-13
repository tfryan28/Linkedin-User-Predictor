import pandas as pd
import numpy as np
import streamlit as st
import joblib

# Load the pre-trained model (ensure model.pkl is in the same folder as the app)
model = joblib.load('model.pkl')

# Title of the app
st.title('LinkedIn Usage Prediction')

# Description of the app
st.write("This app predicts whether a person is likely to use LinkedIn based on certain features.")

# Take input from user for each feature
income = st.slider('Income', min_value=1, max_value=9, value=5)
education = st.slider('Education Level', min_value=1, max_value=8, value=4)
parent = st.radio('Are you a parent?', options=[0, 1])
married = st.radio('Are you married?', options=[0, 1])
female = st.radio('Are you female?', options=[0, 1])
age = st.slider('Age', min_value=18, max_value=99, value=25)

# Create a DataFrame with the user input
user_input = pd.DataFrame({
    'income': [income],
    'education': [education],
    'parent': [parent],
    'married': [married],
    'female': [female],
    'age': [age]
})

# Use the model to predict whether the person uses LinkedIn
prediction = model.predict(user_input)
probability = model.predict_proba(user_input)

# Show the prediction
if prediction[0] == 1:
    st.write("**Prediction**: The user is likely to use LinkedIn.")
else:
    st.write("**Prediction**: The user is unlikely to use LinkedIn.")

# Show the probability
st.write(f"**Probability of using LinkedIn**: {round(probability[0][1], 2)}")