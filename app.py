<<<<<<< HEAD
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
=======
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
>>>>>>> 02020ab96c33525822433807ce5b056223ab9645
