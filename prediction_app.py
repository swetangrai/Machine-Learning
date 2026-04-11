import streamlit as st
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle

# Load the model and preprocessors
model = keras.models.load_model('ann_churn_model.h5')
le = pickle.load(open('pickle/label_encoder.pkl', 'rb'))
ohe = pickle.load(open('pickle/onehot_encoder.pkl', 'rb'))
sc = pickle.load(open('pickle/standard_scaler.pkl', 'rb'))

# Streamlit UI
st.title('Customer Churn Prediction App')
st.markdown('Enter customer details below to predict the likelihood of churn.')

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider('Credit Score', min_value=300, max_value=850, value=600)
    geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
    age = st.slider('Age', min_value=18, max_value=100, value=30)
    balance = st.slider('Balance', min_value=0.0, max_value=250000.0, value=0.0, step=1000.0)
    has_cr_card = st.radio('Has Credit Card', ['No', 'Yes'])
    salary = st.slider('Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0, step=5000.0)

with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    tenure = st.slider('Tenure (years)', min_value=0, max_value=10, value=5)
    num_products = st.slider('Number of Products', min_value=1, max_value=4, value=1)
    is_active = st.radio('Is Active Member', ['No', 'Yes'])

# Convert radio buttons to 0/1
has_cr_card = 1 if has_cr_card == 'Yes' else 0
is_active = 1 if is_active == 'Yes' else 0

if st.button('Predict Churn'):
    # Preprocess inputs
    geo_encoded = le.transform([geography])[0]
    gender_encoded = ohe.transform([[gender]])[0]
    features = np.array([[credit_score, geo_encoded, age, tenure, balance, num_products, has_cr_card, is_active, salary, gender_encoded[0]]])
    features_scaled = sc.transform(features)

    # Make prediction
    prediction_prob = model.predict(features_scaled)[0][0]
    prediction = (prediction_prob > 0.5).astype(int)

    # Display result
    st.info(f'Churn Probability: {prediction_prob:.2f}')
    if prediction == 1:
        st.error('Prediction: Customer will churn')
    else:
        st.success('Prediction: Customer will not churn')