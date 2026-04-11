### Detailed Instructions for AI Agent: Create Prediction Script (Streamlit App)

**Objective:** Create a Streamlit app that provides a user interface for predicting customer churn using the trained ANN model. The app should take user inputs, preprocess them using saved encoders/scalers, make predictions, and display results.

**Requirements:**
- Use Streamlit for the web app.
- Load the saved model (`ann_churn_model.h5`), label encoder, one-hot encoder, and standard scaler from pickle files.
- Provide input fields for all features: CreditScore (number), Geography (select: France, Germany, Spain), Gender (select: Male, Female), Age (number), Tenure (number), Balance (number), NumOfProducts (number), HasCrCard (select: 0/1), IsActiveMember (select: 0/1), EstimatedSalary (number).
- Preprocess inputs: Encode Geography and Gender, standardize all features.
- Predict churn probability and class (0/1).
- Display prediction result clearly (e.g., "Customer will churn" or "Customer will not churn" with probability).

**Step-by-Step Instructions for the AI Agent:**

1. **Import Libraries:**
   - Import `streamlit`, `tensorflow.keras`, `pandas`, `numpy`, `pickle`.
   - Code example: `import streamlit as st; from tensorflow import keras; import pandas as pd; import numpy as np; import pickle`

2. **Load Model and Preprocessors:**
   - Load the model: `model = keras.models.load_model('ann_churn_model.h5')`
   - Load pickles: `le = pickle.load(open('pickle/label_encoder.pkl', 'rb')); ohe = pickle.load(open('pickle/onehot_encoder.pkl', 'rb')); sc = pickle.load(open('pickle/standard_scaler.pkl', 'rb'))`
   - Code example: As above.

3. **Create Streamlit UI:**
   - Title: `st.title('Customer Churn Prediction')`
   - Input fields:
     - `credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)`
     - `geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])`
     - `gender = st.selectbox('Gender', ['Male', 'Female'])`
     - `age = st.number_input('Age', min_value=18, max_value=100, value=30)`
     - `tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)`
     - `balance = st.number_input('Balance', value=0.0)`
     - `num_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)`
     - `has_cr_card = st.selectbox('Has Credit Card', [0, 1])`
     - `is_active = st.selectbox('Is Active Member', [0, 1])`
     - `salary = st.number_input('Estimated Salary', value=50000.0)`
   - Predict button: `if st.button('Predict Churn'):`

4. **Preprocess Inputs:**
   - Encode Geography: `geo_encoded = le.transform([geography])[0]`
   - Encode Gender: `gender_encoded = ohe.transform([[gender]])[0]` (note: ohe expects 2D, and since drop='first', it's [0] for Male)
   - Create feature array: `features = np.array([[credit_score, geo_encoded, age, tenure, balance, num_products, has_cr_card, is_active, salary, gender_encoded[0]]])`
   - Standardize: `features_scaled = sc.transform(features)`

5. **Make Prediction:**
   - Predict: `prediction_prob = model.predict(features_scaled)[0][0]`
   - Class: `prediction = (prediction_prob > 0.5).astype(int)`
   - Display: `st.write(f'Churn Probability: {prediction_prob:.2f}'); st.write('Prediction: Customer will churn' if prediction == 1 else 'Prediction: Customer will not churn')`

6. **Run the App:**
   - At the end: `if __name__ == '__main__': st.run()`

**Additional Guidelines:**
- Structure the app with clear sections.
- Handle errors (e.g., invalid inputs).
- Assume files are in correct paths.
- Test the app for functionality.
- Expected output: Interactive Streamlit app for predictions.

**Expected Output:**
- A complete `prediction_app.py` script.
- Run with `streamlit run prediction_app.py`.