import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

st.title('🏠 London House Price Predictor')
st.write('Predict average house prices across London boroughs using a Random Forest model trained on real data.')

# Input fields
area = st.selectbox('Borough', sorted(le.classes_))
year = st.slider('Year', min_value=2000, max_value=2030, value=2023)
median_salary = st.number_input('Median Salary (£)', min_value=10000, max_value=100000, value=35000, step=1000)
mean_salary = st.number_input('Mean Salary (£)', min_value=10000, max_value=100000, value=40000, step=1000)
borough_flag = st.selectbox('Borough Flag', [0, 1], help='1 = official London borough')
houses_sold = st.number_input('Houses Sold per Month', min_value=0, max_value=1000, value=100)
no_of_crimes = st.number_input('Number of Crimes', min_value=0, max_value=10000, value=300)
population_size = st.number_input('Population Size', min_value=0, max_value=1000000, value=250000, step=1000)
number_of_jobs = st.number_input('Number of Jobs', min_value=0, max_value=1000000, value=100000, step=1000)
recycling_pct = st.number_input('Recycling %', min_value=0, max_value=100, value=25)

# Predict
if st.button('Predict Price'):
    area_encoded = le.transform([area])[0]

    input_data = pd.DataFrame([{
        'year': year,
        'median_salary': median_salary,
        'mean_salary': mean_salary,
        'borough_flag_x': borough_flag,
        'area_encoded': area_encoded,
        'houses_sold': houses_sold,
        'no_of_crimes': no_of_crimes,
        'population_size': population_size,
        'number_of_jobs': number_of_jobs,
        'recycling_pct': recycling_pct
    }])

    prediction = model.predict(input_data)[0]

    st.success(f'### Predicted Average Price: £{prediction:,.0f}')
    st.caption('This prediction is based on historical London housing data and a Random Forest model with R² = 0.990')