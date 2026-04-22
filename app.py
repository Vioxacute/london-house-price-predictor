import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="London House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f4f0;
    color: #1a1a1a;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding: 48px 48px 80px !important;
    max-width: 860px !important;
}

/* Top bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 64px;
}

.topbar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: -0.3px;
    color: #1a1a1a;
}

.topbar-badge {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888;
    background: #ebebeb;
    padding: 6px 14px;
    border-radius: 100px;
}

/* Hero */
.hero-eyebrow {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 16px;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(40px, 7vw, 64px);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -2px;
    color: #1a1a1a;
    margin: 0 0 24px;
}

.hero-title em {
    font-style: normal;
    color: #2563eb;
}

.hero-desc {
    font-size: 15px;
    font-weight: 400;
    color: #666;
    line-height: 1.75;
    max-width: 500px;
    margin-bottom: 48px;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 0;
    margin-bottom: 64px;
    border: 1px solid #e0dfd9;
    border-radius: 12px;
    overflow: hidden;
    background: white;
}

.stat-item {
    flex: 1;
    padding: 24px 28px;
    border-right: 1px solid #e0dfd9;
}

.stat-item:last-child {
    border-right: none;
}

.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #2563eb;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}

.stat-lbl {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #aaa;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #e0dfd9;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 16px;
}

.card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid #f0eeea;
}

/* Result */
.result-card {
    background: #2563eb;
    border-radius: 16px;
    padding: 40px 32px;
    text-align: center;
    margin-top: 16px;
}

.result-eyebrow {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin-bottom: 12px;
}

.result-price {
    font-family: 'Syne', sans-serif;
    font-size: 56px;
    font-weight: 800;
    color: white;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 10px;
}

.result-meta {
    font-size: 13px;
    color: rgba(255,255,255,0.5);
    font-weight: 400;
}

/* Override Streamlit widgets */
div[data-baseweb="select"] > div {
    background: #f8f8f6 !important;
    border: 1px solid #e0dfd9 !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
}

.stNumberInput > div > div > input {
    background: #f8f8f6 !important;
    border: 1px solid #e0dfd9 !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    font-family: 'Inter', sans-serif !important;
}

label, .stSelectbox label, .stNumberInput label {
    color: #555 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton > button {
    background: #1a1a1a !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 16px 40px !important;
    width: 100% !important;
    margin-top: 8px !important;
    transition: background 0.2s !important;
}

.stButton > button:hover {
    background: #2563eb !important;
}

/* Footer */
.footer {
    margin-top: 64px;
    padding-top: 24px;
    border-top: 1px solid #e0dfd9;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.footer-left {
    font-size: 12px;
    color: #aaa;
}

.footer-right {
    font-size: 12px;
    color: #aaa;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

# Top bar
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">LHPP</div>
    <div class="topbar-badge">Random Forest Model</div>
</div>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-eyebrow">London Property Intelligence</div>
<h1 class="hero-title">Predict Any<br><em>Borough Price</em></h1>
<p class="hero-desc">Enter borough characteristics below to generate an instant average house price prediction, powered by a model trained on real Land Registry and ONS data.</p>

<div class="stats-row">
    <div class="stat-item">
        <div class="stat-num">0.990</div>
        <div class="stat-lbl">R² Score</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">£8,312</div>
        <div class="stat-lbl">Avg. Error</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">32</div>
        <div class="stat-lbl">Boroughs</div>
    </div>
    <div class="stat-item">
        <div class="stat-num">25yrs</div>
        <div class="stat-lbl">of Data</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Form card 1
st.markdown('<div class="card"><div class="card-title">Borough & Time</div>', unsafe_allow_html=True)
area = st.selectbox('Borough', sorted(le.classes_))
year = st.slider('Year', min_value=2000, max_value=2030, value=2023)
borough_flag = st.selectbox('Borough Type', [1, 0],
                             format_func=lambda x: 'Official London Borough' if x == 1 else 'Outer Area')
st.markdown('</div>', unsafe_allow_html=True)

# Form card 2
st.markdown('<div class="card"><div class="card-title">Economy</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    median_salary = st.number_input('Median Salary (£)', min_value=10000, max_value=100000, value=35000, step=500)
with col2:
    mean_salary = st.number_input('Mean Salary (£)', min_value=10000, max_value=100000, value=40000, step=500)
number_of_jobs = st.number_input('Number of Jobs', min_value=0, max_value=1000000, value=100000, step=1000)
st.markdown('</div>', unsafe_allow_html=True)

# Form card 3
st.markdown('<div class="card"><div class="card-title">Demographics & Activity</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    population_size = st.number_input('Population Size', min_value=0, max_value=1000000, value=250000, step=1000)
    houses_sold = st.number_input('Houses Sold / Month', min_value=0, max_value=1000, value=100)
with col4:
    no_of_crimes = st.number_input('Number of Crimes', min_value=0, max_value=10000, value=300)
    recycling_pct = st.number_input('Recycling %', min_value=0, max_value=100, value=25)
st.markdown('</div>', unsafe_allow_html=True)

# Predict button
predict = st.button('Generate Prediction →')

if predict:
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
    st.markdown(f"""
    <div class="result-card">
        <div class="result-eyebrow">Predicted Average Price</div>
        <div class="result-price">£{prediction:,.0f}</div>
        <div class="result-meta">{area.title()} &nbsp;·&nbsp; {year} &nbsp;·&nbsp; R² 0.990</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div class="footer-left">Trained on London housing data 1995–2020</div>
    <div class="footer-right">Random Forest · scikit-learn</div>
</div>
""", unsafe_allow_html=True)