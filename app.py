import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="London House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #f0ede8;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #12121e 50%, #0a0a0f 100%);
    border-bottom: 1px solid rgba(212, 175, 95, 0.2);
    padding: 60px 80px 50px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(212, 175, 95, 0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #d4af5f;
    border: 1px solid rgba(212, 175, 95, 0.3);
    padding: 6px 16px;
    border-radius: 2px;
    margin-bottom: 24px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(42px, 6vw, 72px);
    font-weight: 900;
    line-height: 1.05;
    color: #f0ede8;
    margin: 0 0 16px;
    letter-spacing: -1px;
}

.hero-title span {
    color: #d4af5f;
}

.hero-subtitle {
    font-size: 16px;
    font-weight: 300;
    color: rgba(240, 237, 232, 0.55);
    max-width: 520px;
    line-height: 1.7;
    margin: 0;
}

.hero-stats {
    display: flex;
    gap: 48px;
    margin-top: 48px;
}

.stat {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 700;
    color: #d4af5f;
}

.stat-label {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(240, 237, 232, 0.4);
}

/* Main content */
.main-content {
    padding: 60px 80px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 60px;
    max-width: 1400px;
}

/* Section headers */
.section-header {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #d4af5f;
    margin-bottom: 28px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(212, 175, 95, 0.15);
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(212, 175, 95, 0.08), rgba(212, 175, 95, 0.03));
    border: 1px solid rgba(212, 175, 95, 0.25);
    border-radius: 4px;
    padding: 40px;
    text-align: center;
    margin-top: 24px;
}

.result-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(240, 237, 232, 0.45);
    margin-bottom: 12px;
}

.result-price {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 900;
    color: #d4af5f;
    line-height: 1;
    margin-bottom: 8px;
}

.result-note {
    font-size: 12px;
    color: rgba(240, 237, 232, 0.35);
    font-weight: 300;
}

/* Streamlit widget overrides */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(212, 175, 95, 0.2) !important;
    border-radius: 3px !important;
    color: #f0ede8 !important;
}

.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(212, 175, 95, 0.2) !important;
    border-radius: 3px !important;
    color: #f0ede8 !important;
}

.stSlider > div > div > div > div {
    background: #d4af5f !important;
}

label {
    color: rgba(240, 237, 232, 0.7) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    letter-spacing: 0.5px !important;
}

.stButton > button {
    background: #d4af5f !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 14px 40px !important;
    width: 100% !important;
    margin-top: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #e8c96e !important;
    transform: translateY(-1px) !important;
}

/* Divider */
.divider {
    height: 1px;
    background: rgba(212, 175, 95, 0.1);
    margin: 0 80px;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('model.pkl')
le = joblib.load('label_encoder.pkl')

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-tag">Machine Learning · London Property</div>
    <h1 class="hero-title">London House<br><span>Price Predictor</span></h1>
    <p class="hero-subtitle">Predict average property prices across London boroughs using a Random Forest model trained on real Land Registry and ONS data.</p>
    <div class="hero-stats">
        <div class="stat">
            <span class="stat-value">0.990</span>
            <span class="stat-label">R² Score</span>
        </div>
        <div class="stat">
            <span class="stat-value">£8,312</span>
            <span class="stat-label">Avg. Error</span>
        </div>
        <div class="stat">
            <span class="stat-value">32</span>
            <span class="stat-label">Boroughs</span>
        </div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# Layout
st.markdown("<div style='padding: 60px 80px 0;'>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="section-header">Borough & Time</div>', unsafe_allow_html=True)
    area = st.selectbox('Borough', sorted(le.classes_))
    year = st.slider('Year', min_value=2000, max_value=2030, value=2023)
    borough_flag = st.selectbox('Borough Type', [1, 0],
                                 format_func=lambda x: 'Official London Borough' if x == 1 else 'Outer Area')

    st.markdown('<div class="section-header" style="margin-top:32px;">Economy</div>', unsafe_allow_html=True)
    median_salary = st.number_input('Median Salary (£)', min_value=10000, max_value=100000, value=35000, step=500)
    mean_salary = st.number_input('Mean Salary (£)', min_value=10000, max_value=100000, value=40000, step=500)
    number_of_jobs = st.number_input('Number of Jobs', min_value=0, max_value=1000000, value=100000, step=1000)

with col2:
    st.markdown('<div class="section-header">Demographics & Activity</div>', unsafe_allow_html=True)
    population_size = st.number_input('Population Size', min_value=0, max_value=1000000, value=250000, step=1000)
    houses_sold = st.number_input('Houses Sold per Month', min_value=0, max_value=1000, value=100)
    no_of_crimes = st.number_input('Number of Crimes', min_value=0, max_value=10000, value=300)
    recycling_pct = st.number_input('Recycling %', min_value=0, max_value=100, value=25)

    st.markdown("<div style='margin-top: 32px;'>", unsafe_allow_html=True)
    predict = st.button('Generate Prediction →')
    st.markdown("</div>", unsafe_allow_html=True)

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
        <div class="result-box">
            <div class="result-label">Predicted Average Price</div>
            <div class="result-price">£{prediction:,.0f}</div>
            <div class="result-note">{area.title()} · {year} · R² 0.990</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)