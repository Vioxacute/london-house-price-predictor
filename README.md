# 🏠 London House Price Predictor

A machine learning web app that predicts average house prices across London boroughs using real historical data.

🔗 **Live App:** [london-house-price-predictor.onrender.com](https://london-house-price-predictor.onrender.com)

---

## Overview

This project uses a Random Forest regression model trained on real London housing data to predict average property prices by borough. Users can input borough characteristics such as salary, crime rate, and population to receive an instant price prediction.

The model achieves an **R² score of 0.990** and a **Mean Absolute Error of £8,312** on unseen test data.

---

## Features

- Predict average house prices for any London borough
- Interactive input controls for key borough features
- Trained on real ONS and Land Registry housing data
- Deployed live as a web app — no installation required

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas & numpy | Data cleaning and manipulation |
| scikit-learn | Model training and evaluation |
| matplotlib & seaborn | Data visualisation |
| Streamlit | Web app interface |
| Render | Cloud deployment |

---

## Dataset

Two datasets from Kaggle's [Housing in London](https://www.kaggle.com/datasets/justinas/housing-in-london) collection were used:

- `housing_in_london_monthly_variables.csv` — monthly average prices, houses sold, crime rates per borough
- `housing_in_london_yearly_variables.csv` — yearly borough stats including median salary, population size, number of jobs

The two datasets were merged on `area` and `year` to create a rich feature set for training.

---

## Model

- **Algorithm:** Random Forest Regressor (100 estimators)
- **Train/Test Split:** 80/20
- **R² Score:** 0.990
- **Mean Absolute Error:** £8,312

**Top predictive features:**
1. Year
2. Median salary
3. Mean salary
4. Borough flag
5. Area (encoded)

---

## Project Structure

```
london-house-price-predictor/
│
├── Data/
│   ├── housing_in_london_monthly_variables.csv
│   └── housing_in_london_yearly_variables.csv
│
├── app.py              # Streamlit web app
├── train.py            # Model training script
├── explore.ipynb       # Data exploration notebook
├── requirements.txt    # Dependencies
└── render.yaml         # Render deployment config
```

---

## Run Locally

1. Clone the repository:
```bash
git clone https://github.com/Vioxacute/london-house-price-predictor.git
cd london-house-price-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train.py
```

4. Run the app:
```bash
streamlit run app.py
```

---

## What I Learned

- Cleaning and merging real-world datasets with missing values
- Feature engineering and encoding categorical variables
- Training and evaluating a Random Forest regression model
- Building and deploying an interactive ML web app

---

## Author

**James** — aspiring engineer  
GitHub: [Vioxacute](https://github.com/Vioxacute)
