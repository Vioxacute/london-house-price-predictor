import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
monthly = pd.read_csv('data/housing_in_london_monthly_variables.csv')
yearly = pd.read_csv('data/housing_in_london_yearly_variables.csv')

# Parse dates and extract year
monthly['date'] = pd.to_datetime(monthly['date'])
yearly['date'] = pd.to_datetime(yearly['date'])
monthly['year'] = monthly['date'].dt.year
yearly['year'] = yearly['date'].dt.year

# Clean yearly
yearly['mean_salary'] = pd.to_numeric(yearly['mean_salary'], errors='coerce')
yearly['recycling_pct'] = pd.to_numeric(yearly['recycling_pct'], errors='coerce')
yearly_clean = yearly.drop(columns=['life_satisfaction', 'area_size', 'no_of_houses', 'date', 'code'])

# Merge
df = pd.merge(monthly, yearly_clean, on=['area', 'year'], how='left')

# Drop unnecessary columns
df = df.drop(columns=['date_x', 'date_y', 'code_x', 'code_y',
                       'life_satisfaction', 'area_size', 'no_of_houses',
                       'borough_flag_y'], errors='ignore')

# Drop missing targets and fill features
df = df.dropna(subset=['average_price'])
for col in ['houses_sold', 'no_of_crimes', 'median_salary', 'mean_salary',
            'recycling_pct', 'population_size', 'number_of_jobs']:
    df[col] = df[col].fillna(df[col].median())

# Encode area
le = LabelEncoder()
df['area_encoded'] = le.fit_transform(df['area'])

# Train
features = ['year', 'median_salary', 'mean_salary', 'borough_flag_x',
            'area_encoded', 'houses_sold', 'no_of_crimes',
            'population_size', 'number_of_jobs', 'recycling_pct']

X = df[features]
y = df['average_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model trained and saved!")