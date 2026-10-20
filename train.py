import json

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

TEST_FROM_YEAR = 2016  # the honest test: train on earlier years, predict these

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

features = ['year', 'median_salary', 'mean_salary', 'borough_flag_x',
            'area_encoded', 'houses_sold', 'no_of_crimes',
            'population_size', 'number_of_jobs', 'recycling_pct']
X = df[features]
y = df['average_price']


def new_model():
    return RandomForestRegressor(n_estimators=100, random_state=42)


def scores(y_true, y_pred):
    return {'r2': round(r2_score(y_true, y_pred), 3), 'mae': round(mean_absolute_error(y_true, y_pred))}


# Evaluate on later years, training only on earlier ones
train, test = df[df['year'] < TEST_FROM_YEAR], df[df['year'] >= TEST_FROM_YEAR]
time_split = scores(test['average_price'], new_model().fit(train[features], train['average_price']).predict(test[features]))

# Baseline to beat: each area's average price in the last training year, carried forward
last_price = train[train['year'] == train['year'].max()].groupby('area')['average_price'].mean()
baseline = scores(test['average_price'], test['area'].map(last_price))

# For comparison only: a random 80/20 split of monthly rows. Neighbouring
# months of the same area land in both train and test, so it mostly measures
# interpolation and overstates accuracy.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
random_split = scores(y_test, new_model().fit(X_train, y_train).predict(X_test))

metrics = {
    f'time_split_test_{TEST_FROM_YEAR}_onwards': time_split,
    'baseline_last_known_price': baseline,
    'random_split_overstated': random_split,
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
for name, m in metrics.items():
    print(f"{name:<36} R2 {m['r2']:.3f}   MAE £{m['mae']:,}")

# Final model for the app, trained on every year
model = new_model().fit(X, y)
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model trained and saved!")