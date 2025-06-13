import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib

# Load data
original_df = pd.read_excel('NaturalGasPrice_Input.xlsx', engine='openpyxl')
original_df['Month'] = pd.to_datetime(original_df['Month'])
original_df.set_index('Month', inplace=True)

# Set new independent variables and target
independent_vars = [
    'Production (in BCM)',
    'Residential Consumption',
    'Commercial Consumption',
    'Industrial Consumption',
    'Electric Power Consumption',
    'Other Consumption',
    'NG working underground storage (BCM) ',  # Trailing space restored
    'Exports (in BCM)'
]
target_col = 'Henryhub NG prices (USD/MMBtu)'
X = original_df[independent_vars]
y = original_df[target_col]

# Train-test split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.save')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest (basic)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, 'natural_gas_price_rf_model.pkl')
rf_pred = rf_model.predict(X_test_scaled)
rf_results = pd.DataFrame({'Month': y_test.index, 'Actual': y_test.values, 'Predicted': rf_pred})
rf_results.to_excel('rf_test_vs_prediction_results.xlsx', index=False)
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, rf_pred, label='RF Predicted')
plt.legend()
plt.title('Random Forest: Test vs Prediction (No Feature Engineering)')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.tight_layout()
plt.savefig('rf_test_vs_prediction.png')
plt.close()

# --- Feature Engineering: Lag, Rolling, Cyclical ---
LAGS = [1, 2]
ROLLS = [3]

df = original_df.copy()
# Lag features for target
target_lag_cols = []
for lag in LAGS:
    col = f'{target_col}_lag{lag}'
    df[col] = df[target_col].shift(lag)
    target_lag_cols.append(col)
# Rolling mean features for target
roll_cols = []
for roll in ROLLS:
    col = f'{target_col}_roll{roll}'
    df[col] = df[target_col].rolling(roll).mean().shift(1)
    roll_cols.append(col)
# Cyclical month features
df['month'] = df.index.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Drop rows with NA (from lag/rolling)
df = df.dropna().copy()

# New feature list
feature_cols = independent_vars + target_lag_cols + roll_cols + ['month_sin', 'month_cos']
X = df[feature_cols]
y = df[target_col]

# Train-test split (no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardization
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.save')
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost (with features)
xgb_model = xgb.XGBRegressor(n_estimators=300, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, 'natural_gas_price_xgb_model.pkl')
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_results = pd.DataFrame({'Month': y_test.index, 'Actual': y_test.values, 'Predicted': xgb_pred})
xgb_results.to_excel('xgb_test_vs_prediction_results.xlsx', index=False)
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, xgb_pred, label='XGB Predicted')
plt.legend()
plt.title('XGBoost: Test vs Prediction (Lag/Roll/Cyclical Features)')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.tight_layout()
plt.savefig('xgb_test_vs_prediction.png')
plt.close()

# Save feature names for Flask app
with open('feature_names.txt', 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

# Save last N months of actuals for recursive forecasting
N = max(LAGS + ROLLS)
last_actuals = df.iloc[-N:][[target_col]].copy()
last_actuals.to_csv('last_actuals.csv')

print('XGBoost with lag/rolling/cyclical features trained and results saved.')
