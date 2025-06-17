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

# --- Enhanced Feature Engineering: More Lags, Rolling Stats, Cyclical, Deltas ---
LAGS = [1, 2, 3, 6, 12]
ROLLS = [3, 6, 12]

# Add lag/rolling features for all independent variables and target
df = original_df.copy()
feature_cols = independent_vars.copy()
for var in [target_col] + independent_vars:
    # Lags
    for lag in LAGS:
        col = f'{var}_lag{lag}'
        df[col] = df[var].shift(lag)
        feature_cols.append(col)
    # Rolling means
    for roll in ROLLS:
        col = f'{var}_roll{roll}'
        df[col] = df[var].rolling(roll).mean().shift(1)
        feature_cols.append(col)
    # Rolling std
    for roll in ROLLS:
        col = f'{var}_rollstd{roll}'
        df[col] = df[var].rolling(roll).std().shift(1)
        feature_cols.append(col)
# Cyclical month features
df['month'] = df.index.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['quarter'] = df.index.quarter
df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
feature_cols += ['month_sin', 'month_cos', 'quarter_sin', 'quarter_cos']
# Month-over-month and year-over-year changes for target
df['target_mom'] = df[target_col].pct_change(1)
df['target_yoy'] = df[target_col].pct_change(12)
feature_cols += ['target_mom', 'target_yoy']
# Drop rows with NA (from lag/rolling)
df = df.dropna().copy()
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

# XGBoost with tuned hyperparameters
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=2,
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, 'natural_gas_price_xgb_model.pkl')
xgb_pred = xgb_model.predict(X_test_scaled)

# Random Forest (ensemble)
rf_model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, 'natural_gas_price_rf_model.pkl')
rf_pred = rf_model.predict(X_test_scaled)

# Ensemble: average predictions
ensemble_pred = (xgb_pred + rf_pred) / 2
# Post-processing: apply a floor (e.g., minimum price = 2.5)
ensemble_pred = np.maximum(ensemble_pred, 2.5)

# Save results
results = pd.DataFrame({'Month': y_test.index, 'Actual': y_test.values, 'XGB': xgb_pred, 'RF': rf_pred, 'Ensemble': ensemble_pred})
results.to_excel('ensemble_test_vs_prediction_results.xlsx', index=False)
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, ensemble_pred, label='Ensemble Forecast', linestyle='--')
plt.plot(y_test.index, xgb_pred, label='XGB Only', alpha=0.5)
plt.plot(y_test.index, rf_pred, label='RF Only', alpha=0.5)
plt.legend()
plt.title('Ensemble: Test vs Prediction (Enhanced Features)')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.tight_layout()
plt.savefig('ensemble_test_vs_prediction.png')
plt.close()

# Save feature names for Flask app
with open('feature_names.txt', 'w') as f:
    for col in feature_cols:
        f.write(f"{col}\n")

# Save last N months of actuals for recursive forecasting
N = max(LAGS + ROLLS + [12])
last_actuals = df.iloc[-N:][[target_col]].copy()
last_actuals.to_csv('last_actuals.csv')

print('Ensemble model with enhanced features trained and results saved.')
