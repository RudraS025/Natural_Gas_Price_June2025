from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import os

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = 'natural_gas_xgb_model.pkl'
SCALER_PATH = 'scaler.save'
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Load feature names from file
with open('feature_names.txt') as f:
    FEATURES = [line.strip() for line in f if line.strip()]

target_col = 'India total Consumption of Natural Gas (in BCM)'

def add_time_features(df):
    df['Year'] = pd.to_datetime(df['Month']).dt.year
    df['Month_num'] = pd.to_datetime(df['Month']).dt.month
    df['Quarter'] = pd.to_datetime(df['Month']).dt.quarter
    df['Month_sin'] = np.sin(2 * np.pi * df['Month_num'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month_num'] / 12)
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    if request.method == 'POST':
        input_data = []
        months = []
        for i in range(10):
            row = []
            month_val = request.form.get(f'date_{i}')
            months.append(month_val)
            for feat in FEATURES:
                val = request.form.get(f'{feat}_{i}', type=float)
                row.append(val)
            if any([v is not None for v in row]) and month_val:
                input_data.append(row)
        if input_data:
            df_input = pd.DataFrame(input_data, columns=FEATURES)
            df_input['Month'] = months[:len(df_input)]
            df_input = add_time_features(df_input)
            # Ensure columns order matches training
            X_pred = df_input[FEATURES + ['Year','Month_num','Quarter','Month_sin','Month_cos']]
            X_pred_scaled = scaler.transform(X_pred)
            preds = model.predict(X_pred_scaled)
            forecast = preds
    return render_template('index.html', features=FEATURES, forecast=forecast)

if __name__ == '__main__':
    app.run(debug=True)
