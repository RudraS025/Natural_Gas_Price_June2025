from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
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

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    error = None
    input_df = None
    if request.method == 'POST':
        # Handle Excel upload
        if 'excel_file' in request.files and request.files['excel_file'].filename:
            file = request.files['excel_file']
            try:
                df = pd.read_excel(file)
                # Normalize column names (strip, lower, replace spaces)
                df.columns = [str(col).strip().replace("  ", " ").replace(" ", "_").replace("-", "_") for col in df.columns]
                # Map expected features to uploaded columns
                feature_map = {}
                for feat in FEATURES:
                    for col in df.columns:
                        if feat.replace(" ", "_").lower() == col.lower():
                            feature_map[feat] = col
                            break
                if len(feature_map) != len(FEATURES):
                    error = f"Excel file must contain columns: {', '.join(FEATURES)}"
                else:
                    # Only keep Month and mapped features
                    cols_needed = ['Month'] + [feature_map[feat] for feat in FEATURES]
                    if 'Month' not in df.columns:
                        # Try to find a column that matches 'Month' (case-insensitive)
                        for col in df.columns:
                            if col.lower() == 'month':
                                df.rename(columns={col: 'Month'}, inplace=True)
                                break
                    df = df[cols_needed]
                    if len(df) > 10:
                        df = df.iloc[:10]
                    input_df = df.copy()
                    input_df.columns = ['Month'] + FEATURES
            except Exception as e:
                error = f"Error reading Excel file: {e}"
        else:
            # Manual entry fallback
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
                input_df = pd.DataFrame(input_data, columns=FEATURES)
                input_df['Month'] = months[:len(input_df)]
        # Forecast if we have valid input_df
        if input_df is not None and error is None:
            try:
                X_pred = input_df[FEATURES]
                # Fix for single-row input (reshape if needed)
                if X_pred.shape[0] == 1:
                    X_pred_scaled = scaler.transform(X_pred.values.reshape(1, -1))
                else:
                    X_pred_scaled = scaler.transform(X_pred)
                preds = model.predict(X_pred_scaled)
                forecast = list(zip(input_df['Month'], preds))
            except Exception as e:
                error = f"Error during forecasting: {e}"
    return render_template('index.html', features=FEATURES, forecast=forecast, error=error)

if __name__ == '__main__':
    app.run(debug=True)
