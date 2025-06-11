from flask import Flask, render_template, request, redirect
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
                # Only keep Month and FEATURES columns
                cols_needed = ['Month'] + FEATURES
                df = df[[col for col in cols_needed if col in df.columns]]
                if len(df) > 10:
                    df = df.iloc[:10]
                if not all(feat in df.columns for feat in FEATURES):
                    error = f"Excel file must contain columns: {', '.join(FEATURES)}"
                else:
                    input_df = df.copy()
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
                X_pred_scaled = scaler.transform(X_pred)
                preds = model.predict(X_pred_scaled)
                forecast = list(zip(input_df['Month'], preds))
            except Exception as e:
                error = f"Error during forecasting: {e}"
    return render_template('index.html', features=FEATURES, forecast=forecast, error=error)

if __name__ == '__main__':
    app.run(debug=True)
