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
    input_values = [[None for _ in FEATURES] for _ in range(10)]
    input_dates = [None for _ in range(10)]
    if request.method == 'POST':
        # Handle Excel upload
        if 'excel_file' in request.files and request.files['excel_file'].filename:
            file = request.files['excel_file']
            try:
                df = pd.read_excel(file)
                # Normalize column names (strip, lower, replace spaces, remove dots)
                df.columns = [str(col).strip().replace("  ", " ").replace(" ", "_").replace("-", "_").replace(".", "") for col in df.columns]
                # Map expected features to uploaded columns (robust to whitespace, case, dots)
                feature_map = {}
                for feat in FEATURES:
                    feat_norm = feat.strip().replace(" ", "_").replace("-", "_").replace(".", "").lower()
                    for col in df.columns:
                        if feat_norm == col.lower():
                            feature_map[feat] = col
                            break
                # Find Month column
                month_col = None
                for col in df.columns:
                    if col.lower() == 'month':
                        month_col = col
                        break
                if len(feature_map) != len(FEATURES) or not month_col:
                    error = f"Excel file must contain columns: Month, {', '.join(FEATURES)}"
                else:
                    # Only keep Month and mapped features, in correct order
                    cols_needed = [month_col] + [feature_map[feat] for feat in FEATURES]
                    df = df[cols_needed]
                    if len(df) > 10:
                        df = df.iloc[:10]
                    input_df = df.copy()
                    input_df.columns = ['Month'] + FEATURES
                    # Prepare values for pre-filling manual entry
                    for i, row in input_df.iterrows():
                        if i < 10:
                            input_dates[i] = row['Month'] if pd.notnull(row['Month']) else None
                            for j, feat in enumerate(FEATURES):
                                input_values[i][j] = row[feat] if pd.notnull(row[feat]) else None
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
                # For pre-filling
                input_dates[i] = month_val
                for j, v in enumerate(row):
                    input_values[i][j] = v
            if input_data:
                input_df = pd.DataFrame(input_data, columns=FEATURES)
                input_df['Month'] = months[:len(input_df)]
        # Forecast if we have valid input_df
        if input_df is not None and error is None:
            try:
                X_pred = input_df[FEATURES]
                # Ensure always 2D
                if X_pred.shape[0] == 1:
                    X_pred_scaled = scaler.transform(X_pred.values.reshape(1, -1))
                else:
                    X_pred_scaled = scaler.transform(X_pred)
                # XGBoost bug workaround: use predict with validate_features=False
                try:
                    preds = model.predict(X_pred_scaled, validate_features=False)
                except Exception:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(X_pred_scaled)
                    preds = model.get_booster().predict(dmatrix)
                forecast = list(zip(input_df['Month'], preds))
            except Exception as e:
                error = f"Error during forecasting: {e}"
    return render_template('index.html', features=FEATURES, forecast=forecast, error=error, input_values=input_values, input_dates=input_dates)

if __name__ == '__main__':
    app.run(debug=True)
