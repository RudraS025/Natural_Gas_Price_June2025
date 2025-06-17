from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import xgboost as xgb

app = Flask(__name__)

# Load model and scaler
MODEL_PATH = 'natural_gas_price_xgb_model.pkl'
SCALER_PATH = 'scaler.save'
model = joblib.load(MODEL_PATH)
# Final bulletproof patch: monkey-patch gpu_id property if missing
if isinstance(model, xgb.XGBModel):
    if not hasattr(model, 'gpu_id'):
        try:
            model.gpu_id = 0
        except Exception:
            pass
scaler = joblib.load(SCALER_PATH)

# Load feature names from file
with open('feature_names.txt') as f:
    FEATURES = [line.strip() for line in f if line.strip()]

# Load last N actuals for recursive forecasting
LAST_ACTUALS_PATH = 'last_actuals.csv'
last_actuals_df = pd.read_csv(LAST_ACTUALS_PATH, parse_dates=['Month'])

# Define target column name (must match training script)
target_col = 'Henryhub NG prices (USD/MMBtu)'

# Helper to get cyclical month features
def get_month_cyclical_features(date_str):
    dt = pd.to_datetime(date_str)
    month = dt.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    return month_sin, month_cos

def excel_date_to_str(val):
    # Convert Excel/Excel string/Excel Timestamp to YYYY-MM-DD for HTML date input
    if pd.isnull(val):
        return ''
    if isinstance(val, pd.Timestamp):
        return val.strftime('%Y-%m-%d')
    try:
        return pd.to_datetime(val).strftime('%Y-%m-%d')
    except Exception:
        return str(val)

# List of exogenous variables (no lag/roll/cyclical)
EXOGENOUS_VARS = [
    'Production (in BCM)',
    'Residential Consumption',
    'Commercial Consumption',
    'Industrial Consumption',
    'Electric Power Consumption',
    'Other Consumption',
    'NG working underground storage (BCM)',  # No trailing space
    'Exports (in BCM)'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    forecast = None
    error = None
    input_df = None
    chart_data = None  # For charting last 15 actuals + forecast
    # Only exogenous vars for manual/Excel entry
    input_values = [[None for _ in EXOGENOUS_VARS] for _ in range(10)]
    input_dates = [None for _ in range(10)]
    preview = False
    if request.method == 'POST':
        action = request.form.get('action', 'preview')
        # Excel upload: preview mode
        if 'excel_file' in request.files and request.files['excel_file'].filename and action == 'preview':
            file = request.files['excel_file']
            try:
                df = pd.read_excel(file)
                # Strip whitespace from all column names
                df.columns = [str(col).strip() for col in df.columns]
                feature_map = {}
                for feat in EXOGENOUS_VARS:
                    for col in df.columns:
                        if feat == col:
                            feature_map[feat] = col
                            break
                month_col = None
                for col in df.columns:
                    if col.lower() == 'month':
                        month_col = col
                        break
                if len(feature_map) != len(EXOGENOUS_VARS) or not month_col:
                    error = f"Excel file must contain columns: Month, {', '.join(EXOGENOUS_VARS)}"
                else:
                    cols_needed = [month_col] + [feature_map[feat] for feat in EXOGENOUS_VARS]
                    df = df[cols_needed]
                    if len(df) > 10:
                        df = df.iloc[:10]
                    input_df = df.copy()
                    input_df.columns = ['Month'] + EXOGENOUS_VARS
                    for i, row in input_df.iterrows():
                        if i < 10:
                            input_dates[i] = excel_date_to_str(row['Month'])
                            for j, feat in enumerate(EXOGENOUS_VARS):
                                input_values[i][j] = row[feat] if pd.notnull(row[feat]) else None
                    preview = True
            except Exception as e:
                error = f"Error reading Excel file: {e}"
        else:
            # Forecast action: use manual entry or pre-filled values
            input_data = []
            months = []
            for i in range(10):
                month_val = request.form.get(f'date_{i}')
                months.append(month_val)
                row = []
                for feat in EXOGENOUS_VARS:
                    val = request.form.get(f'{feat}_{i}', type=float)
                    row.append(val)
                if any([v is not None for v in row]) and month_val:
                    input_data.append(row)
                input_dates[i] = month_val
                for j, v in enumerate(row):
                    input_values[i][j] = v
            if input_data:
                input_df = pd.DataFrame(input_data, columns=EXOGENOUS_VARS)
                input_df['Month'] = months[:len(input_df)]
        # --- Forecasting: use recursive feature generation for both Excel/manual ---
        if input_df is not None and error is None and action == 'forecast':
            try:
                # Strip whitespace from DataFrame columns to match model features
                input_df.columns = [str(col).strip() for col in input_df.columns]
                lag_cols = [col for col in FEATURES if '_lag' in col]
                # Only include roll columns that do NOT contain 'std' (i.e., not '_rollstd')
                roll_cols = [col for col in FEATURES if '_roll' in col and 'std' not in col]
                cyc_cols = ['month_sin', 'month_cos']
                exog_cols = EXOGENOUS_VARS
                history = last_actuals_df.copy()
                preds = []
                # --- Seasonality and noise injection setup ---
                # Use last 10 years of same-month values for seasonality
                hist_months = history['Month'].dt.month.values
                hist_prices = history[history.columns[-1]].values
                # Build a dict: month -> list of historical prices for that month
                month_hist = {}
                for m, p in zip(hist_months, hist_prices):
                    if m not in month_hist:
                        month_hist[m] = []
                    month_hist[m].append(p)
                # Start AR(1) noise at 0
                prev_noise = 0
                noise_std = np.std(np.diff(hist_prices)[-24:]) if len(hist_prices) > 24 else 0.15
                # Main recursive forecast loop
                for i in range(len(input_df)):
                    row = input_df.iloc[i].copy()
                    feat_row = {}
                    for col in exog_cols:
                        feat_row[col] = row.get(col, np.nan)
                    for lag_col in lag_cols:
                        try:
                            lag_n = int(lag_col.split('_lag')[-1])
                        except Exception:
                            continue
                        feat_row[lag_col] = history[history.columns[-1]].iloc[-lag_n]
                    for roll_col in roll_cols:
                        try:
                            roll_n = int(roll_col.split('_roll')[-1])
                        except Exception:
                            continue
                        feat_row[roll_col] = history[history.columns[-1]].iloc[-roll_n:].mean()
                    month_sin, month_cos = get_month_cyclical_features(row['Month'])
                    feat_row['month_sin'] = month_sin
                    feat_row['month_cos'] = month_cos
                    feat_vec = [feat_row[f] if f in feat_row else 0 for f in FEATURES]
                    feat_vec_scaled = scaler.transform([feat_vec])
                    y_pred = model.predict(feat_vec_scaled)[0]
                    # --- Inject historical seasonality and AR(1) noise ---
                    forecast_month = pd.to_datetime(row['Month']).month
                    # Historical mean for this month
                    if forecast_month in month_hist and len(month_hist[forecast_month]) > 2:
                        month_seasonal = np.mean(month_hist[forecast_month][-10:])
                    else:
                        month_seasonal = np.mean(hist_prices[-12:])
                    # AR(1) noise: new_noise = 0.7*prev_noise + N(0, noise_std)
                    new_noise = 0.7 * prev_noise + np.random.normal(0, noise_std)
                    prev_noise = new_noise
                    # Blend model, seasonality, and noise
                    y_blend = 0.65 * y_pred + 0.3 * month_seasonal + 0.05 * new_noise
                    # Clamp to plausible range
                    y_blend = float(np.clip(y_blend, 1.5, 6.0))
                    preds.append(y_blend)
                    new_row = {'Month': pd.to_datetime(row['Month']), history.columns[-1]: y_blend}
                    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
                forecast = list(zip(input_df['Month'], preds))
                # Prepare chart data: last 15 non-NaN actuals + forecast, and connect last actual to first forecast
                full_actuals = last_actuals_df[['Month', last_actuals_df.columns[-1]]].copy()
                # Only keep rows with non-NaN values for the target
                valid_actuals = full_actuals.dropna(subset=[full_actuals.columns[-1]])
                last_15_actuals = valid_actuals.tail(15)
                # Connect last actual to first forecast for smooth line
                if len(forecast) > 0 and len(last_15_actuals) > 0:
                    forecast_months = [str(m)[:10] for m, _ in forecast]
                    forecast_values = [float(f) for _, f in forecast]
                    # Insert the last actual value as the first forecast value
                    forecast_months = [last_15_actuals['Month'].iloc[-1].strftime('%Y-%m')] + forecast_months
                    forecast_values = [last_15_actuals[last_15_actuals.columns[-1]].iloc[-1]] + forecast_values
                else:
                    forecast_months = [str(m)[:10] for m, _ in forecast]
                    forecast_values = [float(f) for _, f in forecast]
                chart_data = {
                    'actual_months': last_15_actuals['Month'].dt.strftime('%Y-%m').tolist(),
                    'actual_values': last_15_actuals[last_15_actuals.columns[-1]].tolist(),
                    'forecast_months': forecast_months,
                    'forecast_values': forecast_values
                }
            except Exception as e:
                error = f"Error during forecasting: {e}"
    return render_template('index.html', features=EXOGENOUS_VARS, forecast=forecast, error=error, input_values=input_values, input_dates=input_dates, preview=preview, chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)
