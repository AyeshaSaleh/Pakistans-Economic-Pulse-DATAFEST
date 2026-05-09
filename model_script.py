import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import warnings

warnings.filterwarnings("ignore")

# --- UPDATE THESE FOR CLOUD ENVIRONMENTS ---
TRADE_PATH = "TRADE DATASET.csv"
QIM_PATH = "QIM DATASET.xlsx"
CLIMATE_PATH = "CLEANED CLIMATE DATASET.xlsx"

FORECAST_HORIZON = 12
OVERLAP_MONTHS = 3
RANDOM_SEED = 42

# --- LOAD DATA ---
trade = pd.read_csv(TRADE_PATH)
qim = pd.read_excel(QIM_PATH)
climate = pd.read_excel(CLIMATE_PATH)

# --- HELPER FUNCTIONS ---

def normalize_month(df, date_col_guess=None):
    df = df.copy()
    if 'Observation Date' in df.columns:
        date_col_guess = 'Observation Date'
    elif 'Date' in df.columns:
        date_col_guess = 'Date'

    if date_col_guess and date_col_guess in df.columns:
        df['Month'] = pd.to_datetime(df[date_col_guess], errors='coerce').dt.to_period('M').dt.to_timestamp()
    else:
        df['Month'] = pd.to_datetime(df.iloc[:,0], errors='coerce').dt.to_period('M').dt.to_timestamp()

    df = df.dropna(subset=['Month']).sort_values('Month').reset_index(drop=True)
    return df

def prepare_exog(df_indexed, exog_cols, scaler=None, fill_method='ffill'):
    X = df_indexed[exog_cols].copy()
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(method=fill_method, inplace=True)
    X.fillna(method='bfill', inplace=True)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler

# --- DATA PREPARATION ---
trade = normalize_month(trade)
qim = normalize_month(qim)
climate = normalize_month(climate)

trade_monthly = trade.groupby(['Month','Trade Type'])['Observation Value'].sum().reset_index()
trade_pivot = trade_monthly.pivot(index='Month', columns='Trade Type', values='Observation Value').reset_index()
trade_pivot.columns.name = None

rename_map = {}
for c in trade_pivot.columns:
    if isinstance(c, str):
        lc = c.lower()
        if 'import' in lc: rename_map[c] = 'Total_Imports'
        elif 'export' in lc: rename_map[c] = 'Total_Exports'
trade_pivot = trade_pivot.rename(columns=rename_map)

numeric_climate_cols = ['Month'] + [c for c in climate.columns if climate[c].dtype.kind in 'fi' and c!='Month']
climate_monthly = climate[numeric_climate_cols].drop_duplicates(subset='Month').reset_index(drop=True)

final_merged = trade_pivot.merge(climate_monthly, on='Month', how='left').fillna(method='ffill').fillna(method='bfill')
df = final_merged.copy()
df['month'] = df['Month'].dt.month

# --- FINAL MODEL TRAINING & FORECASTING ---
exog_vars = ['month']

# 1. Imports Model
X_final_imp, scaler_final_imp = prepare_exog(df.set_index('Month'), exog_vars)
y_target_imp = df.set_index('Month')['Total_Imports'].loc[X_final_imp.index]
y_scaler_imp = MinMaxScaler().fit(y_target_imp.values.reshape(-1, 1))
y_scaled_imp = pd.Series(y_scaler_imp.transform(y_target_imp.values.reshape(-1, 1)).flatten(), index=y_target_imp.index)

res_imp = SARIMAX(y_scaled_imp, exog=X_final_imp, order=(1,1,1), seasonal_order=(0,1,1,12)).fit(disp=False)
fitted_imp = y_scaler_imp.inverse_transform(res_imp.fittedvalues.values.reshape(-1, 1)).flatten()

# 2. Exports Model
X_final_exp, scaler_final_exp = prepare_exog(df.set_index('Month'), exog_vars)
y_target_exp = df.set_index('Month')['Total_Exports'].loc[X_final_exp.index]
y_scaler_exp = MinMaxScaler().fit(y_target_exp.values.reshape(-1, 1))
y_scaled_exp = pd.Series(y_scaler_exp.transform(y_target_exp.values.reshape(-1, 1)).flatten(), index=y_target_exp.index)

res_exp = SARIMAX(y_scaled_exp, exog=X_final_exp, order=(1,1,1), seasonal_order=(0,1,1,12)).fit(disp=False)
fitted_exp = y_scaler_exp.inverse_transform(res_exp.fittedvalues.values.reshape(-1, 1)).flatten()

# 3. Future Forecast with Confidence Intervals
future_dates = pd.date_range(start=df['Month'].max(), periods=FORECAST_HORIZON + 1, freq='MS')[1:]
X_future = pd.DataFrame({'month': future_dates.month}, index=future_dates)
X_future_scaled_imp, _ = prepare_exog(X_future, exog_vars, scaler=scaler_final_imp)
X_future_scaled_exp, _ = prepare_exog(X_future, exog_vars, scaler=scaler_final_exp)

# Imports Forecast Obj
fcast_obj_imp = res_imp.get_forecast(steps=FORECAST_HORIZON, exog=X_future_scaled_imp)
fcast_imp = y_scaler_imp.inverse_transform(fcast_obj_imp.predicted_mean.values.reshape(-1, 1)).flatten()
conf_imp = fcast_obj_imp.conf_int()
low_imp = y_scaler_imp.inverse_transform(conf_imp.iloc[:, 0].values.reshape(-1, 1)).flatten()
up_imp = y_scaler_imp.inverse_transform(conf_imp.iloc[:, 1].values.reshape(-1, 1)).flatten()

# Exports Forecast Obj
fcast_obj_exp = res_exp.get_forecast(steps=FORECAST_HORIZON, exog=X_future_scaled_exp)
fcast_exp = y_scaler_exp.inverse_transform(fcast_obj_exp.predicted_mean.values.reshape(-1, 1)).flatten()
conf_exp = fcast_obj_exp.conf_int()
low_exp = y_scaler_exp.inverse_transform(conf_exp.iloc[:, 0].values.reshape(-1, 1)).flatten()
up_exp = y_scaler_exp.inverse_transform(conf_exp.iloc[:, 1].values.reshape(-1, 1)).flatten()

# --- CONSTRUCT FINAL_OUTPUT_DF ---
actuals_df = df[['Month', 'Total_Imports', 'Total_Exports']].copy().rename(columns={'Total_Imports': 'Imports', 'Total_Exports': 'Exports'})
actuals_df['Data_Type'] = 'Actual'

overlap_df = pd.DataFrame({
    'Month': df['Month'].iloc[-OVERLAP_MONTHS:],
    'Imports': fitted_imp[-OVERLAP_MONTHS:],
    'Exports': fitted_exp[-OVERLAP_MONTHS:]
})
future_df = pd.DataFrame({'Month': future_dates, 'Imports': fcast_imp, 'Exports': fcast_exp})
full_forecast_df = pd.concat([overlap_df, future_df], ignore_index=True)
full_forecast_df['Data_Type'] = 'Forecast'

final_output_df = pd.concat([actuals_df, full_forecast_df], ignore_index=True)
final_output_df['Balance_of_Trade'] = final_output_df['Exports'] - final_output_df['Imports']

# --- VISUALIZATION ---
actuals_plot = final_output_df[final_output_df['Data_Type'] == 'Actual']
forecast_plot = final_output_df[final_output_df['Data_Type'] == 'Forecast']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

# Exports Chart
ax1.plot(actuals_plot['Month'], actuals_plot['Exports'], label='Actual Exports', color='blue', linewidth=2)
ax1.plot(future_dates, fcast_exp, label='Forecast Exports', color='red', linestyle='--', linewidth=2)
ax1.fill_between(future_dates, low_exp, up_exp, color='red', alpha=0.15, label='95% Confidence Interval')
ax1.set_title('Exports: Historical vs Forecast (with Confidence Intervals)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Imports Chart
ax2.plot(actuals_plot['Month'], actuals_plot['Imports'], label='Actual Imports', color='green', linewidth=2)
ax2.plot(future_dates, fcast_imp, label='Forecast Imports', color='orange', linestyle='--', linewidth=2)
ax2.fill_between(future_dates, low_imp, up_imp, color='orange', alpha=0.15, label='95% Confidence Interval')
ax2.set_title('Imports: Historical vs Forecast (with Confidence Intervals)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

final_output_df