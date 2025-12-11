# AMS-595_Group_Project

# Walmart Weekly Sales – Forecasting, Seasonality and Anomaly Detection

This project analyzes Walmart weekly store‐level sales and builds several
time–series and machine learning models to understand patterns, forecast demand,
and detect unusual (anomalous) weeks.

The code in this repository:

- Cleans and enriches the original **Walmart_Sales.csv** file  
- Engineers rich calendar, economic and lag features  
- Trains and compares **ARIMA**, **Prophet**, **XGBoost**, and **LSTM** models  
- Performs **anomaly detection** using residuals, z-scores and Isolation Forest  
- Fits **polynomial** and **spline** curves to study overfitting vs smooth trends  
- Uses **Random Forest + SHAP** to interpret promotion / macro drivers of sales  

---

## 1. Data

- **File:** `Walmart_Sales.csv`
- **Grain:** Weekly sales per store
- **Key columns:**
  - `Store`, `Date`, `Weekly_Sales`
  - `Holiday_Flag`
  - Economic drivers: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`

The script:

- Parses `Date` as `DD-MM-YYYY`  
- Drops corrupted store `Store = 36`  
- Sorts by `Store`, `Date`

---

## 2. Feature Engineering

Calendar / seasonality features:

- `Year`, `Month`, `Week`, `WeekOfYear`
- Fixed holiday flags:
  - Super Bowl, Labor Day, Thanksgiving, Black Friday, Christmas
- Additional events:
  - `Easter_Flag`, `TaxRefund_Flag`, `July4_Flag`, `BackToSchool_Flag`
  - `Christmas_Leadup` (3 weeks before Christmas)

Lag and rolling features (per store):

- Lags: `Prev_Week_Sales`, `Sales_2Weeks_Ago`, `Sales_3Weeks_Ago`, `Lag_52`
- Rolling statistics over 3 weeks:
  - `RollingMean_3Weeks`, `RollingStd_3Weeks`, `RollingTrend_3Weeks`
- Holiday-aware rolling mean:
  - `RollingMean_3Weeks_noHoliday` (excludes holiday weeks)

Economic features:

- Week-to-week deltas: `Delta_Fuel`, `Delta_CPI`, `Delta_Unemp`, `Delta_Temp`
- 4-week rolling averages: `Fuel_4wk`, `Unemp_4wk`, `CPI_4wk`, `Temp_4wk`
- Store-wise standardized versions: `*_z` for CPI, temperature, fuel price, unemployment

A correlation heatmap summarizes relationships between `Weekly_Sales` and
all engineered features.

---

## 3. Exploratory Analysis

For a randomly selected store:

- Time series of weekly sales
- Average sales on **holiday vs non-holiday** weeks and holiday lift %
- **Seasonality plots:**
  - Average weekly sales by `WeekOfYear`
  - Average weekly sales by `Month`

Economic relationships:

- Pearson correlations between `Weekly_Sales` and economic variables  
- 2×2 grid of scatter plots: sales vs `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`

---

## 4. Forecasting Models

All models are trained and evaluated on a single chosen store.

### 4.1 ARIMA (pmdarima)

- Uses `auto_arima` with:
  - Weekly frequency (`m=52`) and seasonal component
  - AICc for model selection
- Train–test split: last 10 weeks as hold-out
- Metrics: **RMSE**, **MAE**
- Plot: actual vs ARIMA forecast for the last 10 weeks

### 4.2 Prophet

- Prophet setup:
  - Weekly and yearly seasonality enabled
  - Daily seasonality disabled
- Uses the same 10-week horizon and date index as ARIMA
- Metrics: **RMSE**, **MAE**
- Prophet’s own forecast plot plus a comparison plot:
  - Actual vs ARIMA vs Prophet forecasts

### 4.3 XGBoost Regressor

- Features used (subset of engineered features):

  `Week, Month, Year,
   Holiday_Flag, fixed_holiday_flag,
   Prev_Week_Sales, Sales_2Weeks_Ago, Sales_3Weeks_Ago, Lag_52,
   RollingMean_3Weeks, RollingStd_3Weeks, RollingTrend_3Weeks,
   CPI_z, Temperature_z, Fuel_Price_z, Unemployment_z`

- Temporal split:
  - Train: dates `< 2012-01-01`
  - Test: dates `>= 2012-01-01`
- Model: `XGBRegressor` with tuned depth, learning rate and subsampling
- Metrics: **RMSE**, **MAE**
- Plot: actual vs XGBoost predictions over time

### 4.4 LSTM (Keras)

- Univariate sequence model using only `Weekly_Sales`
- Pipeline:
  - Scale sales to [0, 1] with `MinMaxScaler`
  - Use the previous `SEQ_LEN = 4` weeks to predict the next week
  - Split on `2012-01-01` (train vs test)
- Model:
  - `LSTM(32)` → `Dense(1)`
  - Loss: MSE, optimizer: Adam
- Metrics: **RMSE**, **MAE** (in original sales units)
- Plot: actual vs LSTM predictions on the test period

### 4.5 Model Comparison

A summary table compares all four models:

- `ARIMA`
- `Prophet`
- `XGBoost`
- `LSTM`

using RMSE and MAE, along with a bar chart for RMSE.

---

## 5. Anomaly Detection

Based on XGBoost residuals:

1. **Z-score method**
   - Compute residuals: `Weekly_Sales - Pred_XGB`
   - Standardize to `Residual_z`
   - Flag anomalies where `|z| > 3`
   - Plot: weekly sales with red markers on z-score anomalies

2. **Isolation Forest**
   - Fit `IsolationForest` on residuals
   - Map predictions to `Anomaly_IF` (0/1)
   - Plot: weekly sales with purple markers on Isolation Forest anomalies

A combined plot shows both anomaly types on the same timeline.

---

## 6. Promotion & Macro Drivers (Linear + RF + SHAP)

- Features: `Holiday_Flag`, `Fuel_Price`, `CPI`, `Unemployment`  
- **Linear Regression**:
  - Fits a simple model and prints coefficients and intercept
- **Random Forest Regressor**:
  - Learns nonlinear influence of these drivers
  - Prints global feature importances
- **SHAP analysis**:
  - Uses `shap.TreeExplainer`
  - Bar summary plot of feature importance contributions

This block explains which macro variables and holiday signals are most strongly
associated with sales changes.

---

## 7. Polynomial vs Spline Fits (Week 6)

For one selected store:

- Create a time index and scale it to `[0, 1]`
- Fit polynomial regressions with degrees **3, 5, 10, 15**
  - Compute MSE and condition number of the Vandermonde matrix
  - Plot actual vs each polynomial fit
- Fit a **cubic spline** (`UnivariateSpline`)
  - Evaluate MSE
  - Plot actual vs spline
- Comparison table: polynomial degrees, MSE, condition number, and spline MSE

This section illustrates overfitting, numerical instability, and the benefits of
smooth spline fits for long time series.

---

## 8. Environment and Requirements

### Python version

- Python 3.9+ (tested with 3.10)

### Key libraries

```bash
pip install numpy pandas matplotlib seaborn
pip install scikit-learn pmdarima prophet xgboost shap
pip install tensorflow  # or tensorflow-macos if on Apple Silicon
pip install scipy
