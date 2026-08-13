# Version 1.5 Plan: 15-Minute Price Prediction with XGBoost

This file is the detailed plan for upgrading from hourly (version 1.0) to 15-minute (version 1.5) resolution.

---

## 0. Understanding Version 1.5

### What is different from Version 1.0?

**Version 1.0 (already done):**

- Price resolution: hourly (每小时)
- Features: temperature, wind speed, wind direction
- Aggregation: 10-minute weather data -> hourly mean

**Version 1.5 (new):**

- Price resolution: 15-minute (15分钟)
- Features: same (temperature, wind speed, wind direction)
- Aggregation: 10-minute weather data -> 15-minute mean

### Why upgrade to 15-minute?

1. More granular data (更细的时间粒度).
2. More training samples (更多训练样本) — 4x more rows for the same date range.
3. Captures intraday patterns (捕捉日内模式) better.
4. Closer to real trading practice (更接近实际交易).

### Key challenge in version 1.5

The main difficulty is **resampling correctly** (正确的重采样):

- You have 10-minute weather data.
- You need 15-minute aligned price data.
- 15-minute is not a multiple of 10-minute, so **careful padding** (小心的填充) is needed.

---

## 1. Prerequisite: Understand What You Already Know

### Before starting version 1.5, you must explain:

1. What is a datetime index in Pandas?
2. How does `resample()` work?
3. Why can't you directly average wind direction angles?
4. What is a lag feature (滞后特征)?
5. What is data leakage (数据泄漏)?

### Self-check question:

"If I have 10-minute data and want 15-minute data, what happens to rows that don't align?"

**Answer:** Pandas `resample()` groups data by the new time bucket (新的时间桶) and aggregates (聚合) with the chosen method (mean, sum, etc.).

**If you cannot answer this clearly, STOP and review the materials below before proceeding:**

- Pandas resample tutorial (搜索: pandas resample tutorial)
- Your version 1.0 data cleaning notebook

---

## 2. Phase Overview: What You Will Build

### Phase 1: Data Loading and Inspection (数据加载与检查)

- Read 15-minute price file
- Read 10-minute temperature and wind files
- Check timestamps, timezones, missing values

### Phase 2: Data Cleaning (数据清洗)

- Standardize column names
- Convert timestamps to datetime
- Handle timezones (likely UTC already)
- Remove duplicates and invalid rows

### Phase 3: Resampling to 15-Minute (重采样到15分钟)

- Resample temperature from 10-min to 15-min (using mean)
- Resample wind speed from 10-min to 15-min (using mean)
- Resample wind direction from 10-min to 15-min (using sin/cos averaging first, then convert back)
- Verify alignment with price data

### Phase 4: Feature Engineering (特征工程)

- Encode wind direction (sin/cos)
- Create lag features (滞后特征): price_lag_1, price_lag_4 (15-min), price_lag_96 (24-hour)
- Create rolling features (滚动特征): rolling mean/std for 1h, 4h, 24h windows
- Create calendar features (日历特征): hour, minute, day_of_week, is_weekend, etc.

### Phase 5: Train/Test Split by Time (按时间分割)

- 80% training, 20% testing
- No random shuffle
- Training period: 2023-01-01 to 2024-08 (approximately)
- Testing period: 2024-09 to 2025-12 (approximately)

### Phase 6: Model Training (模型训练)

- Train XGBoost on 15-minute training set
- Evaluate on 15-minute test set
- Compare with version 1.0 baseline

### Phase 7: Validation and Export (验证与导出)

- Check predictions look reasonable
- Save model and predictions
- Save final dataset for production use

---

## 3. Step-by-Step Detailed Plan

### STEP 1: Load and Inspect 15-Minute Price Data

**Goal:** Understand the 15-minute price file structure.

**What you do:**

```python
import pandas as pd
import numpy as np

# Read the 15-minute price file
price_15min = pd.read_csv("data/originalData/electricPrices/15min_2023_2025.csv")

# Inspect
print("Shape:", price_15min.shape)
print("Columns:", price_15min.columns.tolist())
print("First 10 rows:")
print(price_15min.head(10))
print("\nData types:")
print(price_15min.dtypes)
print("\nMissing values:")
print(price_15min.isna().sum())
```

**What to check:**

1. Are there exactly 4 rows per hour (每小时4行)? (Because 60 min / 15 = 4)
2. Do the timestamps start at :00, :15, :30, :45 minutes?
3. Is there a timezone in the timestamp?
4. Any missing timestamps?

**Checklist before moving on:**

- [ ] I can explain the shape and structure of the 15-min price data.
- [ ] I know what timezone the timestamps are in.
- [ ] I can see how many rows there are per day.

**If stuck:** Refer to basic Pandas `read_csv`, `head()`, `dtypes`, `shape`.

---

### STEP 2: Load and Inspect 10-Minute Weather Data

**Goal:** Understand temperature and wind files (you likely did this in version 1.0, but now from the 15-min perspective).

**What you do:**

```python
# Read temperature files
temp_files = sorted(glob.glob("data/originalData/Temperature/*.csv"))
# Filter out any cleaned files; keep only the raw airport data
temp_files = [f for f in temp_files if "airport" in f.lower()]

# Read one file to check
temp_sample = pd.read_csv(temp_files[0])
print("Temperature columns:", temp_sample.columns.tolist())
print("Sample:")
print(temp_sample.head())

# Read wind files
wind_files = sorted(glob.glob("data/originalData/WindDirection&Speed/*.csv"))
wind_sample = pd.read_csv(wind_files[0])
print("\nWind columns:", wind_sample.columns.tolist())
print("Sample:")
print(wind_sample.head())
```

**What to check:**

1. Are both 10-minute resolution (10分钟)?
2. Do columns match what you expect?
3. Do timestamps use the same format?

**Checklist before moving on:**

- [ ] I understand the structure of both temperature and wind files.
- [ ] I know both are 10-minute resolution.
- [ ] I have a list of all files to read.

---

### STEP 3: Understand 15-Minute Alignment Challenge

**Goal:** Know exactly what happens when you resample 10-min to 15-min.

**Why this matters:**

With hourly (60 min), every hour has exactly 6 ten-minute buckets: 0-10, 10-20, ..., 50-60.  
With 15-minute, the alignment is **NOT perfect** (不完美对齐):

- 00:00-00:15 → contains 10-min rows at 00:00, 00:10
- 00:15-00:30 → contains 10-min rows at 00:15
- 00:30-00:45 → contains 10-min rows at 00:30, 00:40
- 00:45-01:00 → contains 10-min rows at 00:50

Pandas `resample()` will use the label (标签) of the bucket and aggregate rows that fall into it.

**Hands-on check:**

```python
# Create a small sample to see how 10-min resamples to 15-min
import pandas as pd

dates = pd.date_range("2023-01-01", periods=10, freq="10min")
sample_df = pd.DataFrame({
    "timestamp": dates,
    "value": np.random.randn(10)
})
sample_df = sample_df.set_index("timestamp")

print("Original 10-min data:")
print(sample_df)

resampled_15min = sample_df.resample("15min").mean()
print("\nResampled to 15-min:")
print(resampled_15min)
```

**What to understand:**

1. The first 15-minute bucket may have 1 or 2 rows from 10-minute data.
2. `resample()` groups all rows in that 15-minute window and applies the aggregation function.

**Checklist before moving on:**

- [ ] I understand that 10-min doesn't align perfectly with 15-min.
- [ ] I can run the sample resample code and see the result.

---

### STEP 4: Clean and Standardize All Data

**Goal:** Prepare all three data sources (price, temperature, wind) with consistent column names and datetime handling.

**Part A: Clean 15-Minute Price Data**

```python
# Read
price_15min = pd.read_csv("data/originalData/electricPrices/15min_2023_2025.csv")

# Rename columns
price_15min = price_15min.rename(columns={
    "timestamp": "timestamp",
    "value": "price_eur_mwh"
})

# Convert timestamp to datetime (it likely has timezone info already)
price_15min["timestamp"] = pd.to_datetime(price_15min["timestamp"])

# If timezone is not present, add it (假设芬兰时区)
if price_15min["timestamp"].dt.tz is None:
    price_15min["timestamp"] = price_15min["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Helsinki")
else:
    # If it already has UTC, convert to local
    price_15min["timestamp"] = price_15min["timestamp"].dt.tz_convert("Europe/Helsinki")

# Sort and remove duplicates
price_15min = price_15min.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

# Set index
price_15min = price_15min.set_index("timestamp")

print("Price 15-min cleaned. Shape:", price_15min.shape)
```

**Part B: Clean Temperature Data (combine all files)**

```python
import glob

temp_files = sorted(glob.glob("data/originalData/Temperature/*.csv"))
# Filter to keep only airport raw data
temp_files = [f for f in temp_files if "airport" in f.lower() and "clean" not in f.lower()]

temp_list = []
for f in temp_files:
    tmp = pd.read_csv(f)
    tmp = tmp.rename(columns={
        "Observation station": "station_name",
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "Time [Local time]": "time_local",
        "Air temperature mean [°C]": "temperature_c"
    })

    # Create timestamp from year, month, day, time_local
    tmp["timestamp"] = pd.to_datetime(
        tmp["year"].astype(str) + "-" +
        tmp["month"].astype(str).str.zfill(2) + "-" +
        tmp["day"].astype(str).str.zfill(2) + " " +
        tmp["time_local"].astype(str)
    )

    # Assume local time (芬兰时区)
    tmp["timestamp"] = tmp["timestamp"].dt.tz_localize("Europe/Helsinki")

    # Keep only timestamp and temperature
    tmp = tmp[["timestamp", "temperature_c"]]
    temp_list.append(tmp)

# Combine all temperature files
temp_df = pd.concat(temp_list, ignore_index=True)
temp_df = temp_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
temp_df = temp_df.set_index("timestamp")

print("Temperature cleaned. Shape:", temp_df.shape)
```

**Part C: Clean Wind Data (combine all files)**

```python
wind_files = sorted(glob.glob("data/originalData/WindDirection&Speed/*.csv"))

wind_list = []
for f in wind_files:
    w = pd.read_csv(f)
    w = w.rename(columns={
        "Observation station": "station_name",
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "Time [Local time]": "time_local",
        "Wind direction mean [°]": "wind_direction_deg",
        "Wind speed mean [m/s]": "wind_speed_ms"
    })

    w["timestamp"] = pd.to_datetime(
        w["year"].astype(str) + "-" +
        w["month"].astype(str).str.zfill(2) + "-" +
        w["day"].astype(str).str.zfill(2) + " " +
        w["time_local"].astype(str)
    )

    w["timestamp"] = w["timestamp"].dt.tz_localize("Europe/Helsinki")

    w = w[["timestamp", "wind_direction_deg", "wind_speed_ms"]]
    wind_list.append(w)

wind_df = pd.concat(wind_list, ignore_index=True)
wind_df = wind_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
wind_df = wind_df.set_index("timestamp")

print("Wind cleaned. Shape:", wind_df.shape)
```

**Checklist before moving on:**

- [ ] I can run all three cleaning blocks without errors.
- [ ] All three dataframes have datetime index with timezone.
- [ ] All three dataframes are sorted and deduplicated.

---

### STEP 5: Resample Weather Data to 15-Minute

**Goal:** Convert 10-minute weather data to 15-minute for alignment with price.

**Important:** Wind direction (风向) is circular, so we must handle it specially.

**Part A: Resample Temperature and Wind Speed (simple mean)**

```python
# Resample to 15-minute using mean
temp_15min = temp_df.resample("15min").mean()
wind_speed_15min = wind_df[["wind_speed_ms"]].resample("15min").mean()

print("Temperature 15-min shape:", temp_15min.shape)
print("Wind speed 15-min shape:", wind_speed_15min.shape)
```

**Part B: Resample Wind Direction (circular mean using sin/cos)**

```python
# Wind direction must be converted to sin/cos first
rad = np.deg2rad(wind_df["wind_direction_deg"].astype(float))
wind_df["wind_dir_sin"] = np.sin(rad)
wind_df["wind_dir_cos"] = np.cos(rad)

# Resample sin and cos
wind_dir_sin_15min = wind_df[["wind_dir_sin"]].resample("15min").mean()
wind_dir_cos_15min = wind_df[["wind_dir_cos"]].resample("15min").mean()

# Reconstruct angle from mean sin/cos
wind_dir_deg_15min = np.rad2deg(
    np.arctan2(wind_dir_sin_15min["wind_dir_sin"], wind_dir_cos_15min["wind_dir_cos"])
) % 360

wind_dir_deg_15min = pd.DataFrame({
    "wind_direction_deg": wind_dir_deg_15min,
    "wind_dir_sin": wind_dir_sin_15min["wind_dir_sin"],
    "wind_dir_cos": wind_dir_cos_15min["wind_dir_cos"]
})

print("Wind direction 15-min shape:", wind_dir_deg_15min.shape)
```

**Checklist before moving on:**

- [ ] Temperature 15-min data has correct shape (approx 4x price 15-min rows).
- [ ] Wind speed 15-min data aligns with temperature.
- [ ] Wind direction sin/cos are properly averaged.

---

### STEP 6: Merge Price and Weather at 15-Minute Resolution

**Goal:** Combine all three data sources into one master table.

```python
# Combine weather into one dataframe
weather_15min = pd.concat([
    temp_15min[["temperature_c"]],
    wind_speed_15min[["wind_speed_ms"]],
    wind_dir_deg_15min
], axis=1)

# Merge with price
final_df = price_15min.join(weather_15min, how="left")

# Check alignment
print("Final merged shape:", final_df.shape)
print("Missing values per column:")
print(final_df.isna().sum())
print("\nFirst rows:")
print(final_df.head())
```

**What to check:**

1. Do price and weather rows match (same number of rows after merge)?
2. How many missing values are there? (应该很少或为0)
3. Do the timestamps make sense?

**If missing values are high:**

- Use `fill_value=method` or interpolate carefully.
- Check if data ranges don't overlap.

**Checklist before moving on:**

- [ ] No more than a few missing values (< 1%).
- [ ] The merged table has price, temperature, wind_speed, wind_direction columns.
- [ ] Timestamps are continuous (or have expected gaps).

---

### STEP 7: Add Time and Calendar Features

**Goal:** Create features that help the model learn daily and seasonal patterns.

```python
# Add basic time features
final_df["hour"] = final_df.index.hour
final_df["minute"] = final_df.index.minute
final_df["day_of_week"] = final_df.index.dayofweek  # Monday=0, Sunday=6
final_df["day_of_year"] = final_df.index.dayofyear
final_df["month"] = final_df.index.month
final_df["is_weekend"] = final_df["day_of_week"].isin([5, 6]).astype(int)

# Create time-of-day feature (0-95, representing each 15-min slot in a day)
final_df["time_of_day"] = final_df["hour"] * 4 + final_df["minute"] // 15

# Add cyclical encoding for hour (使用sin/cos编码时间周期)
final_df["hour_sin"] = np.sin(2 * np.pi * final_df["hour"] / 24)
final_df["hour_cos"] = np.cos(2 * np.pi * final_df["hour"] / 24)

final_df["month_sin"] = np.sin(2 * np.pi * final_df["month"] / 12)
final_df["month_cos"] = np.cos(2 * np.pi * final_df["month"] / 12)

print("Calendar features added.")
print(final_df.head())
```

**Checklist before moving on:**

- [ ] Calendar features are added correctly.
- [ ] No NaN values in these new features.

---

### STEP 8: Create Lag Features (滞后特征)

**Goal:** Add past price information so the model can learn from recent history.

**Why lag features matter:**

- Electricity prices have strong intraday patterns (日内模式).
- Today's price depends on recent past.
- Lag features help capture these patterns without data leakage.

```python
# Lag features for price (prices from past time steps)
# 15-min lags:
final_df["price_lag_1"] = final_df["price_eur_mwh"].shift(1)   # 15 min ago
final_df["price_lag_4"] = final_df["price_eur_mwh"].shift(4)   # 1 hour ago
final_df["price_lag_24"] = final_df["price_eur_mwh"].shift(24) # 6 hours ago
final_df["price_lag_96"] = final_df["price_eur_mwh"].shift(96) # 24 hours ago
final_df["price_lag_672"] = final_df["price_eur_mwh"].shift(672) # 7 days ago

# Lag features for weather
final_df["temp_lag_1"] = final_df["temperature_c"].shift(1)
final_df["temp_lag_4"] = final_df["temperature_c"].shift(4)
final_df["wind_lag_1"] = final_df["wind_speed_ms"].shift(1)
final_df["wind_lag_4"] = final_df["wind_speed_ms"].shift(4)

print("Lag features added.")
print(final_df.head(100))  # Show first 100 to see when NaN disappears
```

**Key point about lag features:**

- Rows 0-671 will have NaN values (因为我们需要过去7天的数据).
- You will remove these rows later when creating train/test splits.

**Checklist before moving on:**

- [ ] Lag features are created with `.shift()`.
- [ ] I understand that early rows will have NaN.

---

### STEP 9: Create Rolling Features (滚动特征)

**Goal:** Add statistics over windows (时间窗口统计) to capture trends.

```python
# Rolling mean for price (价格滚动均值)
final_df["price_rolling_4"] = final_df["price_eur_mwh"].rolling(window=4).mean()     # 1h
final_df["price_rolling_96"] = final_df["price_eur_mwh"].rolling(window=96).mean()   # 24h

# Rolling std for price (价格滚动标准差 = volatility measure)
final_df["price_rolling_std_4"] = final_df["price_eur_mwh"].rolling(window=4).std()
final_df["price_rolling_std_96"] = final_df["price_eur_mwh"].rolling(window=96).std()

# Rolling mean for weather
final_df["temp_rolling_4"] = final_df["temperature_c"].rolling(window=4).mean()
final_df["temp_rolling_96"] = final_df["temperature_c"].rolling(window=96).mean()

final_df["wind_rolling_4"] = final_df["wind_speed_ms"].rolling(window=4).mean()
final_df["wind_rolling_96"] = final_df["wind_speed_ms"].rolling(window=96).mean()

print("Rolling features added.")
print(final_df.describe())
```

**Checklist before moving on:**

- [ ] Rolling features are created.
- [ ] I understand that rolling window creates NaN for early rows.

---

### STEP 10: Handle Missing Values and Remove Incomplete Rows

**Goal:** Clean up rows with NaN values before training.

```python
# Check missing values
print("Missing values before cleaning:")
print(final_df.isna().sum())

# Drop rows with any NaN (because of lag and rolling features)
final_df = final_df.dropna()

print("\nMissing values after cleaning:")
print(final_df.isna().sum())

print("\nFinal shape:", final_df.shape)
print("Date range:", final_df.index.min(), "to", final_df.index.max())

# Save cleaned data
final_df.to_csv("data/processed/final_master_table_15min.csv", index=True)
print("\nSaved to: data/processed/final_master_table_15min.csv")
```

**Checklist before moving on:**

- [ ] All NaN values are removed.
- [ ] I know the date range of the cleaned data.
- [ ] Data is saved.

---

### STEP 11: Create Train and Test Sets (按时间分割，不要随机打乱)

**Goal:** Split data into training and testing without shuffling.

**Important rule:** For time series, always split by time, never randomly shuffle (时间序列必须按时间分割，不能随机打乱).

```python
# Calculate split point (80% training, 20% testing)
split_idx = int(len(final_df) * 0.8)

# Split
train_df = final_df.iloc[:split_idx].copy()
test_df = final_df.iloc[split_idx:].copy()

print("Training set shape:", train_df.shape)
print("Training date range:", train_df.index.min(), "to", train_df.index.max())
print("\nTest set shape:", test_df.shape)
print("Test date range:", test_df.index.min(), "to", test_df.index.max())

# Define feature columns (exclude price, time features that are already encoded)
feature_cols = [col for col in final_df.columns if col != "price_eur_mwh"]

X_train = train_df[feature_cols]
y_train = train_df["price_eur_mwh"]

X_test = test_df[feature_cols]
y_test = test_df["price_eur_mwh"]

print("\nFeature columns:", feature_cols)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
```

**Checklist before moving on:**

- [ ] Training and test sets do not overlap.
- [ ] Test set is later in time than training set.
- [ ] Feature columns are correct (no price in X).

---

### STEP 12: Train XGBoost Model

**Goal:** Train a gradient boosting model on 15-minute data.

```python
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create and train model
model = xgb.XGBRegressor(
    n_estimators=100,       # Number of trees
    max_depth=6,            # Tree depth
    learning_rate=0.1,      # Learning rate (学习率)
    objective="reg:squarederror",  # Regression task
    random_state=42,
    verbosity=1
)

print("Training XGBoost model...")
model.fit(X_train, y_train)
print("Training complete!")

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n=== Model Performance ===")
print(f"Training MAE: {train_mae:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Top 10 Feature Importances ===")
print(feature_importance.head(10))
```

**Checklist before moving on:**

- [ ] Model trains without errors.
- [ ] Test MAE is reasonable (better than a naive baseline).
- [ ] Feature importances make sense (price lags should be high).

---

### STEP 13: Visualize Results

**Goal:** Plot predictions vs actual to see if the model works.

```python
import matplotlib.pyplot as plt

# Plot a sample of test predictions
plt.figure(figsize=(15, 6))
sample_test = test_df.iloc[:1000]  # First 1000 test rows
plt.plot(sample_test.index, sample_test["price_eur_mwh"], label="Actual", color="blue", linewidth=2)
plt.plot(sample_test.index, y_pred_test[:1000], label="Predicted", color="red", linewidth=1, alpha=0.7)
plt.xlabel("Timestamp")
plt.ylabel("Price (EUR/MWh)")
plt.title("15-Minute Price Prediction: Actual vs Predicted (Sample)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/predictions_15min_sample.png", dpi=100)
plt.show()

# Plot error distribution
plt.figure(figsize=(12, 4))
errors = y_test.values - y_pred_test
plt.subplot(1, 2, 1)
plt.hist(errors, bins=50, edgecolor='black')
plt.xlabel("Prediction Error (EUR/MWh)")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, errors, alpha=0.3, s=1)
plt.xlabel("Actual Price")
plt.ylabel("Error")
plt.title("Error vs Actual Price")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/error_analysis_15min.png", dpi=100)
plt.show()
```

**Checklist before moving on:**

- [ ] Predictions follow actual price movements.
- [ ] Errors look randomly distributed (not biased).

---

### STEP 14: Save Model and Results

**Goal:** Save the trained model for production use.

```python
import joblib

# Save model
model.save_model("models/xgboost_15min_v1_5.json")
print("Model saved to: models/xgboost_15min_v1_5.json")

# Save predictions
results_df = test_df.copy()
results_df["predicted_price"] = y_pred_test
results_df["error"] = results_df["price_eur_mwh"] - results_df["predicted_price"]
results_df.to_csv("results/predictions_15min.csv", index=True)
print("Predictions saved to: results/predictions_15min.csv")

# Save feature importance
feature_importance.to_csv("results/feature_importance_15min.csv", index=False)
print("Feature importance saved.")
```

**Checklist before moving on:**

- [ ] Model is saved.
- [ ] Predictions are saved.
- [ ] Feature importance is saved.

---

## 4. Comparison: Version 1.0 vs Version 1.5

| Aspect               | Version 1.0               | Version 1.5                          |
| -------------------- | ------------------------- | ------------------------------------ |
| Resolution           | Hourly (60 min)           | 15-Minute                            |
| Number of samples    | ~18,000 rows              | ~72,000 rows (4x more)               |
| Data granularity     | Coarse                    | Fine                                 |
| Resampling challenge | Simple (10 min -> 60 min) | Complex (10 min -> 15 min)           |
| Expected accuracy    | Baseline                  | Should improve with more samples     |
| Training time        | Fast                      | Slower (more data, more features)    |
| Inference latency    | N/A                       | Lower latency for real-time forecast |

---

## 5. Potential Issues and How to Fix Them

### Issue 1: Too many NaN values after lag/rolling features

**Cause:** Using lags of 672 creates many NaN rows.
**Fix:** Reduce to lag_96 and rolling_96 if needed, or keep and just drop NaN.

### Issue 2: Data leakage (数据泄漏)

**Cause:** Using future weather or future price as a feature.
**Fix:** Always use `.shift()` or ensure features are from time <= current time.

### Issue 3: Low accuracy on test set

**Cause:**

- Missing important features.
- Poor hyperparameter tuning.
- Data quality issues.
  **Fix:**
- Add more lag features.
- Try different max_depth and learning_rate.
- Check for data gaps or outliers.

### Issue 4: Train accuracy much higher than test accuracy

**Cause:** Overfitting (过拟合).
**Fix:**

- Reduce max_depth.
- Reduce n_estimators.
- Add regularization (更正项).
- Use more data.

---

## 6. What You Should Know Before Each Step

### Before STEP 1-3 (Data Loading):

- You should know what a CSV file is.
- You should know how to open and inspect files.

### Before STEP 4-6 (Data Cleaning):

- You should understand Pandas datetime handling.
- You should know what `.resample()` does.
- You should understand `.join()` and `.merge()`.

### Before STEP 7-9 (Feature Engineering):

- You should know what "features" are.
- You should understand lag and rolling window concepts.
- You should know why cyclical encoding (循环编码) is needed for time.

### Before STEP 10-11 (Train/Test Split):

- You should understand why random shuffle is bad for time series.
- You should know the difference between train and test sets.

### Before STEP 12-14 (Model Training):

- You should understand supervised learning basics.
- You should know what MAE and RMSE mean.
- You should have XGBoost installed (`pip install xgboost`).

---

## 7. Final Checklist: Can You Do This Before Proceeding?

- [ ] I can explain why 10-min resampling to 15-min is different from 10-min to 60-min.
- [ ] I can explain what data leakage is and why it's bad.
- [ ] I can explain the difference between lag features and rolling features.
- [ ] I can run Pandas `.resample()`, `.shift()`, `.rolling()` methods correctly.
- [ ] I understand that I should NOT shuffle time-series data when splitting.
- [ ] I know how to use `.join()` to combine multiple dataframes by datetime index.
- [ ] I can explain why wind direction needs sin/cos encoding.

If you cannot answer all of these, **STOP** and review the appropriate sections above before continuing.

---

## 8. Next Steps After Version 1.5 Is Complete

1. Compare version 1.0 and version 1.5 performance.
2. If version 1.5 is better, keep the 15-minute approach.
3. Try adding more features (e.g., external variables, holidays).
4. Try hyperparameter tuning for XGBoost.
5. Consider ensemble models (combining predictions).
6. Deploy to production when satisfied.

---

## 9. Learning References

**Pandas:**

- `pd.to_datetime()` — Convert strings to datetime
- `.resample()` — Change time frequency
- `.shift()` — Create lag features
- `.rolling()` — Create rolling window features
- `.set_index()` — Set datetime as index
- `.tz_localize()` and `.tz_convert()` — Timezone handling

**XGBoost:**

- `xgb.XGBRegressor` — Regression model
- `.fit()` — Train the model
- `.predict()` — Make predictions
- `feature_importances_` — See which features matter most

**Concepts:**

- Lag feature (滞后特征) — Past value from a previous time step
- Rolling feature (滚动特征) — Statistic over a time window
- Cyclical encoding (循环编码) — Sin/cos for circular values like time and direction
- Data leakage (数据泄漏) — Using information that wouldn't be available in real prediction
- Time-series split (时间序列切分) — Splitting by time, not randomly
