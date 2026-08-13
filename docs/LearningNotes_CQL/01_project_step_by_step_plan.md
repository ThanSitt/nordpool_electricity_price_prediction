# Nordpool 7-Day Price Prediction Plan

This file is your main roadmap.

## 1. Short answer first

For your current skill level, the best choice is a hybrid path:

1. Use simple baseline models first.
2. Then use XGBoost (XGBoost，梯度提升树) as the main model.
3. Do not start by learning calculus, derivatives, matrices, or gradients in depth.

You can learn those later if you want, but you do not need them first to finish this project.

## 2. Your project goal

The goal is to predict Finland electricity spot prices (芬兰电价预测) for the next 7 days.

This means:

1. You collect weather and market data.
2. You clean and align the data.
3. You train a model.
4. You evaluate the model.
5. You make a prediction pipeline that can run again in production (生产环境).

## 3. My recommendation for your first model

Start with this first version:

1. Target: electricity price.
2. Main features: temperature (温度), wind speed (风速), wind direction (风向).
3. Main model: XGBoost.
4. First time resolution: hourly (每小时), if your price file is hourly.

Why this is best for you:

1. It is easier to understand.
2. It is easier to clean and merge.
3. It is easier to debug.
4. It already gives a strong practical result.

If later you want 15-minute prediction (15分钟预测), you can upgrade the whole pipeline.

## 4. Important project idea

Your project is not only about model training.

The real work is the data pipeline (数据管道):

1. Read data.
2. Clean data.
3. Align time stamps.
4. Create features.
5. Train model.
6. Test model.
7. Save results.

This is the most important part of the whole summer project.

## 5. Full step-by-step roadmap

### Step 1. Understand the data sources

Read each source one by one.

You currently have:

1. Electricity prices in Helsinki.
2. Temperature data from Helsinki-Vantaa airport.
3. Wind speed and wind direction data from Oulu Vihreäsaari harbour.

For each file, answer these questions:

1. What is the time column?
2. What is the unit?
3. What is the time resolution?
4. Are there missing values?
5. Are timestamps local time or UTC (世界协调时间)?

Do not train anything yet.

### Step 2. Choose one common time resolution

For the first version, choose one time base.

If your price data is hourly, then make temperature and wind hourly too.

If later you change to 15-minute Nord Pool data, then convert all sources to 15-minute resolution.

For a beginner, hourly is the safer start.

### Step 3. Clean each source alone

Clean one file at a time.

For each file:

1. Read the CSV file.
2. Convert the time column into datetime (日期时间).
3. Sort by time.
4. Remove duplicated rows.
5. Check missing rows.
6. Rename columns into a simple style.
7. Save a cleaned version.

Do not merge files before each file is clean.

### Step 4. Fix wind direction

Wind direction (风向) is not a normal number.

You should not use raw degrees directly without thinking.

Better choices:

1. Convert wind direction to sin and cos.
2. Or convert wind speed and wind direction to u and v components.

This is important because 359 degrees and 1 degree are close, but plain numbers do not show that clearly.

### Step 5. Align all data by time

After cleaning, put all sources on the same time index.

Example:

1. Price data at hour 10:00.
2. Temperature at hour 10:00.
3. Wind at hour 10:00.

Now they can be merged into one table.

### Step 6. Create simple features

Start with easy features:

1. Hour of day (一天中的小时).
2. Day of week (星期几).
3. Weekend flag (周末标记).
4. Temperature.
5. Wind speed.
6. Wind direction features.
7. Lag price features (滞后特征), such as last hour price.

Do not create too many features in the beginning.

### Step 7. Build a baseline model

Before XGBoost, use a simple baseline model (基线模型).

Examples:

1. Predict the next price as the last observed price.
2. Use a simple linear regression (线性回归).

Why this matters:

1. It tells you if the real model is better than a simple guess.
2. It helps you understand whether your features are useful.

### Step 8. Train XGBoost

When the data is clean and features are ready, train XGBoost.

XGBoost is a tree-based model (树模型).

You do not need to understand the full math first.

What you need to know first:

1. Input features go in.
2. Price target goes out.
3. The model learns patterns from past data.

### Step 9. Evaluate the model

Use time-series evaluation (时间序列评估).

Do not shuffle the data.

Use these metrics (评估指标):

1. MAE (平均绝对误差).
2. RMSE (均方根误差).
3. MAPE (平均绝对百分比误差), if it makes sense for your data.

### Step 10. Make plots and understand errors

Plot the results.

Good plots:

1. Real price vs predicted price.
2. Temperature vs price.
3. Wind speed vs price.
4. Missing value chart.
5. Before and after resampling chart.

This is how you build understanding, not only numbers.

### Step 11. Build production style workflow

Later, make a repeatable workflow:

1. Load new data.
2. Clean new data.
3. Create features.
4. Load trained model.
5. Predict next 7 days.
6. Save output.

This is the production (生产) stage.

## 6. Learning order for you

Learn in this order:

1. Pandas basics (Pandas 基础).
2. Time-series basics (时间序列基础).
3. Data cleaning (数据清洗).
4. Simple visualization (数据可视化).
5. Baseline model.
6. XGBoost.
7. Evaluation.
8. Production pipeline.

Do not jump to advanced math first.

## 7. What you should be able to do before moving on

Before the next step, you should be able to explain:

1. What the time column means.
2. Why your data must have one common resolution.
3. Why wind direction needs special treatment.
4. Why the model should not be trained before cleaning.
5. Why a baseline is needed.

If you cannot explain one of these points, stop and review.

## 8. Final recommendation

For your first working version, use this path:

1. Hourly data.
2. Temperature + wind speed + wind direction.
3. Wind direction transformed into sin/cos or u/v.
4. Baseline model first.
5. XGBoost second.
6. Time-series split for evaluation.

This is the best balance between difficulty and progress for your current level.
