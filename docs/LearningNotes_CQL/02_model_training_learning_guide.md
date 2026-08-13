# Model Training Learning Guide

This file teaches you how to train the model step by step.

## 1. What training means

Training (训练) means the model learns from past data.

In your project:

1. Input data = temperature, wind, time features.
2. Output label = electricity price.

The model learns a pattern between the input and the output.

## 2. Which model should you use

My recommendation:

1. First baseline model (基线模型).
2. Then XGBoost (XGBoost，梯度提升树) as the main model.

Why XGBoost is good for you:

1. It works well on tabular data (表格数据).
2. It does not require you to understand advanced calculus first.
3. It often performs strongly on price prediction problems.
4. It can handle non-linear patterns (非线性关系).

You do not need to start with neural networks (神经网络).

## 3. What you must understand before training

Before training, learn these ideas:

1. Feature (特征): a column used as input.
2. Label or target (标签/目标值): the value you want to predict.
3. Train set (训练集): data used to teach the model.
4. Test set (测试集): data used to check the model.
5. Data leakage (数据泄漏): using future information by mistake.
6. Lag feature (滞后特征): past values used as input.
7. Time-series split (时间序列切分): split data by time, not randomly.

If you understand only these ideas, you can already start.

## 4. Learning path for model training

### Part A. Learn the simplest prediction idea

Start with one very simple question:

Can the next price be predicted by the last price?

This is your first baseline.

If this simple idea already works badly, that is useful information.

### Part B. Learn one-step prediction first

Do not start with a full 7-day model on day one.

Start with:

1. Next hour prediction.
2. Then next day prediction.
3. Then 7-day ahead prediction.

This sequence is much easier to understand.

### Part C. Learn how to evaluate correctly

For time series, you must keep time order.

That means:

1. Train on past data.
2. Test on later data.
3. Never mix future data into the past.

## 5. Training workflow

### Step 1. Prepare one clean table

Before model training, create one final dataset.

This dataset should contain:

1. Time column.
2. Price target.
3. Temperature.
4. Wind speed.
5. Wind direction features.
6. Time-based features.
7. Lag features.

### Step 2. Define the target

You need to decide what the model predicts.

For example:

1. Price at the next hour.
2. Price 24 hours later.
3. Price 7 days later.

For a beginner, one-step prediction is easier.

### Step 3. Split data by time

Use a time-based split (按时间切分).

Example idea:

1. First period = training.
2. Middle period = validation.
3. Last period = test.

Do not random shuffle.

### Step 4. Train the baseline

Before XGBoost, train a baseline.

Possible baselines:

1. Last value model.
2. Simple linear regression (线性回归).

The baseline is not the final answer.

It is only your comparison point.

### Step 5. Train XGBoost

XGBoost is your main model.

Training idea:

1. Use your feature columns as X.
2. Use the price column as y.
3. Fit the model on the training data.
4. Predict on validation and test data.

At this point, you only need to understand the workflow, not the inner math.

### Step 6. Check the result

Use these metrics:

1. MAE (平均绝对误差).
2. RMSE (均方根误差).
3. MAPE (平均绝对百分比误差), if useful.

Also check the plot:

1. Real value line.
2. Predicted value line.
3. Error over time.

### Step 7. Improve carefully

If the result is weak, improve in this order:

1. Clean the data better.
2. Add better lag features.
3. Add rolling mean features (滚动平均特征).
4. Add calendar features.
5. Tune XGBoost parameters (参数调优).

Do not try to change everything at the same time.

## 6. What to learn first as a beginner

Focus on these topics first:

1. Pandas data loading.
2. Datetime handling.
3. Resampling.
4. Merge and join.
5. Missing value handling.
6. Basic plotting.
7. Train/test split for time series.
8. XGBoost fitting.

Do not start with deep matrix algebra (矩阵代数).

## 7. What to understand about 7-day prediction

There are two common ways:

### Option 1. Direct forecast

Train the model to predict a specific future horizon.

Examples:

1. 1 hour ahead.
2. 24 hours ahead.
3. 168 hours ahead.

This is easier to explain.

### Option 2. Recursive forecast

Predict one step, then use that prediction as input for the next step.

This is harder and can accumulate error.

For your first project, I recommend the direct approach first.

## 8. Good habits during training

1. Keep a notebook of what you changed.
2. Change one thing at a time.
3. Save each model version.
4. Compare every new result with the baseline.
5. Write down the best settings.

This will make the project much easier to defend in front of your supervisor.

## 9. When you are ready to move forward

You are ready for the next step when you can do all of this:

1. Load the final dataset.
2. Split it by time.
3. Train a baseline model.
4. Train XGBoost.
5. Explain the result in simple words.

If you can do these five things, you are ready for evaluation and later production.
