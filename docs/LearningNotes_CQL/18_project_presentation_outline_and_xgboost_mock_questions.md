# Project Presentation Outline and XGBoost Mock Questions

This note is written for a student who already understands basic Python, lists, dictionaries, and loops, but is still learning how Pandas (数据分析库) and machine learning code fit together.

The goal is to help you present this project clearly, using simple English (英语) with Chinese hints (中文提示) for technical terms.

---

## 1. How I Would Present This Project as the Student

Use one simple story:

Goal -> Data -> Feature Engineering (特征工程) -> Model Training (模型训练) -> Better Tuning (调参) -> Live Prediction (实时预测) -> Lesson

If you keep that order, your presentation will feel organized and easy to follow.

### Slide 1. Project Goal

What to say:

- This project predicts Finland Nord Pool day-ahead electricity prices.
- The target is price in EUR/MWh.
- The project has two paths: training in notebooks and prediction in the live source code.

Files to show:

- [README.md](/e:/Github/nordpool_electricity_price_prediction/README.md)
- [docs/Project-Learning-Notes.md](/e:/Github/nordpool_electricity_price_prediction/docs/Project-Learning-Notes.md)

What you should emphasize:

- The project is not only about training a model.
- It also includes live prediction (实时预测) and automation (自动化).

### Slide 2. Data and Project Evolution

What to say:

- The project started with weather only.
- V1 used hourly data.
- V1.5 changed the time resolution (时间分辨率) to 15 minutes.
- V2 added feature engineering (特征工程), which was the real breakthrough.

Files to show:

- [xgboost_models/modelV1.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.ipynb)
- [xgboost_models/modelV1.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.5.ipynb)
- [xgboost_models/modelV2.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.ipynb)
- [docs/LearningNotes_CQL/05_version_1_5_plan.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/05_version_1_5_plan.md)

Useful code anchors:

- [xgboost_models/modelV1.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.ipynb) shows the first hourly baseline.
- [xgboost_models/modelV1.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.5.ipynb) shows the 15-minute baseline.
- [xgboost_models/modelV2.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.ipynb) shows the jump from raw features to engineered features.

What you should emphasize:

- Resolution alone did not help much.
- Better features helped much more than just more rows.

### Slide 3. Why Feature Engineering Mattered

What to say:

- Electricity price is time-based data (时间序列数据).
- The model needs past prices, rolling statistics (滚动统计), and calendar features (日历特征).
- These features help the model learn price patterns.

Files to show:

- [data/convertData/V2.5_15min_features.csv](/e:/Github/nordpool_electricity_price_prediction/data/convertData/V2.5_15min_features.csv)
- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py#L32)
- [xgboost_models/modelV2.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.ipynb)

Useful code anchors:

- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py#L32) defines the price buffer (价格缓冲器).
- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py#L84) defines the weather buffer (天气缓冲器).
- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py#L253) starts the feature builder (特征构建器).

What you should emphasize:

- Lag features (滞后特征) and rolling features (滚动特征) are the key reason V2 and V2.5 improved.
- Wind direction is encoded with sin/cos because degrees are circular (循环的).

### Slide 4. Why V2.5 Became the Best Early Model

What to say:

- V2.5 used 15-minute data plus engineered features.
- It had many more training rows (训练样本) than hourly models.
- It achieved much better MAE (平均绝对误差), RMSE (均方根误差), and R2 (决定系数).

Files to show:

- [xgboost_models/modelV2.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.ipynb)
- [docs/Project-Learning-Notes.md](/e:/Github/nordpool_electricity_price_prediction/docs/Project-Learning-Notes.md)

What you should emphasize:

- V2.5 is the first model version that feels strong enough for real use.
- The model still needs careful testing because recursive forecasting (递归预测) can make errors grow.

### Slide 5. Why Tuning Was More Important Than Adding More Features

What to say:

- V2.5.2 compared XGBoost and LightGBM fairly.
- V2.5.3 tuned XGBoost with Optuna (超参数自动搜索).
- Good tuning improved XGBoost more than simply adding extra features.

Files to show:

- [xgboost_models/modelV2.5.2.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.2.ipynb)
- [xgboost_models/modelV2.5.3.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.3.ipynb)
- [docs/LearningNotes_CQL/12_why_more_features_did_not_help_and_xgboost_optuna_guide.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/12_why_more_features_did_not_help_and_xgboost_optuna_guide.md)

What you should emphasize:

- A weak model can make a good feature look useless.
- You should tune the model first, then test new features again.

### Slide 6. What Happened With Grid and Nuclear Features

What to say:

- Grid flow features and nuclear output features were added later.
- Some of them helped the model offline.
- But only features that can also exist in live prediction can be deployed safely.

Files to show:

- [xgboost_models/modelV3.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV3.ipynb)
- [xgboost_models/modelV3.1.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV3.1.ipynb)
- [xgboost_models/modelV4.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV4.ipynb)
- [lightgbm_models/modelV3.1.ipynb](/e:/Github/nordpool_electricity_price_prediction/lightgbm_models/modelV3.1.ipynb)
- [docs/LearningNotes_CQL/15_nuclear_data_and_v4_learning_guide.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/15_nuclear_data_and_v4_learning_guide.md)
- [docs/LearningNotes_CQL/16_models_folder_saved_vs_experiments_learning_guide.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/16_models_folder_saved_vs_experiments_learning_guide.md)

Useful code anchors:

- [src/config.py](/e:/Github/nordpool_electricity_price_prediction/src/config.py#L11) defines model paths and API settings.
- [src/config.py](/e:/Github/nordpool_electricity_price_prediction/src/config.py#L19) defines the 7-day forecast window.
- [src/config.py](/e:/Github/nordpool_electricity_price_prediction/src/config.py#L38) defines the grid history window.

What you should emphasize:

- Train and serve (训练和部署) must match.
- A feature that works in a notebook is not always safe in live prediction.

### Slide 7. Live Prediction Pipeline

What to say:

- The live system uses public APIs (公开接口) to fetch prices and weather.
- Then it builds the same features as training.
- Finally it loads saved models and writes forecast CSV files.

Files to show:

- [src/fetch_live.py](/e:/Github/nordpool_electricity_price_prediction/src/fetch_live.py)
- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py)
- [src/predict_system.py](/e:/Github/nordpool_electricity_price_prediction/src/predict_system.py)
- [tests/test_predict_system.py](/e:/Github/nordpool_electricity_price_prediction/tests/test_predict_system.py)
- [docs/LearningNotes_CQL/08_from_training_to_automation_learning_guide.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/08_from_training_to_automation_learning_guide.md)

Useful code anchors:

- [src/predict_system.py](/e:/Github/nordpool_electricity_price_prediction/src/predict_system.py#L31) loads all model bundles.
- [src/predict_system.py](/e:/Github/nordpool_electricity_price_prediction/src/predict_system.py#L44) runs the recursive forecast.
- [src/predict_system.py](/e:/Github/nordpool_electricity_price_prediction/src/predict_system.py#L86) back-fills actual prices.
- [src/predict_system.py](/e:/Github/nordpool_electricity_price_prediction/src/predict_system.py#L122) is the main workflow.

What you should emphasize:

- The script is the bridge from notebook training to daily use.
- GitHub Actions only starts the script; the script does the real work.

### Slide 8. Why the Tests Matter

What to say:

- The tests make sure the time logic is correct.
- They check that hourly models use hourly means.
- They also check that 15-minute models keep all four quarter-hour steps.

Files to show:

- [tests/test_predict_system.py](/e:/Github/nordpool_electricity_price_prediction/tests/test_predict_system.py)
- [tests/test_features.py](/e:/Github/nordpool_electricity_price_prediction/tests/test_features.py)

What you should emphasize:

- Time-series data cannot be handled like random rows.
- The tests protect the project from subtle mistakes.

### Slide 9. Final Result and Business Value

What to say:

- The final project is a working system, not only a model experiment.
- It can train, save, load, forecast, and evaluate.
- The best model version is a strong tuned XGBoost model, and LightGBM is also competitive.

Files to show:

- [README.md](/e:/Github/nordpool_electricity_price_prediction/README.md)
- [docs/Project-Learning-Notes.md](/e:/Github/nordpool_electricity_price_prediction/docs/Project-Learning-Notes.md)

What you should emphasize:

- You learned both model development and system thinking.
- That is the real value of the project.

---

## 2. Teacher Questions About Your XGBoost Work

Below are 10 likely questions your teacher could ask if you focus on XGBoost (XGBoost，梯度提升树). I also include short sample answers you can practice.

### 1. Why did V1 and V1.5 perform much worse than V2 and V2.5?

Sample answer:

V1 and V1.5 used only weather (天气) features, so the model had very little information about price behavior. V2 and V2.5 added lag features (滞后特征), rolling statistics (滚动统计), and calendar features (日历特征). These features let the model see the autoregressive (自回归) nature of electricity prices, so the error dropped a lot.

### 2. Why did you choose XGBoost instead of a neural network (神经网络)?

Sample answer:

I chose XGBoost because this is tabular data (表格数据), not images or text. XGBoost works well on structured rows and columns, and it is easier for me to understand and explain. For my current level, it is a strong and practical choice.

### 3. Why did you use shuffle=False in the train-test split?

Sample answer:

Because this is time-series data (时间序列数据). If I shuffle the data, future rows can leak (泄漏) into the training set, which gives a false result. Keeping time order makes the test set more realistic.

### 4. What is the most important feature group in V2.5?

Sample answer:

The most important group is price lag features (价格滞后特征), especially short lags and daily lags. Electricity price often depends on its recent history. Rolling mean and rolling standard deviation (滚动标准差) also help because they capture trend and volatility (波动).

### 5. Why did Optuna tuning help XGBoost so much?

Sample answer:

Optuna (自动超参数搜索) helped because the default XGBoost settings were not fully adapted to this dataset. By tuning learning rate (学习率), tree depth (树深度), and regularization (正则化), the model learned better and overfit less. This was more useful than simply adding extra weak features.

### 6. What is recursive forecasting (递归预测), and what problem does it create?

Sample answer:

Recursive forecasting means I predict one future step, then use that predicted value as input for the next step. It is necessary because future prices do not exist yet. The problem is that small errors can grow step by step, so the 7-day live forecast can be much worse than the offline test score.

### 7. Why did the high_volatility_prob feature not help?

Sample answer:

Because the classifier (分类器) that generated it was weak on the minority class. It carried some signal (信号), but the signal was not strong enough to improve the final price model. This shows that a feature can look meaningful alone but still hurt the full model.

### 8. How do you make sure the training features and live features match?

Sample answer:

I keep the live feature code in src/features.py, and it mirrors the training notebooks. The model bundle also stores feature_cols, so each model uses the exact column list it was trained with. This reduces train/serve gap (训练/部署差距).

### 9. Why did you add grid and nuclear features later?

Sample answer:

I added them because they are supply-side (供给侧) signals and can contain real information about price. Grid flow (电网流量) and nuclear output (核电出力) are closer to the market structure than weather alone. But I also had to check whether they were available in live prediction.

### 10. If you had to explain your best XGBoost result in one sentence, what would you say?

Sample answer:

My best XGBoost model is strong because it combines good feature engineering (特征工程), careful time-series validation (时间序列验证), and Optuna tuning (超参数搜索), while keeping the live prediction pipeline consistent.

---

## 3. Short Speaking Tips for the Presentation

1. Use simple sentences.
2. Say one idea at a time.
3. Always connect the notebook result to the business reason.
4. Do not say too much math (数学) if the teacher asks for a project story.
5. Use this pattern often: problem -> method -> result -> lesson.

Example sentence templates:

- "At first, I used only weather features."
- "Then I added lag features and rolling statistics."
- "The MAE became much lower after feature engineering."
- "After tuning, the model got even better."
- "The live system uses the same feature logic as training."

---

## 4. Best Files to Open Before the Presentation

If you want to review only a few files, open these first:

- [README.md](/e:/Github/nordpool_electricity_price_prediction/README.md)
- [docs/Project-Learning-Notes.md](/e:/Github/nordpool_electricity_price_prediction/docs/Project-Learning-Notes.md)
- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py)
- [src/predict_system.py](/e:/Github/nordpool_electricity_price_prediction/src/predict_system.py)
- [xgboost_models/modelV2.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.ipynb)
- [xgboost_models/modelV2.5.3.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.3.ipynb)
- [xgboost_models/modelV4.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV4.ipynb)

---

## 5. One-Sentence Summary You Can Use at the End

This project shows that electricity price prediction improves most when I combine time-series (时间序列) feature engineering, careful XGBoost tuning, and a live prediction system that uses the same logic as training.
