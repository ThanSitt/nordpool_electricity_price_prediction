# 10 Model & Forecast Visualization — Learning Guide

> Corresponding notebook: `notebooks/model_visualization.ipynb`
> This guide explains, step by step, WHY each part of the code is written and HOW it works.
> Technical terms (专业术语) are annotated with Chinese (中文) so you can learn both languages at once.

---

## 0. The Big Picture (整体思路)

The trained model files (`.pkl`, 模型文件) are like "frozen brains". This notebook does TWO things:

| Part | English               | 中文        | Goal (目标)                                                  |
| ---- | --------------------- | ----------- | ------------------------------------------------------------ |
| A    | Model Evaluation      | 模型评估    | Prove the model is trustworthy on HISTORICAL data (历史数据) |
| B    | Future 7-Day Forecast | 未来7天预测 | Show what the model predicts for the NEXT 7 days (未来7天)   |

Why both? First prove the model works (A), then trust its forecast (B). This is the professional workflow.

---

## Part A — Model Evaluation (模型评估)

### A0. Imports (导入库)

```python
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

- `joblib` — to LOAD (加载) the saved `.pkl` model files (deserialize, 反序列化)
- `plotly.express (px)` — quick interactive charts (交互式图表)
- `plotly.graph_objects (go)` — low-level control (e.g. adding extra lines)
- `sklearn.metrics` — to compute MAE / RMSE / R² (评估指标)
- `Path` — to build file paths safely across Windows/macOS/Linux (跨平台路径)

### A0b. Load features + chronological split (加载数据 + 按时间切分)

```python
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Europe/Helsinki')
...
X_test  = X.iloc[train_end:]
y_test  = y.iloc[train_end:]
```

WHY (为什么):

1. **Timezone fix (时区修复)** — the CSV has mixed offsets (mixed timezones, 混合时区: +02:00 winter / +03:00 summer). `utc=True` unifies them first, then we convert to Helsinki time. (This is the exact bug we fixed before.)
2. **Chronological split (按时间切分)** — `test_size = 20%`, take the LAST 20% as the test set. We NEVER shuffle (打乱), because time-series data must keep time order — the future must not leak into the past (data leakage, 数据泄漏).
3. The last 20% = the most recent period = a fair "exam" the model has never seen during training (训练时没见过).

### A0c. Load model + predict + metrics (加载模型 + 预测 + 指标)

```python
meta = joblib.load(MODELS_DIR / f'{model_name}.pkl')
model = meta['model']
feature_cols = meta['feature_cols']
y_pred = model.predict(X_test[feature_cols])
```

WHY:

1. `joblib.load` opens the saved model bundle (模型文件). The bundle contains 3 things: `model`, `feature_cols`, `step_min` (we saw this in an earlier guide).
2. `X_test[feature_cols]` — we feed the model ONLY the columns it was trained on (特征列). If we passed extra/missing columns, the prediction would be wrong.
3. Metrics (指标):
   - **MAE (平均绝对误差)** = average |actual − predicted|
   - **RMSE (均方根误差)** = sqrt of mean squared error (punishes big errors more)
   - **R² (决定系数)** = fraction of price variance explained (0~1, closer to 1 is better)

### A1. Actual vs Predicted — Time Series (时间序列折线图)

```python
steps = 7 * 24 * 4   # 7 days * 24 h * 4 quarters = 672 points
fig.add_trace(go.Scatter(x=test_dt[:steps], y=y_test[:steps], name='Actual'))
fig.add_trace(go.Scatter(x=test_dt[:steps], y=y_pred[:steps], name='Predicted'))
```

WHY: drawing real price and predicted price together over the first 7 days of the test set. If the two lines track each other, the model is reliable (可靠). We show only 7 days so the lines are readable (too many points would look like a solid blur).

### A2. Actual vs Predicted — Scatter Plot (散点图)

```python
idx = rng.choice(len(y_test), size=20000, replace=False)   # sample (抽样) for speed
fig = px.scatter(x=y_test.values[idx], y=y_pred[idx], opacity=0.3)
lims = [min(...), max(...)]
fig.add_trace(go.Scatter(x=lims, y=lims, name='Perfect fit (y=x)', ...))
```

WHY:

- One point per test sample. X = actual, Y = predicted.
- The red **identity line (y=x, 对角线参照线)** is "perfect prediction". Points near it = good. Above it = over-predict (预测过高); below = under-predict.
- `opacity=0.3` (透明度) because thousands of overlapping points would hide density. `sample` keeps the interactive chart fast (21k points is heavy).

### A3. Residual Distribution (残差分布直方图)

```python
residuals = y_test.values - y_pred     # residual (残差) = actual - predicted
fig = px.histogram(residuals, nbins=80)
fig.add_vline(x=0, ...)                # zero line (0线)
```

WHY:

- Residual (残差) = prediction error at each point.
- A good model: residuals centered near 0 (mean ≈ 0) and spread symmetrically (对称). We got `mean=0.095`, which is very close to 0 → no systematic bias (系统性偏差).
- A long tail (长尾巴) means the model sometimes makes big errors (e.g. during price spikes, 价格尖峰).

### A4. Feature Importance (特征重要性)

```python
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
fig = px.bar(imp.tail(20), orientation='h', ...)
```

WHY: XGBoost/LightGBM can tell us which input features (特征) they rely on most. This explains WHAT drives the price, and guides future feature engineering (特征工程). `tail(20)` shows the top 20.

---

## Part B — Future 7-Day Forecast (未来7天预测)

### B0. Load forecast CSVs (加载预测结果)

```python
for path in sorted(PREDICTIONS_DIR.glob('*_forecasts.csv')):
    f = pd.read_csv(path, parse_dates=['run_date', 'target_datetime'])
    ...
```

WHY: The daily forecast is generated by `src/predict_system.py`, which loads the SAME `.pkl` models and saves one CSV per model into `predictions/`. So these CSVs ARE "the model's 7-day forecast". We just read them — no need to re-run the heavy prediction.

Each CSV has:
| Column | 中文 | Meaning |
| --- | --- | --- |
| `run_date` | 运行日期 | when the forecast was made |
| `target_datetime` | 目标时间 | the predicted time slot |
| `predicted_price` | 预测价格 | forecasted price |
| `actual_price` | 真实价格 | real price (backfilled, 回填, once the day passed) |
| `abs_error` | 绝对误差 | \|actual − predicted\| |

### B1. All models — 7-day forecast (多模型对比)

```python
for name, f in forecasts.items():
    latest = f['run_date'].max()               # take this model's newest run
    f_latest = f[f['run_date'] == latest]
    fig.add_trace(go.Scatter(x=..., y=f_latest['predicted_price'], name=name))
```

WHY: draw every model's most recent 7-day prediction on one chart. You immediately see which models agree and which is an outlier (离群). Click the legend to show/hide lines (图例交互).

### B2. Best model forecast vs actual (最佳模型对比真实)

```python
name = 'xgboost_v2_5'                          # best model per README
...
actual = f_latest.dropna(subset=['actual_price'])
if not actual.empty:
    fig.add_trace(go.Scatter(..., name='Actual (backfilled)', ...))
```

WHY: overlay the REAL price (black dashed line) once it becomes available. After a few days you can visually judge how accurate this week's forecast was. `dropna` removes time slots whose actual price isn't published yet (还没公布).

---

## 5. How to Use (使用方法)

1. Run Part A cells first — verify the metrics match the README (e.g. xgboost_v2_5: MAE≈2.8, RMSE≈8.2, R²≈0.97).
2. Change `model_name = 'xgboost_v2_5'` to any of the 8 saved models (e.g. `lightgbm_v2_5`, `xgboost_v2_5_2`) and re-run Part A to compare.
3. Re-run Part B whenever `predictions/` gets a new daily run — the charts update automatically.

---

## 6. Glossary (术语表 中英对照)

| English               | 中文          | One-line explanation                         |
| --------------------- | ------------- | -------------------------------------------- |
| Model artifact / .pkl | 模型文件      | the saved trained model (joblib)             |
| Deserialize           | 反序列化      | load a file back into memory (`joblib.load`) |
| Time-series split     | 时间序列切分  | split by time, never shuffle                 |
| Data leakage          | 数据泄漏      | future info leaking into training            |
| Test set              | 测试集        | held-out data the model never trained on     |
| MAE                   | 平均绝对误差  | average absolute error                       |
| RMSE                  | 均方根误差    | root mean squared error                      |
| R² score              | 决定系数      | fraction of variance explained               |
| Residual              | 残差          | actual − predicted                           |
| Identity line         | 对角线参照线  | the y = x perfect-fit line                   |
| Scatter plot          | 散点图        | one dot per sample                           |
| Histogram             | 直方图        | counts of values in bins                     |
| Feature importance    | 特征重要性    | how much each feature matters                |
| Interactive chart     | 交互式图表    | hover / zoom / toggle                        |
| Backfill              | 回填          | fill actual prices after the day passes      |
| Outlier               | 离群点/异常值 | a point far from the rest                    |

---

_This guide corresponds to `notebooks/model_visualization.ipynb`. Key idea: Part A proves the model works (evaluation), Part B shows the model's 7-day forecast — together they answer "can I trust this model to predict the next week?" (我能信任这个模型预测未来一周吗？)_
