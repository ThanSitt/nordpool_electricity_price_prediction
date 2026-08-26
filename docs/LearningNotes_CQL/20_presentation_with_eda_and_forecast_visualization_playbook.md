# 20 演讲实战手册：结合 EDA 与 Forecast Visualization 的完整 Presentation 方案

> 目标：帮助你在课堂或答辩里，把数据可视化 (data visualization)、模型评估 (model evaluation)、以及 XGBoost (XGBoost) vs LightGBM (LightGBM) 对比讲清楚。
>
> 文档结构：
>
> 1. 中文回答你的关键问题（每个专业词后带英文）
> 2. 现场动态演示 (live demo) 的逐步操作清单
> 3. 可直接朗读的英文完整演讲稿 (full English speech script)

---

## 1. 先回答你的关键问题（中文）

### 1.1 你对两个笔记本的理解是否正确？

你的理解方向是正确的，但可以更准确：

- A 笔记本 [data_visualization/2.0_eda.ipynb](e:/Github/nordpool_electricity_price_prediction/data_visualization/2.0_eda.ipynb) 的作用是探索性数据分析 (Exploratory Data Analysis, EDA)。
- 它不是只在“特征工程之前”用一次，而是每次数据版本 (data version) 变化后，都应该重新看一次。
- B 笔记本 [data_visualization/2.0_forecast_visualization.ipynb](e:/Github/nordpool_electricity_price_prediction/data_visualization/2.0_forecast_visualization.ipynb) 的作用是模型评估 (model evaluation) + 7 天预测可视化 (7-day forecast visualization)。

更通俗地说：

- A 负责回答“数据长什么样，为什么要这样做特征工程 (feature engineering)”。
- B 负责回答“模型到底准不准，未来 7 天各模型预测有什么差异”。

### 1.2 XGBoost 和 LightGBM 对比时，参数要不要完全一样？

**不要要求“参数名称和数值完全一样”**，因为它们是两种不同算法 (algorithm)：

- XGBoost (XGBoost) 和 LightGBM (LightGBM) 的很多超参数 (hyperparameter) 本来就不一一对应。
- 例如 LightGBM 有 `num_leaves`，XGBoost 没有完全等价项。

真正的“公平比较 (fair comparison)”应该保持这些一致：

1. 同一个数据集 (dataset)
2. 同一个时间切分 (time split)
3. 同一个损失目标 (objective)，例如都优化 MAE (Mean Absolute Error)
4. 同一类交叉验证 (cross validation)，例如 TimeSeriesSplit
5. 相近的调参预算 (tuning budget)，例如 trial 数量接近

总结：

- **比较条件要一致**，但**参数形式不必完全一致**。

### 1.3 现在最新模型是不是就该比 XGBoost V4 和 LightGBM V3.1？

是，作为“当前最佳版本对比 (best-vs-best comparison)”你应该重点比：

- XGBoost V4（你的 XGBoost 最新强模型）
- LightGBM V3.1（当前项目总体最优）

但是演讲不能只比这两个，还要讲“演化路径 (evolution path)”：

- V1 / V1.5（天气基线）
- V2 / V2.5（特征工程带来的大提升）
- V2.5.3（XGBoost 调参）
- V3 / V4（供给侧特征：电网与核电）

这样老师会看到你不是只会“报最终分数”，而是理解了整个实验逻辑 (experiment logic)。

### 1.4 2.1 All Models Forecast 是本地预测线，怎么和实际价格比？

你提的问题非常专业。

当前 2.1 图（All Models）主要显示每个模型最近一次预测线 (predicted_price)，没有直接叠加实际线 (actual_price)。

你可以用两种方式对比：

1. 指标对比 (metric comparison)
2. 曲线对比 (curve comparison)

#### 方法 A：指标对比（推荐课堂用）

在预测 CSV 里已经有：

- `actual_price`
- `abs_error`

所以可以直接按模型统计：

- 已评估样本数 (evaluated rows)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

这能快速回答“谁更准”。

#### 方法 B：曲线对比（推荐演示用）

从每个模型最新预测里，筛选那些已经有 `actual_price` 的时间点，把 `predicted_price` 和 `actual_price` 画在同一张图。

这样你可以现场讲：

- 哪个模型跟踪真实走势 (trend tracking) 更好
- 哪个模型在尖峰时段 (spike period) 偏差更大

后面第 3 节我给你可直接复制到 notebook 的代码。

---

## 2. 演讲该讲什么：推荐结构（15-20 分钟）

### 2.1 建议时间分配

1. 项目目标与问题定义（2 分钟）
2. 数据与 EDA 关键发现（4 分钟）
3. 模型演化与公平对比逻辑（4 分钟）
4. Forecast 可视化与线上意义（4 分钟）
5. 结论 + 改进计划（2-4 分钟）

### 2.2 每一段你要展示什么

#### 第一段：项目目标

展示文件：

- [README.md](e:/Github/nordpool_electricity_price_prediction/README.md)

你要说清楚：

- 任务是短期电价预测 (short-term electricity price forecasting)
- 为什么要用天气 (weather) + 电网 (grid flow) + 核电 (nuclear output)
- 为什么要同时比较 XGBoost 和 LightGBM

#### 第二段：EDA（A 笔记本）

展示文件：

- [data_visualization/2.0_eda.ipynb](e:/Github/nordpool_electricity_price_prediction/data_visualization/2.0_eda.ipynb)

最值得讲的图：

1. 1.2 Price Distribution：说明价格分布偏态 (skewed distribution)
2. 1.3 Price by Hour：说明日内周期 (intraday seasonality)
3. 1.4a/1.4b：说明工作日与季节规律 (calendar effect)
4. 1.6 Correlation Heatmap：连接到特征工程设计
5. 1.7/1.8：解释 V3.1 新增供给侧特征 (supply-side features)

你要说清楚：

- EDA 不是“可有可无的画图”，而是后续特征工程 (feature engineering) 的证据链 (evidence chain)。

#### 第三段：模型演化

展示文件：

- [README.md](e:/Github/nordpool_electricity_price_prediction/README.md)
- [xgboost_models/modelV2.5.3.ipynb](e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.3.ipynb)
- [xgboost_models/modelV4.ipynb](e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV4.ipynb)
- [lightgbm_models/modelV3.1.ipynb](e:/Github/nordpool_electricity_price_prediction/lightgbm_models/modelV3.1.ipynb)

你要说清楚：

- 从基线 (baseline) 到特征工程，再到调参 (hyperparameter tuning) 的渐进过程
- 什么叫公平比较 (fair comparison)
- 最终 best-vs-best：XGBoost V4 vs LightGBM V3.1

#### 第四段：Forecast Visualization（B 笔记本）

展示文件：

- [data_visualization/2.0_forecast_visualization.ipynb](e:/Github/nordpool_electricity_price_prediction/data_visualization/2.0_forecast_visualization.ipynb)

最值得讲的图：

1. 1.1 Actual vs Predicted（走势跟踪）
2. 1.2 Scatter + y=x（系统偏差）
3. 1.3 Residual Histogram（误差分布）
4. 1.4 Feature Importance（模型解释）
5. 2.1 7-Day Forecast All Models（多模型分歧）

你要说清楚：

- Section 1 看历史测试集性能 (offline evaluation)
- Section 2 看实时预测输出 (operational forecast)
- 两者结合才是完整判断

#### 第五段：结论

你要明确给出：

- 谁是总体最优（LightGBM V3.1）
- 谁是 XGBoost 最优（XGBoost V4）
- 你的学习收获（从语法到系统化建模）
- 下一步计划（例如滚动重训练、特征漂移监控）

---

## 3. 动态演示：一步一步怎么操作

下面是你现场“按键式”操作流程。

### 3.1 演示前准备（2 分钟）

1. 打开 [data_visualization/2.0_eda.ipynb](e:/Github/nordpool_electricity_price_prediction/data_visualization/2.0_eda.ipynb)
2. 打开 [data_visualization/2.0_forecast_visualization.ipynb](e:/Github/nordpool_electricity_price_prediction/data_visualization/2.0_forecast_visualization.ipynb)
3. 确认 [models/saved](e:/Github/nordpool_electricity_price_prediction/models/saved) 下有模型文件 (model artifacts)
4. 确认 [predictions](e:/Github/nordpool_electricity_price_prediction/predictions) 下有预测 CSV

### 3.2 先演示 EDA（A 笔记本）

按顺序运行 A 笔记本的所有代码单元 (code cell)，重点停留在：

1. 图 1.2（价格分布）
2. 图 1.3（按小时箱线图）
3. 图 1.4（按周/按月均价）
4. 图 1.6（相关性热力图）
5. 图 1.7/1.8（核电与电网）

讲解策略：

- 每张图只回答一个问题。
- 不要一张图讲三分钟，控制在 30-45 秒。

### 3.3 再演示模型评估（B 笔记本 Section 1）

先运行 B 笔记本前 6 个单元，确认打印出模型指标 (MAE/RMSE/R2)。

然后按模型切换演示：

1. 在“加载模型”的单元，把 `MODEL_NAME` 设为 `lightgbm_v3_1`，运行 Section 1 图表单元。
2. 再把 `MODEL_NAME` 改为 `xgboost_v4`，重复运行 Section 1 图表单元。

你要现场说：

- “现在我用相同测试集 (same test split) 对比两个最新模型。”
- “我只改模型名，其他条件不变，所以可比较性 (comparability) 是好的。”

### 3.4 演示 7 天预测（B 笔记本 Section 2）

运行 Section 2 的两个单元：

1. 先加载所有 `*_forecasts.csv`
2. 再画 2.1 All Models 预测图

你要现场说：

- 这张图是“每个模型最近一次 run_date 的未来 7 天预测线”。
- 线越接近说明模型意见更一致；分叉越大说明不确定性 (uncertainty) 更高。

### 3.5 追加一个“预测 vs 实际”对比（建议现场加分）

在 B 笔记本 Section 2 后面新增一个代码单元，粘贴下面代码：

```python
import numpy as np
import pandas as pd
import plotly.express as px

# 汇总每个模型已评估样本（actual_price 不为空）
rows = []
for name, f in forecasts.items():
    eval_f = f.dropna(subset=['actual_price']).copy()
    if eval_f.empty:
        continue
    err = eval_f['predicted_price'] - eval_f['actual_price']
    rows.append({
        'model': name,
        'evaluated_rows': len(eval_f),
        'mae': np.mean(np.abs(err)),
        'rmse': np.sqrt(np.mean(err ** 2)),
    })

leaderboard = pd.DataFrame(rows).sort_values('mae')
display(leaderboard)

fig = px.bar(
    leaderboard,
    x='model',
    y='mae',
    title='Evaluated Forecast MAE by Model (lower is better)',
    labels={'mae': 'MAE', 'model': 'Model'}
)
fig.show()
```

这段会直接回答你的问题：

- “All Models Forecast 只是预测线，那和真实价格怎么比？”
- 答案：用 `actual_price` 和 `predicted_price` 计算每个模型的 MAE/RMSE，并画排行图。

---

## 4. 演讲时建议重点对比的内容

### 4.1 模型层面的对比

1. 基线阶段：V1.5 vs V2.5
2. 调参阶段：V2.5 vs V2.5.3
3. 最新阶段：XGBoost V4 vs LightGBM V3.1

### 4.2 图表层面的对比

1. Actual vs Predicted 折线重合度
2. Scatter 点云对角线贴合度
3. Residual 分布是否居中、是否长尾
4. 7-Day Forecast 各模型分歧区间

### 4.3 结论层面的对比

- 准确率 (accuracy) 结论：谁 MAE 更低
- 稳定性 (stability) 结论：谁残差更集中
- 业务可用性 (operational usability) 结论：谁在预测区间更平滑、异常更少

---

## 5. 你最关心问题的简短口语答案（中文）

### Q1：是不是参数必须一模一样才能比？

不是。要保证的是比较协议 (comparison protocol) 一致，而不是参数名字一致。

### Q2：是不是直接比 XGBoost V4 和 LightGBM V3.1 就够了？

如果是“当前最优对决 (best-vs-best)”可以这样做，但演讲里最好先讲演化过程 (evolution)，再讲最终对决。

### Q3：All Models Forecast 怎么和实际价格比？

用 `actual_price` 与 `predicted_price` 做后评估 (backtesting-like evaluation)，计算每个模型 MAE/RMSE，再配合曲线图展示。

---

## 6. Full English Speech Script (可直接朗读)

> 下面这份演讲稿按 15-20 分钟设计，全部英文。你可以直接照读，也可以按你的语速删减。

### 6.1 Opening (about 1 minute)

Good morning everyone.

Today I will present my electricity price forecasting project for Finland Nord Pool day-ahead market.

My focus is not only model training, but also the full workflow: data understanding, feature engineering, model comparison, and operational forecasting.

I will show two visualization notebooks:

First, EDA, where we understand data patterns.

Second, forecast visualization, where we evaluate model quality and compare 7-day forecasts across models.

### 6.2 Project Goal and Why This Matters (about 1 minute)

The project goal is to predict short-term electricity prices at 15-minute resolution.

This is important because electricity prices are highly dynamic and affected by demand, weather, cross-border grid flows, and nuclear supply.

In this project, we compare two gradient boosting algorithms:

XGBoost and LightGBM.

### 6.3 EDA Story (about 4 minutes)

Now I start with EDA.

In this notebook, we use the V3.1 dataset, which includes weather, lag features, rolling features, grid flow features, and nuclear features.

First, I show the price distribution.

It is not symmetric. It has a heavy right tail, which means extreme high-price events exist.

This tells us why forecasting is difficult, especially during spikes.

Second, I show the boxplot by hour.

We clearly see intraday seasonality:

morning and evening hours are generally higher, and night hours are lower.

This supports using calendar and time-based features.

Third, I show average price by weekday and month.

This confirms calendar effects and seasonal effects.

Winter months tend to be higher, which is consistent with heating demand.

Fourth, I show the correlation heatmap.

Price lags and rolling features are strongly informative.

This is why feature engineering gives a major performance jump from V1/V1.5 to V2/V2.5.

Finally, I show new V3.1 supply-side features:

nuclear output and cross-border grid flows.

These plots justify why we tested V3 and V4 model families.

So the key message from EDA is:

we do not engineer features blindly.

We engineer features based on observed structure in the data.

### 6.4 Model Evolution and Fair Comparison (about 4 minutes)

Now I explain the model evolution.

V1 and V1.5 are weather-only baselines.

They are useful starting points but weak in accuracy.

V2 and V2.5 add feature engineering and improve performance significantly.

Then we move to tuning and supply-side extensions.

An important question is:

How should we compare XGBoost and LightGBM fairly?

Fair comparison does not mean identical parameter names,

because the two algorithms are different.

Fair comparison means same data, same split logic, same objective focus, and similar tuning budget.

For current best-vs-best comparison, we focus on:

XGBoost V4 and LightGBM V3.1.

### 6.5 Forecast Visualization and Dynamic Demo (about 5 minutes)

Now I open the forecast visualization notebook.

In Section 1, I evaluate one selected model on the same held-out test set.

I first run LightGBM V3.1, then switch MODEL_NAME to XGBoost V4.

I compare four views:

Actual vs Predicted line chart,

Actual vs Predicted scatter with y=x line,

Residual histogram,

and Top feature importance.

This gives both performance and interpretability.

In Section 2, I load all latest forecast CSV files and show 7-day forecast lines for all models.

This chart is useful for seeing model disagreement and forecast spread.

But to compare with real prices, forecast lines alone are not enough.

So I add one extra step:

I use actual_price and predicted_price in forecast CSVs,

calculate MAE and RMSE per model,

and build a ranked bar chart.

This turns visualization into quantitative model monitoring.

### 6.6 Final Conclusion (about 2 minutes)

I want to conclude with three points.

First, feature engineering is the main reason for the big performance improvement.

Second, fair model comparison requires consistent protocol, not identical parameter names.

Third, combining offline evaluation and rolling forecast monitoring gives a complete view of model quality.

At this stage, LightGBM V3.1 is the best overall model,

and XGBoost V4 is the best XGBoost model.

My next steps are model drift monitoring and periodic retraining.

Thank you.

---

## 7. 演讲结束时可加的一句英文

The key value of this project is not only building one accurate model, but building a transparent and reproducible forecasting system from data understanding to operational monitoring.
