# Model Comparison & Automation Guide

## XGBoost vs LightGBM：有什么区别，哪个更好？

### 一句话对比

| 特性 | XGBoost | LightGBM |
|------|---------|----------|
| 生长方式 | 按层生长（level-wise） | 按叶子生长（leaf-wise） |
| 训练速度 | 中等 | 更快 |
| 内存占用 | 较高 | 较低 |
| 小数据集 | 更稳定 | 容易过拟合 |
| 大数据集 | 较慢 | 更快 |
| 调参难度 | 较简单 | 需要更精细 |

### 生长方式图解

**XGBoost（level-wise，按层生长）：**
```
第1层:         [根节点]
              /       \
第2层:    [节点]     [节点]
          /   \     /   \
第3层:  [叶] [叶] [叶] [叶]
```
**特点：** 每层所有节点同时分裂，树更平衡，不容易过拟合。

**LightGBM（leaf-wise，按叶子生长）：**
```
第1步:       [根节点]
              /
第2步:    [节点]
          /   \
第3步:  [叶] [节点]
              /   \
第4步:     [叶] [叶]
```
**特点：** 每次只分裂损失最大的叶子，树不对称，收敛更快但容易过拟合。

### 在本项目中的使用

- **XGBoost** 放在 `xgboost_models/` 文件夹，使用默认参数，适合初学者
- **LightGBM** 放在 `lightgbm_models/` 文件夹，使用 Optuna 自动调参，性能略优
- 两个模型的训练数据、特征完全一样，只是模型算法不同
- 训练完成后都保存到 `models/saved/`，供实时预测使用

### 哪个更好？

**对于你的项目：XGBoost 更容易上手，LightGBM 调优后更准。**
两者的 V2.5 版本差距不大（RMSE 都在 8-9 之间）。建议先跑通 XGBoost，再用 LightGBM 做对比。

---

## 实时预测自动化：项目是怎么自动运行的？

### 核心文件

| 文件 | 功能 |
|------|------|
| `src/config.py` | 设置路径、API 地址、预测天数 |
| `src/fetch_live.py` | 从网络上获取最新的电价和天气预报 |
| `src/features.py` | 构建特征（和训练 notebook 完全一致） |
| `src/predict_system.py` | 主程序：加载模型 → 跑预测 → 保存结果 |
| `src/utils.py` | 辅助函数 |

### 整体流程

```
                        ┌─────────────────────┐
                        │  src/fetch_live.py   │
                        │  获取实时数据         │
                        └──────────┬──────────┘
                                   │ 电价 + 天气预报
                                   ▼
                        ┌─────────────────────┐
                        │  src/features.py     │
                        │  构建特征             │
                        └──────────┬──────────┘
                                   │ 特征表格
                                   ▼
                        ┌─────────────────────┐
                        │  src/predict_system.py│
                        │  加载 saved/*.pkl    │
                        │  递归预测 7 天        │
                        │  保存到 predictions/  │
                        └──────────┬──────────┘
                                   │ CSV 文件
                                   ▼
                        ┌─────────────────────┐
                        │  predictions/        │
                        │  XGBoost_V2_pred.csv │
                        │  LightGBM_V2.csv ... │
                        └─────────────────────┘
```

### 步骤 1：获取实时数据（fetch_live.py）

`fetch_live.py` 从两个免费 API 获取数据：

1. **电价** — 从 Elering 的 NPS 接口获取芬兰日前电价（15分钟分辨率，免费，无需 API Key）
2. **天气预报** — 从两个来源获取：
   - FMI（芬兰气象研究所）：短时天气预报（未来约 2 天）
   - Open-Meteo（免费天气 API）：长期天气预报（剩余天数）

### 步骤 2：构建特征（features.py）

`features.py` 的代码**完全复制**了训练 notebook 中的特征工程步骤：
- 时间特征（hour, minute, day_of_week, season...）
- 周期编码（hour_sin/cos, month_sin/cos...）
- 节假日标志（is_holiday）
- 滞后特征（price_lag_1, price_lag_4, price_lag_96...）
- 滚动特征（rolling_mean, rolling_std...）
- 天气衍生特征（HDD, wind_power_proxy...）

为什么需要单独一个 `features.py`？因为训练时用 notebook 造特征，但实时预测时不能打开 notebook，所以必须用 `.py` 脚本做同样的事。

### 步骤 3：主程序（predict_system.py）

`predict_system.py` 做三件事：

**A. 加载所有训练好的模型**
```python
# 从 models/saved/ 加载所有 .pkl 文件
models = {}
for pkl_file in models/saved/*.pkl:
    model = joblib.load(pkl_file)
    # 每个模型都包含了训练时用的特征列表
```

**B. 递归预测 7 天**
```
真实历史价格 → 预测第 1 小时 → 把预测值加入"历史" → 预测第 2 小时 → ... → 直到 7 天填满
```
这个过程叫 **递归预测**（recursive forecasting），因为预测值会被当作下一轮的特征输入。

**C. 填入实际电价并计算误差**
- 当预测的时间已经过去（有了真实电价），自动填入 `actual_price` 列
- 计算 `abs_error = |actual - predicted|`
- 下次再跑时不会重复插入，而是更新已有行

### 步骤 4：GitHub Actions 自动运行

`.github/workflows/daily_forecast.yml` 是一个配置文件，告诉 GitHub：
- **什么时候运行**：每天 UTC 11:00（芬兰时间 13:00 / 14:00）
- **做什么**：
  1. 安装环境
  2. 运行测试
  3. 执行 `python src/predict_system.py`
  4. 把新的预测结果自动提交到仓库

**这样每天你不需要手动跑脚本，GitHub 会自动帮你跑。**

---

## 模型文件（.pkl）是什么？

`.pkl` 文件是用 `joblib` 保存的训练好的模型，包含：
- 模型参数和权重（树的结构）
- 训练时使用的特征列表（`feature_cols`）
- 模型的分辨率（`step_min`，小时=60，15分钟=15）

### 怎么保存模型？

```python
import joblib

# 训练完成后
model_meta = {
    'model': model_v25,          # 训练好的 XGBoost/LightGBM 对象
    'feature_cols': feature_cols, # 特征列名列表
    'step_min': 15,               # 15分钟分辨率
}
joblib.dump(model_meta, 'models/saved/xgboost_v2_5.pkl')
```

### 怎么加载模型做预测？

```python
import joblib
meta = joblib.load('models/saved/xgboost_v2_5.pkl')
model = meta['model']
feature_cols = meta['feature_cols']
step_min = meta['step_min']
```

### 现有的 6 个模型

| 文件名 | 算法 | 版本 |
|--------|------|------|
| `xgboost_v1.pkl` | XGBoost | V1（小时/仅天气） |
| `xgboost_v1_5.pkl` | XGBoost | V1.5（15分钟/仅天气） |
| `xgboost_v2.pkl` | XGBoost | V2（小时/工程特征） |
| `xgboost_v2_5.pkl` | XGBoost | V2.5（15分钟/工程特征） |
| `lightgbm_v2.pkl` | LightGBM | V2（小时/工程特征） |
| `lightgbm_v2_5.pkl` | LightGBM | V2.5（15分钟/工程特征） |

---

## 如何重新训练并更新模型？

1. 打开对应的 notebook（如 `xgboost_models/modelV2.5.ipynb`）
2. 逐格运行，训练完成后最后会保存模型到 `models/saved/`
3. 下次 `python src/predict_system.py` 会自动加载新模型

**注意：** 如果特征变了，必须同步更新 `src/features.py`，否则预测会出错。

---

## 项目中的两套 Notebook 文件夹

| 位置 | 用途 |
|------|------|
| `xgboost_models/` | XGBoost 训练 notebook（V1, V1.5, V2, V2.5） |
| `lightgbm_models/` | LightGBM 训练 notebook（V2, V2.5） |
| `models/modelV*.ipynb` | 旧版 notebook，早期小时级模型的备份 |

`xgboost_models/` 和 `lightgbm_models/` 是最终版本，`models/` 中的是早期版本。

---

## 快速问答

### Q：什么是"递归预测"？
A：因为模型需要"昨天的价格"来预测今天，未来 7 天的价格还不存在，所以先把第 1 小时的预测值当成"历史"，再去预测第 2 小时，以此类推。

### Q：为什么需要 `src/features.py`？
A：训练时在 notebook 中造特征，实时预测时不能在 notebook 中一步步操作，所以把造特征的代码写成 `.py` 脚本供主程序调用。

### Q：GitHub Actions 是什么意思？
A：它是 GitHub 提供的自动运行功能，只要配置好 `.github/workflows/` 文件，GitHub 就会按设定的时间自动执行命令，不需要你手动在电脑上跑。

### Q：predictions/ 文件夹里的 CSV 怎么看？
A：每个 CSV 包含以下列：
- `run_date` — 运行预测的日期
- `target_datetime` — 预测的目标时间
- `predicted_price` — 预测的电价（EUR/MWh）
- `actual_price` — 真实电价（已发生的时间会填入）
- `abs_error` — 绝对误差（`|actual - predicted|`）
