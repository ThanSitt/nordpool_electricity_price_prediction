# 15. Nuclear Power Data + XGBoost V4 — The Three Notebooks Explained

> Date: 2026-08-14
> Related notebooks:
>
> 1. `data/originalData/Nuclear/15min_NucleatData.ipynb` — download nuclear 15-min data
> 2. `data/convertData/V3.1_15min_feature_engineering.ipynb` — build the SHARED dataset
> 3. `xgboost_models/modelV4.ipynb` — train XGBoost V3 (grid) & V4 (grid + nuclear)
>    技术术语配中文注释。

---

## 0. TL;DR — 一个故事

项目要加**核电发电量**作为新特征。我和伙伴分工：**伙伴调 LightGBM，我调 XGBoost**，
但**共用一个数据集**。于是：

1. 先下载核电 15 分钟实测数据（Fingrid）。
2. 造一个**共享数据集** `V3.1_15min_features.csv` = V2.5 + 网格 + 核电（伙伴的 LightGBM 和我的 XGBoost 都用它）。
3. 我训练 XGBoost：**V3 = 加网格**，**V4 = 加核电**（在 V3 网格基础上）。

**结果**：核电特征真的有效！V4 MAE 2.6993（比 V2.5.3 的 2.7236 更好），是当前**最好的 XGBoost 模型**。

```
Fingrid 核电实时数据（~3min）
        │  ① 15min_NucleatData.ipynb 下载 + 重采样
        ▼
nuclear_measured_15min.csv（15min）
        │  ② V3.1_15min_feature_engineering.ipynb
        ▼
V3.1_15min_features.csv（共享数据集 = V2.5 + 网格 + 核电，70 列）
        │  ③ modelV4.ipynb
        ▼
XGBoost V3（网格） MAE 2.7152 → XGBoost V4（+核电） MAE 2.6993 ★
```

---

## 1. Notebook ①：`15min_NucleatData.ipynb` — 下载核电数据

**目标**：拿到芬兰 2023–2025 的 15 分钟核电发电量（MW）。

### 1.1 为什么是"实测"数据 + 15 分钟？

- **实测（measured）**：这是已发生的真实出力。用它做特征时要注意——预测**未来**时刻时，
  当期实测值拿不到，所以只能做**滞后/滚动特征**（`nuclear_lag_96` 等）。
- **15 分钟**：为了和已有的 `V2.5/V3_15min_features.csv`（105,216 行）**对齐**。

### 1.2 数据来源：Fingrid 数据集 188

- API：`https://data.fingrid.fi/api`（需要免费 API key，`FINGRID_API_KEY` 环境变量）。
- **数据集 188** = "Nuclear power production - real-time data"（约 3 分钟粒度）。
- notebook 里有一个 **discovery 单元**：搜索 Fingrid 目录，打印所有含 "nuclear" 的数据集，
  确认 ID=188 正确（防止用错数据集）。

### 1.3 关键代码思路

```python
API_KEY = os.environ.get('FINGRID_API_KEY', '')   # 从环境变量读，绝不硬编码
if not API_KEY:
    raise EnvironmentError('...')                  # 没 key 就明确报错
```

- **分页 + 429 重试**：`fetch_fingrid()` 每次拉 10,000 行，遇 429 限流等 60s×次数 再试
  （复用 `15min_GridTransmission.ipynb` 的成熟模式）。
- **3 分钟 → 15 分钟**：`resample('15min').mean()`（15 分钟内的均值）。
- **覆盖度检查**：对比期望的 15 分钟网格，算覆盖率；如果 2023–2025 不完整，
  提示改用 ENTSO-E "Actual Generation per Production Type"（2015 年起有完整历史）。

### 1.4 产出

`data/originalData/Nuclear/nuclear_measured_15min.csv`

- 105,216 行 × 15 分钟（和 V2.5/V3 完全对齐 ✓）
- 数值合理：mean ≈ 3611 MW，max ≈ 4409 MW（芬兰 5 台机组总容量）

> **安全提醒**：这次 key 用环境变量，**没有**像 `15min_GridTransmission.ipynb` 那样写死在代码里。

---

## 2. Notebook ②：`V3.1_15min_feature_engineering.ipynb` — 造共享数据集

**目标**：把核电特征并进 V3（V2.5 + 网格），产出**共享数据集** `V3.1_15min_features.csv`。

### 2.1 合并

```python
df = df.merge(nuc[['datetime', 'nuclear_power_mw']], on='datetime', how='left')
df['nuclear_power_mw'] = df['nuclear_power_mw'].ffill().bfill()
```

- **左连接**：以 V3 的 105,216 行为主，缺的核电值用 `ffill().bfill()` 填充
  （258 个 NaN → 0）。核电是稳定基荷，填充安全。

### 2.2 为什么造这些核电特征？

| 特征                       | 含义          | 为什么                                     |
| -------------------------- | ------------- | ------------------------------------------ |
| `nuclear_power_mw`         | 当期核电出力  | 训练时已知，模型学"核电高/低 → 电价"的关系 |
| `nuclear_lag_96`           | 24 小时前     | **线上预测时可算**（昨天已实测）           |
| `nuclear_lag_672`          | 7 天前        | 同上，周同比                               |
| `nuclear_rolling_mean_24h` | 过去 24h 均值 | 平滑噪声，反映基荷水平                     |
| `nuclear_rolling_mean_7d`  | 过去 7 天均值 | 更长周期趋势                               |
| `nuclear_change_1d`        | 与 24h 前的差 | 捕获机组停/启的变化冲击                    |

> **关键点（train/serve gap）**：`nuclear_power_mw`（当期）训练时有用，但**线上预测未来时拿不到**。
> 真正线上可用的是**滞后特征**。这正是之前 V3.1_live 实验（网格）得到的教训——训练有用的部分
> 不一定是线上可用的部分。

### 2.3 产出

`data/convertData/V3.1_15min_features.csv` — **105,216 行 × 70 列**（V3 的 64 + 6 核电）
= **共享数据集**。伙伴的 LightGBM 直接读它训练（= 他的 "V3"），我的 XGBoost V4 也用它。

---

## 3. Notebook ③：`modelV4.ipynb` — XGBoost V3 vs V4

**目标**：在同一数据、同一切分、同一超参数下，对比 **V3（加网格）** 和 **V4（加核电）**，
判断核电特征是否真的提升。

### 3.1 三个特征集（同一个 80/20 切分 → 公平对比）

```python
baseline_cols = [c for c in df.columns if c not in ['datetime','price']
                 and not c.startswith('fi_') and not c.startswith('nuclear_')]  # 49 V2.5
grid_cols    = [c for c in df.columns if c.startswith('fi_')]      # 13 网格
nuclear_cols = [c for c in df.columns if c.startswith('nuclear_')] # 6 核电

v25_cols = baseline_cols                      # 49
v3_cols  = baseline_cols + grid_cols          # 62
v4_cols  = baseline_cols + grid_cols + nuclear_cols   # 68
```

### 3.2 用调好的超参数（V2.5.3 的 Optuna 最优）

```python
tuned = dict(
    objective='reg:absoluteerror', n_estimators=2000,
    learning_rate=0.00983, max_depth=12, min_child_weight=31,
    subsample=0.7997, colsample_bytree=0.9983,
    reg_lambda=0.0129, reg_alpha=0.4381, random_state=42)
```

> 为什么要"调好参再测特征"？—— 因为之前 V3 实验在**默认**超参数下测网格是"变差"（+0.027），
> 但**调好参后**测是"变好"（-0.0084）。弱模型测特征不可信（见学习笔记 12）。

### 3.3 结果

| 模型            | 特征 | MAE        | RMSE   | R²         | 增量           |
| --------------- | ---- | ---------- | ------ | ---------- | -------------- |
| V2.5（参考）    | 49   | 2.7236     | 8.1642 | 0.9722     | —              |
| V3（+网格）     | 62   | 2.7152     | 8.0699 | 0.9728     | −0.0084 ✅     |
| **V4（+核电）** | 68   | **2.6993** | 8.0321 | **0.9731** | **−0.0159 ✅** |

- **核电特征真的有效**（和"高波动概率"负结果形成对比——因为核电是真实的供给侧信号，
  而高波动概率来自弱分类器）。
- V4 是**当前最好的 XGBoost**（超过 V2.5.3 的 2.7236）。
- V2.5 基线精确复现 V2.5.3 的 MAE → 对比干净。

### 3.4 保存位置：`models/experiments/`（不是 `models/saved/`）

```python
fit_and_save(v3_cols, 'xgboost_v3.pkl')
fit_and_save(v4_cols, 'xgboost_v4.pkl')
```

> 因为网格/核电特征**不在 `src/` 线上管线里**（之前决定回退网格接入），
> 这些模型如果放进 `models/saved/`，每日自动预测会加载但算不出这些特征 → 静默劣化。
> 所以放在 `models/experiments/` 作为**实验记录**，等以后决定接入再迁移。

---

## 4. 关键收获（Key Takeaways）

1. **实测数据只能做滞后特征**——当期值预测未来时拿不到（train/serve gap）。
2. **共享数据集让分工清晰**：伙伴和我读同一个 `V3.1_15min_features.csv`，
   各自训练自己的模型，版本号可以不同（他叫 V3，我叫 V4），互不冲突。
3. **真实供给侧信号 ≠ 噪声特征**：核电（真实、稳定、有物理意义）**有效**；
   高波动概率（弱分类器产物）**无效**。特征"质量"比"数量"重要。
4. **调好参再测特征**：V3 网格从"默认下变差"变成"调参后变好"，证明了这点。
5. **API key 用环境变量**，不写死进 notebook（吸取 GridTransmission 的教训）。
6. **模型保存位置 = 部署能力**：`models/saved/` = 线上会用；`models/experiments/` = 只是实验记录。
