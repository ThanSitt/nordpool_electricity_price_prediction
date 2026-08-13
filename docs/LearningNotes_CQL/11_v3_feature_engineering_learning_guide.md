# 11 Feature Engineering Learning Guide — The V3 Notebook, Explained from Zero (特征工程学习指南)

> This guide teaches you **feature engineering (特征工程)** from the absolute beginning, then walks through `data/convertData/V3_15min_feature_engineering.ipynb` step by step, and finally explains **how to engineer a new nuclear-power feature (核电特征工程)**.
> Every technical term (专业术语) is followed by its Chinese meaning (中文释义) so you can learn the English words and the concepts at the same time.
> Related files:
>
> - `data/convertData/V3_15min_feature_engineering.ipynb` (the notebook we analyse)
> - `data/convertData/V2.5_15min_features.csv` (the input table)
> - `data/convertData/V3_15min_features.csv` (the output table)
> - `data/originalData/GridTransmission/grid_transmission_15min.csv` (the raw grid data)
> - `data/originalData/Nuclear/15min_Nuclear.ipynb` (the future nuclear download notebook)

---

## Part 0 — For the Complete Beginner (给完全初学者的基础)

### 0.1 What is a "feature"? (什么是"特征"?)

A machine-learning model (机器学习模型) sees the world as a **table (表格)** with rows and columns:

| Row = one moment | Feature A | Feature B | Target |
| ---------------- | --------- | --------- | ------ |
| 2023-01-01 00:00 | 10.0      | 2.5       | 61.20  |
| 2023-01-01 00:15 | 9.8       | 2.6       | 58.40  |
| ...              | ...       | ...       | ...    |

- **Row (行)** = one sample (样本), in our project: one **15-minute time slot (15分钟时间片)**.
- **Column (列)** = one **feature (特征)** = one number that describes that moment.
- **Target / label (目标/标签)** = the thing we want to predict (预测): here the electricity **price (电价)**.

So a "feature" is simply **a useful number that describes a moment in time**. That's all.

### 0.2 What is "feature engineering"? (什么是"特征工程"?)

**Feature engineering (特征工程)** = the work of turning **raw data (原始数据)** into **numbers that make the model's job easy (容易学习的数字)**.

**A simple analogy (类比):**

Imagine teaching a child (模型) to guess tomorrow's weather. You have two choices:

| Choice                           | What you give the child                                                                                    | Result                                                                                                        |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Raw data (原始数据)              | "The date is 2026-08-13 14:30, wind from 359°, temperature -1.2°C"                                         | The child is confused: it doesn't know 14:30 is afternoon, doesn't know 359°≈1°, doesn't know winter is cold. |
| Engineered features (工程化特征) | "hour=14, is_weekend=0, temperature=-1.2, heating_degree_day=18.2, wind_dir_sin=0.017, wind_dir_cos=0.999" | The child can now see clear patterns: "cold + afternoon + weekday = high demand".                             |

**In one sentence:** the model cannot "understand" the real world (dates, degrees, physics) — it only understands numbers. Feature engineering converts real-world knowledge into numbers the model can learn from.

### 0.3 Why can't we just feed raw data? (为什么不能直接喂原始数据?)

1. **A timestamp (时间戳) like "2023-06-15 14:30" is just text.** The model has no idea that 14:30 is the evening peak (晚高峰) or that June is summer. We must split it into `hour`, `month`, `season`... and even better, into **cyclic features (周期特征)**.

2. **Wind direction in degrees has the "359° problem".** 359° and 1° are 2 degrees apart physically (basically the same wind) but 358 numbers apart mathematically. The model would treat them as completely different. We fix this with **sin/cos encoding (正弦/余弦编码)**.

3. **The model can't see time.** Electricity prices are **autoregressive (自回归)** — today's price depends strongly on yesterday's price. But a raw table has no "yesterday" column. We must create **lag features (滞后特征)** that bring the past into the present row.

4. **The model can't see calendars or physics.** It doesn't know weekends are cheap or that cold weather raises demand. We create **calendar flags (日历标志)** like `is_weekend`, and **weather-derived features (天气衍生特征)** like `HDD` (heating-degree-day).

### 0.4 The Golden Rule (黄金法则) of this project

You may have noticed in the README:

> Feature engineering (特征工程) matters FAR more than resolution (分辨率).
>
> - Adding resolution only: R² 0.107 → 0.125 (barely improves)
> - Adding engineered features: R² 0.107 → 0.911 (huge jump)

**This is why we spend so much effort on feature engineering** — it is the difference between a useless model and a great model.

---

## Part 1 — The Building Blocks You Will Meet (你会遇到的构件)

| English (英文)                             | 中文           | One-line explanation (一句话解释)                                    |
| ------------------------------------------ | -------------- | -------------------------------------------------------------------- |
| Feature (特征)                             | 特征           | A number describing one moment, used as model input                  |
| Target / label (目标/标签)                 | 目标/标签      | The value we want to predict (here: price)                           |
| Tabular data (表格数据)                    | 表格数据       | Data arranged in rows and columns                                    |
| DataFrame (数据框)                         | 数据框         | pandas' name for a table                                             |
| Row / sample (行/样本)                     | 行/样本        | One time slot                                                        |
| Column (列)                                | 列             | One feature                                                          |
| Merge / join (合并/连接)                   | 合并/连接      | Combine two tables using a shared key                                |
| Left join (左连接)                         | 左连接         | Keep all rows of the left table, add matching columns from the right |
| Resampling (重采样)                        | 重采样         | Change the time resolution (e.g. 3-min → 15-min)                     |
| Lag feature (滞后特征)                     | 滞后特征       | The value from N steps in the past                                   |
| Rolling window (滚动窗口)                  | 滚动窗口       | A statistic over the last N values (e.g. mean/std)                   |
| Time series (时间序列)                     | 时间序列       | Data ordered by time                                                 |
| Chronological split (时序切分)             | 按时间顺序切分 | Train on past, test on future — never shuffle                        |
| Data leakage (数据泄漏)                    | 数据泄漏       | Future information accidentally leaking into training                |
| NaN / missing value (缺失值)               | 缺失值         | An empty/unknown cell in the table                                   |
| Back-fill (回填) / forward-fill (前向填充) | 回填/前向填充  | Fill an empty cell with the next/previous value                      |
| Cyclic encoding (周期编码)                 | 周期编码       | Encode a repeating quantity (hour/month/angle) with sin & cos        |
| One-hot encoding (独热编码)                | 独热编码       | Turn a category into 0/1 columns                                     |
| Resolution (分辨率)                        | 分辨率         | The time granularity (15-min, hourly)                                |
| Shift (移位)                               | 移位           | Move a column up/down by N rows (creates lags)                       |

Don't worry if some are unclear now — each one is explained where it first appears below.

---

## Part 2 — Step-by-Step Walkthrough of `V3_15min_feature_engineering.ipynb`

### The Big Picture first (先看全局)

```
V2.5_15min_features.csv      ← already contains ALL features from V1→V2.5
      │                          (weather, price, lags, rolling, calendar...)
      ▼
grid_transmission_15min.csv  ← NEW raw data: Finland's cross-border flows
      │
      ▼
      MERGE (left join)  +  LAG FEATURES  +  SANITY CHECK  +  SAVE
      ▼
V3_15min_features.csv      ← ready to train the V3 model
```

**In one sentence:** the notebook takes the already-finished V2.5 feature table, adds the new **grid transmission (跨境输电)** data, engineers it into lag features, checks the result, and saves a new table. **It never touches the old features — it only ADDS new ones.**

---

### Step 1 — Import libraries (导入库) — Cell 1

```python
import pandas as pd
import numpy as np
from pathlib import Path
```

- `pandas (pandas库)` — the tool for tables: reading CSVs, merging, shifting, rolling.
- `numpy (numpy库)` — math helpers (mean, std, arrays).
- `Path (路径对象)` — builds file paths that work on Windows/macOS/Linux (跨平台路径).

---

### Step 2 — Load V2.5 Features (加载 V2.5 特征) — Cells 2–3

```python
df = pd.read_csv('V2.5_15min_features.csv')
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Europe/Helsinki')
df = df.sort_values('datetime').reset_index(drop=True)
```

**What it does:** reads the previous version's complete feature table, converts the time column to the Helsinki timezone (赫尔辛基时区), and sorts by time.

**WHY (为什么)?**

1. **V3 = V2.5 + grid.** Instead of rebuilding all 49 features from scratch, we load the finished table and add new columns. This is called **building on top of the previous version (在上一版基础上叠加)**.
2. **Timezone fix (时区修复).** The CSV contains **mixed UTC offsets (混合时区偏移)**: +02:00 in winter (冬令时) and +03:00 in summer (夏令时). We first parse everything as **UTC (世界协调时)** with `utc=True`, then convert to `Europe/Helsinki` — the local time of the Finnish electricity market. This is the _exact bug_ the project fixed before: mixing timezones silently misaligns rows.
3. **Sort by time (按时间排序).** Features like lags and rolling windows only make sense on time-ordered data.

**What is already inside this table?** All 49 features from V2.5, grouped as:

| Group (分组)               | Examples (例子)                                  | Meaning (含义)                                    |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------- |
| Temporal (时间特征)        | `hour`, `minute`, `day_of_week`, `season`        | When is this moment?                              |
| Cyclic (周期特征)          | `hour_sin`, `hour_cos`, `month_sin/cos`          | The "circle" version of time (smooth 23:00→00:00) |
| Calendar (日历特征)        | `is_weekend`, `is_holiday`, `is_peak_hour`       | Is it a special day?                              |
| Weather (天气特征)         | `temp`, `wind_speed`, `wind_direction_deg`       | The weather at this moment                        |
| Weather derived (天气衍生) | `HDD`, `wind_power_proxy`, `wind_dir_sin/cos`    | Physical meanings extracted from weather          |
| Price lags (价格滞后)      | `price_lag_1`, `price_lag_96`, `price_lag_672`   | The price N steps ago                             |
| Price rolling (价格滚动)   | `price_rolling_mean_1h`, `price_rolling_std_24h` | Recent price statistics                           |

---

### Step 3 — Load Grid Transmission Data (加载跨境输电数据) — Cells 4–5

```python
grid_path = Path('../originalData/GridTransmission/grid_transmission_15min.csv')
grid = pd.read_csv(grid_path)
grid['datetime'] = pd.to_datetime(grid['datetime'], utc=True).dt.tz_convert('Europe/Helsinki')
grid = grid.sort_values('datetime').reset_index(drop=True)
```

**What it does:** loads the new raw data — Finland's **cross-border electricity flows (跨境输电功率)** in MW.

Columns include:

- `fi_ee` — Finland ↔ Estonia (Estlink 1+2)
- `fi_no` — Finland ↔ Norway
- `fi_se_north` / `fi_se_central` — Finland ↔ Sweden (two corridors)
- `fi_se_total` — combined Sweden flow
- `fi_total_net` — total net flow (all corridors)
- `fi_se_abs` — |Sweden flow|, a congestion proxy

**Sign convention (符号约定):** positive = Finland exports (出口), negative = Finland imports (进口).

**WHY does this matter for price (为什么影响电价)?** Electricity price is set by **supply and demand (供给与需求)**. Finland is not an island — when domestic supply is tight, it imports more; when there is surplus, it exports. **The cross-border flow is a live picture of the supply/demand balance (供需平衡的实时写照).** If imports are constrained, prices tend to rise.

**BUT — here is the critical trap:** this is a **realized (已实现/事后)** measurement. At prediction time we don't know _today's_ or _tomorrow's_ actual flow. That is exactly why Step 4 uses **lag features**, not the current value.

---

### Step 4 — Merge on Datetime (按时间合并) — Cells 6–7

```python
grid_cols = ['datetime', 'fi_ee', 'fi_no', 'fi_se_north', 'fi_se_central',
             'fi_se_total', 'fi_total_net', 'fi_se_abs']
df = df.merge(grid[grid_cols], on='datetime', how='left')
df[grid_flow_cols] = df[grid_flow_cols].bfill()
```

**What it does:** performs a **left join (左连接)** — for every row of `df` (the V2.5 table), it finds the grid row with the same `datetime` and copies the grid columns next to it.

**WHY (为什么)?**

1. **One row per moment (每个时刻一行).** A model needs ALL information about one moment in a single row. The V2.5 features live in one table, the grid data in another — merging (合并) brings them together into one wide row.
2. **Left join (左连接) = "keep everything I already have, add what matches."** We never want to lose existing rows, so the left (V2.5) table is the master.

**Why back-fill (回填) 8 NaN rows at the start?** The grid CSV has 8 empty cells at the very beginning (2023-01-01 00:00–01:45, before the UTC midnight boundary). `bfill()` fills each empty cell with the **next valid value (后一个有效值)**. This is a **one-off edge case (一次性的边界情况)** with negligible effect on training.

**Key idea: not every NaN needs to be "fixed".** In machine learning, missing values (缺失值) are normal. We only fill when the gap is tiny and meaningless.

---

### Step 5 — Grid Lag Features (滞后特征) — THE MOST IMPORTANT CELL — Cells 8–9

```python
for col in ['fi_total_net', 'fi_se_total', 'fi_ee']:
    df[f'{col}_lag_96']  = df[col].shift(96)
    df[f'{col}_lag_672'] = df[col].shift(672)
```

**What it does:** creates `fi_total_net_lag_96` (= the flow **96 steps ago** = 24 hours ago, because 96 × 15 min = 24 h) and `fi_total_net_lag_672` (= **672 steps ago** = 7 days ago).

**This is the heart of the whole notebook, so let's understand it deeply.**

#### 5.1 What is a lag feature? (什么是滞后特征?)

`df[col].shift(96)` moves the column **down by 96 rows**. So the value that appears in today's row is the value from 96 rows (24 hours) earlier:

```
time             fi_total_net   fi_total_net_lag_96
2023-06-15 00:00     500              (NaN — no data yet)
2023-06-15 00:15     480              (NaN)
... first 96 rows are empty (NaN) ...
2023-06-16 00:00     620              500      ← 24h ago value
2023-06-16 00:15     610              480      ← 24h ago value
```

So **a lag feature (滞后特征) tells the model: "what was the value at this same time yesterday (昨天同时刻) / a week ago (一周前)?"**

#### 5.2 Why lag and NOT the current value? (为什么用滞后而不是当前值?)

This is the **single most important idea in time-series ML**, and it comes down to one word: **data leakage (数据泄漏)**.

- A model for **day-ahead prices (日前电价)** must predict _tomorrow's_ price using only information available _today_.
- At "today", we do **not** know tomorrow's grid flow (it hasn't happened yet).
- If we used `fi_total_net` (the same-time value) as a feature, the model would "cheat" — during training it would see tomorrow's answer, so it would look great in the test set but **fail completely in real life** (which is what happened to the live system before — see the project's Known Problems).
- **Solution: only use PAST values as features.** The past is always available. Hence: **lag features (滞后特征)**.

> **Golden rule (黄金法则):** A feature must be a value you could actually know at prediction time. If it isn't, it leaks the future (泄漏未来信息).

#### 5.3 Why 96 and 672 specifically? (为什么偏偏是 96 和 672?)

Because our data is **15-minute resolution (15分钟分辨率)**:

- 96 × 15 min = 24 hours → `_lag_96` = "yesterday at the same time"
- 672 × 15 min = 168 hours = 7 days → `_lag_672` = "last week at the same time"

**Why these two?** Electricity prices and flows are strongly **periodic (周期性)**:

- A **daily cycle (日周期)** — every day looks like the last day.
- A **weekly cycle (周周期)** — every week looks like last week (weekday vs weekend patterns).

Lags at 24h and 7d let the tree model (树模型) "see" these cycles, which is far more informative than just the last 15-minute value.

#### 5.4 Why `fi_total_net`, `fi_se_total`, `fi_ee` but not the others?

The notebook lags the three most meaningful aggregate flows: the **total net flow**, the **Sweden total**, and the **Estonia (Estlink) flow**. The individual corridors (`fi_se_north`, `fi_se_central`) are already folded into `fi_se_total`, so lagging them again would add redundant (冗余) features.

---

### Step 6 — Sanity Check (合理性检查) — Cells 10–12

```python
print(f'Total columns: {len(df.columns)}')
print(f'Total rows:    {len(df)}')
new_cols = [c for c in df.columns if c not in v25_cols]   # what did we ADD?
nan_counts = df.isnull().sum()                            # where are the gaps?
df[grid_flow_cols].describe()                             # stats of new features
```

**What it does:** checks the shape (形状), the list of newly-added columns, the **NaN (缺失值)** counts, and summary statistics.

**WHY (为什么)?** Feature engineering can silently break:

- Wrong number of rows → the merge failed.
- Unexpected NaN everywhere → the timezone or the key didn't match.
- A column full of zeros or absurd values → bad sign convention or wrong dataset.

**What is "expected" here?** The new lag columns should have exactly **96 NaN** rows (first 96 rows have no "yesterday") and **672 NaN** rows (first 672 rows have no "last week"). If you see NaN counts that match this pattern, the lags were built correctly. This is called a **sanity check (合理性检查)** — verifying the obvious things before trusting the data.

---

### Step 7 — Plot (画图) — Cell 13

```python
df.set_index('datetime')['fi_se_total'].plot(...)   # one subplot per flow
```

**What it does:** draws a **time-series line plot (时间序列折线图)** of each flow over the whole period.

**WHY (为什么)?** A picture reveals problems numbers hide: sudden cliffs (断崖), long flat lines (长平线 = possibly a bug or a frozen sensor), and the sign convention (positive = export vs import). Plotting is a **visual sanity check (可视化合理性检查)**.

---

### Step 8 — Save (保存) — Cells 14–15

```python
out_path = 'V3_15min_features.csv'
df.to_csv(out_path, index=False)
```

**What it does:** saves the final table.

**WHY (为什么)?** Each version keeps **one reproducible dataset (可复现数据集)** on disk, so training (训练) is a separate, simple step: just `pd.read_csv('V3_15min_features.csv')` and train. This is the project's whole philosophy — **data pipeline (数据管道) first, training second.**

---

## Part 3 — Deeper: WHY each feature type exists (每种特征为什么存在)

Now that you've seen the mechanics, here's the "physics" of why each feature group helps predict electricity prices:

| Feature group                     | Why it predicts price (为什么能预测电价)                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Time (时间)**                   | Demand (需求) follows daily rhythms — mornings and evenings peak, nights are cheap.                              |
| **Cyclic sin/cos (周期正弦余弦)** | Makes "23:00" and "00:00" numerically close (they're an hour apart, not 23 apart).                               |
| **Calendar (日历)**               | Weekends and holidays have low industrial demand → cheaper prices.                                               |
| **Weather (天气)**                | Cold weather → heating demand → higher price. Wind → more cheap wind power → lower price.                        |
| **HDD (采暖度日)**                | A physical "how cold is it really" number: `max(0, 17 - temp)`. 17°C is the heating threshold.                   |
| **wind_power_proxy (风电代理)**   | Wind power ∝ wind³. Cubic wind speed is a simple physics model of wind generation.                               |
| **Price lags (价格滞后)**         | Prices are autoregressive — yesterday is the best predictor of today. The single strongest signal.               |
| **Rolling stats (滚动统计)**      | The recent **level (水平)** (mean) and **volatility (波动)** (std) of price — a high std means a nervous market. |
| **Grid flows (跨境输电)**         | The supply/demand balance — imports tight = price up.                                                            |
| **Nuclear (核电, future)**        | Base-load supply — a nuclear unit going offline = less cheap supply = price up.                                  |

**A pattern to notice (一个规律):** almost every feature converts a _concept the model can't see_ (time, calendar, physics, the past) into a _number the model can read_. That is feature engineering in a nutshell.

---

## Part 4 — How to Engineer a Nuclear Feature (核电特征工程怎么做)

### 4.1 The raw nuclear data (核电原始数据)

- **Source (来源):** Fingrid "Nuclear power production – real-time data" (核电实时发电量), 2023–2025.
- **What it measures:** Finland's actual nuclear output in **MW (兆瓦)** — the realized (已实现) generation.
- **Resolution (分辨率):** the real-time series updates every **3 minutes (3分钟)**. To align with the 15-min model we must **resample (重采样)** to 15-minute means: `df.resample('15min').mean()`.

### 4.2 The single most important rule: lag only (只做滞后)

Remember Step 5 of the V3 notebook? The same logic applies, even more strongly:

- Nuclear output is **realized after the fact (事后才知道)** — today you do NOT know tomorrow's nuclear output.
- Therefore, **just like the grid features, use ONLY lag features (滞后特征).**
- **NEVER** put the same-timestamp nuclear value into the model — that would be **data leakage (数据泄漏)**.

> 中文总结：核电是"事后"数据，只能做滞后特征，绝不能用同刻值。

### 4.3 Suggested nuclear features (建议的核电特征) — mirror the V3 grid pattern

| New feature                | Formula (公式)                | Meaning (含义)                                          |
| -------------------------- | ----------------------------- | ------------------------------------------------------- |
| `nuclear_lag_96`           | `nuclear.shift(96)`           | Nuclear output 24 h ago (日周期)                        |
| `nuclear_lag_672`          | `nuclear.shift(672)`          | Nuclear output 7 days ago (周周期)                      |
| `nuclear_rolling_mean_24h` | `nuclear.rolling(96).mean()`  | Average nuclear over last 24 h                          |
| `nuclear_rolling_mean_7d`  | `nuclear.rolling(672).mean()` | Average nuclear over last 7 days                        |
| `nuclear_change_1d`        | `nuclear - nuclear.shift(96)` | **Change vs yesterday — the outage signal (停机信号)!** |

In code (this is the whole feature-engineering core):

```python
nuc = nuclear_df.set_index('datetime')['nuclear_mw']   # 15-min series
df['nuclear_lag_96']            = nuc.shift(96)
df['nuclear_lag_672']           = nuc.shift(672)
df['nuclear_rolling_mean_24h']  = nuc.rolling(96).mean()
df['nuclear_rolling_mean_7d']   = nuc.rolling(672).mean()
df['nuclear_change_1d']         = nuc - nuc.shift(96)
```

### 4.4 Why does nuclear affect price, and what is the real signal? (核电为什么影响电价?)

- Nuclear is **base load (基荷电源)**: it runs 24/7 and is cheap. Its absolute level is fairly constant.
- The interesting event is **when a unit goes offline (停机/检修)** — supply suddenly drops, so price jumps.
- That's why **`nuclear_change_1d` (较昨日变化)** is likely your most informative nuclear feature: it spikes exactly when something happened.
- The **absolute level (绝对水平)** matters less; the **change and the rolling mean (变化与滚动均值)** capture the state of the fleet.

### 4.5 Concrete notebook steps for nuclear (mirror V3 exactly)

Following the V3 notebook structure, the new `V4_15min_feature_engineering.ipynb` would be:

```
Step 1  Load V3_15min_features.csv          (start from the finished V3 table)
Step 2  Load nuclear_measured_15min.csv     (raw nuclear, resampled to 15-min)
Step 3  Merge on datetime (left join)       (bring nuclear into each row)
Step 4  Create lag/rolling features         (shift(96), shift(672), rolling...)
Step 5  Sanity check                        (NaN counts: 96 and 672 at start)
Step 6  Plot the nuclear series             (see outages as dips)
Step 7  Save V4_15min_features.csv          (reproducible dataset)
```

### 4.6 Golden rules for nuclear (核电特征工程的黄金法则)

1. **Lag only (只做滞后)** — never use same-time nuclear values (no leakage).
2. **Verify NaN counts (检查缺失值数量)** — expect 96 and 672 NaNs at the start of the lag columns.
3. **Controlled comparison (受控对比)** — train with and without the nuclear features using the same split (切分) and same hyperparameters (超参数), and compare MAE/RMSE/R². Do not assume "more features = better".
4. **Remember the V2.5.1 lesson (记住 V2.5.1 的教训)** — an intuitively sensible feature (`high_volatility_prob`) made the model WORSE. Always test, never assume.

---

## Part 5 — Glossary (术语表 中英对照)

| English (英文)        | 中文       | One-line explanation (一句话解释)                  |
| --------------------- | ---------- | -------------------------------------------------- |
| Feature               | 特征       | An input number describing a moment                |
| Target / label        | 目标/标签  | The value to predict (price)                       |
| Feature engineering   | 特征工程   | Turning raw data into useful model inputs          |
| Raw data              | 原始数据   | Unprocessed source data (CSV as-is)                |
| Tabular data          | 表格数据   | Rows-and-columns data                              |
| Row / sample          | 行/样本    | One time slot                                      |
| Column                | 列         | One feature                                        |
| Merge / join          | 合并/连接  | Combine two tables on a key                        |
| Left join             | 左连接     | Keep left table's rows, add matching right columns |
| Resampling            | 重采样     | Change time resolution (3-min → 15-min)            |
| Lag feature           | 滞后特征   | Value from N steps in the past                     |
| Shift                 | 移位       | Move a column by N rows (`shift(96)`)              |
| Rolling window        | 滚动窗口   | Statistic over the last N values                   |
| Time series           | 时间序列   | Time-ordered data                                  |
| Chronological split   | 时序切分   | Train past, test future, no shuffle                |
| Data leakage          | 数据泄漏   | Future info accidentally used in training          |
| NaN / missing value   | 缺失值     | Empty cell                                         |
| Back-fill             | 回填       | Fill empty cell with the next value                |
| Forward-fill          | 前向填充   | Fill empty cell with the previous value            |
| Cyclic encoding       | 周期编码   | sin/cos encoding of hour/month/angle               |
| One-hot encoding      | 独热编码   | Categories → 0/1 columns                           |
| Resolution            | 分辨率     | Time granularity (15-min, hourly)                  |
| Base load             | 基荷       | Always-on, cheap generation (nuclear)              |
| Outage                | 停机       | A unit temporarily offline                         |
| Autoregressive        | 自回归     | Value depends on its own past values               |
| Supply / demand       | 供给/需求  | The balance that sets the price                    |
| Controlled experiment | 受控实验   | Change only one thing at a time                    |
| Hyperparameters       | 超参数     | Settings chosen before training                    |
| Sanity check          | 合理性检查 | Verify shapes/NaN/ranges before trusting data      |

---

## Part 6 — One-Page Summary (一页总结)

1. **Feature (特征)** = a useful number describing one 15-minute moment; **target (目标)** = the price.
2. **Feature engineering (特征工程)** = turning raw data (time, weather, physics, the past) into numbers a model can learn from.
3. **V3 notebook = ADD only.** It loads the finished V2.5 table, merges (合并) grid flows, and creates **lag features (滞后特征)** at 96 (24 h) and 672 (7 d).
4. **Why lag (为什么滞后)?** Because of **data leakage (数据泄漏)** — at prediction time you only know the past, so features must be past values.
5. **Sanity check (合理性检查)** before saving: shape, new columns, NaN counts (expect 96 & 672), plots.
6. **Nuclear (核电)** = realized data → **lag features only** (`nuclear_lag_96/672`, rolling means, `nuclear_change_1d`), then a **controlled comparison (受控对比)** against the version without it.
7. **The project's core lesson (核心教训):** more features ≠ better — measure it (V2.5.1 proved a "good" feature can hurt).

> Next step for you: after downloading `nuclear_measured_15min.csv` into `data/originalData/Nuclear/`, build `V4_15min_feature_engineering.ipynb` following Section 4.5, and train the V4 XGBoost model with a controlled comparison vs V3.
