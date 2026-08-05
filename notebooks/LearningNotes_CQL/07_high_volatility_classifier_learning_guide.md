# 07 高波动概率分类模型（High-Volatility Probability Classifier）— 零基础学习指南

> 本指南对应笔记本：`data/convertData/05_feature_high_volatility.ipynb`
> 目标读者：完全没有数学和机器学习基础的学生
> 说明：所有中文专有名词都配上英文，方便你对照学习

---

## 0. 这篇指南是干什么的？你要做什么？

你的目标是：**训练一个小模型（classifier，分类器），让它根据"天气 + 时间"判断"未来会不会出现电价剧烈波动"，然后把这个判断结果（一个 0~1 的概率值）作为一个新的特征（feature），喂给原来的价格预测模型。**

简单说，你想给原来的模型加一个"**风暴预警器**"：

> 原模型：明天电价多少？（回归，regression）
> 新模型：明天电价会不会疯涨？（分类，classification）→ 把"会不会疯涨"当成一个线索，帮助原模型预测得更准。

用生活比喻：原模型像"天气预报员"（报温度），新模型像"风暴预警员"（报有没有台风）。你让预报员参考"台风预警信号"来报温度，可能会更准。

---

## 1. 零基础核心概念（先看懂再写代码）

### 1.1 回归（Regression）vs 分类（Classification）

|                | 回归（Regression）        | 分类（Classification）              |
| -------------- | ------------------------- | ----------------------------------- |
| 要预测什么     | 一个**数字**（连续值）    | 一个**类别**（离散的几类）          |
| 例子           | 明天电价 = 45.7 EUR/MWh   | 明天会不会高波动 = 是/否（1/0）     |
| 输出           | 数字，如 45.7             | 概率，如 0.87（=87%概率是"高波动"） |
| 你项目里已有的 | XGBoost/LightGBM 价格预测 | **本笔记本要做的**                  |

> 关键区别：回归输出"多少"，分类输出"哪一类 / 多大概率是哪一类"。

### 1.2 什么是"标签"（Label / Target，标签/目标值）

分类模型需要"标准答案"来学习。这个答案就叫**标签（label）**或**目标（target）**。

你的笔记本用一行代码制造了标签：

```python
df['is_high_volatility'] = (df['price_roll_std_6h'] >= vol_threshold).astype(int)
```

意思是：把"过去 6 小时价格波动（标准差，standard deviation）"和某个警戒线（threshold，阈值）比较：

- 波动 ≥ 警戒线 → 标签 = 1（高波动，疯涨/暴跌时刻）
- 波动 < 警戒线 → 标签 = 0（平稳）

这就把"价格波动"这件事，变成了一列 0/1 的答案，分类模型才好学习。

### 1.3 什么是"特征"（Feature，特征）

特征是**喂给模型看的"线索"**。分类模型只能看到这些线索，然后猜标签。

你的笔记本选的特征（修正后）：

```python
vol_features = ['temp', 'wind_speed', 'day_of_week', 'hour', 'month']
```

- 温度（temp）
- 风速（wind_speed）
- 星期几（day_of_week）
- 几点（hour）
- 月份（month）

> 直觉：极端天气（很冷、大风）常常导致电价大波动；周末/深夜通常平稳。所以这些"线索"确实和"波动"有关系。

### 1.4 什么是"阈值"（Threshold，阈值/警戒线）

阈值就是"一条分界线"。你的笔记本用：

```python
vol_threshold = df['price_roll_std_6h'].quantile(0.85)
```

`quantile(0.85)` = **85% 分位数**（85th percentile）。意思是：把所有时刻的波动值排序，取"第 85% 的位置"当分界线。

- 也就是说：**波动最大的那 15% 时刻 = 高波动（标签1）**，剩下 85% = 平稳（标签0）。

> 为什么用 85% 分位数？因为"高波动"本来就是相对概念，用电价历史自己的分布来定界线，比拍脑袋定一个数更合理。

### 1.5 什么是"概率"（Probability）和 predict_proba（概率）

这是整个笔记本**最精髓**的地方。

普通的 `model.predict()` 会直接告诉你"是 1 还是 0"。
但 `model.predict_proba()` 会给一个**柔和的概率**（0~1 之间的数）：

```python
risk_model.predict_proba(X_vol)[:, 1]
```

- 返回两列：`[:, 0]` = "是平稳"的概率，`[:, 1]` = "是高波动"的概率
- 比如返回 `0.87`，意思就是"**87% 的把握这会是高波动时刻**"

> 为什么用概率而不是 0/1？因为 0.87 和 0.55 的"危险程度"完全不同，概率保留了这种**细腻程度（granularity）**，作为新特征更有信息量。

### 1.6 什么是训练集/测试集（Training Set / Test Set）

- **训练集（training set）**：给模型做作业用的（教它规律）
- **测试集（test set）**：闭卷考试用的（考它学得怎么样）

你的笔记本用了：

```python
train_test_split(X_vol, y_vol, test_size=0.2, shuffle=False)
```

- `test_size=0.2`：20% 当考试题
- `shuffle=False`：**不打乱**，因为时间序列必须按时间顺序，未来不能泄漏到过去（这是铁律！）

### 1.7 核心直觉：为什么能用"天气"预测"价格波动"？

你可能想问：**天气又不是价格，凭什么天气能预测价格波动？**

答案：**因果关系**。电价剧烈波动往往由**突发供需变化**引起，而天气是主要推手：

- 温度骤降 → 取暖需求暴增 → 价格飙涨
- 大风突然来袭/停歇 → 风电出力剧变 → 价格波动
- 极端天气影响水电站、输电线路

所以"天气 + 时间"里藏着"会不会高波动"的信号。分类模型就是去**自动发现**这些信号（而不是我们人肉总结）。

---

## 2. 你的笔记本在干什么（3 个单元格逐个讲）

### 单元格 1：制造标签（Labeling，打标签）

```
读取 V2.5 宽表 → 按时间排序 → 用价格算6小时滚动标准差 → 删掉头部NaN
→ 用85%分位数当阈值 → 生成0/1标签
```

做的事情：把"连续的价格波动"变成"0/1 的判断题答案"。

### 单元格 2：训练风险分类器（Training the classifier）

```
选天气+时间特征 → 80/20按时间切分 → XGBClassifier训练
→ predict_proba算出概率 → 写回表格新列 high_volatility_prob
```

做的事情：教一个"风暴预警器"，并让它在每个时刻输出"高波动概率"。

### 单元格 3：保存（Saving）

```
把带新特征的表另存为 V3.0_15min_Risk_Enhanced_Dataset.csv
```

做的事情：生成一份"增强版"数据表，供价格模型重训使用。

---

## 3. ⚠️ 重要：你的代码里有"列名和路径错误"（我帮你逐项查过了）

### 3.1 特征列名不匹配（会直接报错 KeyError）

| 你笔记本里用的名字             | 数据表里**真实存在**的名字    | 说明                             |
| ------------------------------ | ----------------------------- | -------------------------------- |
| `air_temp_mean`                | `temp`                        | 温度列                           |
| `wind_speed_mean`              | `wind_speed`                  | 风速列                           |
| `wind_lag_24h`                 | ❌ 不存在                     | 没有"24小时前风速"这一列         |
| `temp_lag_24h`                 | `temp_lag_4` 或 `temp_lag_96` | 只有 1小时前/24小时前(用96) 两种 |
| `dayofweek`                    | `day_of_week`                 | 注意下划线写法                   |
| `price_roll_std_24h`（检查用） | `price_rolling_std_24h`       | 多了 "ing"，且那是24小时窗口     |

> 📌 学习要点：**写代码前，永远先用 `df.columns.tolist()` 打印表头确认列名**。这是初学者最容易踩的坑，也是最应该养成的好习惯。

### 3.2 路径错误（相对路径（relative path）指向错误）

你的笔记本**住在 `data/convertData/` 文件夹里**，所以：

- 读取 `'../data/convertData/V2.5_15min_features.csv'` 会变成 `data/data/...` ❌
- 保存 `'../data/convertData/V3.0...csv'` 也会变成 `data/data/...` ❌

**修正**：因为你就在 `data/convertData/` 里，直接用文件名即可：

```python
df = pd.read_csv('V2.5_15min_features.csv')          # 同一文件夹
df.to_csv('V3.0_15min_Risk_Enhanced_Dataset.csv')    # 同一文件夹
```

> 📌 学习要点：**相对路径是从"当前文件所在文件夹"出发的**。搞清楚你的笔记本放在哪一层，再决定 `../` 要用几个。

---

## 4. 修正后的完整代码（可以直接用）

把下面的单元格替换进你的笔记本（路径和列名都已修正，并新增了"评估分类器"一步）：

### 单元格 1 — 制造标签（Labeling）

```python
import pandas as pd
import numpy as np

print("1. Loading V2.5 15-Minute Dataset...")
# 修正①：路径 —— 笔记本在 data/convertData 里，直接写文件名
df = pd.read_csv('V2.5_15min_features.csv')

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

print("2. Defining 'High Volatility' Threshold...")
# 用真实价格算 6 小时滚动标准差（波动率），15分钟粒度 → 24 行一个窗口
# 注意：直接新算一列，不依赖表里已有的 price_rolling_std_24h（那个是24小时窗口）
df['price_roll_std_6h'] = df['price'].rolling(window=24).std()

# 删掉滚动窗口开头的 NaN 缺失值
df = df.dropna(subset=['price_roll_std_6h']).copy()

vol_threshold = df['price_roll_std_6h'].quantile(0.85)
print(f"   -> Top 15% Volatility Threshold: {vol_threshold:.2f}")

print("3. Creating the Binary Classification Target (0 or 1)...")
df['is_high_volatility'] = (df['price_roll_std_6h'] >= vol_threshold).astype(int)
print(df['is_high_volatility'].value_counts(normalize=True) * 100)
```

### 单元格 2 — 训练风险分类器（修正列名 + 新增评估）

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("4. Selecting Safe Features for the Classifier...")
# 修正②：列名全部改成数据表里真实存在的名字
# temp_lag_4 = 1小时前的温度（15分钟×4=1小时）
vol_features = ['temp', 'wind_speed', 'wind_direction_deg', 'temp_lag_4',
                'hour', 'day_of_week', 'month']

X_vol = df[vol_features]
y_vol = df['is_high_volatility']

# 保险：再删一次特征里的 NaN（XGB 其实能容忍，但干净点好）
mask = X_vol.notna().all(axis=1)
X_vol, y_vol = X_vol[mask], y_vol[mask]

X_vol_train, X_vol_test, y_vol_train, y_vol_test = train_test_split(
    X_vol, y_vol, test_size=0.2, shuffle=False
)

print("5. Initializing and Training the XGBoost Risk Classifier...")
risk_model = XGBClassifier(
    n_estimators=100,       # 100棵警戒树
    learning_rate=0.05,     # 学慢一点，稳一点
    max_depth=5,            # 深度5，防止死记硬背（过拟合）
    objective='binary:logistic',  # 明确告诉它：这是二分类任务
    random_state=42         # 锁死随机种子，结果可复现
)
risk_model.fit(X_vol_train, y_vol_train)

print("6. Evaluating the Risk Classifier (NEW STEP)...")
y_pred = risk_model.predict(X_vol_test)
print('Test accuracy:', round(accuracy_score(y_vol_test, y_pred), 4))
print(classification_report(y_vol_test, y_pred))

print("7. Extracting the High-Volatility Probabilities...")
# predict_proba 返回 [平稳概率, 高波动概率]，取第二列 [:, 1]
predicted_probabilities = risk_model.predict_proba(X_vol)[:, 1]
df['high_volatility_prob'] = predicted_probabilities

print("   -> New feature 'high_volatility_prob' created!")
df[['datetime', 'price', 'is_high_volatility', 'high_volatility_prob']].tail(10)
```

### 单元格 3 — 保存增强版数据表（修正路径）

```python
print("8. Saving the new V3 Matrix (V2.5 + Risk Feature)...")
# 修正③：路径 —— 直接保存到当前文件夹
save_path = 'V3.0_15min_Risk_Enhanced_Dataset.csv'
df.to_csv(save_path, index=False)
print(f"Matrix saved to: {save_path}")
```

---

## 5. 这个新特征怎么用进价格模型（V3）？

新特征 `high_volatility_prob` 造出来后，要让价格模型用上它，需要**三步**：

```
步骤1：本笔记本 → 生成 V3.0_15min_Risk_Enhanced_Dataset.csv（已含新特征）
步骤2：在 V3 对比笔记本里，训练价格模型时把新列加进特征
       比如：df.drop(columns=['price','datetime']) 后，模型会自动看到新列
步骤3：⚠️ 必须同步修改 src/features.py（见第6节）
```

在 V3 对比笔记本里，你只需要：

```python
df = pd.read_csv('../data/convertData/V3.0_15min_Risk_Enhanced_Dataset.csv')
X = df.drop(columns=['price', 'datetime', 'is_high_volatility', 'price_roll_std_6h'])
```

> 注意：`is_high_volatility`（0/1标签）和 `price_roll_std_6h`（波动值）**不能**当特征喂给价格模型——它们本身就是"从价格算出来的答案"，会泄漏真实信息（数据泄漏，data leakage）。**只有 `high_volatility_prob`（天气推断出的概率）才能当特征。**

---

## 6. 几个必须知道的"坑"（Caveats，注意事项）

### 6.1 全局阈值 = 轻微数据泄漏（Slight Data Leakage）

`vol_threshold = quantile(0.85)` 是用**整张表（包括测试期）**算的。这等于阈值"偷看了未来"。

- 影响：轻微，因为阈值只是一个全局分界线
- 改进（进阶做法）：只在**训练部分**算阈值：
  ```python
  train_part = df.iloc[:int(len(df)*0.8)]
  vol_threshold = train_part['price_roll_std_6h'].quantile(0.85)
  ```

### 6.2 类别不平衡（Class Imbalance，类别不平衡）

你的标签是 85% 平稳 / 15% 高波动——**不均衡**。

- 后果：模型可能"偷懒"，全都猜 0，准确率照样 85%
- 所以：**别只看准确率（accuracy）**，要看 `precision`（精确率）、`recall`（召回率）。我用 `classification_report` 帮你打印了这些指标。
- 改进：`XGBClassifier(scale_pos_weight=...)` 可以惩罚"偷懒"。

### 6.3 滚动窗口（Rolling Window）包含当前值吗？

`df['price'].rolling(24).std()` 用的是**从当前时刻往前数 24 个**（含当前）。

- 对你的**标签**来说：没问题（我们就是想标记"当前这一刻"的波动）
- 但注意：表里原有的 `price_rolling_std_1h` 是 `shift(1)` 之后再 rolling（**不含当前**），两种语义不同，别混用

### 6.4 ⚠️ 最关键的坑：实时预测（Live Prediction）怎么得到这个特征？

这是整个项目里最重要的一个约束，也是 README 反复强调的：

> **训练时造的特征，必须在实时预测时能一模一样地算出来。**

好消息是：你的设计很巧妙——分类器的输入只有**天气和时间**，而未来 7 天的天气（天气预报）和时间（日历）都是**提前可知的**。所以未来每个时间点的 `high_volatility_prob` 都能算出来，可以用于实时预测。✅

但坏消息是：**目前 `src/features.py` 里没有这个逻辑**。如果你只改了训练数据、不改 `src/features.py`，就会造成"训练和预测特征不一致"——预测结果会错。所以：

- 进阶任务：把 `high_volatility_prob` 的计算也写进 `src/features.py`，并在 `src/predict_system.py` 里加载 `risk_model`
- 这是"让特征真正上线"的关键一步（你现在可以先不写，但要**知道**这件事）

### 6.5 分类器本身怎么评价才算"好"？

- 好分类器：高波动时刻的 `high_volatility_prob` 明显比平稳时刻高
- 你可以画图验证：
  ```python
  import matplotlib.pyplot as plt
  high = df[df['is_high_volatility']==1]['high_volatility_prob']
  low  = df[df['is_high_volatility']==0]['high_volatility_prob']
  plt.hist([low, high], label=['平稳', '高波动'], bins=30, alpha=0.6)
  plt.legend(); plt.show()
  ```
  如果两个直方图分得开，说明这个特征有用；如果完全重叠，说明这个特征没信息量。

---

## 7. 学习路径：接下来学什么

| 阶段            | 内容                                                 | 优先级 |
| --------------- | ---------------------------------------------------- | ------ |
| 1️⃣ 先跑通       | 用修正后的代码把 3 个单元格跑起来                    | 必须   |
| 2️⃣ 看懂         | 回答：标签怎么来的？predict_proba 返回什么？         | 必须   |
| 3️⃣ 验证         | 跑第 6.5 节的直方图，看特征有没有用                  | 重要   |
| 4️⃣ 集成         | 把新特征加进 V3 价格模型重训，对比有没有变准         | 重要   |
| 5️⃣ 上线（进阶） | 把逻辑镜像进 `src/features.py` + `predict_system.py` | 进阶   |
| 6️⃣ 优化（进阶） | 修数据泄漏、处理类别不平衡、试更多特征               | 进阶   |

---

## 8. 术语表（Glossary 中英对照）

| 中文            | English               | 一句话解释                             |
| --------------- | --------------------- | -------------------------------------- |
| 分类            | Classification        | 预测"是哪一类"，而不是"是多少"         |
| 二分类          | Binary Classification | 只有两个类（0/1，如 平稳/高波动）      |
| 回归            | Regression            | 预测一个连续数字（如电价 45.7）        |
| 标签 / 目标     | Label / Target        | 模型要学的"标准答案"                   |
| 特征            | Feature               | 喂给模型的"线索"（天气、时间）         |
| 阈值 / 警戒线   | Threshold             | 判断"算不算高波动"的分界线             |
| 分位数          | Percentile / Quantile | 把数据排序后取某个百分比位置的值       |
| 标准差          | Standard Deviation    | 衡量数据"波动/离散"程度                |
| 滚动窗口        | Rolling Window        | 滑动地看过去 N 个时间点                |
| 概率            | Probability           | 0~1 之间的"把握程度"                   |
| 预测概率        | predict_proba         | 输出概率而不是硬分类结果               |
| 训练集 / 测试集 | Training / Test Set   | 做作业的数据 / 考试的数据              |
| 准确率          | Accuracy              | 猜对的占比                             |
| 精确率          | Precision             | 猜成"高波动"里，真的高波动的比例       |
| 召回率          | Recall                | 真正的高波动里，被找出来的比例         |
| 类别不平衡      | Class Imbalance       | 两类样本数量悬殊                       |
| 数据泄漏        | Data Leakage          | 模型偷看了不该看的"未来/答案"信息      |
| 过拟合          | Overfitting           | 死记硬背训练题，换新题就垮             |
| 随机种子        | Random Seed           | 锁死随机，让结果可复现                 |
| 相对路径        | Relative Path         | 从当前文件位置出发找文件               |
| 无监督/有监督   | Supervised Learning   | 有标准答案的学习方式（本项目是有监督） |

---

_本指南由项目背景 + 你的笔记本内容整理而成。核心一句话：先用 85% 分位数把波动变成 0/1 标签，再用天气+时间训练一个分类器输出"高波动概率"，把这个概率当成新特征，喂给价格预测模型。_
