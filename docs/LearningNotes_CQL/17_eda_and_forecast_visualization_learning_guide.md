# 17 EDA 与预测可视化学习指南 — 逐行讲解版

> 对应笔记本（notebook）：
>
> - `data_visualization/eda_1.1.ipynb`（EDA，使用最新 V3.1 数据集）
> - `data_visualization/forecast_visualization.ipynb`（模型评估 + 7天预测）
>
> 阅读前提：你完全没有 EDA（探索性数据分析）基础，所以本文从零讲起，包括：为什么拆分、EDA 是什么、为什么选这些图、每一行代码是做什么的、每个图该怎么看。
>
> **词汇说明**：中文讲解为主；凡是可能超出你雅思 3500 词汇量的技术词/专业词，我都在后面用括号标注了英文原文，方便你对照学习。

---

## 0. 背景：为什么要把 EDA 和预测可视化分开？

原来的 `eda_and_forecast_visualization.ipynb` 一个笔记本（notebook）里塞了三件事：

1. **EDA**（Exploratory Data Analysis，探索性数据分析）——分析数据集本身
2. **模型评估**（Model Evaluation）——看训练好的模型在测试集上准不准
3. **7天预测**（7-Day Forecast）——看每天的预测结果

现在拆成两个，各干各的，理由如下：

| 拆分后                         | 负责什么                   | 用哪个数据集                                           |
| ------------------------------ | -------------------------- | ------------------------------------------------------ |
| `eda_1.1.ipynb`                | 只做 EDA（探索性数据分析） | **最新的 V3.1**（70 列，含网格+核电）                  |
| `forecast_visualization.ipynb` | 模型评估 + 7天预测         | V3.1 数据 + `models/saved/*.pkl` + `predictions/*.csv` |

**为什么拆？**

- **职责单一**（single responsibility）：一个文件只做一类事，好维护、好理解。
- **EDA 要跟着数据走**：现在有了 V3.1 新特征（跨境输电 `fi_*`、核电 `nuclear_*`），EDA 需要重新跑一遍来看这些新特征。
- **预测可视化要跟着模型/预测结果走**：它每天都要刷新（每天有新的预测 CSV）。

> 我检查过 `docs/LearningNotes_CQL/` 里已有的解释：**10 号指南**（`10_model_visualization_learning_guide.md`）解释了"模型评估 + 未来预测"的概念（英文为主），但**没有逐行讲解代码，也没有讲解 EDA 那 8 张图（1.1–1.8）**。所以本文是新的、更详细的版本——尤其补上了"模型评估这一部分该怎么看"（见第 4 节）。

---

## 1. EDA 是什么？为什么要做 EDA？（零基础篇）

**EDA** = 在训练模型之前，先用图表和数据统计"认识"你的数据集。

打个比方（analogy）：你要给一个朋友介绍你自己。你有两个选择——

- 直接把你的人生履历（几万字）丢给他 → 他看完还是晕的。
- 用几张图（身高体重、爱好分布、时间分配饼图）快速让他看懂 → 他一下就有概念了。

**EDA 就是"用图表认识数据"**。它的目标：

1. **看分布**（distribution）——价格大多在什么范围？有没有特别离谱的值（异常值，outlier）？
2. **看趋势**（trend）——价格随时间怎么变？冬天贵夏天便宜吗？
3. **看周期**（seasonality/cycle）——每天几点贵？星期几便宜？每月呢？
4. **看相关性**（correlation）——温度和价格有关系吗？昨天价格和今天价格有关系吗？
5. **看数据质量**——有没有大量缺失值（missing values）？有没有明显错误？

**为什么先做 EDA 再训练？** 因为：

- 不做 EDA 就训练，等于"盲人摸象"——你不知道数据长什么样，出了问题（比如某列全是 0）你根本发现不了。
- EDA 会告诉你该做哪些特征工程（feature engineering）——比如发现电价有明显日周期，就该做滞后特征（lag feature）。
- 这是专业流程：**先探索（EDA）→ 再工程（feature engineering）→ 再训练（training）**。

---

## 2. `eda_1.1.ipynb` 逐行讲解

### 2.0 Setup（准备工作）——第 2、3 个 cell

**第 2 个 cell（markdown）**：标题 `## 0. Setup`，只是分区说明。

**第 3 个 cell（python）**，逐行：

```python
from pathlib import Path
```

- 从 `pathlib`（路径库，Python 标准库）导入 `Path` 类。
- 作用：用 `Path` 构造文件路径，Windows / Mac / Linux 都能用（跨平台，cross-platform）。

```python
import numpy as np
import pandas as pd
```

- 导入 `numpy`（数值计算库）和 `pandas`（表格数据处理库），是 Python 数据科学的"地基"。
- `as np` / `as pd` 是给库起**别名**（alias），以后写 `np.mean` 而不是 `numpy.mean`。

```python
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
```

- `plotly`（交互式绘图库）的三种导入：
  - `px` = **高级接口**（high-level），一句话就能画常见的图（折线、柱状、散点）。
  - `go` = **低级接口**（low-level），可以精确控制每一条线、每一个细节。
  - `pio` = 负责把图**保存成图片文件**（PNG）。
- 交互式（interactive）的意思是：图可以鼠标悬停看数值、缩放、拖拽。

```python
DATA_PATH   = Path('../data/convertData/V3.1_15min_features.csv')
CHARTS_DIR  = Path('../charts')
CHARTS_DIR.mkdir(exist_ok=True)
HELSINKI = 'Europe/Helsinki'
```

- `DATA_PATH`：指向**最新的 V3.1 数据集**（这是本次拆分的关键——EDA 用 V3.1 而不是旧的 V2.5）。
- `CHARTS_DIR`：图片保存目录（`charts/`）。
- `CHARTS_DIR.mkdir(exist_ok=True)`：创建目录；`exist_ok=True` 表示"如果目录已存在也不报错"（幂等，idempotent）。
- `HELSINKI`：时区名（timezone），`Europe/Helsinki` 芬兰赫尔辛基时区。

```python
def show_and_save(fig, name, width=1200, height=600, scale=2):
    """Render a figure in the notebook and persist a static PNG into charts/."""
    fig.show()
    pio.write_image(fig, CHARTS_DIR / f'{name}.png', width=width, height=height, scale=scale)
    print(f'  saved -> {CHARTS_DIR / f"{name}.png"}')
```

- 定义一个**辅助函数**（helper function）`show_and_save`，接收一个图 `fig` 和名字 `name`：
  - `fig.show()`：在笔记本里**交互式显示**这个图。
  - `pio.write_image(...)`：把图**保存成静态 PNG** 到 `charts/` 目录，文件名是 `name.png`。
  - `width/height` 是图片尺寸（像素，pixel）；`scale=2` 是 2 倍分辨率（清晰度）。
  - `print(...)`：打印保存路径，方便确认。
- 这样每张图都"既能在笔记本里看，又能在 `charts/` 里留一份永久副本"。

---

### 2.1 加载数据——第 4 个 cell

```python
df = pd.read_csv(DATA_PATH)
```

- `pd.read_csv(...)`：读取 CSV 文件成一个**数据框**（DataFrame，表格）。`df` 是 DataFrame 的惯例命名。

```python
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(HELSINKI)
```

- `pd.to_datetime(...)`：把 `datetime` 这一列从文本转成真正的**时间类型**（datetime）。
- `utc=True`：先统一按**世界协调时**（UTC, Coordinated Universal Time）解析——因为原始 CSV 里时区偏移不统一（冬令时 +02:00、夏令时 +03:00，混合时区 mixed timezone）。
- `.dt.tz_convert(HELSINKI)`：再转换到赫尔辛基时区。这样所有时间才对齐。
- 这是项目里修过的一个真实 bug：时区不对会让行错位。

```python
df = df.sort_values('datetime').reset_index(drop=True)
```

- `sort_values('datetime')`：按时间**排序**（时间序列必须有序）。
- `.reset_index(drop=True)`：排序后重新编号行索引（index），`drop=True` 表示不要旧索引列。

```python
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
```

- 从时间列**提取**出 `hour`（小时，0–23）、`day_of_week`（星期几，0=周一 … 6=周日）、`month`（月份，1–12）。
- 为什么？后面画"按小时/星期/月份"的图需要这些列。（原始表里其实已有这些特征列，这里重新生成是为了保证和 datetime 完全一致，方便分组。）

```python
print(f'Rows: {len(df):,} | {df["datetime"].min()} -> {df["datetime"].max()}')
print(f'Columns: {len(df.columns)}')
```

- 打印行数（`len(df)`，`,` 是千分位分隔符）、时间范围（最早到最晚）、列数。这相当于一个快速"体检报告"。

---

### 2.2 每张图的"为什么选它 + 怎么看"（第 5–14 个 cell）

#### 图 1.1 价格随时间变化（折线图，line chart）——第 6 个 cell

```python
sample = df.sample(n=20000, random_state=42).sort_values('datetime')
fig = px.line(sample, x='datetime', y='price',
              title='1.1 Electricity Price Over Time (V3.1)',
              labels={'datetime': 'Time', 'price': 'Price (EUR/MWh)'})
show_and_save(fig, '1.1_price_over_time')
```

**逐行解释：**

- `df.sample(n=20000, random_state=42)`：随机**抽样**（sample）2 万行。为什么抽？因为全表 10.5 万行全画出来，交互式图会卡。`random_state=42` 是**随机种子**（random seed），保证每次抽样结果一样（可复现，reproducible）。
- `.sort_values('datetime')`：抽样后再按时间排序（否则折线会乱）。
- `px.line(sample, x='datetime', y='price', ...)`：画**折线图**——横轴（x-axis）是时间，纵轴（y-axis）是价格。
- `title=...`：图的标题。`labels={...}`：把横轴/纵轴的显示名字改成更友好的（"Time"、"Price (EUR/MWh)"）。
- `show_and_save(...)`：显示 + 保存。

**为什么选折线图？** 因为要看"价格随时间怎么走"——趋势（trend）、波动（volatility）、有没有尖峰（spike）。

**怎么看？**

1. 看整体水平：价格大概在什么范围波动？
2. 看尖峰（spike）：哪些时刻价格冲到很高？（电价会因缺电、极寒冲高）
3. 看季节：冬天（1–3月、11–12月）是不是明显更贵？
4. 看是否稳定：波动大 = 市场紧张；平稳 = 市场宽松。

#### 图 1.2 价格分布（直方图，histogram）——第 6 个 cell（同一个 cell 的第二张图）

```python
fig = px.histogram(df, x='price', nbins=120, title='1.2 Price Distribution (V3.1)',
                   labels={'price': 'Price (EUR/MWh)', 'count': 'Count'})
fig.update_layout(showlegend=False)
show_and_save(fig, '1.2_price_distribution')
```

**逐行解释：**

- `px.histogram(df, x='price', nbins=120, ...)`：画**直方图**。把价格分成 120 个"桶"（bin，区间），统计每个区间里有多少个样本。
- `fig.update_layout(showlegend=False)`：去掉图例（legend），因为直方图不需要。

**为什么选直方图？** 看价格**分布**长什么样——是集中在某个区间，还是尾巴很长。

**怎么看？**

1. 看中心：价格集中在哪个范围（比如 20–60 之间）？
2. 看**右尾**（right tail）：是不是有很多特别高的价格？尾巴长 = 偶尔有极端高价。
3. 看是否对称。不对称说明有偏（skewed）。
4. 这张图会告诉你：价格**不是正态分布**（normal distribution），而是"中间多、右尾长"——这对建模有影响（模型会难以预测尖峰）。

#### 图 1.3 按小时分布（箱线图，box plot）——第 7 个 cell

```python
fig = px.box(df, x='hour', y='price', title='1.3 Price by Hour of Day (V3.1)',
             labels={'hour': 'Hour', 'price': 'Price (EUR/MWh)'})
show_and_save(fig, '1.3_price_by_hour')
```

**逐行解释：**

- `px.box(df, x='hour', y='price', ...)`：画**箱线图**。横轴是小时（0–23），纵轴是价格；每个小时画一个"箱子"。

**箱线图怎么读（重要基础）：** 一个箱子包含五个数——

- 中位数（median，箱子里那条横线）
- 下四分位（Q1，箱子下边）和上四分位（Q3，箱子上边）——中间包含中间 50% 的数据
- 上下"须"（whisker）——正常范围的边界
- 箱子外面的点 = **异常值**（outlier）

**为什么选箱线图？** 看"一天中不同时间价格是否有规律"——这是电价最重要的**日内周期**（intraday pattern, 日内模式）。

**怎么看？**

1. 早上（7–9 点）和傍晚（17–20 点）是不是贵？→ 因为用电高峰（peak hours）。
2. 深夜（0–6 点）是不是便宜？→ 因为需求低（off-peak）。
3. 这就是为什么特征里有 `is_peak_hour`（是否高峰）——EDA 证明了这个特征有道理。

#### 图 1.4 星期 / 月份的平均价格（柱状图，bar chart）——第 8 个 cell

```python
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df['day_name'] = df['day_of_week'].map(dict(enumerate(day_names)))

avg_day = df.groupby('day_name', as_index=False)['price'].mean()
fig = px.bar(avg_day, x='day_name', y='price', category_orders={'day_name': day_names},
             title='1.4a Average Price by Weekday (V3.1)',
             labels={'day_name': 'Weekday', 'price': 'Avg Price (EUR/MWh)'})
show_and_save(fig, '1.4a_price_by_weekday')

avg_month = df.groupby('month', as_index=False)['price'].mean()
fig = px.bar(avg_month, x='month', y='price',
             title='1.4b Average Price by Month (V3.1)',
             labels={'month': 'Month', 'price': 'Avg Price (EUR/MWh)'})
show_and_save(fig, '1.4b_price_by_month')
```

**逐行解释：**

- `day_names = [...]`：星期的英文名列表。
- `dict(enumerate(day_names))`：生成 `{0:'Mon', 1:'Tue', ...}` 的映射字典（dictionary）。
- `df['day_name'].map(...)`：把数字星期几（0–6）**映射**成英文名（Mon–Sun），方便看图。
- `df.groupby('day_name', as_index=False)['price'].mean()`：**按星期分组**（groupby），每组求价格**平均值**（mean）。`as_index=False` 表示分组字段保留为普通列而不是索引。
- `px.bar(...)`：画**柱状图**。`category_orders` 指定柱子顺序按星期排（否则可能按字母排乱掉）。
- 第二个柱状图同理：按月份分组求平均价格。

**为什么选柱状图？** 柱状图最适合"比较几个类别的平均值"——周几贵、几月贵。

**怎么看？**

1. **星期维度**：周末（Sat/Sun）是不是比工作日便宜？→ 工业用电少。
2. **月份维度**：冬季（12、1、2、3 月）是不是明显贵？→ 供暖需求大（这就是 `HDD` 特征、`season` 特征的依据）。
3. 这两张图直接支撑了特征工程里"日历特征（calendar feature）+ 季节（season）"的设计。

#### 图 1.5 天气 vs 价格（散点图，scatter plot）——第 10 个 cell

```python
sample = df.sample(n=20000, random_state=42)

fig = px.scatter(sample, x='temp', y='price', opacity=0.3,
                 title='1.5a Temperature vs Price (V3.1)',
                 labels={'temp': 'Temperature (°C)', 'price': 'Price (EUR/MWh)'})
show_and_save(fig, '1.5a_temp_vs_price')

fig = px.scatter(sample, x='wind_speed', y='price', opacity=0.3,
                 title='1.5b Wind Speed vs Price (V3.1)',
                 labels={'wind_speed': 'Wind Speed (m/s)', 'price': 'Price (EUR/MWh)'})
show_and_save(fig, '1.5b_wind_vs_price')
```

**逐行解释：**

- `px.scatter(sample, x='temp', y='price', opacity=0.3, ...)`：画**散点图**——每个点是一个时刻，横轴温度、纵轴价格。
- `opacity=0.3`：点的**透明度**（transparency/opacity）。为什么要透明？因为 2 万个点叠在一起会变成一团黑；透明后能看出**密度**（哪里点密哪里亮）。

**为什么选散点图？** 看两个**连续变量**（continuous variables）之间有没有关系（关系 = relationship）。

**怎么看？**

1. 温度 vs 价格：温度越低（左边），价格是不是越高？→ 冷 = 供暖需求大 = 价格高。如果左高右低，说明**负相关**（negative correlation）。
2. 风速 vs 价格：风速越大，风电越多，价格应该越低（**负相关**）。
3. 注意：散点图往往有大量重叠，看的是"总体趋势"而不是单个点。
4. 这张图验证了 `HDD`（采暖度日，heating degree day）这个特征为什么合理。

#### 图 1.6 相关性热力图（heatmap）——第 11 个 cell

```python
cols = ['price', 'temp', 'wind_speed', 'HDD', 'wind_power_proxy',
        'price_lag_1', 'price_lag_96', 'price_rolling_mean_1h', 'hour', 'day_of_week',
        'fi_total_net', 'fi_ee', 'nuclear_power_mw', 'nuclear_change_1d']
corr = df[cols].corr()

fig = px.imshow(corr, text_auto=True, aspect='auto',
                color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                title='1.6 Correlation Heatmap (V3.1)')
show_and_save(fig, '1.6_correlation_heatmap')
```

**逐行解释：**

- `cols = [...]`：**挑选**要参与相关分析的列。注意：这里特意加了 V3.1 的新特征（`fi_total_net` 跨境净流量、`fi_ee` 爱沙尼亚流、`nuclear_power_mw` 核电、`nuclear_change_1d` 核电日变化），因为 EDA 要分析新特征。
- `df[cols]`：只取这些列。`.corr()`：计算**相关矩阵**（correlation matrix）——两两之间的相关系数（correlation coefficient，范围 -1 到 +1）。
- `px.imshow(corr, text_auto=True, ...)`：把相关矩阵画成**热力图**（颜色深浅表示数值大小）。
  - `text_auto=True`：在格子里直接显示数字。
  - `aspect='auto'`：自动调整格子宽高。
  - `color_continuous_scale='RdBu_r'`：颜色方案——红（正相关）+ 蓝（负相关）。
  - `zmin=-1, zmax=1`：颜色标尺范围固定为 -1 到 1（相关性的理论范围）。

**怎么看？**

1. **对角线上全是 1**（自己和自己完全相关），正常，忽略。
2. 看 `price` 那一行（或列）：哪些特征和价格相关性强？
   - `price_lag_1`（昨天价格）如果接近 1 → 价格**强自相关**（autocorrelation），这就是滞后特征有效的原因。
   - `HDD` 如果正相关 → 越冷越贵。
   - `nuclear_change_1d` 如果负相关 → 核电下降（停机）时价格上升，方向符合直觉。
3. 颜色越深（越红 +1 或越蓝 -1）关系越强；接近 0（白色）表示几乎无关。
4. **注意**：相关性 ≠ 因果（correlation ≠ causation），且相关性强也不一定代表"加了就有用"（这正是 V2.5.1 实验的教训）。

#### 图 1.7 核电 vs 价格（V3.1 新特征）——第 13 个 cell

```python
sample = df.sample(n=20000, random_state=42).sort_values('datetime')

fig = px.line(sample, x='datetime', y='nuclear_power_mw',
              title='1.7a Nuclear Power Output Over Time (V3.1)',
              labels={'datetime': 'Time', 'nuclear_power_mw': 'Nuclear (MW)'})
show_and_save(fig, '1.7a_nuclear_over_time')

fig = px.scatter(sample, x='nuclear_power_mw', y='price', opacity=0.3,
                 title='1.7b Nuclear Output vs Price (V3.1)',
                 labels={'nuclear_power_mw': 'Nuclear (MW)', 'price': 'Price (EUR/MWh)'})
show_and_save(fig, '1.7b_nuclear_vs_price')
```

**逐行解释：** 和前面一样：折线图看核电随时间的变化；散点图看核电和价格的关系。

**为什么加这两张图？** 因为核电是 **V3.1 新加的特征**，EDA 必须"认识"它——看它有没有值、波动大不大、和价格有没有关系。

**怎么看？**

1. **折线图（1.7a）**：核电出力（nuclear power）是不是大部分时间是一条平稳的"基荷"（base load）线？有没有突然掉下去的地方？——**掉下去的地方 = 有核电机组停堆/检修（outage）**，这正是最重要的信号。
2. **散点图（1.7b）**：核电高的时候价格是不是偏低？核电低（停机）时价格是不是偏高？如果有这种"左下-右上"或"左上-右下"的形状，说明有关系。
3. 结论用于指导特征设计：核电是"事后"数据，只能做**滞后特征**（lag feature），如 `nuclear_lag_96`、`nuclear_change_1d`。

#### 图 1.8 跨境输电流量（V3.1 新特征）——第 14 个 cell

```python
sample = df.sample(n=20000, random_state=42).sort_values('datetime')

fig = px.line(sample, x='datetime', y='fi_total_net',
              title='1.8a Finland Total Net Flow (MW) — positive=export',
              labels={'datetime': 'Time', 'fi_total_net': 'Net Flow (MW)'})
fig.add_hline(y=0, line_dash='dash', line_color='red')
show_and_save(fig, '1.8a_fi_total_net')

fig = px.line(sample, x='datetime', y='fi_ee',
              title='1.8b FI↔Estonia Estlink Flow (MW) — positive=export',
              labels={'datetime': 'Time', 'fi_ee': 'Estlink Flow (MW)'})
show_and_save(fig, '1.8b_fi_ee')
```

**逐行解释：**

- 折线图看两个流：`fi_total_net`（芬兰总净流量）和 `fi_ee`（芬兰—爱沙尼亚 Estlink 线流量）。
- `fig.add_hline(y=0, line_dash='dash', line_color='red')`：在 y=0 处加一条**红色虚线**。为什么？因为 0 是"出口/进口"的分界线——上面是出口，下面是进口，虚线帮你一眼看清方向。

**怎么看？**

1. **正负号**：大于 0 = 芬兰在**出口**（export），小于 0 = 芬兰在**进口**（import）。
2. 看这条线在 0 上下摆动的情况——芬兰什么时候是净出口、什么时候净进口。
3. 这和电价的关系：进口多说明自己不够用，可能推高价格。EDA 阶段先用折线图确认数据合理、符号约定正确（符号约定 = sign convention）。

---

## 3. `forecast_visualization.ipynb` 逐行讲解（Setup + 数据准备）

### 3.0 Setup——第 2、3 个 cell

和第 2 节几乎一样，区别：

- 多了 `import joblib`（用来读取保存的模型 .pkl 文件，反序列化 deserialize）。
- 多了 `from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error`（计算模型评估指标）。
- 多了两个路径：`MODELS_DIR`（模型目录 `models/saved`）、`PREDICTIONS`（预测结果目录 `predictions`）。

### 3.1 加载数据 + 时序切分（chronological split）——第 5 个 cell

```python
df = pd.read_csv(DATA_PATH)
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert(HELSINKI)
df = df.sort_values('datetime').reset_index(drop=True)

X = df.drop(columns=['price', 'datetime'])
y = df['price']

n = len(df)
train_end = n - int(n * 0.20)
X_test  = X.iloc[train_end:].reset_index(drop=True)
y_test  = y.iloc[train_end:].reset_index(drop=True)
test_dt = df['datetime'].iloc[train_end:].reset_index(drop=True)
print(f'Train rows: {train_end:,} | Test rows: {n - train_end:,}')
```

**逐行解释：**

- `X = df.drop(columns=['price', 'datetime'])`：**X 是特征**（features）——去掉 `price`（目标）和 `datetime`（不是特征），剩下的都是模型的输入。`X` 大写是惯例（矩阵 Matrix）。
- `y = df['price']`：**y 是目标**（target/label）——要预测的电价。
- `train_end = n - int(n * 0.20)`：算出"前 80% 结束"的行号（`int` 取整）。这是 **80/20 切分**。
- `X.iloc[train_end:]`：取**最后 20%** 作为测试集（test set）。`.iloc` 是按位置取行。
- **为什么取最后 20% 而不是随机取？** 因为这是**时间序列**（time series），必须**按时间顺序切分**（chronological split）——训练用过去，测试用未来，**绝不打乱**（never shuffle）。如果打乱，未来信息会泄漏进训练（数据泄漏，data leakage），模型成绩是假的。

### 3.2 加载模型 + 预测 + 算指标——第 6 个 cell

```python
MODEL_NAME = 'xgboost_v2_5'
meta = joblib.load(MODELS_DIR / f'{MODEL_NAME}.pkl')
model = meta['model']
feature_cols = meta['feature_cols']

y_pred = model.predict(X_test[feature_cols])

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f'{MODEL_NAME}: MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}')
```

**逐行解释：**

- `MODEL_NAME = 'xgboost_v2_5'`：选哪个模型。你可以改成 `lightgbm_v2_5`、`xgboost_v2_5_3` 等（看 `models/saved/` 里有什么）。
- `joblib.load(...)`：读取模型文件（反序列化）。`meta` 是一个**字典**（dictionary），包含三个东西：`model`（模型本体）、`feature_cols`（训练时用的特征列名）、`step_min`（分辨率）。
- `X_test[feature_cols]`：**只喂模型它训练时用的那些列**（顺序也要对）——如果多传或少传列，预测会错。
- `model.predict(...)`：在测试集上做预测，得到 `y_pred`（预测值数组）。
- **三个指标：**
  - **MAE**（Mean Absolute Error，平均绝对误差）= 平均每个预测差多少（EUR/MWh）。越小越好。好理解：平均差 2.8 欧。
  - **RMSE**（Root Mean Squared Error，均方根误差）= 先求误差平方的平均再开根号。它对**大误差惩罚更重**（因为先平方了）。RMSE 明显大于 MAE 说明存在少数大错误。
  - **R²**（R squared，决定系数）= 模型解释了价格波动的百分比（0–1，越接近 1 越好）。0.97 表示解释了 97% 的波动。
- `:.4f` 是格式化：保留 4 位小数。

---

## 4. Model Evaluation（模型评估）——这个部分怎么看？（重点补讲）

这是你要求重点补充的部分。**核心思想：先证明模型可信，再相信它的预测。**

### 4.1 为什么先看模型评估？

在 `forecast_visualization.ipynb` 里，模型评估（Section 1）在前，7 天预测（Section 2）在后。逻辑是：

- **评估**：拿"历史数据里模型没见过的最后 20%"来考它（像期末考），看它准不准。
- **预测**：如果考试合格，才相信它对"未来 7 天"的预测。

### 4.2 四个图分别怎么看？

**图 1.1 实际 vs 预测（时间序列折线图）**

- 两条线：蓝/红 = 真实价格（Actual），另一条 = 预测（Predicted），只画测试集前 7 天（`steps = 7 * 24 * 4`，672 个点）。
- **怎么看**：两条线是否基本重合（overlap）？重合 = 模型好；分开 = 模型差。特别看**尖峰时刻**两条线差多少——模型往往跟不上尖峰。

**图 1.2 实际 vs 预测（散点图 + 对角线）**

- 每个点：横轴 = 真实价，纵轴 = 预测价。红色虚线是 **y=x**（完美预测线，identity line）。
- **怎么看**：点越靠拢红线越好。点在线上方 = 预测过高（over-predict）；下方 = 预测过低（under-predict）。整体呈一条斜带说明预测有真实价值；如果是一团圆形则说明预测和实际无关。

**图 1.3 残差分布（residual distribution）**

- **残差（residual）** = 真实值 − 预测值 = 每个点的预测误差。
- 直方图：统计所有误差的分布。
- **怎么看**：理想模型误差围绕 0 对称分布（中心在 0，两侧对称）。
  - 均值（mean）接近 0 → 没有系统性偏差（systematic bias，系统性偏差）——模型不会"整体偏高"或"整体偏低"。
  - 标准差（std）小 → 误差集中、稳定。
  - 有很长的尾巴 → 存在少数巨大误差（通常是价格尖峰时刻）。
  - 均值明显不是 0 → 模型有偏，需要检查。

**图 1.4 特征重要性（feature importance）**

- 横条：每个特征对模型的"贡献"有多大（`model.feature_importances_`）。
- `imp.tail(20)`：取**最重要的 20 个**（按重要性升序排好后取最后 20 个），`orientation='h'` 横着画（方便读名字）。
- **怎么看**：排在最上面的特征就是模型最依赖的。比如如果 `price_lag_1`、`price_rolling_mean_1h` 排最前 → 电价强自相关，滞后特征就是主力。如果 `nuclear_*`、`fi_*` 排在很后面 → 说明 V3.1 新特征对当前模型贡献小（这本身就是一个有价值的 EDA 结论）。

### 4.3 模型评估的局限（一定要知道）

- **测试集分数 ≠ 实际表现**。本项目实测过：测试集 MAE 只有 ~2.8，但**线上递归预测**（recursive forecasting，递归预测）MAE 高达 ~30（因为预测值会喂回模型当特征，误差会累积 + 2026 年价格环境变了）。
- 所以模型评估图只能证明"模型在历史数据上靠谱"，**不能**保证"未来一定准"。要持续监控每天的预测 vs 实际（看 `predictions/*.csv` 里的 `abs_error`）。

---

## 5. 7-Day Forecast（7天预测）——怎么看（第 12–14 个 cell）

### 5.1 预测 CSV 从哪来？

`predictions/*.csv` 是 `src/predict_system.py` 每天自动生成的（GitHub Actions 每天 UTC 11:00 跑）。每个模型一个 CSV，列有：

- `run_date`：哪天跑的
- `target_datetime`：预测的是哪个时刻
- `predicted_price`：预测价
- `actual_price`：真实价（时间过去后自动回填 backfill）
- `abs_error`：绝对误差

### 5.2 `load_forecasts()` 函数——第 12 个 cell

```python
def load_forecasts():
    frames = {}
    for path in sorted(PREDICTIONS.glob('*_forecasts.csv')):
        f = pd.read_csv(path, parse_dates=['run_date', 'target_datetime'])
        name = path.stem.replace('_forecasts', '')
        if f['target_datetime'].dt.tz is None:
            f['target_datetime'] = f['target_datetime'].dt.tz_localize('UTC').dt.tz_convert(HELSINKI)
        else:
            f['target_datetime'] = f['target_datetime'].dt.tz_convert(HELSINKI)
        frames[name] = f
    return frames
```

**逐行解释：**

- `frames = {}`：建一个空**字典**，用来装"模型名 → 预测数据框"。
- `PREDICTIONS.glob('*_forecasts.csv')`：**通配符**（wildcard）匹配 `predictions/` 下所有 `xxx_forecasts.csv` 文件。
- `pd.read_csv(path, parse_dates=['run_date','target_datetime'])`：读取 CSV，并把两列时间列解析成时间类型。
- `path.stem.replace('_forecasts', '')`：从文件名提取模型名（如 `xgboost_v2_5_forecasts` → `xgboost_v2_5`）。
- 时区处理：如果没有时区就按 UTC 解析再转赫尔辛基（和前面一样的原因）。
- `frames[name] = f`：放进字典。

### 5.3 图 2.1 所有模型对比——第 13 个 cell

```python
for name, f in forecasts.items():
    f_latest = f[f['run_date'] == f['run_date'].max()]
    fig.add_trace(go.Scatter(x=f_latest['target_datetime'], y=f_latest['predicted_price'],
                             name=name, mode='lines'))
```

**逐行解释：**

- `forecasts.items()`：遍历每个模型。
- `f['run_date'].max()`：找到**最近一次**运行日期。
- `f[f['run_date'] == ...]`：只取最近这一次的预测（布尔过滤，boolean filtering）。
- `go.Scatter(..., mode='lines')`：加一条折线。`mode='lines'` = 只画线不画点。

**怎么看：**

1. 所有模型的 7 天预测画在一张图上，一眼看出谁和谁接近、谁是**离群**（outlier，离群值）。
2. 如果某条线明显偏离大多数 → 那个模型可能有问题或风格不同。
3. 交互式图：点图例（legend）可以单独显示/隐藏某条线。

### 5.4 图 2.2 最佳模型 vs 真实价——第 14 个 cell

```python
name = 'xgboost_v2_5'
f_latest = f[f['run_date'] == f['run_date'].max()]
fig.add_trace(... predicted ...)
actual = f_latest.dropna(subset=['actual_price'])
if not actual.empty:
    fig.add_trace(... actual (black dashed) ...)
```

**逐行解释：**

- 默认看最佳模型 `xgboost_v2_5`（你也可以改）。
- `dropna(subset=['actual_price'])`：**删掉** `actual_price` 为空的行（这些时刻的真实价还没公布）。
- 如果已有真实价（`actual` 非空），就加一条**黑色虚线**叠上去。

**怎么看：**

- 预测线 vs 黑色真实线：差多少一眼可见。
- 刚跑完的前几天真实价还没公布（虚线还没出来），过几天回填（backfill）后就能对比。
- 这是对"昨天的预测准不准"的**持续体检**——每天跑一次就能看到历史预测的误差。

---

## 6. 词汇表（本指南出现的技术词，中英对照）

| 中文           | English                          | 一句话解释                    |
| -------------- | -------------------------------- | ----------------------------- |
| 探索性数据分析 | Exploratory Data Analysis (EDA)  | 训练前用图表认识数据          |
| 特征           | feature                          | 描述一个时刻的输入数值        |
| 目标/标签      | target / label                   | 要预测的值（电价）            |
| 数据框         | DataFrame                        | pandas 里的表格               |
| 时区           | timezone                         | 时间标准（赫尔辛基时区）      |
| 世界协调时     | UTC (Coordinated Universal Time) | 全球统一时间基准              |
| 抽样           | sample                           | 随机取一部分                  |
| 随机种子       | random seed                      | 让随机结果可复现的固定数字    |
| 直方图         | histogram                        | 统计数值落在各区间数量的图    |
| 箱线图         | box plot                         | 展示中位数/四分位/异常值的图  |
| 散点图         | scatter plot                     | 一个点一个样本，看两变量关系  |
| 柱状图         | bar chart                        | 比较类别平均值的图            |
| 折线图         | line chart                       | 看随时间变化的图              |
| 热力图         | heatmap                          | 用颜色深浅表示数值的矩阵图    |
| 相关矩阵       | correlation matrix               | 两两变量相关系数的表格        |
| 相关系数       | correlation coefficient          | 两变量线性关系强度（-1 到 1） |
| 自相关         | autocorrelation                  | 自己和自己的过去相关          |
| 分布           | distribution                     | 数值的散布情况                |
| 异常值         | outlier                          | 离大多数很远的值              |
| 中位数         | median                           | 排序后正中间的数              |
| 四分位         | quartile                         | 把数据四等分的分界点          |
| 右尾           | right tail                       | 分布右侧很长的部分            |
| 负相关         | negative correlation             | 一个升另一个降                |
| 透明度         | opacity / transparency           | 点的透明程度                  |
| 交互式         | interactive                      | 可悬停、缩放、拖拽            |
| 时间序列       | time series                      | 按时间排序的数据              |
| 时序切分       | chronological split              | 按时间顺序切分训练/测试       |
| 数据泄漏       | data leakage                     | 未来信息泄漏进训练            |
| 残差           | residual                         | 真实值 − 预测值               |
| 系统性偏差     | systematic bias                  | 模型整体偏高或偏低            |
| 特征重要性     | feature importance               | 每个特征对模型的贡献          |
| 递归预测       | recursive forecasting            | 预测值喂回模型继续预测        |
| 回填           | backfill                         | 真实值公布后填入              |
| 离群           | outlier                          | 明显偏离整体的点              |
| 图例           | legend                           | 图中区分线条的说明            |
| 反序列化       | deserialize                      | 把文件读回内存对象            |
| 数据质量       | data quality                     | 数据是否干净可信              |

---

## 7. 一页总结（复习用）

1. **EDA（探索性数据分析）** = 训练前用图表认识数据：分布、趋势、周期、相关性、异常值。不做 EDA 就训练 = 盲人摸象。
2. **为什么拆**：EDA 跟着数据走（现在用 V3.1 新特征），预测可视化跟着模型/预测走，职责分离更好维护。
3. **每张图一个目的**：折线看趋势、直方图看分布、箱线图看日内周期、柱状图看星期/月份、散点图看相关性、热力图看整体相关矩阵、核电/网格图认识 V3.1 新特征。
4. **模型评估怎么看**：散点贴近 y=x 好；残差围绕 0 对称好；均值≈0 无系统性偏差；重要性排最前的就是模型最依赖的特征。
5. **切记**：测试集分数 ≠ 实际表现（本项目线上递归预测误差远大于测试集），要持续监控每日预测误差。
