# 13. The `src/` Folder — Complete Guide（src 文件夹详解 & 与 GitHub Actions 的联动）

> Date: 2026-08-13
> **Extends** 指南 `06_model_comparison_and_automation_guide.md`（讲 src 概览）和
> `08_from_training_to_automation_learning_guide.md`（讲"训练→预测"心智模型）。
> 本篇是**深入每一个文件**的版本，并在最后加入最新加入的**网格传输（grid）**接入。
> 技术术语配中文注释。

---

## 0. TL;DR — 一句话

`src/` 是**线上预测系统**（live prediction system）：它每天从公开 API 抓取最新
电价 + 天气（+ 可选网格数据），用训练好的 `.pkl` 模型**递归预测未来 7 天**，
把结果写成 CSV。**GitHub Actions 只是每天定时执行一句 `python src/predict_system.py`**，
`src/` 里的代码自己完成所有工作。

```
训练世界（一次性）                预测世界（每天自动）
notebook 造特征→训练→存 .pkl  ──►  src/ 加载 .pkl → 抓数据 → 造特征 → 预测 → CSV → commit
                                   ▲
                                   └─ 被 .github/workflows/daily_forecast.yml 唤醒
```

---

## 1. `src/` 文件夹地图（每个文件干什么）

| 文件                  | 角色     | 核心内容                                          | 是否必需       |
| --------------------- | -------- | ------------------------------------------------- | -------------- |
| `config.py`           | 配置中心 | 路径、API 地址、预测天数、网格配置                | ✅ 必需        |
| `features.py`         | 特征工厂 | `build_features()` + 3 个 buffer 类               | ✅ 必需        |
| `fetch_live.py`       | 数据采集 | Elering 电价 + FMI/Open-Meteo 天气 + Fingrid 网格 | ✅ 必需        |
| `predict_system.py`   | 指挥中心 | 主程序：抓数据→预测→存 CSV                        | ✅ 必需        |
| `plot_predictions.py` | 可视化   | 把预测 CSV 画成 `charts/*.png`                    | 可选（跑完图） |
| `test_fmi.py`         | 诊断工具 | 手动测试 FMI 接口返回什么                         | 调试用         |
| `utils.py`            | 工具函数 | 目前是空的占位                                    | 预留           |
| `__init__.py`         | 包标记   | 让 `src` 成为 Python 包                           | 约定           |

### 文件依赖关系（谁 import 谁）

```
predict_system.py ──► fetch_live.py ──► config.py
      │                    │
      └──► features.py ────┘
            │
      （被 predict_system 调用）
plot_predictions.py ──► config.py
```

- `config.py` 被几乎所有模块 import（它不含密码，只有公开 URL + 环境变量读取）。
- `features.py` / `fetch_live.py` 都只依赖 `config.py`。
- `predict_system.py` 是唯一"指挥"模块：它调用 `fetch_live`（抓数据）、`features`（造特征）、`config`（路径）。

---

## 2. `config.py` — 配置中心（Configuration Hub）

**功能**：所有路径、API 地址、超参数常量集中在**一个文件**，方便审计和修改。

```python
ROOT = Path(__file__).resolve().parent.parent   # 项目根目录（自动定位，不写死）
SAVED_MODELS_DIR = ROOT / 'models' / 'saved'    # 模型仓库
PREDICTIONS_DIR  = ROOT / 'predictions'         # 预测输出
HELSINKI = 'Europe/Helsinki'                    # 全局时区（冬 +02 / 夏 +03）

ELERING_PRICE_URL = 'https://dashboard.elering.ee/api/nps/price'   # 电价（无需 key）
OPEN_METEO_FORECAST_URL = 'https://api.open-meteo.com/v1/forecast' # 长期天气

FORECAST_HOURS = 7 * 24        # 预测 7 天 = 168 小时
PRICE_HISTORY_HOURS = 200      # 电价历史窗口（~8.3 天，供滞后特征用）
```

**2026-08 新增 — 网格传输配置**：

```python
FINGRID_API_URL = 'https://data.fingrid.fi/api'          # Fingrid 开放数据
FINGRID_API_KEY = os.environ.get('FINGRID_API_KEY', '')  # 从环境变量读 key！

# 4 条跨境线路的 dataset ID（与 15min_GridTransmission.ipynb 一致）
FINGRID_GRID_DATASETS = {'fi_ee': 55, 'fi_no': 57,
                         'fi_se_north': 60, 'fi_se_central': 61}
GRID_FLOW_COLS  = ['fi_ee','fi_no','fi_se_north','fi_se_central',
                   'fi_se_total','fi_total_net','fi_se_abs']
GRID_LAG_COLS   = ['fi_total_net_lag_96','fi_total_net_lag_672',
                   'fi_se_total_lag_96','fi_se_total_lag_672',
                   'fi_ee_lag_96','fi_ee_lag_672']
```

> **设计要点**：`config.py` **不含任何写死的密钥**（no credentials）。
> Fingrid 需要免费 API key，通过环境变量 `FINGRID_API_KEY` 在运行时提供；
> 没有 key 时网格功能自动跳过（网格模型会被安全忽略）。

---

## 3. `features.py` — 特征工厂（Feature Factory）

**核心思想**：把训练 notebook 里的特征工程**用代码原样复刻**，这样线上预测时的
特征和训练时**完全一致**（否则模型预测无效）。

### 3.1 三个 Buffer（数据缓冲器）

| 类                 | 作用                                        | 关键方法                     |
| ------------------ | ------------------------------------------- | ---------------------------- |
| `PriceBuffer`      | 存 15 分钟电价值（历史 + 递归填回的预测值） | `lag_steps`, `rolling_steps` |
| `WeatherBuffer`    | 存小时天气，滚动/滞后窗口                   | `get`, `rolling_hours`       |
| `GridBuffer`（新） | 存 15 分钟跨境流量                          | `get`（当期流）, `lag_steps` |

**为什么用 buffer 而不是 DataFrame？**
因为递归预测时，每个未来时刻的"滞后特征"依赖**之前时刻的值**（包括刚预测出来的值）。
PriceBuffer 用字典按时间快照存储，`add(ts, price)` 把预测值填回去，下一个时刻就能取到。

### 3.2 `build_features(dt, price_buf, wx_buf, grid=None)` — 特征工厂主函数

对**任意一个时刻 dt** 返回一个包含**所有特征**的字典：

- 时间/日历/周期编码（hour, day_of_week, is_holiday, sin/cos...）
- 天气（temp, wind, HDD, wind_power_proxy, 温度滚动/滞后）
- 电价滞后（price_lag_1~672）+ 滚动（mean/std/min/max）
- **网格（V3，可选）**：
  - 当期流（`fi_ee` 等）→ **预测未来时刻时拿不到 → NaN**
  - 滞后（`fi_*_lag_96/672`）→ 从历史可算（24h/7d 前）

**为什么一个函数能服务所有模型？**
因为每个 `.pkl` 保存了自己的 `feature_cols` 清单，`build_features` 返回所有特征，
每个模型只取自己训练时用到的那些列：

```python
row = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_cols}])
```

### 3.3 关键设计：当期网格流 = NaN（train/serve 差距）

训练时模型见过 `fi_ee(t)` = **t 时刻实际**跨境流量；但线上预测**未来** t 时，
`fi_ee(t)` 还没发生 → 拿不到。所以 `GridBuffer.get(future)` 返回 NaN。
这正是为什么**能部署的网格模型只能用滞后特征**（见 12 节实验结论）。

---

## 4. `fetch_live.py` — 数据采集（Data Fetchers）

### 4.1 电价 — `fetch_prices(start, end)`

- 来源：**Elering NPS**（公开、免费、**无需 API key**）。
- 返回：FI 日前电价，15 分钟分辨率，`Europe/Helsinki` 时区。
- 防坑：如果返回全 0（曾踩过 Fingrid dataset 105 是"下调备用量"而非电价的坑）→ 直接报错拒绝。

### 4.2 天气 — `fetch_weather(start, end)`

组合**两个来源**保证 7 天都有数据：

| 来源              | 提供                                           | 覆盖                          |
| ----------------- | ---------------------------------------------- | ----------------------------- |
| FMI（芬兰气象所） | 观测（`place=Helsinki/Oulu`）+ HIRLAM 短时预报 | 过去 ~54 小时 → 未来 ~54 小时 |
| Open-Meteo        | 长期预报（10 天）                              | 剩余天数                      |

**防坑**：如果长期预报覆盖不完整，`fetch_weather` **直接报错**而不是静默用旧数据
前向填充（绝不"伪造"7 天天气）。

### 4.3 网格 — `fetch_grid_transmission(start, end)`（新增）

- 来源：**Fingrid Open Data**（需要 `FINGRID_API_KEY`）。
- 复用训练 notebook 的分页 + 429 重试模式。
- 抓 4 条线路（EE/NO/SE1/SE3）→ 派生 7 列（`fi_se_total`、`fi_total_net`、`fi_se_abs`）→ 重采样 15 分钟。

### 4.4 `test_fmi.py` — 为什么单独一个文件？

这是**调试工具**：手动测试 FMI 接口各种参数（latlon / place / fmisid / bbox / timestep）
到底返回什么、有没有 `ExceptionReport`。它不是预测的一部分，只在排查数据问题时用。

---

## 5. `predict_system.py` — 指挥中心（The Conductor）

### 主流程 `main()`

```
now = 赫尔辛基当前时间
predict_start = 下一个本地午夜（明天 00:00）      ← 从不从"今天中间"开始
predict_end   = predict_start + 168h - 15min

1. fetch_prices(now-200h, predict_start+24h)      → 电价历史（滞后特征用）
2. fetch_weather(now-200h, predict_end)           → 小时天气
3. (可选) fetch_grid_transmission(...) → GridBuffer
4. load_models()                                   → 扫描 models/saved/*.pkl
5. 对每个模型：
     fill_actuals(...)     # 把已发生的真实电价回填到旧预测
     删除"今天"这天的行（重跑不重复）
     run_forecast(...)     # 递归预测 7 天
     追加新行 → 去重 → 排序 → save_csv(...)
6. 打印每个模型的线上 MAE 摘要
```

### 关键函数

| 函数             | 作用                                                                          |
| ---------------- | ----------------------------------------------------------------------------- |
| `load_models()`  | 扫描 `models/saved/*.pkl`，全部加载。**新增模型无需改代码，放进去就自动被用** |
| `run_forecast()` | **递归预测**：预测 t → 把预测值塞回 PriceBuffer → 预测 t+1（误差会累积）      |
| `fill_actuals()` | 预测时间已过、真实电价已公布 → 回填 `actual_price` + `abs_error`              |
| `save_csv()`     | 每个模型写一个 `predictions/<model>_forecasts.csv`                            |

### 新增：网格模型的"安全跳过"

```python
needs_grid = any(c.startswith('fi_') for c in meta['feature_cols'])
if needs_grid and grid_buf is None:
    print(f'  [skip] {model_name}: needs grid features but no grid data available')
    continue
```

没有 `FINGRID_API_KEY` 或网格抓取失败 → 需要网格特征的模型被**安全跳过**，
其余模型照常预测。

---

## 6. `plot_predictions.py` — 可视化（图表）

**功能**：读取 `predictions/*_forecasts.csv`，给每个模型画一张 7 天预测折线图
（`charts/<model>.png`），再画一张所有模型对比图（`charts/comparison.png`）。

- 只**读 CSV + 画图**，不做预测。
- 真实电价（黑色虚线）在有值的地方叠加。
- 跑法：`python src/plot_predictions.py`（在 `predict_system.py` 之后跑）。

---

## 7. `utils.py` 和 `__init__.py`

- `utils.py`：目前是**空占位**（预留放通用小函数）。别的项目喜欢把"读 CSV 的公共函数"
  放这里，本项目暂时不需要。
- `__init__.py`：空文件，作用是告诉 Python"这是一个包"。`src` 被 `predict_system.py`
  通过 `sys.path.insert(0, str(Path(__file__).parent))` 加入路径后，
  `import config / features / fetch_live` 才找得到。

---

## 8. 与 GitHub Actions 的联动（How it all links together）

`.github/workflows/daily_forecast.yml` 是**闹钟 + 搬运工**：

```yaml
on:
  schedule:
    - cron: '0 11 * * *'     # 每天 UTC 11:00 = 芬兰 13/14:00（电价公布后）
  workflow_dispatch:          # 也支持手动触发

jobs:
  forecast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4            # 1. 拉取仓库（含 src/ 和 models/saved/）
      - uses: actions/setup-python@v5        # 2. Python 3.11
      - run: pip install -r requirements.txt # 3. 装依赖（版本锁定，保证 pkl 能反序列化）
      - run: python -m unittest discover -s tests -v   # 4. 先跑测试（坏了就不发预测）
      - run: python src/predict_system.py    # 5. ★ 唯一真正执行预测的命令
      - run: git add predictions/ && git commit -m "forecast: YYYY-MM-DD daily update" && git push
```

**联动本质**：

```
GitHub Actions（定时） ──运行──► python src/predict_system.py
                                    │
                                    ├─ src/fetch_live.py   抓最新电价/天气/(网格)
                                    ├─ src/features.py     造特征（和训练一致）
                                    ├─ src/predict_system.py 加载 pkl → 递归预测 7 天
                                    └─ 写 predictions/*.csv
                                              │
                                      git commit + push（预测结果进仓库）
                                              │
                                      （第二天）用户可打开 charts / CSV 查看
```

**关键点**：

1. **workflow 本身不写任何预测逻辑**，它只是"在特定时间运行这一条命令"。
2. `src/` 代码和测试**都提交在仓库里**，所以云端能复现。
3. 依赖**版本锁定**（requirements.txt）是为了 `.pkl` 能跨环境反序列化。
4. 预测结果提交进仓库 → 有历史、可追踪、可画图。
5. **要跑网格模型**：需要在 GitHub 仓库 Settings → Secrets 里加
   `FINGRID_API_KEY`（对应 `config.py` 读的环境变量）。

---

## 9. 怎么运行和测试

```powershell
# 1. 跑测试（14 个，含网格新测试）
python -m unittest discover -s tests -v

# 2. 跑一次实时预测（联网）
python src/predict_system.py

# 3. 把预测画成图（可选）
python src/plot_predictions.py

# 4. 排查 FMI 接口（可选）
python src/test_fmi.py
```

---

## 10. 关键设计决策（值得记住）

| 决策                               | 原因                                           |
| ---------------------------------- | ---------------------------------------------- |
| 所有东西用 `Europe/Helsinki` 时区  | 训练/预测数据对齐，避免冬夏令时错位            |
| `config.py` 无写死密钥             | 公开数据源可免配置运行；Fingrid key 走环境变量 |
| 预测从"下一个午夜"开始             | 今天电价可能已公布，从中间开始会重复/混乱      |
| 递归预测（预测值当历史）           | 模型需要滞后特征，未来没有 → 只能自己填回去    |
| 滞后/滚动窗口**不含当前时刻**      | 防泄漏：特征只能用过去的信息                   |
| 网格"当期流"在预测时 = NaN         | 未来实测流量不存在；能部署的只有滞后特征       |
| 重跑不重复插入（按 run_date 去重） | 手动重跑只更新今天的行                         |
| 网格模型在无数据时**安全跳过**     | 没 key/抓取失败时不产出垃圾预测                |

---

## 11. 与指南 06 / 08 的关系（我扩展了什么）

| 指南 | 讲了                                               | 本篇新增/深化                                                          |
| ---- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| 06   | src 概览 + 三步流程 + pkl 解释                     | 逐文件深挖、`plot_predictions.py`、`test_fmi.py`、依赖关系、网格接入   |
| 08   | .yml 是什么 + pkl 三件套 + predict_system 心智模型 | 模块内部实现、buffer 机制、安全跳过逻辑、GitHub Secrets、10 条设计决策 |

---

## 12. 最新实验结论：网格特征能否进线上？（2026-08-13）

| 版本               | 特征                         | MAE    | 结论                                      |
| ------------------ | ---------------------------- | ------ | ----------------------------------------- |
| 完整 V3.1          | 49 + 7 当期流 + 6 滞后（62） | 2.7152 | 变好（-0.0084），但**当期流预测时拿不到** |
| **live-safe V3.1** | 49 + 6 滞后（55）            | 2.7416 | **变差（+0.0180）**，不能部署             |
| 调好的 V2.5.3      | 49                           | 2.7236 | ✅ 生产模型                               |

**核心教训（train/serve gap，训练/服务差距）**：在训练时带来提升的"当期网格流"，
恰恰是线上预测时**拿不到**的那部分；线上可行的"滞后"网格特征反而没用。
所以 `src/` 里网格接入的**代码已就绪并通过测试**，但**目前没有部署任何网格模型**
（`models/saved/` 中没有带 `fi_*` 特征的 pkl）。一旦未来出现"线上可用且有帮助"的
网格特征，这套基础设施可以直接启用。
