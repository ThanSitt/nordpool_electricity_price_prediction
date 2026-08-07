# 08 从"训练模型"到"自动预测"：完整认知转换指南

> 对应你的问题："我训练完了 XGBoost V2，然后呢？怎么变成 GitHub Actions 自动预测的？"
> 目标读者：只做过训练、还没理解"预测"和"部署"的初学者
> 所有中文专有名词都配英文

---

## 0. 先给你一张"全局地图"（最重要！先看这个）

```
┌──────────── 你熟悉的世界：训练（Training）────────────┐
│                                                       │
│  数据 → 特征 → XGBoost 训练 → 得到"模型"              │
│  （notebook 里做的事）          ↓                      │
│                          model.pkl（模型文件）        │
└───────────────────────────────────────────────────────┘
                        ↓ 进入"部署"（Deployment）世界
┌──────────── 你还不熟的世界：预测 + 自动化 ─────────────┐
│                                                       │
│  GitHub Actions（.yml）每天定时喊：                    │
│      python src/predict_system.py                     │
│        ↓ 读取所有 model.pkl                           │
│        ↓ 拿最新天气/电价 → 造特征 → 预测7天            │
│        ↓ 生成 predictions/*.csv                       │
│        ↓ git commit（forecast: 日期 daily update）     │
└───────────────────────────────────────────────────────┘
```

**核心一句话**：训练产生"模型文件（.pkl）"，预测程序"使用"这个模型文件去算未来；GitHub Actions 只是每天自动喊预测程序起床。**模型不是 CSV，也不是 notebook。**

---

## 1. .yml 是什么？（你的问题 1、2、3）

### 1.1 .yml 是程序吗？是 Python 吗？

**不是程序，也不是 Python。** `.yml`（也叫 `.yaml`）是一种**配置文件格式**（config file format），就像 JSON、TOML 一样，用来**描述"一些设定"**，而不是"执行逻辑"。

| 对比          | 编程语言（Python）        | 配置格式（YAML）               |
| ------------- | ------------------------- | ------------------------------ |
| 干什么        | 写"怎么算"的逻辑          | 写"有哪些设定/步骤"的描述      |
| 例子          | `for i in range(10): ...` | `schedule: cron: '0 11 * * *'` |
| 有没有 if/for | 有                        | 没有（只是数据）               |
| 结尾          | `.py`                     | `.yml` / `.yaml`               |

YAML 全称是 "**YAML Ain't Markup Language**"（YAML 不是标记语言）——一个递归缩写，意思是"我就是个简单的人类可读格式"。

### 1.2 为什么用 .yml 结尾？

`.yml` 是 YAML 格式文件的后缀名（`.yaml` 是另一种常见写法，两者一样）。用它能告诉"编辑器/工具"：请按 YAML 语法高亮和检查这个文件。

### 1.3 这是 GitHub 专属格式吗？

**YAML 本身不是 GitHub 专属**——很多工具都用 YAML 写配置，比如：

- Docker（`docker-compose.yml`）
- Kubernetes（k8s）
- Ansible（自动化运维）
- **你自己的项目也有！`environment.yml` 就是 YAML 格式！**

但 **`.github/workflows/*.yml` 这种"工作流文件"是 GitHub Actions 专属的约定**：GitHub 会在你仓库的 `.github/workflows/` 文件夹里找 YAML 文件，按里面的设定自动执行。

> 📌 一句话：**YAML 是通用格式，`daily_forecast.yml` 是"用 YAML 写的一份 GitHub 自动化说明书"。**

---

## 2. 训练完的模型到底是个什么东西？（你的问题 9 核心）

你问："训练完成的模型是 CSV 吗？是 notebook 吗？能改 notebook 里的数据来预测吗？"

**都不是。** 让我彻底讲清楚：

### 2.1 训练完得到的是 .pkl 文件（模型文件）

训练完成后，代码会执行这一句（在 notebook 最后）：

```python
joblib.dump({
    'model': model_v25,            # ① 训练好的模型本体
    'feature_cols': X_train.columns.tolist(),  # ② 用了哪些特征
    'step_min': 15,                # ③ 分辨率（15分钟）
}, 'models/saved/xgboost_v2_5.pkl')
```

`joblib.dump()` 把内存里那个"训练好的大脑"**打包保存成磁盘文件**，后缀是 `.pkl`。

**类比**：训练好的模型就像"一个学完了所有知识的大脑"。`.pkl` 文件就是把这个大脑**冷冻保存**起来的容器。`.pkl` = pickle（泡菜/腌渍）——意思是"把对象腌起来保存"。

### 2.2 .pkl 里到底装了哪三样东西？

| 字段           | 中文       | 作用                                 |
| -------------- | ---------- | ------------------------------------ |
| `model`        | 模型本体   | 真正用来预测的"大脑"                 |
| `feature_cols` | 特征列名单 | 预测时要给模型喂哪些列（顺序不能错） |
| `step_min`     | 时间分辨率 | 60=小时，15=15分钟                   |

**这就是为什么不是"改 notebook 数据就能预测"**：notebook 是"学校"（用来训练），预测时你不需要学校，你只需要**解冻模型文件**（`.pkl`），然后用新数据喂给它。

### 2.3 预测 = 加载 .pkl + 调用 predict()

```python
import joblib
meta = joblib.load('models/saved/xgboost_v2_5.pkl')   # 解冻大脑
model = meta['model']                                   # 拿出模型
pred = model.predict(新的一行特征)                       # 用大脑算未来
```

**你的思维需要升级的地方**：训练和预测是两个阶段——

- 训练：教大脑（notebook，一次性的）
- 预测：用大脑（predict_system.py，每天重复的）

---

## 3. models/ 文件夹、.gitkeep、.pkl 都是什么？（你的问题 8）

### 3.1 三个概念

| 东西            | 是什么                  | 作用                 |
| --------------- | ----------------------- | -------------------- |
| `models/saved/` | 模型仓库文件夹          | 存放所有训练好的模型 |
| `*.pkl`         | 训练好的模型文件        | 预测程序读取它们     |
| `.gitkeep`      | 占位文件（placeholder） | 让 git 保留空文件夹  |

### 3.2 .gitkeep 是干嘛的？

**git 不跟踪"空文件夹"**——如果文件夹里什么都没，git 提交时会忽略它。`.gitkeep` 是一个约定俗成的**空占位文件**，让文件夹"看起来有东西"，git 就会把它带进仓库。

> 📌 它本身没任何作用，就是"占个位"。你可以把它理解成：在空房间里放一把椅子，这样 git 才会"记住"这个房间存在。

### 3.3 ⚠️ 我帮你发现的坑：V3 模型没有被 git 跟踪！

我看了你的 `.gitignore`，里面有这样几行：

```gitignore
*.pkl
!models/saved/xgboost_v*.pkl          # 放行原版 XGBoost
!models/saved/lightgbm_v2.pkl         # 放行 lightgbm_v2
!models/saved/lightgbm_v2_5.pkl       # 放行 lightgbm_v2_5
```

含义是：

- `*.pkl` = 默认所有 .pkl **都不提交**（模型文件很大，不该随便提交）
- `!xxx` = 但**这几个例外要提交**（放行）

**结果**：

- ✅ 会提交：`xgboost_v1/v1_5/v2/v2_5.pkl`、`lightgbm_v2.pkl`、`lightgbm_v2_5.pkl`（共6个）
- ❌ **不会提交**：你刚生成的 `xgboost_v3.pkl` 和 `lightgbm_v3.pkl`（不在放行名单！）

**后果**：你的 V3 模型只在本地，推送到 GitHub 后**远程仓库没有它们**，所以 GitHub Actions 每天自动跑的时候**找不到 V3 模型，只会跑那 6 个原版模型**。

> 如果你想 GitHub 上也跑 V3，需要在 `.gitignore` 里加两行：
>
> ```gitignore
> !models/saved/xgboost_v3.pkl
> !models/saved/lightgbm_v3.pkl
> ```

---

## 4. predict_system.py 是什么？（你的问题 7）

你说"我只知道训练模型"——对，`predict_system.py` 就是**训练之外的另一半：预测系统**。

### 4.1 它在哪里？

```
src/
├── config.py          ← 配置（路径、API、参数）
├── fetch_live.py      ← 从3个API拿最新数据
├── features.py        ← 造特征（和训练时一模一样！）
├── predict_system.py  ← ★主程序：预测的总指挥
└── utils.py           ← 工具（空）
```

### 4.2 它的核心职责（对照代码）

| 函数             | 做什么                                      |
| ---------------- | ------------------------------------------- |
| `main()`         | 总指挥：算时间、调数据、调模型、存结果      |
| `load_models()`  | ★**扫描并加载 `models/saved/` 里所有 .pkl** |
| `run_forecast()` | ★**递归预测未来7天**（672个15分钟点）       |
| `fill_actuals()` | 过去的时间点补上真实价格                    |
| `save_csv()`     | 每个模型存一个 CSV                          |

关键代码（你问的"模型在哪启动"）：

```python
def load_models():
    models = {}
    for pkl in sorted(config.SAVED_MODELS_DIR.glob('*.pkl')):  # 扫描所有.pkl
        meta = joblib.load(pkl)                                 # 逐个解冻
        models[pkl.stem] = meta                                 # 放进字典
    return models
```

> 📌 **所以"启动所有模型"的代码就在 `src/predict_system.py` 的 `load_models()` 里**——它自动扫描 `models/saved/` 文件夹，把里面每一个 `.pkl` 都加载进来。**不需要手动指定模型**，放进去多少它就加载多少。

递归预测的核心（`run_forecast`）：

```python
for i in range(FORECAST_HOURS * (60 // step_min)):   # 比如 672 次
    features = build_features(timestamp, price_buf, wx_buf)  # 造这一时刻的特征
    prediction = model.predict(row)[0]                        # 预测
    price_buf.add(timestamp, prediction)                      # 把预测当"历史"喂回
```

---

## 5. 工作流（.yml）是怎么"启动所有模型"的？（你的问题 6）

关键认知：**工作流本身不直接加载模型**。它只做一件事：**喊一句命令**。

```yaml
- name: Run prediction system
  run: python src/predict_system.py # ← 唯一的关键命令
```

然后 `predict_system.py` 内部自己会去加载模型（见第 4 节）。整个链条是：

```
daily_forecast.yml
   └─> 运行 python src/predict_system.py
          ├─> load_models()  扫 models/saved/*.pkl → 加载全部
          ├─> fetch_live     拿天气+电价
          ├─> run_forecast   每个模型预测7天
          └─> save_csv       每个模型存一个CSV
```

> 📌 就像你的本地命令 `python src/predict_system.py` 一样——GitHub 每天跑的就是**同一条命令**，只是它在云端自动跑。

---

## 6. 预测结果：几个 CSV？里面是什么？（你的问题 4、5）

### 6.1 几个 CSV？

**6 个 CSV（不是 7 个）**，每个模型一个：

```
predictions/
├── xgboost_v1_forecasts.csv       ← XGBoost V1 的预测
├── xgboost_v1_5_forecasts.csv     ← XGBoost V1.5
├── xgboost_v2_forecasts.csv       ← XGBoost V2
├── xgboost_v2_5_forecasts.csv     ← XGBoost V2.5（15分钟）
├── lightgbm_v2_forecasts.csv      ← LightGBM V2
└── lightgbm_v2_5_forecasts.csv    ← LightGBM V2.5
```

> 你可能会说"7个"——也许你把 7 天和文件数搞混了。目前提交的模型是 6 个 → 6 个 CSV。**每个 CSV 里都装着未来 7 天的预测**。

### 6.2 一个 CSV 里装的是"未来 7 天"，粒度和模型有关

| 模型类型                  | 时间粒度           | 7天多少行         |
| ------------------------- | ------------------ | ----------------- |
| 15 分钟模型（V1.5、V2.5） | 每 15 分钟一个预测 | 7×96 = **672 行** |
| 小时模型（V1、V2）        | 每小时一个预测     | 7×24 = **168 行** |

每一行格式：

```
run_date           = 哪一天做的预测
target_datetime    = 预测的是哪个时刻
predicted_price    = 预测电价（EUR/MWh）
actual_price       = 真实电价（过几天自动回填）
abs_error          = 误差 |真实-预测|
```

### 6.3 每天怎么积累？

（上一轮讲过，这里一句话复习）：**同一个文件每天"追加"新一批 `run_date`**，不是删旧建新。所以一个文件里跑了一个月，就有 30 批不同的 `run_date`。

---

## 7. 完整心智模型（从训练到自动预测）

```
① 训练（你会的）
   notebook → 数据/特征/模型 → 存成 model.pkl
                    ↓
② 提交（git）
   model.pkl + 预测代码 + 工作流 → push 到 GitHub
                    ↓
③ 部署（GitHub Actions 每天自动）
   .yml 定时器 → python src/predict_system.py
              → 加载 model.pkl → 拿数据 → 造特征 → 预测7天
              → 存 predictions/*.csv → git commit
                    ↓
④ 查看结果
   打开 predictions/*.csv 或画折线图
```

**对照你现在的情况（只做完了①）**：

- ✅ 你已经会训练 XGBoost V2 → 它会生成 `xgboost_v2.pkl`
- ❌ 你还没理解 ③④：预测程序读取 .pkl → 自动出 CSV → GitHub 每天自动跑

---

## 8. 你现在缺的"下一步"清单

| 步骤 | 做什么                                                              | 状态        |
| ---- | ------------------------------------------------------------------- | ----------- |
| 1    | 本地运行 `python src/predict_system.py` 看它加载模型、出 CSV        | 你做过了 ✅ |
| 2    | 理解 `src/predict_system.py` 的 `load_models()` 和 `run_forecast()` | 本文第4节   |
| 3    | 把训练好的新模型（如 V3）放行到 `.gitignore`，才能被 GitHub 跑      | ⚠️ 需要你改 |
| 4    | 推送代码到 GitHub，让 `daily_forecast.yml` 每天自动跑               | 待做        |
| 5    | 到 GitHub 网页 → Actions 标签页，看每天的运行日志                   | 待做        |

---

## 9. 术语表（中英对照）

| 中文      | English                             | 一句话解释                         |
| --------- | ----------------------------------- | ---------------------------------- |
| 训练      | Training                            | 让模型从历史数据学规律（一次性）   |
| 预测      | Prediction / Inference              | 用训练好的模型算未来（反复用）     |
| 部署      | Deployment                          | 把训练好的模型放进能自动运行的环境 |
| 模型文件  | Model artifact                      | 训练结果的"打包文件"（.pkl）       |
| 序列化    | Serialization                       | 把内存对象存成文件（joblib.dump）  |
| 反序列化  | Deserialization                     | 把文件读回内存（joblib.load）      |
| 配置文件  | Config file                         | 描述设定的文件（YAML/JSON）        |
| 工作流    | Workflow                            | GitHub 上定义自动步骤的 YAML 文件  |
| 定时任务  | Scheduled job                       | 到点自动执行的任务（cron）         |
| CI/CD     | Continuous Integration / Deployment | 自动化"集成+部署"的一套实践        |
| 占位文件  | Placeholder                         | 让 git 保留空文件夹的 .gitkeep     |
| git 忽略  | Gitignore                           | 告诉 git 哪些文件不提交            |
| 放行/例外 | Allowlist / Exception               | gitignore 里 `!` 开头的例外规则    |
| 递归预测  | Recursive forecasting               | 用上一步预测当下一步的历史         |

---

_本指南结合 `.github/workflows/daily_forecast.yml`、`src/predict_system.py`、`models/saved/`、`.gitignore` 写成。核心一句话：训练产生 .pkl 模型文件，predict_system.py 读取它来预测，.yml 每天自动喊 predict_system.py 跑，结果存成 6 个（每模型一个）包含未来 7 天的 CSV。_
