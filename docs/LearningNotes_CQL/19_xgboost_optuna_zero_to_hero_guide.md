# XGBoost (XGBoost，梯度提升树) 与 Optuna (自动超参数搜索) 零基础学习笔记

> 适合已经会 Python (Python) 基础语法、列表 (list)、字典 (dict)、循环 (loop)，但还不熟悉 Pandas (Pandas，数据分析库)、机器学习 (machine learning，机器学习) 和调参 (hyperparameter tuning，超参数调优) 的同学。
>
> 这篇笔记以本项目为例，讲清楚：XGBoost 是什么、Optuna 是什么、代码应该怎么写、每次加新特征要不要重新调参、以及你当前 XGBoost (XGBoost，梯度提升树) 项目里有哪些调参问题可以改进。

---

## 0. 先给结论

如果你只想先记住一句话：

**XGBoost (XGBoost，梯度提升树) 是一个很适合表格数据 (tabular data，表格数据) 的模型；Optuna (Optuna，自动超参数搜索) 是帮它自动找更好超参数 (hyperparameters，超参数) 的工具。你这个项目里，问题不只是“特征够不够”，还包括“模型有没有调好”。**

更具体一点：

- V1 / V1.5 主要是天气 (weather，天气) 基线 (baseline，基线) 模型。
- V2 / V2.5 加入了特征工程 (feature engineering，特征工程) 后，效果大幅提升。
- V2.5.2 / V2.5.3 说明了一个关键事实：**同样的特征，如果调参更认真，模型还能再进步。**
- V3 / V4 说明了另一件事：**有些新特征确实有用，但前提是模型先调好，而且这些特征在上线时也必须能拿到。**

---

## 1. XGBoost (XGBoost，梯度提升树) 到底是什么？

### 1.1 先用一句最简单的话理解

XGBoost (XGBoost，梯度提升树) 是一种树模型 (tree model，树模型)。它不是像线性回归 (linear regression，线性回归) 那样直接画一条直线，而是把很多棵小树 (decision tree，决策树) 一棵一棵加起来，让它们一起做预测 (prediction，预测)。

### 1.2 它为什么适合你的项目？

因为你的项目数据是这种形式：

- 每一行是一段时间 (timestamp，时间戳) 的数据。
- 每一列是一个特征 (feature，特征)，比如温度 (temperature，温度)、风速 (wind speed，风速)、滞后价格 (lagged price，滞后价格)、滚动均值 (rolling mean，滚动均值) 等。

这类数据非常适合树模型 (tree model，树模型)，因为树模型很擅长处理：

- 数值列 (numeric columns，数值列)
- 类别或标志列 (categorical / flag columns，类别/标志列)
- 非线性关系 (non-linear relationship，非线性关系)
- 特征之间复杂交互 (interaction，交互)

### 1.3 XGBoost (XGBoost，梯度提升树) 的工作方式

你可以把它想成“不断改错”的过程：

1. 第 1 棵树先做一个粗略预测。
2. 第 2 棵树专门修正第 1 棵树的错误。
3. 第 3 棵树再修正前面的错误。
4. 一直重复很多次，最后把所有树的结果加起来。

这种方法叫 boosting (提升)，意思是“让很多弱学习器 (weak learner，弱学习器) 组合成一个强模型 (strong model，强模型)”。

### 1.4 你项目里的 XGBoost (XGBoost，梯度提升树) 用在什么地方？

你的项目主要把 XGBoost 用在回归 (regression，回归) 上，也就是预测连续数值 (continuous value，连续值) 的电价 (electricity price，电价)。

相关 notebook (notebook，交互式笔记本) 包括：

- [xgboost_models/modelV1.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.ipynb)
- [xgboost_models/modelV1.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.5.ipynb)
- [xgboost_models/modelV2.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.ipynb)
- [xgboost_models/modelV2.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.ipynb)
- [xgboost_models/modelV2.5.2.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.2.ipynb)
- [xgboost_models/modelV2.5.3.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.3.ipynb)
- [xgboost_models/modelV3.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV3.ipynb)
- [xgboost_models/modelV3.1.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV3.1.ipynb)
- [xgboost_models/modelV4.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV4.ipynb)

---

## 2. 先看你这个项目里的 XGBoost (XGBoost，梯度提升树) 演化

### 2.1 V1：小时级 (hourly，按小时) 的天气基线

V1 是最基础的版本，特征只有天气 (weather，天气) 相关列，比如温度 (temperature，温度) 和风速 (wind speed，风速)。

它的意义不是“最好”，而是“起点”：

- 先确认最基础的预测路线能不能跑通。
- 先建立一个对照组 (control group，对照组)。

相关文件：

- [xgboost_models/modelV1.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.ipynb)
- [docs/LearningNotes_CQL/02_model_training_learning_guide.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/02_model_training_learning_guide.md)

### 2.2 V1.5：把时间分辨率 (time resolution，时间分辨率) 提高到 15 分钟

V1.5 没有增加很多新特征，只是把数据从小时级 (hourly，按小时) 改成 15 分钟级 (15-minute，15分钟)。

它的作用是：

- 验证“更细粒度 (granularity，粒度)”是否一定更好。
- 结果告诉你：只改变分辨率 (resolution，分辨率) 并不会神奇变强。

相关文件：

- [xgboost_models/modelV1.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV1.5.ipynb)
- [docs/LearningNotes_CQL/05_version_1_5_plan.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/05_version_1_5_plan.md)

### 2.3 V2：特征工程 (feature engineering，特征工程) 是真正的突破

V2 加入了很多“从历史和时间里提炼出来的列”，比如：

- 滞后特征 (lag features，滞后特征)
- 滚动均值 (rolling mean，滚动均值)
- 滚动标准差 (rolling standard deviation，滚动标准差)
- 时间特征 (time features，时间特征)
- 节假日特征 (holiday features，节假日特征)

这一步的意义非常大，因为电价 (electricity price，电价) 本身就是一个强时间序列 (time series，时间序列) 问题。

相关文件：

- [xgboost_models/modelV2.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.ipynb)
- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py)
- [data/convertData/V2.5_15min_features.csv](/e:/Github/nordpool_electricity_price_prediction/data/convertData/V2.5_15min_features.csv)

### 2.4 V2.5：15 分钟 + 特征工程 = 早期最佳模型

V2.5 把 V2 的思路放到 15 分钟数据上，效果进一步提升。

它说明：

- 更细时间粒度 + 更好的特征，通常比“只加更多数据行”更有价值。
- 15 分钟模型里，短期波动 (short-term fluctuation，短期波动) 能被更好地捕捉。

相关文件：

- [xgboost_models/modelV2.5.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.ipynb)
- [data/convertData/V2.5_15min_features.csv](/e:/Github/nordpool_electricity_price_prediction/data/convertData/V2.5_15min_features.csv)

### 2.5 V2.5.2：公平比较 XGBoost (XGBoost，梯度提升树) 和 LightGBM (LightGBM，轻量梯度提升树)

V2.5.2 的核心不是“做更强的新模型”，而是“做公平比较 (fair comparison，公平比较)”。

它回答的问题是：

> 如果 XGBoost 和 LightGBM 都认真调参 (hyperparameter tuning，超参数调优)，谁更强？

这个版本非常重要，因为它告诉你：

- 以前看起来 LightGBM 更强，部分原因是它调得更认真。
- XGBoost 不是不能好，只是默认参数 (default parameters，默认参数) 太普通。

### 2.6 V2.5.3：XGBoost 的 Optuna (Optuna，自动超参数搜索) 调参版本

V2.5.3 是你项目里非常关键的一版，因为它说明：

- XGBoost (XGBoost，梯度提升树) 经过更系统的调参后，能明显进步。
- 它不再只是“默认参数跑一下”，而是“认真找最优参数组合 (best parameter combination，最优参数组合)”。

相关文件：

- [xgboost_models/modelV2.5.3.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV2.5.3.ipynb)
- [docs/LearningNotes_CQL/12_why_more_features_did_not_help_and_xgboost_optuna_guide.md](/e:/Github/nordpool_electricity_price_prediction/docs/LearningNotes_CQL/12_why_more_features_did_not_help_and_xgboost_optuna_guide.md)

### 2.7 V3 / V3.1 / V4：加入电网 (grid，电网) 和核电 (nuclear power，核电)

这几版说明一个非常重要的原则：

**不是所有“看上去很强”的特征，都一定能直接上线 (deployment，部署)。**

原因是：

- 有些特征在 notebook (notebook，交互式笔记本) 里能算出来。
- 但线上预测 (live prediction，实时预测) 时，未来时刻的真实值还没发生，拿不到。

所以你在 V3 / V4 里看到的一个核心主题就是：

- 训练时 (training，训练) 有用，不代表上线时 (serve，服务/部署) 也有用。
- 这就是 train/serve gap (训练/部署差距，训练和部署差距)。

相关文件：

- [xgboost_models/modelV3.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV3.ipynb)
- [xgboost_models/modelV3.1.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV3.1.ipynb)
- [xgboost_models/modelV4.ipynb](/e:/Github/nordpool_electricity_price_prediction/xgboost_models/modelV4.ipynb)
- [src/features.py](/e:/Github/nordpool_electricity_price_prediction/src/features.py)
- [src/predict_system.py](/e:/Github/nordpool_electricity_price_prediction/src/predict_system.py)

---

## 3. Optuna (Optuna，自动超参数搜索) 到底是什么？

### 3.1 先把“超参数 (hyperparameter，超参数)”和“参数 (parameter，参数)”分清楚

这两个词很容易混。

- 参数 (parameter，参数)：模型训练后自己学出来的东西，比如树内部结构相关结果。
- 超参数 (hyperparameter，超参数)：你训练前手动指定的东西，比如学习率 (learning rate，学习率)、树深度 (max depth，最大深度)、树的数量 (number of trees，树数量) 等。

你可以理解成：

- 参数 (parameter，参数) 是模型自己学的。
- 超参数 (hyperparameter，超参数) 是你帮它选的。

### 3.2 Optuna (Optuna，自动超参数搜索) 是做什么的？

Optuna 是一个自动找超参数 (automatic hyperparameter tuning，自动超参数调优) 的工具。

它的工作方式可以理解成：

1. 给它一堆候选范围 (search space，搜索空间)。
2. 它自动试很多组参数 (trial，试验)。
3. 每试一组，就训练一次模型 (model training，模型训练)。
4. 然后算一个分数 (score，分数)，比如 MAE (Mean Absolute Error，平均绝对误差)。
5. 它记住哪组更好，再继续往更好的方向试。

### 3.3 Optuna (Optuna，自动超参数搜索) 为什么比“手动乱试”更好？

因为手动调参常常会有这几个问题：

- 你不知道该先改哪个参数。
- 你很容易只试几组就放弃。
- 你可能在一个偶然很好的验证区间 (validation window，验证窗口) 上“看起来很强”，但其实是过拟合 (overfitting，过拟合)。

Optuna 的优点是：

- 更系统。
- 更省时间。
- 更容易重复 (reproducible，可重复)。

### 3.4 Optuna (Optuna，自动超参数搜索) 的核心概念

你只需要先记住 5 个词：

| 英文词                        | 中文解释                      | 你可以怎么理解         |
| ----------------------------- | ----------------------------- | ---------------------- |
| search space (搜索空间)       | 每个参数允许的范围            | “这个参数可以从哪里选” |
| trial (试验)                  | 一次参数尝试                  | “先试一组看看”         |
| objective function (目标函数) | 打分函数                      | “这组参数到底好不好”   |
| study (调参任务)              | 整个搜索过程                  | “总的调参项目”         |
| sampler (采样器)              | Optuna 决定下一组怎么试的方式 | “下一次该试哪组”       |

### 3.5 Optuna (Optuna，自动超参数搜索) 是不是有固定参数？

答案是：**有一部分是固定的，有一部分是要搜索的。**

固定的通常是：

- 任务类型 (task type，任务类型)：回归 (regression，回归) 还是分类 (classification，分类)
- 评价指标 (metric，评价指标)：比如 MAE (Mean Absolute Error，平均绝对误差)
- 训练轮数上限 (n_estimators，树数量)
- 随机种子 (random seed，随机种子)

要搜索的通常是：

- 学习率 (learning rate，学习率)
- 树深度 (max depth，最大深度)
- 子采样比例 (subsample，样本子采样)
- 列采样比例 (colsample_bytree，按树列采样)
- 正则化系数 (regularization，正则化)
- 最小叶子样本权重 (min child weight，最小子节点权重)

### 3.6 Optuna (Optuna，自动超参数搜索) 最重要的原则

不是“你把所有参数都交给它乱搜”，而是：

- 你先决定哪些参数值得搜索。
- 你再给每个参数一个合理范围。
- 你让 Optuna 在这个范围内自动找更优组合。

换句话说：

**Optuna 不是魔法，它是一个更聪明的搜索工具 (search tool，搜索工具)。**

---

## 4. 在你的项目里，Optuna (Optuna，自动超参数搜索) 应该怎么写？

### 4.1 最基本的写法框架

你的 notebook (notebook，交互式笔记本) 里可以按照这个结构写：

```python
import optuna
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        'objective': 'reg:absoluteerror',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'random_state': 42,
    }

    fold_maes = []
    for tr_idx, va_idx in tscv.split(X_train):
        model = XGBRegressor(**params, verbosity=0)
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[va_idx])
        fold_maes.append(mean_absolute_error(y_train.iloc[va_idx], pred))

    return float(np.mean(fold_maes))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

best_params = study.best_params
```

### 4.2 这段代码每一部分到底在干什么？

#### 1. `TimeSeriesSplit (时序交叉验证)`

因为你的数据是时间序列 (time series，时间序列)，不能随机切分 (random split，随机切分)。

`TimeSeriesSplit` 会按时间顺序切分训练集 (training set，训练集) 和验证集 (validation set，验证集)，这样更真实。

#### 2. `objective(trial)`

这是目标函数 (objective function，目标函数)。

每次 Optuna 给你一组参数，你都要：

- 用这组参数训练模型。
- 在验证集上算 MAE (Mean Absolute Error，平均绝对误差)。
- 返回这个 MAE。

#### 3. `trial.suggest_float(...)` / `trial.suggest_int(...)`

这就是告诉 Optuna：

- 这个参数是浮点数 (float，浮点数) 还是整数 (integer，整数)
- 它的范围是多少

例如：

- `learning_rate` 用 `suggest_float`，因为它是小数。
- `max_depth` 用 `suggest_int`，因为它是整数。

#### 4. `direction='minimize'`

因为你要最小化 (minimize，最小化) MAE (Mean Absolute Error，平均绝对误差)。

也就是说：MAE 越小越好。

### 4.3 训练完以后怎么拿到最优参数？

```python
best_params = study.best_params
best_score = study.best_value
```

- `best_params` 是最好的参数组合。
- `best_value` 是最好的分数 (score，分数)，这里就是最小 MAE (Mean Absolute Error，平均绝对误差)。

### 4.4 最后怎么训练最终模型？

```python
model = XGBRegressor(
    objective='reg:absoluteerror',
    n_estimators=2000,
    random_state=42,
    verbosity=0,
    **best_params
)
model.fit(X_train, y_train)
```

这一步叫最终训练 (final training，最终训练)：

- 前面是“找参数”。
- 这里是“用最好的参数在全部训练集上重新训练一次”。

---

## 5. 你这个项目里，Optuna (Optuna，自动超参数搜索) 应该怎么设置？

### 5.1 哪些参数适合搜？

从你现在的 notebook 看，比较适合搜的有：

- `learning_rate` (学习率)
- `max_depth` (最大深度)
- `min_child_weight` (最小子节点权重)
- `subsample` (样本子采样)
- `colsample_bytree` (按树列采样)
- `reg_lambda` (L2 正则化)
- `reg_alpha` (L1 正则化)

这些就是你 V2.5.3 里面已经在搜的内容。

### 5.2 哪些参数通常先固定？

在你这个项目里，通常建议先固定这些：

- `objective='reg:absoluteerror'`，因为你最终关心 MAE (Mean Absolute Error，平均绝对误差)
- `n_estimators=2000`，因为树少了容易欠拟合 (underfitting，欠拟合)
- `random_state=42`，让结果可重复 (reproducible，可重复)
- `verbosity=0`，减少输出噪音 (noise，噪音)

### 5.3 `n_estimators` (树数量) 要不要放进 Optuna 搜索？

在你这个项目里，一般不建议一开始就把 `n_estimators` 放进去搜。

原因：

- 你已经知道 100 棵树太少。
- 先把 `n_estimators` 提到一个较大的值 (比如 2000)，再用其他参数去控制模型复杂度 (model complexity，模型复杂度) 更稳。

### 5.4 搜索范围怎么定？

搜索范围 (search range，搜索范围) 不要乱定。

一个实用原则是：

- 小学习率 (learning rate，学习率) 通常配更多树 (trees，树)。
- 深度 (depth，深度) 不要一开始就太大。
- 正则化 (regularization，正则化) 范围要给足。

你现在的范围大体是合理的，但有两个方向可以继续改进：

1. 让搜索更稳定。
2. 让搜索更符合时间序列 (time series，时间序列) 的真实验证方式。

---

## 6. 以你的项目为例，为什么“每加一个新特征都必须重新调参”并不是绝对的？

### 6.1 正确答案：不是“必须”，但“通常应该重新检查”

你问的是一个很好的问题。

答案不是简单的“是”或“不是”，而是：

**不一定每次都要从零重新跑完整 Optuna (Optuna，自动超参数搜索)，但只要特征集合 (feature set，特征集合) 有明显变化，就应该重新验证，最好重新调参或至少做一轮简化搜索。**

### 6.2 为什么？

因为新特征会改变数据分布 (data distribution，数据分布) 和模型学习方式 (learning pattern，学习模式)。

比如：

- 只加一个很弱的新特征，影响可能很小。
- 加一大批新特征，比如网格 (grid，电网) 和核电 (nuclear power，核电)，影响就可能很大。

### 6.3 你这个项目里最典型的例子

#### 例子 1：V2.5 → V2.5.2 / V2.5.3

这里特征没变，但模型调参变认真了，所以效果更好。

这说明：

- 特征没变，模型也可能更强。
- 说明之前不是“特征不行”，而是“模型没调好”。

#### 例子 2：V2.5 → V3 / V4

这里特征变了，所以不能直接假设原来的超参数 (hyperparameters，超参数) 仍然最好。

原因是：

- 新特征会影响最佳树深度 (best depth，最佳深度)
- 也会影响最佳学习率 (best learning rate，最佳学习率)
- 还会影响正则化 (regularization，正则化) 的强弱

所以在 V3 / V4 上重新调参是合理的。

### 6.4 实用建议

如果你只是：

- 改了一两个弱特征

你可以先：

1. 用旧的最优参数跑一次。
2. 再做少量 trial (试验) 的 Optuna (Optuna，自动超参数搜索) 检查。

如果你是：

- 明显增加了特征族 (feature family，特征族)，例如 grid (电网) / nuclear (核电)

那就建议：

1. 重新调参。
2. 或者至少重新跑一轮较小的搜索 (small search，小规模搜索)。

---

## 7. 你这个项目里的 XGBoost (XGBoost，梯度提升树) 调参问题，哪里可以改？

下面我直接指出项目里比较值得改进的地方。

### 7.1 问题 1：早期模型太靠近默认参数 (default parameters，默认参数)

V1、V1.5、V2、V2.5 早期版本里，XGBoost 很大程度上是“先跑起来”，不是“认真调优”。

这个问题的后果是：

- 模型可能欠拟合 (underfitting，欠拟合)
- 新特征的价值被低估

#### 怎么改：

- 给 V2.5 以后更系统的 Optuna (Optuna，自动超参数搜索)
- 尤其是对 XGBoost (XGBoost，梯度提升树) 做和 LightGBM (LightGBM，轻量梯度提升树) 一样认真甚至更认真的搜索

### 7.2 问题 2：V2.5.2 只跑了 10 个 trial (试验)，偏少

V2.5.2 已经证明 XGBoost 可以通过 Optuna (Optuna，自动超参数搜索) 变好，但 10 个 trial 往往偏少。

这会导致：

- 搜索不够充分 (insufficient search，不充分搜索)
- 可能还没找到真正好的参数组合

#### 怎么改：

- 把 trial 数提高到 30 或更多
- 或者先用 10 个 trial 粗搜，再用更小范围做二次精搜 (refined search，精细搜索)

### 7.3 问题 3：训练目标和评估指标 (evaluation metric，评估指标) 最好统一

你在项目里看到两种常见写法：

- `reg:squarederror` (MSE 风格，均方误差风格)
- `reg:absoluteerror` (MAE 风格，平均绝对误差风格)

如果你最终看的是 MAE (Mean Absolute Error，平均绝对误差)，那么训练时也最好直接用 MAE 风格的目标函数 (objective function，目标函数)。

#### 怎么改：

- 优先使用 `objective='reg:absoluteerror'`
- 这样训练目标和评估目标一致 (aligned，对齐)

### 7.4 问题 4：没有明显看到 early stopping (早停)

从 notebook 的代码结构看，很多训练是直接给 `n_estimators=2000` 然后全量训练，没有明显使用 early stopping (early stopping，早停)。

这意味着：

- 训练可能更慢
- 你可能不知道 2000 棵树到底是不是最合适

#### 怎么改：

- 在单次训练里加入验证集 (validation set，验证集)
- 使用 `early_stopping_rounds` 或等价机制

不过要注意：

- 如果你在 Optuna 交叉验证 (cross-validation，交叉验证) 里加 early stopping，代码会更复杂。
- 但是从工程 (engineering，工程) 角度看，这是值得的。

### 7.5 问题 5：`TimeSeriesSplit` (时序交叉验证) 比单一验证窗口更稳，但要注意每折长度

你已经在 V2.5.3 / V4 里使用了 `TimeSeriesSplit(5)`，这是对的。

但要注意：

- 如果每折验证区间太短，MAE (Mean Absolute Error，平均绝对误差) 会比较不稳定。
- 如果数据有明显季节性 (seasonality，季节性)，折数 (number of splits，切分数量) 和每折长度要平衡。

#### 怎么改：

- 可以试 `n_splits=5` 和 `n_splits=3` 做对比
- 看看哪种更稳定

### 7.6 问题 6：每次加新特征都应该“重新做公平对照实验”

你已经做过很好的控制实验 (controlled experiment，受控实验)，比如：

- V2.5.1：高波动概率 (high_volatility_prob，high_volatility_prob) 是否有用
- V3.1：网格特征 (grid features，电网特征) 在调参后是否有用
- V4：核电 (nuclear power，核电) 是否继续带来提升

这说明你的方向是对的。

#### 怎么改：

- 以后新增特征，不要只看一次结果。
- 要和 baseline (基线) 做严格比较。
- 最好保持相同的数据切分 (split，切分)、相同评价指标 (metric，评价指标)、相同训练流程。

### 7.7 问题 7：最后保存的模型最好附带清晰的元信息 (metadata，元数据)

你现在的模型包 (model bundle，模型包) 已经包含：

- `model`
- `feature_cols`
- `step_min`

这很好。

#### 还能再改进什么？

- 给每个模型保存训练日期 (training date，训练日期)
- 保存特征版本 (feature version，特征版本)
- 保存主要超参数 (main hyperparameters，主要超参数)

这样以后你在展示 (presentation，展示) 时会更清楚。

---

## 8. 你现在应该怎么理解“有没有调参方面的问题”？

我直接给你一个清晰判断：

### 8.1 有问题的地方

有，主要是这三类：

1. 早期 XGBoost (XGBoost，梯度提升树) 偏默认 (default，默认) 了。
2. V2.5.2 的调参次数 (trials，试验次数) 偏少。
3. 新特征 (new features，新特征) 和调参 (tuning，调优) 的顺序有时需要更严格地控制。

### 8.2 但这些问题不是“错误”，而是“成长路径”

这点很重要。

你的项目不是失败，而是一个很好的学习过程：

- 先有 baseline (基线)
- 再做 feature engineering (特征工程)
- 再做 Optuna (Optuna，自动超参数搜索)
- 再做 fair comparison (公平比较)
- 再做 train/serve gap (训练/部署差距) 检查

这条路线是非常专业的。

---

## 9. 你以后可以直接照抄的回答模板

### 问题：Optuna (Optuna，自动超参数搜索) 是什么？

回答模板：

Optuna 是一个自动超参数搜索 (automatic hyperparameter tuning，自动超参数调优) 工具。它会在我设定的搜索空间 (search space，搜索空间) 里自动尝试很多组参数 (trial，试验)，并根据目标函数 (objective function，目标函数) 找到最优参数组合 (best parameter combination，最优参数组合)。

### 问题：每加一个新特征都要重新调参吗？

回答模板：

不一定每次都要从零开始完整调参，但只要特征集合 (feature set，特征集合) 有明显变化，就应该重新验证，最好重新调参或者至少做一轮简化搜索。因为新特征会改变数据分布 (data distribution，数据分布) 和模型的最佳超参数 (best hyperparameters，最佳超参数)。

### 问题：你项目里调参最需要改进的地方是什么？

回答模板：

我觉得最需要改进的是：早期模型太接近默认参数，V2.5.2 的 trial 数偏少，以及以后新增特征时要更严格做受控实验 (controlled experiment，受控实验) 和时序交叉验证 (TimeSeriesSplit，时序交叉验证)。

---

## 10. 最后给你一个学习顺序

如果你现在还在从基础走向进阶，我建议你按这个顺序学：

1. 先理解 Pandas (Pandas，数据分析库) 里的 DataFrame (数据框) 是什么。
2. 再理解时间序列 (time series，时间序列) 为什么不能乱打乱 (shuffle，打乱)。
3. 再理解特征工程 (feature engineering，特征工程) 为什么重要。
4. 再理解 XGBoost (XGBoost，梯度提升树) 是怎么“用树修正错误”的。
5. 再理解 Optuna (Optuna，自动超参数搜索) 是怎么帮你找更好的超参数 (hyperparameters，超参数)。
6. 最后把训练 (training，训练) 和部署 (deployment，部署) 连成一个完整系统。

---

## 11. 最重要的一句话

**在你的项目里，XGBoost (XGBoost，梯度提升树) 的提升不是只靠“更多特征”，而是靠“更好的特征工程 (feature engineering，特征工程) + 更认真的 Optuna (Optuna，自动超参数搜索) 调参 + 更严格的时间序列验证 (time-series validation，时间序列验证)”。**

这就是你最应该向老师讲清楚的核心逻辑。
