# 12. Why "More Features" Didn't Help — and How Optuna Fixes the Real Problem

> Date: 2026-08-13
> Related notebooks: `xgboost_models/modelV2.5.ipynb`, `modelV2.5.1.ipynb`, `modelV2.5.2.ipynb`, `modelV3.ipynb`, `lightgbm_models/modelV2.5.ipynb`
> Language note: 技术术语已配中文注释，方便对照学习。

---

## 0. TL;DR — 一句话结论

加了更多特征（跨境网格、高波动概率）并没有让模型变好，因为**瓶颈从来不是"特征数量"，而是"模型配置"**。

- 默认 XGBoost（只有 100 棵树、MSE 损失）是**欠训练**的（undertrained）。
- Optuna 调参后的 XGBoost（2000 棵树 + MAE 损失，**特征没变多**）反而明显更好。
- **所以：完全可以给 XGBoost 加 Optuna**，而且项目里已经有现成证据（V2.5.2）。
- 正确顺序：**先调好模型，再重新检验新特征**——在弱模型上测特征，结论不可靠。

---

## 1. 证据 — 把所有模型放在一起对比

同一份数据（`V2.5_15min_features.csv` / `V3_15min_features.csv`）、同一个 80/20 时序切分。

| Model             | Features             | Loss | Tuning                | MAE             | RMSE   | R²     |
| ----------------- | -------------------- | ---- | --------------------- | --------------- | ------ | ------ |
| XGBoost V2.5      | 49                   | MSE  | 默认（100 棵树）      | 2.82            | 8.22   | 0.972  |
| **XGBoost V3**    | **62（+13 网格）**   | MSE  | 默认                  | **2.847**       | 8.368  | 0.971  |
| XGBoost V2.5.1    | 49 → 50（+风险特征） | MAE  | Optuna 参数           | 2.7555 → 2.7957 | —      | 变差   |
| LightGBM V2.5.1   | 49 → 50（+风险特征） | MAE  | Optuna 参数           | 2.7165 → 2.7426 | —      | 变差   |
| XGBoost V2.5.2    | 49                   | MAE  | Optuna（10 次）       | 2.7652          | 8.2342 | 0.9717 |
| LightGBM V2.5.2   | 49                   | MAE  | Optuna（10 次）       | 2.7167          | 8.0958 | 0.9727 |
| **LightGBM V2.5** | 49                   | MAE  | Optuna（30 次 TS-CV） | **2.6406**      | 7.9216 | 0.9738 |

**"铁证"（smoking gun）**：

1. V3 **加了 13 个网格特征** → 变差（MAE 2.847 vs 2.82）。
2. V2.5.2 **一个特征都没加，只做了 Optuna 调参** → 变好（MAE 2.7652 vs 2.82）。
3. 结论：**调参带来的提升，比加 13 个特征带来的提升还大。**

---

## 2. 为什么"加特征"没起作用？

### 2.1 真正的瓶颈是模型配置，不是特征数量（最核心）

XGBoost V2.5 / V3 用的是几乎"默认"的超参数：

```python
XGBRegressor(objective='reg:squarederror', learning_rate=0.1,
             n_estimators=100, max_depth=6)   # 只有 100 棵树！
```

- `n_estimators=100` + `learning_rate=0.1` 在 84,000 行训练数据上**明显欠训练**（树太少，学不透）。
- 对比：V2.5.2 的 XGBoost 用 2000 棵树 + MAE 损失，**特征数更少（49）却拿到了 2.7652**。
- **证明**：默认 XGBoost 没榨干 49 个特征的潜力，你再加特征当然不会更好——它连旧特征的信号都没学完。

### 2.2 电价是"自回归"的——信号早就被捕获了

49 个特征里已经有 `price_lag_*`（15 分钟到 7 天）和 `price_rolling_*`（1h/6h/24h/7d 的均值/标准差）。

- 电价最可预测的部分（昨天、上周的价格 → 今天）**已经被滞后特征吃掉了**。
- R² 已经到 0.972，剩下 2.8% 方差主要是突发尖峰（市场冲击），**本来就是近乎不可预测的**。
- 新特征只能在"残差"里找肉，而残差主要是噪声 → 边际收益 ≈ 0。

### 2.3 新特征本身"先天就弱"（weak by construction）

**网格特征（fi\_\*）——只能做滞后**：

- 为了不泄漏，V3 只用了**前一天/上周**的网格流（lag_96 / lag_672）。这是对的。
- 但"昨天的跨境流量"对"今天价格"的信息，**大部分已经藏在今天的电价历史里了**（因为电价本身会反映供给）。
- 而且 15 分钟粒度的跨境流量**噪声很大**（波动剧烈）。

**高波动概率（high_volatility_prob）——来自弱分类器**：

- 分类器对高波动时刻的召回率只有 **0.24**（漏掉 76% 的尖峰）。
- 直方图验证：稳定期 0.113 vs 高波动期 0.341，虽然有区分，但**分布大量重叠**。
- 等于给模型喂了"**弱信号 + 噪声**"。

### 2.4 树模型不会"自动奖励"更多特征

XGBoost / LightGBM 是**贪心的、按增益（gain）分裂**的：

- 加一个有用但**与旧特征高度相关**的特征 → 重要性被分散，预测不提升。
- 加一个**噪声**特征 → 可能让某些分裂变得次优，反而略微变差。
- 在**欠训练**或**欠正则化**时，这个负面效应更明显。

### 2.5 损失函数（loss function）不匹配

- V2.5 / V3 用的是 `reg:squarederror`（MSE 损失）。
- 电价有尖峰，MSE 会**过度惩罚尖峰**，把模型往"保守"方向拉。
- LightGBM（和 V2.5.2 的 XGBoost）用的是 `reg:absoluteerror`（MAE 损失）——**直接优化我们要评估的指标**。
- 这是我们评估用 MAE，训练却用 MSE 造成的"免费损失"。

---

## 3. Optuna 到底是什么？（新手友好版）

Optuna 是一个**自动超参数调参库**（automatic hyperparameter tuning library）。核心概念：

| 概念             | 中文     | 作用                                                                                                       |
| ---------------- | -------- | ---------------------------------------------------------------------------------------------------------- |
| **search space** | 搜索空间 | 每个超参数允许的取值范围，用 `trial.suggest_float('learning_rate', 0.005, 0.1, log=True)` 定义             |
| **trial**        | 一次尝试 | 从搜索空间里取一组超参数，训练+评估，得到一个分数                                                          |
| **objective**    | 目标函数 | 一个"打分函数"，返回要最小化的指标（这里 = 5 折 CV 的平均 MAE）                                            |
| **study**        | 调参过程 | 跑很多 trial，记录并追踪最好的一组                                                                         |
| **TPE sampler**  | 采样器   | Optuna 默认的**贝叶斯优化**——先随机试几组，然后"学习"哪些参数组合好，聪明地猜下一组（比 grid/random 高效） |

**它怎么工作（一句话）**：像"猜 + 学习"——试几组 → 记住哪些好 → 往好的方向继续猜 → 循环 30 次。

**为什么必须配 TimeSeriesSplit（时序交叉验证）？**

- 这是项目里 LightGBM V2.5 自己踩过的坑（Run 3 → Run 4）：
  - Run 3 只用**一个固定的验证窗口**调参 → 验证 MAE 1.55 超低，测试 MAE 却 2.71（过拟合到那段"平静期"）。
  - Run 4 改成 `TimeSeriesSplit(5)`（滚动扩展窗口，平均 5 折 MAE）→ 验证 2.84 接近真实测试 2.64，**可信**。
- **教训**：时序数据调参必须用 TS-CV，不能用一个连续验证窗口，否则会"骗过自己"。

---

## 4. 可以给 XGBoost 加 Optuna 吗？可以——而且证据已经存在

**答案：完全可以，而且你的 `modelV2.5.2.ipynb` 里已经做过了。**

V2.5.2 那个 notebook 本身就是"XGBoost + Optuna"：

```python
# 两个模型共享同一个搜索空间
common = {
    'n_estimators': 2000,
    'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
    'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'reg_lambda':       trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
    'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
    'random_state': 42,
}
if model_type == 'xgb':
    params = {**common,
              'objective': 'reg:absoluteerror',            # ← MAE 损失
              'max_depth':        trial.suggest_int('max_depth', 4, 12),
              'min_child_weight': trial.suggest_int('min_child_weight', 1, 50)}
    model = XGBRegressor(**params, verbosity=0)
```

它已经把 XGBoost 的 MAE 从 2.82 拉到 2.7652。**只是当时只跑了 10 次 trial**；而 LightGBM V2.5 跑了 30 次 TS-CV，达到了 2.6406。

> 所以你现在最该做的，不是再加特征，而是：**把 XGBoost 也像 LightGBM V2.5 那样认真调一遍（MAE 损失 + TimeSeriesSplit(5) + 30 次 trial）**。

---

## 5. 推荐做法：先调模型，再重测特征

### 第一步 — 给 XGBoost V2.5 做 Optuna 调参（代码骨架）

```python
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
tscv = TimeSeriesSplit(n_splits=5)   # 时序交叉验证，不要用单一窗口

def objective(trial):
    params = {
        'objective': 'reg:absoluteerror',            # ← 训练目标 = 评估指标 MAE
        'n_estimators': 2000,                        # 足够多，让低学习率收敛
        'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth':        trial.suggest_int('max_depth', 4, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda':       trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
        'random_state': 42,
    }
    fold_maes = []
    for tr_idx, va_idx in tscv.split(X_train):
        m = XGBRegressor(**params, verbosity=0)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_maes.append(mean_absolute_error(
            y_train.iloc[va_idx], m.predict(X_train.iloc[va_idx])))
    return float(np.mean(fold_maes))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
print('Best CV MAE:', study.best_value, study.best_params)

# 用最优参数在整个 80% 训练集上重训，得到最终模型
best = study.best_params
model = XGBRegressor(objective='reg:absoluteerror', n_estimators=2000,
                     random_state=42, verbosity=0, **best)
model.fit(X_train, y_train)
```

### 第二步 — 调参**之后**再重测网格 / 核电 / 风险特征

- 因为"在弱模型上测特征"不可靠：默认 XGBoost 欠训练，负结果分不清是"特征没用"还是"模型没能力用"。
- 用**调好后的 XGBoost** 做 V2.5 vs V3（+网格） vs V4（+核电）的受控对比，结论才可信。
- 如果调参后网格/核电仍然没用 → 那才是真正的"这组特征没用"（也值得记录）。

### 第三步 — 顺便修正损失函数

把评估和训练统一到 **MAE**（`reg:absoluteerror`），和 LightGBM 站在同一起跑线，公平对比。

---

## 6. 预期结果 / 验证

- 参考基准：LightGBM V2.5 = **MAE 2.6406**。
- 期望：Optuna 调参后的 XGBoost V2.5 应达到 **MAE ≈ 2.7 或更好**（V2.5.2 已到 2.7652，加大 trial 数应再降）。
- 验证指标：同一切分下的 MAE / RMSE / R²，与 LightGBM V2.5 并列对比。

---

## 7. 关键收获（Key Takeaways）

1. **特征多 ≠ 模型好**。模型的"消化能力"（超参数、损失函数、正则化）才是瓶颈。
2. **先调参，再测特征**。在弱基线上测新特征，结论不可信。
3. **新特征要"强"才有价值**：前瞻信息（如未来停电计划）比滞后信息（昨天的网格流）更有用；弱分类器产出的概率特征 ≈ 噪声。
4. **时序调参必须用 TimeSeriesSplit**，单一验证窗口会过拟合到某段平静期（项目 Run 3 教训）。
5. **训练损失 = 评估指标**（MAE ↔ MAE），不要用 MSE 训练却用 MAE 评估。
6. Optuna 可以（也应该）用在 XGBoost 上——V2.5.2 就是现成例子，只是当时 trial 太少。
