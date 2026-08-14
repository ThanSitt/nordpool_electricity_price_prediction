# Record — Grid Transmission Integration into `src/` (REVERTED)

> Date: 2026-08-14
> Status: **REVERTED** — grid features are NOT integrated into the live pipeline.
> This document records exactly what was changed and how to restore it later.

---

## 1. 这是什么 / What this is

在 2026-08-13 尝试把**跨境电网传输特征（grid / `fi_*`）**接入每日自动预测
（`src/` 实时管线），希望让 V3.1 模型进入 `python src/predict_system.py` 的每日流程。

**最终决定：回退（revert）**。原因是 live-safe 实验显示线上可行的网格滞后特征
反而让模型变差（见第 5 节），所以网格**不接入每日预测**。

---

## 2. 当时改了哪些文件（6 个）

| 文件                       | 改动内容                                                                                                                           |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `src/config.py`            | 新增 `FINGRID_API_URL`、`FINGRID_API_KEY`（读环境变量）、`FINGRID_GRID_DATASETS`（55/57/60/61）、`GRID_FLOW_COLS`、`GRID_LAG_COLS` |
| `src/features.py`          | 新增 `GridBuffer` 类 + `build_features()` 增加可选参数 `grid`（当期流对未来 = NaN，滞后特征 lag_96/lag_672 从历史可算）            |
| `src/fetch_live.py`        | 新增 `_fetch_fingrid_dataset()` + `fetch_grid_transmission()`（Fingrid API 分页 + 429 重试，派生 7 列流量）                        |
| `src/predict_system.py`    | `main()` 抓网格 → 构造 `GridBuffer` → 传入 `run_forecast`；无网格数据时安全跳过需要 `fi_*` 特征的模型                              |
| `tests/test_features.py`   | 新增 `GridBufferTests`（当期流、未来 NaN、lag_steps、build_features 网格分支）                                                     |
| `tests/test_fetch_live.py` | 新增 `FetchGridTests`（派生列正确性、无 API key 报错）                                                                             |

> 完整 diff 保存在旁边这个 patch 文件里：`grid_src_integration.patch`（19.8 KB）。

---

## 3. 为什么当时要这么做 / Motivation

- V3.1 实验（调好参的 XGBoost）显示网格特征让测试 MAE 从 2.7236 → 2.7152（变好）。
- 于是想把网格特征接入 `src/`，让 V3.1 能进每日自动预测。

---

## 4. 测试结果（当时全绿）

`python -m unittest discover -s tests -v` → **14 个测试全部通过（OK, exit 0）**，
其中 6 个是新加的网格测试。

---

## 5. 为什么最终回退 / Why reverted（关键实验结论）

为了线上可用，重训了 **live-safe V3.1**（只用 6 个网格滞后特征，去掉预测时拿不到的当期流）：

| 模型                              | 特征数 | MAE        | 结论                                  |
| --------------------------------- | ------ | ---------- | ------------------------------------- |
| 完整 V3.1（含 7 当期流 + 6 滞后） | 62     | 2.7152     | 变好（-0.0084），但当期流预测时拿不到 |
| **live-safe V3.1（仅滞后）**      | 55     | **2.7416** | **变差（+0.0180）→ 不部署**           |
| 调好的 V2.5.3（生产模型）         | 49     | 2.7236     | ✅ 保持生产                           |

**核心教训（train/serve gap）**：训练时带来提升的"当期网格流"，恰恰是线上预测时
**拿不到**的部分；线上可行的"滞后"网格特征反而没用。所以网格**不进每日预测**。

---

## 6. 回退后保留了什么 / What was kept

| 保留项        | 位置                                                                    | 说明                                                    |
| ------------- | ----------------------------------------------------------------------- | ------------------------------------------------------- |
| 实验 notebook | `xgboost_models/modelV3.1_live.ipynb`                                   | live-safe 实验结果记录                                  |
| 学习笔记      | `docs/LearningNotes_CQL/13_src_folder_and_automation_learning_guide.md` | src 文件夹详解（含网格接入章节）                        |
| 实验模型      | `models/experiments/xgboost_v3_1.pkl`                                   | 62 特征实验模型（**不在** `models/saved/`，不影响线上） |
| 生产模型      | `models/saved/xgboost_v2_5_3.pkl`                                       | 调好的 XGBoost（49 特征，线上照常使用）                 |

**回退后 `models/saved/` 中没有任何带 `fi_*` 特征的模型** → 线上管线行为与之前完全一致。

---

## 7. 以后想恢复怎么办 / How to restore later

如果未来出现"线上可用且有帮助"的网格特征，用这个 patch 一键恢复全部代码改动：

```powershell
git apply docs/LearningNotes_CQL/grid_src_integration.patch
python -m unittest discover -s tests -v     # 重新跑网格测试验证
```

或者查看本次改动的原始 diff：

```powershell
git apply --stat docs/LearningNotes_CQL/grid_src_integration.patch
```

> 注意：恢复代码后，若要真让网格模型进每日预测，还需：
>
> 1. 用"仅滞后"网格特征重训一个能提升的 live-safe 模型；
> 2. 在 GitHub 仓库 Settings → Secrets 添加 `FINGRID_API_KEY`。
