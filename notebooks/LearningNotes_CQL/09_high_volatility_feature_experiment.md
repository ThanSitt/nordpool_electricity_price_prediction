# 09 The High-Volatility Feature Experiment: Does It Help?

> A controlled experiment (controlled experiment) to test whether adding the `high_volatility_prob` feature improves the electricity-price model.
> Related notebooks:
>
> - `data/convertData/feature_high_volatility.ipynb` (build the classifier + histogram validation)
> - `xgboost_models/modelV3_1_risk_feature_test.ipynb` (the controlled price-model test)
>   Date: 2026-08-07

---

## 0. Background: What Feature Are We Testing?

We built a "storm warning" classifier (XGBClassifier, classification) that takes only **weather + time** and outputs `high_volatility_prob` — a 0~1 probability that "this moment will be highly volatile". We then added this probability as a new feature (feature) to the price-prediction model.

Question: **does this new feature actually make the price model more accurate?**

To answer this honestly, we ran TWO experiments.

---

## 1. Experiment 2 (Done First): Does the Feature Carry Signal?

**Method**: histogram (histogram) — plot the distribution of `high_volatility_prob` separately for:

- stable moments (label = 0)
- high-volatility moments (label = 1)

If the two histograms separate, the feature has information; if they fully overlap, it is useless.

**Results**:

```
Stable periods       mean prob: 0.113   median: 0.070
High-volatility      mean prob: 0.341   median: 0.295
```

**Interpretation**:

- The high-volatility moments clearly have higher probabilities (0.34 vs 0.11) -> **the feature DOES carry signal** (not pure noise). ✅
- But the distributions still overlap -> the signal is weak. This matches the classifier's modest performance (recall of only 0.24 for high volatility).

**Key idea**: a feature can "carry signal" yet still fail to help a model — which is why we need Experiment 1.

---

## 2. Experiment 1 (Done Second): Does the Feature Help the Price Model?

### 2.1 Method: A Controlled Experiment

To isolate the effect of the new feature, we kept **everything else the same**:

| Control variable    | Setting                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------- |
| Data                | the same risk-enhanced dataset (`V2.5.1_15min_Risk_Enhanced_Dataset.csv`, 105,193 rows) |
| Split               | the same chronological 80/20 split                                                      |
| Hyperparameters     | the same (the Optuna best params found in V3)                                           |
| Models              | XGBoost and LightGBM, each with 2000 trees                                              |
| **Only difference** | whether `high_volatility_prob` is included as a feature                                 |

- baseline features: 49 original features (drop all risk columns)
- enhanced features: 49 original features + `high_volatility_prob` (50 features)

Important: we excluded `price_roll_std_6h` and `is_high_volatility` from BOTH versions, because they are "answers derived from price" and would leak information (data leakage).

### 2.2 Results (test set)

| Model    | Without risk feature | With risk feature | MAE change | Verdict |
| -------- | -------------------- | ----------------- | ---------- | ------- |
| XGBoost  | MAE 2.7555           | MAE 2.7957        | +0.0402    | worse   |
| LightGBM | MAE 2.7165           | MAE 2.7426        | +0.0261    | worse   |

R² also dropped slightly for both (e.g. XGBoost 0.9727 -> 0.9716).

### 2.3 Verdict

**Adding `high_volatility_prob` made both models slightly WORSE (MAE increased by about 1%).** So in its current form, this risk feature should NOT be added to the V3 price model.

This "failure" is a valuable, honest result — not every intuitively sensible feature helps.

---

## 3. Why Did It Hurt? (Possible Reasons)

| Possible reason                   | Simple explanation                                                                                                                                  |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| The classifier itself is too weak | it only finds 24% of high-volatility moments (recall 0.24); the probabilities are noisy, so we are feeding "weak signal + noise" to the price model |
| Tree models dislike weak features | XGBoost/LightGBM may make suboptimal splits because of this weak feature, interfering with the 49 features that already work well                   |
| Small effect                      | the change is only ~1% of MAE — a marginal effect, not a disaster                                                                                   |

**Improvement direction**: make the classifier stronger first (e.g. handle class imbalance with `scale_pos_weight`, add more features), then re-run Experiment 1.

---

## 4. What We Learned (Key Takeaways)

1. **The experiment design was correct**: controlling variables (same data, same split, same params, only one extra feature) makes the conclusion trustworthy. ✅
2. **"No help" is also a valid conclusion**: we proved this risk feature is not worth adding to V3, saving future debugging time.
3. **Feature validation has two levels**:
   - Level 1 (Experiment 2): does the feature carry signal? (histogram / correlation)
   - Level 2 (Experiment 1): does the feature improve the model? (controlled test)
   - A feature can pass Level 1 yet fail Level 2.
4. **Scientific mindset**: always measure, never assume. The whole point of a controlled experiment is to get an honest answer.

---

## 5. Next Steps

| Option | What to do                                                                         |
| ------ | ---------------------------------------------------------------------------------- |
| A      | strengthen the classifier (class imbalance, more features) -> re-run the V3.1 test |
| B      | accept the conclusion and move on to other features / tuning                       |
| C      | test other candidate features using the same two-level validation method           |

---

## 6. How to Re-Run This Experiment (Quick Reference)

```python
# 1) Build the risk feature (see feature_high_volatility.ipynb) and save:
df.to_csv('V2.5.1_15min_Risk_Enhanced_Dataset.csv', index=False)

# 2) In modelV3_1_risk_feature_test.ipynb:
df = pd.read_csv('../data/convertData/V2.5.1_15min_Risk_Enhanced_Dataset.csv')
risk_cols = ['price_roll_std_6h', 'is_high_volatility', 'high_volatility_prob']
baseline_cols = [c for c in df.columns if c not in ['price', 'datetime'] + risk_cols]
enhanced_cols = baseline_cols + ['high_volatility_prob']
# ... same chronological 80/20 split, same hyperparameters, train both, compare MAE/RMSE/R2
```
