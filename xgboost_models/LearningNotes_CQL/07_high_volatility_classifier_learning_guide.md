# 07 High-Volatility Probability Classifier — Zero-to-Beginner Learning Guide

> Corresponding notebook: `data/convertData/feature_high_volatility.ipynb`
> Target readers: students with no background in math or machine learning
> Note: This guide has been translated to English for your English note-taking workflow.

---

## 0. What is this guide for? What are you trying to build?

Your goal is: **train a small model (a classifier) that uses "weather + time" to decide whether the electricity price will become highly volatile, then take that decision (a 0~1 probability value) and use it as a new feature for the original price-prediction model.**

In simple words, you want to add a "storm warning system" to the original model:

> Original model: what will tomorrow's price be? (regression)
> New model: will tomorrow's price jump wildly? (classification) → feed "will it jump" as a clue to help the original model predict more accurately.

Analogy: the original model is like a "weather reporter" (reports temperature), the new model is like a "storm warning officer" (reports whether there is a typhoon). Letting the reporter refer to the storm-warning signal might make the temperature report more accurate.

---

## 1. Core Concepts From Zero (Understand Before Writing Code)

### 1.1 Regression vs Classification

|                       | Regression                        | Classification                                              |
| --------------------- | --------------------------------- | ----------------------------------------------------------- |
| What it predicts      | a **number** (continuous value)   | a **category** (a few discrete groups)                      |
| Example               | tomorrow's price = 45.7 EUR/MWh   | high volatility or not = yes/no (1/0)                       |
| Output                | a number, e.g. 45.7               | a probability, e.g. 0.87 (=87% chance of "high volatility") |
| What you already have | XGBoost/LightGBM price prediction | **what this notebook builds**                               |

> Key difference: regression outputs "how much", classification outputs "which category / how likely that category is".

### 1.2 What is a Label / Target?

A classification model needs a "correct answer" to learn from. That answer is called the **label** or **target**.

Your notebook creates the label with one line:

```python
df['is_high_volatility'] = (df['price_roll_std_6h'] >= vol_threshold).astype(int)
```

This compares "the price volatility over the past 6 hours (standard deviation)" with a warning line (threshold):

- volatility >= threshold -> label = 1 (high volatility, wild up/down moment)
- volatility < threshold -> label = 0 (stable)

This turns "price volatility" into a column of 0/1 answers, so the classifier can learn from it.

### 1.3 What is a Feature?

Features are the "clues" you feed the model. The classifier only sees these clues and then guesses the label.

The features your notebook picks (corrected):

```python
vol_features = ['temp', 'wind_speed', 'day_of_week', 'hour', 'month']
```

- temperature (temp)
- wind speed (wind_speed)
- day of week (day_of_week)
- hour of day (hour)
- month (month)

> Intuition: extreme weather (very cold, strong wind) often causes big price swings; weekends / late night are usually calm. So these clues do relate to volatility.

### 1.4 What is a Threshold?

A threshold is a "dividing line". Your notebook uses:

```python
vol_threshold = df['price_roll_std_6h'].quantile(0.85)
```

`quantile(0.85)` = the **85th percentile**. It sorts all volatility values and takes the value at the 85% position as the dividing line.

- In other words: the **top 15% most volatile moments = high volatility (label 1)**, the rest 85% = stable (label 0).

> Why the 85th percentile? Because "high volatility" is a relative concept; setting the line from the price history's own distribution is more reasonable than guessing a number.

### 1.5 What is Probability and predict_proba?

This is the most valuable part of the whole notebook.

Normal `model.predict()` directly tells you "1 or 0".
But `model.predict_proba()` gives a **soft probability** (a number between 0 and 1):

```python
risk_model.predict_proba(X_vol)[:, 1]
```

- It returns two columns: `[:, 0]` = probability of "stable", `[:, 1]` = probability of "high volatility"
- For example, returning `0.87` means "**87% sure this is a high-volatility moment**"

> Why use probability instead of 0/1? Because 0.87 and 0.55 have completely different "danger levels". Probability keeps this fine-grained detail (granularity), so it carries more information as a new feature.

### 1.6 What is a Training Set / Test Set?

- **Training set**: the data used to teach the model (homework)
- **Test set**: the data used to check the model (closed-book exam)

Your notebook uses:

```python
train_test_split(X_vol, y_vol, test_size=0.2, shuffle=False)
```

- `test_size=0.2`: 20% is the exam
- `shuffle=False`: **do not shuffle**, because time-series data must stay in time order — the future must not leak into the past (this is a hard rule!)

### 1.7 Core Intuition: Why can "weather" predict "price volatility"?

You might ask: **weather is not price, so why can weather predict price volatility?**

Answer: **causality**. Sharp electricity-price swings are usually caused by **sudden supply/demand changes**, and weather is a major driver:

- sudden temperature drop -> heating demand surges -> price jumps
- wind suddenly arrives/stops -> wind-power output changes sharply -> price swings
- extreme weather affects hydro plants and transmission lines

So "weather + time" contains signals about "will it be highly volatile". The classifier's job is to **discover these signals automatically** (instead of us summarizing them by hand).

---

## 2. What Your Notebook Does (Cell by Cell)

### Cell 1: Labeling (creating the target)

```
read V2.5 wide table -> sort by time -> compute 6h rolling std of price -> drop leading NaN
-> use the 85th percentile as threshold -> create 0/1 label
```

What it does: turns the continuous "price volatility" into a 0/1 yes/no answer.

### Cell 2: Training the Risk Classifier

```
pick weather + time features -> 80/20 chronological split -> train XGBClassifier
-> compute probabilities with predict_proba -> write back a new column high_volatility_prob
```

What it does: trains a "storm warning system" and makes it output a "high-volatility probability" at every moment.

### Cell 3: Saving

```
save the table with the new feature as V2.5.1_15min_Risk_Enhanced_Dataset.csv
```

What it does: creates an "enhanced" dataset for the price model to retrain on.

---

## 3. Important: Your Code Had "Column Name and Path Errors" (I Checked Them One by One)

### 3.1 Feature Column Name Mismatches (would raise a KeyError)

| Name used in your notebook   | Name that actually exists     | Note                                       |
| ---------------------------- | ----------------------------- | ------------------------------------------ |
| `air_temp_mean`              | `temp`                        | temperature column                         |
| `wind_speed_mean`            | `wind_speed`                  | wind speed column                          |
| `wind_lag_24h`               | does not exist                | there is no "wind 24h ago" column          |
| `temp_lag_24h`               | `temp_lag_4` or `temp_lag_96` | only "1h ago" / "24h ago (use 96)"         |
| `dayofweek`                  | `day_of_week`                 | mind the underscore                        |
| `price_roll_std_24h` (check) | `price_rolling_std_24h`       | missing "ing"; also that is the 24h window |

> Learning point: before writing code, ALWAYS print the header with `df.columns.tolist()` to confirm the column names. This is the most common beginner trap and the best habit to build.

### 3.2 Path Errors (Relative Path Points to the Wrong Place)

Your notebook lives inside the `data/convertData/` folder, so:

- reading `'../data/convertData/V2.5_15min_features.csv'` becomes `data/data/...` (wrong)
- saving `'../data/convertData/V2.5.1...csv'` also becomes `data/data/...` (wrong)

**Fix**: because you are already inside `data/convertData/`, just use the file name directly:

```python
df = pd.read_csv('V2.5_15min_features.csv')          # same folder
df.to_csv('V2.5.1_15min_Risk_Enhanced_Dataset.csv')  # same folder
```

> Learning point: a relative path is resolved from the folder where the current file lives. Know which folder your notebook is in, then decide how many `../` you need.

---

## 4. Corrected Full Code (Ready to Use)

Replace the cells below into your notebook (paths and column names are fixed, and a "classifier evaluation" step was added):

### Cell 1 — Labeling

```python
import pandas as pd
import numpy as np

print("1. Loading V2.5 15-Minute Dataset...")
# Fix 1: path - the notebook is inside data/convertData, so use the file name directly
df = pd.read_csv('V2.5_15min_features.csv')

# convert datetime to UTC, then to Helsinki time (fixes the mixed-timezone error)
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('Europe/Helsinki')
df = df.sort_values('datetime').reset_index(drop=True)

print("2. Defining 'High Volatility' Threshold...")
# compute the 6h rolling std of price (volatility); 15-min granularity -> window of 24 rows
# note: create a fresh column instead of reusing price_rolling_std_24h (that one is a 24h window)
df['price_roll_std_6h'] = df['price'].rolling(window=24).std()

# drop the leading NaN rows created by the rolling window
df = df.dropna(subset=['price_roll_std_6h']).copy()

vol_threshold = df['price_roll_std_6h'].quantile(0.85)
print(f"   -> Top 15% Volatility Threshold: {vol_threshold:.2f}")

print("3. Creating the Binary Classification Target (0 or 1)...")
df['is_high_volatility'] = (df['price_roll_std_6h'] >= vol_threshold).astype(int)
print(df['is_high_volatility'].value_counts(normalize=True) * 100)
```

### Cell 2 — Training the Risk Classifier (fixed columns + added evaluation)

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("4. Selecting Safe Features for the Classifier...")
# Fix 2: use column names that actually exist in the dataset
# temp_lag_4 = temperature 1 hour ago (15 min * 4 = 1 hour)
vol_features = ['temp', 'wind_speed', 'wind_direction_deg', 'temp_lag_4',
                'hour', 'day_of_week', 'month']

X_vol = df[vol_features]
y_vol = df['is_high_volatility']

# safety: drop rows with NaN in the features (XGB can tolerate them, but cleaner is better)
mask = X_vol.notna().all(axis=1)
X_vol, y_vol = X_vol[mask], y_vol[mask]

X_vol_train, X_vol_test, y_vol_train, y_vol_test = train_test_split(
    X_vol, y_vol, test_size=0.2, shuffle=False
)

print("5. Initializing and Training the XGBoost Risk Classifier...")
risk_model = XGBClassifier(
    n_estimators=100,       # 100 warning trees
    learning_rate=0.05,     # learn slowly and steadily
    max_depth=5,            # depth 5, prevents memorizing (overfitting)
    objective='binary:logistic',  # explicitly tell it this is a binary classification task
    random_state=42         # lock the random seed so results are reproducible
)
risk_model.fit(X_vol_train, y_vol_train)

print("6. Evaluating the Risk Classifier (NEW STEP)...")
y_pred = risk_model.predict(X_vol_test)
print('Test accuracy:', round(accuracy_score(y_vol_test, y_pred), 4))
print(classification_report(y_vol_test, y_pred))

print("7. Extracting the High-Volatility Probabilities...")
# predict_proba returns [safe probability, high-volatility probability]; take the second column [:, 1]
# Fix 3: X_vol may be shorter than df (NaN rows removed), so use a Series aligned by index
proba = pd.Series(risk_model.predict_proba(X_vol)[:, 1], index=X_vol.index)
df['high_volatility_prob'] = proba

print("   -> New feature 'high_volatility_prob' created!")
df[['datetime', 'price', 'is_high_volatility', 'high_volatility_prob']].tail(10)
```

### Cell 3 — Saving the Enhanced Dataset (fixed path)

```python
print("8. Saving the new Matrix (V2.5 + Risk Feature)...")
# Fix 4: path - save directly to the current folder
save_path = 'V2.5.1_15min_Risk_Enhanced_Dataset.csv'
df.to_csv(save_path, index=False)
print(f"Matrix saved to: {save_path}")
```

---

## 5. How to Use This New Feature in the Price Model (V2.5.2)?

Once `high_volatility_prob` is created, making the price model use it takes **three steps**:

```
Step 1: this notebook -> generate V2.5.1_15min_Risk_Enhanced_Dataset.csv (already contains the new feature)
Step 2: in the V2.5.2 comparison notebook, include the new column when training the price model
        e.g. after df.drop(columns=['price','datetime']), the model will automatically see the new column
Step 3: WARNING: you must also update src/features.py (see section 6)
```

In the V2.5.2 comparison notebook, you only need:

```python
df = pd.read_csv('../data/convertData/V2.5.1_15min_Risk_Enhanced_Dataset.csv')
X = df.drop(columns=['price', 'datetime', 'is_high_volatility', 'price_roll_std_6h'])
```

> Note: `is_high_volatility` (the 0/1 label) and `price_roll_std_6h` (the volatility value) **cannot** be used as features for the price model — they are "answers derived from price" and would leak real information (data leakage). **Only `high_volatility_prob` (a probability inferred from weather) can be a feature.**

---

## 6. Pitfalls You Must Know (Caveats)

### 6.1 Global Threshold = Slight Data Leakage

`vol_threshold = quantile(0.85)` is computed over the **whole table (including the test period)**. That means the threshold "peeked at the future".

- Impact: slight, because the threshold is only a global dividing line
- Improvement (advanced): compute the threshold only on the training part:
  ```python
  train_part = df.iloc[:int(len(df)*0.8)]
  vol_threshold = train_part['price_roll_std_6h'].quantile(0.85)
  ```

### 6.2 Class Imbalance

Your label is 85% stable / 15% high volatility — **imbalanced**.

- Consequence: the model may be "lazy" and predict 0 for everything, and accuracy is still 85%
- So: **do not only look at accuracy**; look at `precision` and `recall`. The `classification_report` prints these for you.
- Improvement: `XGBClassifier(scale_pos_weight=...)` can penalize the "lazy" behavior.

### 6.3 Does the Rolling Window Include the Current Value?

`df['price'].rolling(24).std()` uses the **previous 24 values including the current one**.

- For your **label**: that is fine (we want to mark the volatility of "this current moment")
- But note: the existing `price_rolling_std_1h` in the table is computed with `shift(1)` before rolling (it **excludes** the current value). The two semantics are different; do not mix them.

### 6.4 WARNING - The Most Important Pitfall: How to Get This Feature in Live Prediction?

This is the most important constraint in the whole project, and it is what the README keeps stressing:

> **Any feature built during training must be computable in exactly the same way during live prediction.**

Good news: your design is clever — the classifier only takes **weather and time** as input, and the next 7 days of weather (forecast) and time (calendar) are **known in advance**. So `high_volatility_prob` can be computed for every future moment and used for live prediction.

Bad news: **`src/features.py` does not contain this logic yet**. If you only change the training data but not `src/features.py`, training and prediction features will be inconsistent — the predictions will be wrong. So:

- Advanced task: also write the `high_volatility_prob` computation into `src/features.py`, and load `risk_model` in `src/predict_system.py`
- This is the key step to make the feature "go live" (you can postpone it for now, but you must **know** about it)

### 6.5 How to Judge Whether the Classifier Itself Is "Good"?

- A good classifier: `high_volatility_prob` is clearly higher for high-volatility moments than for stable moments
- You can verify visually:
  ```python
  import matplotlib.pyplot as plt
  high = df[df['is_high_volatility']==1]['high_volatility_prob']
  low  = df[df['is_high_volatility']==0]['high_volatility_prob']
  plt.hist([low, high], label=['Stable', 'High volatility'], bins=30, alpha=0.6)
  plt.legend(); plt.show()
  ```
  If the two histograms separate, the feature is useful; if they completely overlap, the feature carries no information.

---

## 7. Learning Path: What to Learn Next

| Stage                  | Content                                                                             | Priority  |
| ---------------------- | ----------------------------------------------------------------------------------- | --------- |
| 1. Get it running      | run the 3 cells with the corrected code                                             | must      |
| 2. Understand          | answer: how is the label created? what does predict_proba return?                   | must      |
| 3. Verify              | run the histogram in 6.5 and check whether the feature helps                        | important |
| 4. Integrate           | add the new feature to the V2.5.2 price model, retrain, compare whether it improved | important |
| 5. Go live (advanced)  | mirror the logic into `src/features.py` + `src/predict_system.py`                   | advanced  |
| 6. Optimize (advanced) | fix the data leakage, handle class imbalance, try more features                     | advanced  |

---

## 8. Glossary (English)

| English               | One-line explanation                                                |
| --------------------- | ------------------------------------------------------------------- |
| Classification        | predict "which category", not "how much"                            |
| Binary Classification | only two classes (0/1, e.g. stable / high volatility)               |
| Regression            | predict a continuous number (e.g. price 45.7)                       |
| Label / Target        | the "correct answer" the model learns from                          |
| Feature               | the "clue" fed to the model (weather, time)                         |
| Threshold             | the dividing line for deciding "is it high volatility"              |
| Percentile / Quantile | the value at a certain percentage position after sorting the data   |
| Standard Deviation    | how much the data fluctuates / disperses                            |
| Rolling Window        | slide over the past N time points                                   |
| Probability           | a 0~1 "degree of confidence"                                        |
| predict_proba         | output probabilities instead of a hard class                        |
| Training / Test Set   | homework data / exam data                                           |
| Accuracy              | the fraction of correct guesses                                     |
| Precision             | of the moments predicted "high volatility", how many really are     |
| Recall                | of the truly high-volatility moments, how many were found           |
| Class Imbalance       | the two classes have very different numbers of samples              |
| Data Leakage          | the model peeked at information it should not have (future/answers) |
| Overfitting           | memorizing the training questions and failing on new ones           |
| Random Seed           | lock the randomness so results are reproducible                     |
| Relative Path         | find a file starting from the current file's location               |
| Supervised Learning   | learning with a correct answer (this project is supervised)         |

---

_This guide was written from the project background + your notebook content. Core idea in one sentence: first turn volatility into a 0/1 label with the 85th percentile, then train a classifier on weather + time to output a "high-volatility probability", and feed that probability as a new feature to the price-prediction model._
