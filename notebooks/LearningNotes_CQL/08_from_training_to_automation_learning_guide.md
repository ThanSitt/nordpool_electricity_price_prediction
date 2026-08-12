# 08 From "Training a Model" to "Automated Prediction": A Complete Mental-Model Shift

> Corresponds to your question: "I finished training XGBoost V2, now what? How does it become a GitHub Actions auto-prediction?"
> Target readers: beginners who have only trained models and have not yet understood "prediction" and "deployment"
> Note: This guide has been translated to English for your English note-taking workflow.

---

## 0. Here Is a "Global Map" First (Most Important! Read This First)

```
┌──────────── The world you know: Training ────────────┐
│                                                      │
│  data -> features -> XGBoost training -> a "model"   │
│  (what happens in the notebook)         ↓            │
│                                  model.pkl (model file) │
└──────────────────────────────────────────────────────┘
                       ↓ enter the "Deployment" world
┌──────────── The world you do not know yet: Prediction + Automation ─┐
│                                                                     │
│  GitHub Actions (.yml) calls daily on schedule:                     │
│      python src/predict_system.py                                   │
│        ↓ read all model.pkl                                        │
│        ↓ fetch latest weather/price -> build features -> predict 7 days │
│        ↓ generate predictions/*.csv                                │
│        ↓ git commit (forecast: date daily update)                  │
└────────────────────────────────────────────────────────────────────┘
```

**Core idea in one sentence**: training produces a "model file (.pkl)"; the prediction program "uses" that model file to compute the future; GitHub Actions just wakes the prediction program up every day. **The model is NOT a CSV, and NOT a notebook.**

---

## 1. What Is .yml? (Your Questions 1, 2, 3)

### 1.1 Is .yml a program? Is it Python?

**It is not a program, and not Python.** `.yml` (also written `.yaml`) is a **config file format**, just like JSON or TOML, used to **describe "some settings"**, not to "execute logic".

| Comparison   | Programming language (Python) | Config format (YAML)                    |
| ------------ | ----------------------------- | --------------------------------------- |
| what it does | write "how to compute" logic  | write a description of "settings/steps" |
| example      | `for i in range(10): ...`     | `schedule: cron: '0 11 * * *'`          |
| has if/for?  | yes                           | no (it is only data)                    |
| extension    | `.py`                         | `.yml` / `.yaml`                        |

YAML stands for "**YAML Ain't Markup Language**" — a recursive acronym meaning "I am just a simple human-readable format".

### 1.2 Why does it end in .yml?

`.yml` is the file extension of the YAML format (`.yaml` is another common spelling; they are the same). It tells "editors/tools": please highlight and validate this file as YAML syntax.

### 1.3 Is this a GitHub-only format?

**YAML itself is not GitHub-only** — many tools use YAML for config, for example:

- Docker (`docker-compose.yml`)
- Kubernetes (k8s)
- Ansible (automation)
- **Your own project too! `environment.yml` is YAML format!**

But **`.github/workflows/*.yml` "workflow files" are a GitHub Actions-specific convention**: GitHub looks for YAML files in your repo's `.github/workflows/` folder and executes them according to the settings.

> One sentence: **YAML is a general format; `daily_forecast.yml` is "a GitHub automation manual written in YAML".**

---

## 2. What Exactly Is the Trained Model? (Your Question 9, the Core)

You asked: "Is the trained model a CSV? Is it a notebook? Can I change the data in the notebook to predict?"

**None of those.** Let me explain thoroughly:

### 2.1 After Training You Get a .pkl File (the Model File)

After training finishes, the code runs this line (at the end of the notebook):

```python
joblib.dump({
    'model': model_v25,            # the trained model itself
    'feature_cols': X_train.columns.tolist(),  # which features were used
    'step_min': 15,                # the resolution (15 minutes)
}, 'models/saved/xgboost_v2_5.pkl')
```

`joblib.dump()` **packages the "trained brain" in memory into a disk file**, with the `.pkl` extension.

**Analogy**: a trained model is like "a brain that has learned all the knowledge". The `.pkl` file is the container that **freezes and preserves** this brain. `.pkl` = pickle — meaning "preserve the object like pickling vegetables".

### 2.2 What Three Things Are Inside the .pkl?

| Field          | What it is              | Purpose                                                         |
| -------------- | ----------------------- | --------------------------------------------------------------- |
| `model`        | the model itself        | the "brain" actually used to predict                            |
| `feature_cols` | the feature-column list | which columns to feed the model when predicting (order matters) |
| `step_min`     | the time resolution     | 60 = hourly, 15 = 15 minutes                                    |

**This is why "changing the notebook data" is not how you predict**: the notebook is the "school" (for training). When predicting, you do not need the school; you only need to **thaw the model file (`.pkl`)** and feed new data into it.

### 2.3 Prediction = Load the .pkl + Call predict()

```python
import joblib
meta = joblib.load('models/saved/xgboost_v2_5.pkl')   # thaw the brain
model = meta['model']                                   # take out the model
pred = model.predict(a_new_row_of_features)             # use the brain to compute the future
```

**Where your thinking needs to be upgraded**: training and prediction are two stages —

- training: teach the brain (notebook, one-time)
- prediction: use the brain (predict_system.py, repeated every day)

---

## 3. What Are the models/ Folder, .gitkeep, and .pkl? (Your Question 8)

### 3.1 Three Concepts

| Item            | What it is                 | Purpose                           |
| --------------- | -------------------------- | --------------------------------- |
| `models/saved/` | the model warehouse folder | stores all trained models         |
| `*.pkl`         | the trained model files    | the prediction program reads them |
| `.gitkeep`      | a placeholder file         | lets git keep the empty folder    |

### 3.2 What Is .gitkeep for?

**git does not track "empty folders"** — if a folder is completely empty, git ignores it when committing. `.gitkeep` is a **conventional empty placeholder file** that makes the folder "look non-empty", so git will bring it into the repo.

> It has no real function; it just "holds a place". Think of it as putting a chair in an empty room so git "remembers" that the room exists.

### 3.3 WARNING - A Pitfall I Found For You: The V3 Models Are NOT Tracked by Git!

I looked at your `.gitignore`, which contains these lines:

```gitignore
*.pkl
!models/saved/xgboost_v*.pkl          # allow the original XGBoost models
!models/saved/lightgbm_v2.pkl         # allow lightgbm_v2
!models/saved/lightgbm_v2_5.pkl       # allow lightgbm_v2_5
```

What this means:

- `*.pkl` = by default all .pkl files are **not committed** (model files are big; you should not commit them casually)
- `!xxx` = but **these exceptions should be committed** (allowlisted)

**Result**:

- Will be committed: `xgboost_v1/v1_5/v2/v2_5.pkl`, `lightgbm_v2.pkl`, `lightgbm_v2_5.pkl` (6 in total)
- Will **NOT** be committed: your newly generated `xgboost_v3.pkl` and `lightgbm_v3.pkl` (they are not on the allowlist!)

**Consequence**: your V3 models exist only locally. After pushing to GitHub, the **remote repo does not have them**, so when GitHub Actions runs automatically every day, it **cannot find the V3 models and will only run the 6 original models**.

> If you want GitHub to also run V3, you need to add two lines to `.gitignore`:
>
> ```gitignore
> !models/saved/xgboost_v3.pkl
> !models/saved/lightgbm_v3.pkl
> ```

---

## 4. What Is predict_system.py? (Your Question 7)

You said "I only know how to train models" — right, `predict_system.py` is **the other half besides training: the prediction system**.

### 4.1 Where Does It Live?

```
src/
├── config.py          <- configuration (paths, APIs, parameters)
├── fetch_live.py      <- fetch the latest data from 3 APIs
├── features.py        <- build features (exactly the same as during training!)
├── predict_system.py  <- *main program: the conductor of prediction
└── utils.py           <- utilities (empty)
```

### 4.2 Its Core Responsibilities (Against the Code)

| Function         | What it does                                                       |
| ---------------- | ------------------------------------------------------------------ |
| `main()`         | the conductor: compute time, call data, load models, save results  |
| `load_models()`  | \*scan and load all .pkl files in `models/saved/`                  |
| `run_forecast()` | \*recursively forecast the next 7 days (672 fifteen-minute points) |
| `fill_actuals()` | back-fill real prices for past time points                         |
| `save_csv()`     | save one CSV per model                                             |

The key code (where you asked "how are the models started"):

```python
def load_models():
    models = {}
    for pkl in sorted(config.SAVED_MODELS_DIR.glob('*.pkl')):  # scan all .pkl
        meta = joblib.load(pkl)                                 # thaw each one
        models[pkl.stem] = meta                                 # put into a dict
    return models
```

> So **the "start all models" code is in `src/predict_system.py` inside `load_models()`** — it automatically scans the `models/saved/` folder and loads every `.pkl` in it. **You do not need to specify models manually**; whatever you put in, it loads that many.

The core of recursive forecasting (`run_forecast`):

```python
for i in range(FORECAST_HOURS * (60 // step_min)):   # e.g. 672 times
    features = build_features(timestamp, price_buf, wx_buf)  # build features for this moment
    prediction = model.predict(row)[0]                        # predict
    price_buf.add(timestamp, prediction)                      # feed the prediction back as "history"
```

---

## 5. How Does the Workflow (.yml) "Start All Models"? (Your Question 6)

Key insight: **the workflow itself does not directly load models**. It only does one thing: **run one command**.

```yaml
- name: Run prediction system
  run: python src/predict_system.py # <- the only key command
```

Then `predict_system.py` loads the models by itself (see section 4). The whole chain is:

```
daily_forecast.yml
   └─> run python src/predict_system.py
          ├─> load_models()  scan models/saved/*.pkl -> load all
          ├─> fetch_live     get weather + prices
          ├─> run_forecast   each model predicts 7 days
          └─> save_csv       each model saves one CSV
```

> Just like your local command `python src/predict_system.py` — GitHub runs **the same command** every day, only it runs in the cloud automatically.

---

## 6. The Prediction Results: How Many CSVs? What Is Inside? (Your Questions 4, 5)

### 6.1 How Many CSVs?

**6 CSVs (not 7)**, one per model:

```
predictions/
├── xgboost_v1_forecasts.csv       <- XGBoost V1 forecast
├── xgboost_v1_5_forecasts.csv     <- XGBoost V1.5
├── xgboost_v2_forecasts.csv       <- XGBoost V2
├── xgboost_v2_5_forecasts.csv     <- XGBoost V2.5 (15-minute)
├── lightgbm_v2_forecasts.csv      <- LightGBM V2
└── lightgbm_v2_5_forecasts.csv    <- LightGBM V2.5
```

> You might say "7" — maybe you mixed up the 7 days with the number of files. The committed models are 6 -> 6 CSVs. **Each CSV contains the next 7 days of forecasts.**

### 6.2 One CSV Contains "the Next 7 Days"; the Granularity Depends on the Model

| Model type                    | Time granularity            | Rows for 7 days     |
| ----------------------------- | --------------------------- | ------------------- |
| 15-minute models (V1.5, V2.5) | one prediction every 15 min | 7x96 = **672 rows** |
| hourly models (V1, V2)        | one prediction every hour   | 7x24 = **168 rows** |

Each row format:

```
run_date           = the day the forecast was made
target_datetime    = which moment is being predicted
predicted_price    = predicted price (EUR/MWh)
actual_price       = real price (back-filled automatically after a few days)
abs_error          = error |actual - predicted|
```

### 6.3 How Does It Accumulate Day by Day?

(Review in one sentence from the earlier discussion): **the same file "appends" a new batch of `run_date` every day**; it does not delete old and create new. So if the file has run for a month, it will contain 30 batches of different `run_date` values.

---

## 7. The Complete Mental Model (From Training to Automated Prediction)

```
1. Train (you already know this)
   notebook -> data/features/model -> save as model.pkl
                    |
                    v
2. Commit (git)
   model.pkl + prediction code + workflow -> push to GitHub
                    |
                    v
3. Deploy (GitHub Actions runs automatically every day)
   .yml timer -> python src/predict_system.py
              -> load model.pkl -> fetch data -> build features -> predict 7 days
              -> save predictions/*.csv -> git commit
                    |
                    v
4. View the results
   open predictions/*.csv or draw line charts
```

**Compared to where you are now (you have only done step 1)**:

- You already know how to train XGBoost V2 -> it generates `xgboost_v2.pkl`
- You have not yet understood steps 3 and 4: the prediction program reads the .pkl -> automatically produces CSVs -> GitHub runs it every day

---

## 8. The "Next Steps" Checklist You Are Missing

| Step | What to do                                                                         | Status                  |
| ---- | ---------------------------------------------------------------------------------- | ----------------------- |
| 1    | run `python src/predict_system.py` locally and see it load models and produce CSVs | you already did this    |
| 2    | understand `load_models()` and `run_forecast()` in `src/predict_system.py`         | section 4 of this guide |
| 3    | allowlist your new models (e.g. V3) in `.gitignore` so GitHub can run them         | needs your action       |
| 4    | push the code to GitHub so `daily_forecast.yml` runs every day automatically       | to do                   |
| 5    | open GitHub web page -> Actions tab -> see the daily run logs                      | to do                   |

---

## 9. Glossary (English)

| English                | One-line explanation                                                                                      |
| ---------------------- | --------------------------------------------------------------------------------------------------------- |
| Training               | let the model learn patterns from historical data (one-time)                                              |
| Prediction / Inference | use the trained model to compute the future (repeatedly)                                                  |
| Deployment             | put the trained model into an environment that can run automatically                                      |
| Model artifact         | the "packaged file" of the training result (.pkl)                                                         |
| Serialization          | store an in-memory object to a file (joblib.dump)                                                         |
| Deserialization        | read a file back into memory (joblib.load)                                                                |
| Config file            | a file describing settings (YAML/JSON)                                                                    |
| Workflow               | a YAML file that defines automatic steps on GitHub                                                        |
| Scheduled job          | a task that runs automatically at a set time (cron)                                                       |
| CI/CD                  | Continuous Integration / Continuous Deployment; a set of practices for automated integration + deployment |
| Placeholder            | a .gitkeep file that lets git keep an empty folder                                                        |
| Gitignore              | tells git which files not to commit                                                                       |
| Allowlist / Exception  | the exception rules starting with `!` in .gitignore                                                       |
| Recursive forecasting  | use the previous prediction as the history for the next step                                              |

---

_This guide was written from `.github/workflows/daily_forecast.yml`, `src/predict_system.py`, `models/saved/`, and `.gitignore`. Core idea in one sentence: training produces the .pkl model file, predict_system.py reads it to predict, the .yml wakes predict_system.py every day automatically, and the results are stored as 6 CSVs (one per model), each containing the next 7 days._
