# PredictIQ – System Architecture & Deployment Vision

This document describes how the PredictIQ system is structured, how data flows
from raw sources to predictions, and how the final model could be deployed in a
real-world setting.

---

## 1. High-Level Architecture

At a high level, the system does:

1. **Data Ingestion** – Download the European Soccer Database from Kaggle and read
   the `database.sqlite` file.
2. **Feature Engineering** – Build a leak-free, time-aware match-level dataset
   (`features_matches_main`) using `Team_Attributes` and rolling 5-match history.
3. **Modeling & Evaluation** – Train and evaluate several models
   (LogReg, Random Forest, XGBoost, LightGBM, CatBoost) with a chronological split.
4. **Calibration & Fairness** – Calibrate probabilities (Brier score, reliability curves)
   and measure performance across leagues.
5. **Explainability** – Use SHAP to understand which features drive predictions.
6. **Deployment (Future)** – Serve the tuned XGBoost model as a batch job or API.

---

## 2. Data Flow Diagram

```text
        +----------------------+
        | Kaggle:              |
        | European Soccer DB   |
        | (database.sqlite)    |
        +----------+-----------+
                   |
                   v
        +----------------------+
        |   src/predictiq/     |
        |   data.py            |
        | - download DB        |
        | - open SQLite        |
        | - query tables       |
        +----------+-----------+
                   |
                   v
        +-------------------------------+
        |   src/predictiq/features.py   |
        | - as-of merge Team_Attributes |
        | - build rolling form features |
        | - synthetic team strength     |
        | - save features_matches_main  |
        +----------+--------------------+
                   |
                   v
        +-------------------------------+
        | data/processed/               |
        | - features_matches_main.*     |
        +----------+--------------------+
                   |
                   v
        +--------------------------------------+
        |  notebooks/3_Modeling.ipynb         |
        |  + src/predictiq/modeling.py        |
        | - train/test split by season_index  |
        | - train baseline + ML models        |
        | - tune XGBoost with Optuna          |
        | - calibration & fairness analysis   |
        | - SHAP explainability               |
        +----------------+--------------------+
                         |
                         v
        +--------------------------------------+
        |  Outputs / Reports                  |
        | - MODEL_CARD.md                     |
        | - ARCHITECTURE.md                   |
        | - Plots: F1 comparison, SHAP, etc.  |
        +--------------------------------------+
```

## 3. Components & Responsibilities

### 3.1 Library Modules (`src/predictiq/`)

---

### **`data.py`**

Responsible for accessing the raw database.

**Provides:**
- Functions for downloading the database using `kagglehub`
- Opening SQLite connections
- Running SQL queries safely from Python
- Returning pandas DataFrames for downstream processing

---

### **`features.py`**

Responsible for converting relational soccer tables into a **single ML-ready dataset**.

**Includes:**
- Leak-free **as-of merge** (Match + Team_Attributes)
- Synthetic **team strength features**
- **Tactical difference** features (home − away)
- **Rolling 5-match performance metrics**, such as:
  - `win_rate_last5`
  - `avg_goals_for_last5`
  - `avg_goals_against_last5`
  - `goal_diff_avg_last5`
  - `points_per_game_last5`

**Outputs:**
- `features_matches_main.csv`
- `features_matches_main.parquet`

---

### **`modeling.py`**

Contains reusable ML utilities.

**Provides:**
- Chronological train/val/test splitting
- Training helpers (Logistic Regression, Random Forest, XGBoost)
- Metric computation:
  - Macro-F1  
  - Accuracy  
  - Brier scores  
- Plotting utilities:
  - Calibration curves  
  - Confusion matrix  
  - Model comparison charts  
- SHAP explainability functions (global + local)

---

## 4. End-to-End Workflow

### **Step 1 — Data Exploration**
Notebook: `notebooks/1_EDA.ipynb`

**Performs:**
- Inspect database schema  
- Explore missingness  
- Understand table relationships  

---

### **Step 2 — Feature Engineering**
Notebook: `notebooks/2_FeatureEngineering.ipynb`

**Performs:**
- Opens SQLite DB  
- Performs leak-free as-of merge  
- Computes rolling 5-match features  
- Saves `features_matches_main.parquet`  

---

### **Step 3 — Modeling & Evaluation**
Notebook: `notebooks/3_Modeling.ipynb`

**Pipeline includes:**
- Chronological train/val/test split  
- Baselines (majority class, logistic regression, random forest)  
- XGBoost + Optuna tuning  
- Probability calibration (isotonic)  
- League-wise fairness analysis  
- SHAP global & local explainability  
- Bet365 bookmaker odds benchmark (external)

---

## 5. Deployment Vision

### **5.1 Batch Inference (Offline)**

```text
1. Load upcoming fixtures
2. Generate features using features.py
3. Load trained XGBoost model
4. Predict probabilities
5. Save results (CSV, DB, dashboard)
```

### 5.2 Web API Deployment

A simple FastAPI service could accept:

```json
POST /predict
{
  "home_team_id": 8634,
  "away_team_id": 10281,
  "date": "2015-03-14"
}
```

**Returns:**

```json
{
  "home_win": 0.41,
  "draw": 0.28,
  "away_win": 0.31
}
```

### 5.3 Possible Frontend Dashboard

A future dashboard could include:

- Upcoming match probability table  
- Calibration plots  
- League-wise performance charts  
- SHAP explanation viewer  

---

## 6. Future Improvements

- Add ELO/Glicko dynamic ratings  
- Integrate injury + lineup information  
- Improve draw prediction using specialized loss functions  
- Train per-league models  
- Containerize pipeline using Docker  
- Add unit tests for feature transformations  

---

## 7. Summary

This architecture cleanly separates:

- **Reusable code** → `src/predictiq/`  
- **Experiment notebooks** → `notebooks/`  
- **Documentation** → `README.md`, `MODEL_CARD.md`, `ARCHITECTURE.md`  
- **Processed data** → `data/processed/`  

The system is **modular**, **leak-free**, **reproducible**, and ready for future deployment.

