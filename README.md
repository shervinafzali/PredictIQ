# PredictIQ

# PredictIQ - Football Match Outcome Prediction

This repository contains a complete machine learning pipeline for predicting **football match outcomes** (home win, draw, away win) using the European Soccer Database (Kaggle).  
The project includes:

- SQL data extraction
- Leak-free, time-aware feature engineering
- Training/validation using chronological splits
- Model comparison (LogReg, RF, XGBoost, LightGBM, CatBoost)
- Hyperparameter tuning with Optuna
- Calibration analysis (Brier scores, reliability curves)
- SHAP explainability for model transparency
- A full model card describing the final selected model
- Architecture diagrams for the full project pipeline

---

##  Repository Structure
```text
PredictIQ/
├── src/
│   └── predictiq/
│       ├── data.py              # Data loading utilities
│       ├── features.py          # Reusable feature engineering functions
│       └── modeling.py          # Training, tuning, evaluation helpers
├── notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_FeatureEngineering.ipynb
│   └── 3_Modeling.ipynb
├── MODEL_CARD.md
├── ARCHITECTURE.md
├── README.md
└── LICENSE
```

---

##  Installation

Requires **Python 3.9+**

Install dependencies:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost optuna matplotlib seaborn shap kagglehub
```

Install this project as a package:

```bash
pip install -e .
```

How to Run the Project
Step 1 — Run notebooks in order

1. notebooks/1_EDA.ipynb — explore database
2. notebooks/2_FeatureEngineering.ipynb — build leak-free dataset
3. notebooks/3_Modeling.ipynb — train, tune, evaluate models

These notebooks call reusable functions from:
- src/predictiq/data.py
- src/predictiq/features.py
- src/predictiq/modeling.py



**Final Model Summary**

The best model is tuned XGBoost, achieving:
- Validation Macro-F1: 0.376
- Test Macro-F1: 0.357
The model shows strong calibration and robust performance across leagues.
A detailed model card is available in MODEL_CARD.md.

**Architecture & Deployment Ideas**

See ARCHITECTURE.md for:
- Data flow diagrams
- Full ML pipeline overview
- Deployment scenarios (batch or API inference)

## Dataset Setup

1. Download the **European Soccer Database** from Kaggle:
   - https://www.kaggle.com/datasets/hugomathien/soccer
2. Extract it locally and copy `database.sqlite` into:

   ```text
   data/raw/database.sqlite
   ```

3. The notebooks and predictiq.data.load_db() assume this path and will connect
to data/raw/database.sqlite automatically.

**License:**
MIT License




