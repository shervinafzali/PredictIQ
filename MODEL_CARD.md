# PredictIQ – Football Match Outcome Prediction  
**Model Card for Tuned XGBoost Classifier**

---

## Model Details

**Model Name:** PredictIQ XGBoost Match Outcome Predictor  
**Model Version:** 1.0  
**Model Type:** Multi-class classification (3 classes: home_win, draw, away_win)  
**Developers:** Amir Khajavirad, Shervin Afzali, Rojin Omidvar, and Mana Mohsenzadeh (University of Alberta)  
**Release Date:** 2025-12  
**Repository:** https://github.com/shervinafzali/PredictIQ  

**Algorithm:**  
XGBoost Gradient Boosted Decision Trees (`multi:softprob`)

**Model Artifacts:**  
- Trained model available via notebook (`3_Modeling.ipynb`)  
- Reusable feature + training code in:  
  - `src/predictiq/data.py`  
  - `src/predictiq/features.py`  
  - `src/predictiq/modeling.py`  

---

## Intended Use

### **Primary Use**
Predict pre-match outcomes of football matches using historical structured data from the European Soccer Database (Kaggle).

### **Intended Users**
- Sports analysts  
- Data scientists  
- Academic researchers  
- Betting-model researchers (for simulation/analysis only)  
- Students studying ML modeling or explainability  

### **Out-of-Scope Use Cases**
- Real-money betting  
- Predicting player performance or injuries  
- In-play (live) predictions  
- Using the model in leagues outside the dataset without retraining  
- High-stakes decision-making  

---

## Model/Data Description

### **Dataset**
European Soccer Database (Kaggle) containing:
- 25,000+ matches  
- 11 European leagues  
- Team attributes (tactical ratings, build-up style, chance creation, defense)  
- Match results from 2008–2016  

### **Data Processing**
- Extracted using SQL (`data.py`)  
- All features computed **leak-free** using only information prior to match date  
- Missing values handled via median imputation  
- Chronological split ensures realistic forward prediction  

### **Input Features**
Features fall into three categories:

#### **1. Rolling Form Metrics (Most Important Features)**
- `avg_goal_diff_last5_diff` (home − away)  
- `points_per_game_last5_diff`  
- `home_goal_diff_avg_last5`  
- `away_goal_diff_avg_last5`  
- Goals for/against metrics  

These were the strongest predictors based on SHAP.

#### **2. Tactical Attributes (From Team_Attributes Table)**
- Chance creation: passing, shooting, crossing  
- Build-up play: speed, passing, style  
- Defensive style: aggression, pressure, width  

#### **3. Composite Strength Indicators**
- `team_strength_diff`  
- Normalized ratings per season  

### **Target Variable**
`match_result` ∈ {home_win = 0, draw = 1, away_win = 2}

---

## Training and Evaluation

### **Training Procedure**
- Chronological split:  
  - **Train:** seasons 0–5  
  - **Validation:** season 6  
  - **Test:** season 7  
- Optuna hyperparameter tuning (50 trials)  
- Learning parameters tuned:  
  - `max_depth`, `learning_rate`, `min_child_weight`  
  - `subsample`, `colsample_bytree`, `gamma`  
  - `n_estimators`  

### **Final XGBoost Parameters**
```json
{
  "n_estimators": 238,
  "max_depth": 9,
  "learning_rate": 0.144,
  "subsample": 0.731,
  "colsample_bytree": 0.766,
  "min_child_weight": 3,
  "gamma": 0.452,
  "objective": "multi:softprob",
  "num_class": 3,
  "tree_method": "hist"
}
```

## Evaluation Metrics

### Performance Summary

| Model                  | Val Macro-F1 | Test Macro-F1 |
|------------------------|--------------|---------------|
| Majority Baseline      | —            | 0.203         |
| Logistic Regression    | 0.343        | 0.341         |
| Random Forest          | 0.360        | 0.346         |
| XGBoost (untuned)      | 0.357        | 0.350         |
| **XGBoost (tuned)**    | **0.376**    | **0.357**     |
| Bet365 Odds Baseline*  | —            | **0.380**     |

\*Bet365 odds used only as an external reference baseline.

### Calibration

**Brier Scores (lower = better):**
- Home win: 0.2416
- Draw: 0.2008
- Away win: 0.2036

\*The model is reasonably calibrated, with slight overconfidence at the high end.

### League-wise Performance**

- Macro-F1 ranges from 0.28 to 0.39 depending on league.
- Higher-visibility leagues (Premier League, La Liga) perform better due to larger, more stable data.

## Explainability (SHAP)

### Key Findings

- Recent momentum features dominate (goal difference, points per game).
- Tactical attributes refine predictions but have smaller impact.
- Draw predictions occur when features are balanced near zero (symmetry).
- Model behaves intuitively—consistent with football analytics literature.

### Top 5 Most Important Features**

1. avg_goal_diff_last5_diff
2. points_per_game_last5_diff
3. home_goal_diff_avg_last5
4. away_goal_diff_avg_last5
5. team_strength_diff

## Ethical Considerations
### Fairness & Bias

- Dataset represents European leagues only
- Lower-volume leagues show reduced prediction accuracy
- No personal identifiable data (PII) is used
- No protected demographic attributes present

### Privacy

- Entire dataset is anonymized
- No player-level sensitive data
- No model outputs identify individuals

### Security

- No external API calls
- No sensitive data transfer
- Model is deterministic and offline

## Limitations and Recommendations
### Known Limitations

- Football outcomes are highly stochastic—upper performance bound is low
- Draw prediction remains difficult
- Missing data in some seasons reduces consistency
- No access to injuries, lineups, transfers, weather, or referee data

### Recommendations for Use

- Use probabilities (not hard labels) for decision-making
- Retrain on new seasons if extending to other time periods
- For deployment, apply probability calibration
- Consider augmenting with ELO/Glicko team ratings

## References

1. European Soccer Database (Kaggle)
2. XGBoost: Chen & Guestrin (2016)
3. SHAP: Lundberg & Lee (2017)
4. Model Cards: Mitchell et al. (2018)

## Contact

Developer: Amir Khajavirad, Shervin Afzali, Rojin Omidvar and Mana Mohsenzadeh 
GitHub: 
https://github.com/shervinafzali
https://github.com/Amirkhajavirad
