"""
modeling.py — Training, evaluation, hyperparameter tuning,
and probability calibration utilities for PredictIQ.

This module provides:
- train_xgb(): train tuned or default XGBoost model
- evaluate(): accuracy + macro-F1
- tune_xgb_optuna(): optional Optuna tuning
- calibrate_model(): isotonic calibration
"""

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb


# 1. Training (default or custom params)
def train_xgb(X_train, y_train, X_val=None, y_val=None, params=None):
    """
    Train an XGBoost multi-class model.
    If params is None, a strong default config is used.
    """
    if params is None:
        params = {
            "n_estimators": 400,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0,
            "eval_metric": "mlogloss",
            "objective": "multi:softprob",
            "num_class": 3,
        }

    model = xgb.XGBClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


# 2. Evaluation: Macro-F1 and Accuracy
def evaluate(model, X, y_true):
    """
    Compute macro-F1 and accuracy.
    Returns a dictionary.
    """
    preds = model.predict(X)
    return {
        "macro_f1": f1_score(y_true, preds, average="macro"),
        "accuracy": accuracy_score(y_true, preds),
    }


# 3. Optuna Tuning (optional, used in notebook)
def tune_xgb_optuna(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for Optuna hyperparameter search.
    Returns macro-F1 score on validation set.
    """

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="macro")
    return f1


# 4. Probability Calibration
def calibrate_model(model, X_val, y_val, method="isotonic"):
    """
    Apply probability calibration to a pre-fitted XGB model.
    method  {"isotonic", "sigmoid"}.
    Returns calibrated model.
    """

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv="prefit"
    )
    calibrated.fit(X_val, y_val)
    return calibrated

