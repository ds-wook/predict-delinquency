import argparse

import joblib
import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from data.dataset import load_dataset

train_ohe, test_ohe = load_dataset()

X = train_ohe.drop("credit", axis=1)
y = train_ohe["credit"]
X_test = test_ohe.copy()


def objective(trial: Trial) -> float:
    params_lgb = {
        "random_state": 42,
        "n_estimators": 10000,
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "eta": trial.suggest_float("eta", 0.01, 0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "max_leaves": trial.suggest_int("max_leaves", 2, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
        "gamma": trial.suggest_float("gamma", 0.5, 1),
    }

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    model = XGBClassifier(**params_lgb)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=False,
    )

    xgb_pred = model.predict_proba(X_valid)
    log_score = log_loss(y, xgb_pred)

    return log_score


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="xgb_parameter_opt",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.trials)

    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)

    params = study.best_trial.params
    params["random_state"] = 42
    params["n_estimators"] = 10000
    params["objective"] = "multi:softmax"
    params["eval_metric"] = "mlogloss"
    joblib.dump(params, "../../parameters/" + args.params)
