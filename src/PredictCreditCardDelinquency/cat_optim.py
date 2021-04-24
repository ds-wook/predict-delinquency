import argparse

import joblib
import numpy as np
import optuna
from catboost import CatBoostClassifier
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from data.dataset import load_dataset

train_ohe, test_ohe = load_dataset()

X = train_ohe.drop("credit", axis=1)
y = train_ohe["credit"]
X_test = test_ohe.copy()


def objective(trial: Trial) -> float:
    params_cat = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "od_type": "Iter",
        "od_wait": 500,
        "random_seed": 2021,
        "learning_rate": 0.01,
        "iterations": 10000,
        "cat_features": [col for col in X.columns if X[col].dtype == "uint8"],
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.01, 1),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "bagging_temperature": trial.suggest_int("bagging_temperature", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }

    folds = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    scores = []
    cat_oof = np.zeros((X.shape[0], 3))
    cat_preds = np.zeros((X_test.shape[0], 3))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**params_cat)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )
        cat_oof[valid_idx] = model.predict_proba(X_valid)
        cat_preds += model.predict_proba(X_test) / args.fold

    log_score = log_loss(y, cat_oof)
    scores.append(log_score)
    return np.mean(scores)


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=5)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()
    study = optuna.create_study(
        study_name="cat_parameter_opt",
        direction="minimize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.trials)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    params = study.best_trial.params
    params["random_state"] = 2021
    params["eval_metric"] = "MultiClass"
    params["loss_function"] = "MultiClass"
    params["learning_rate"] = 0.01
    params["od_type"] = "Iter"
    params["od_wait"] = 500
    params["n_estimators"] = 10000
    params["cat_features"] = [col for col in X.columns if X[col].dtype == "uint8"]
    joblib.dump(params, "../../parameters/" + args.params)
