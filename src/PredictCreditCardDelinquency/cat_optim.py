import argparse

import joblib
import numpy as np
import optuna
from catboost import CatBoostClassifier, Pool
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from data.dataset import load_dataset

train, test = load_dataset()

X = train.drop("credit", axis=1)
y = train["credit"]
cat_cols = [c for c in X.columns if X[c].dtypes == "int64"]
X_test = test.copy()


def objective(trial: Trial) -> float:
    folds = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    cat_oof = np.zeros((X.shape[0], 3))
    params_cat = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "od_type": "Iter",
        "od_wait": 500,
        "random_seed": 42,
        "iterations": 10000,
        "cat_features": cat_cols,
        "learning_rate": trial.suggest_uniform("learning_rate", 1e-5, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "bagging_temperature": trial.suggest_int("bagging_temperature", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_data = Pool(data=X_train, label=y_train, cat_features=cat_cols)
        valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_cols)

        model = CatBoostClassifier(**params_cat)
        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=False,
        )

        cat_oof[valid_idx] = model.predict_proba(X_valid)

    log_score = log_loss(y, cat_oof)
    return log_score


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
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
    params["random_state"] = 42
    params["eval_metric"] = "MultiClass"
    params["loss_function"] = "MultiClass"
    params["od_type"] = "Iter"
    params["od_wait"] = 500
    params["iterations"] = 10000
    params["cat_features"] = cat_cols
    joblib.dump(params, "../../parameters/" + args.params)
