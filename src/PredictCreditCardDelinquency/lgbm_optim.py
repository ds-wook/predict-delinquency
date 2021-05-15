import argparse

import joblib
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from data.dataset import load_dataset

train, test = load_dataset()

X = train.drop("credit", axis=1)
y = train["credit"]
X_test = test.copy()


def objective(trial: Trial) -> float:
    params_lgb = {
        "random_state": 42,
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }
    folds = StratifiedKFold(n_splits=args.fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)

    lgb_oof = np.zeros((X.shape[0], 3))
    lgb_preds = np.zeros((X_test.shape[0], 3))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        pre_model = LGBMClassifier(**params_lgb)

        pre_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=False,
        )

        params_lgb2 = params_lgb.copy()
        params_lgb2["learning_rate"] = params_lgb["learning_rate"] * 0.1

        model = LGBMClassifier(**params_lgb2)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=False,
            init_model=pre_model,
        )

        lgb_oof[valid_idx] = model.predict_proba(X_valid)
        lgb_preds += model.predict_proba(X_test) / args.fold

    log_score = log_loss(y, lgb_oof)
    return log_score


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        study_name="lgbm_parameter_opt",
        direction="minimize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.trials)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    params = study.best_trial.params
    params["random_state"] = 42
    params["boosting_type"] = "gbdt"
    params["learning_rate"] = 0.05
    params["n_estimators"] = 10000
    params["objective"] = "multiclass"
    params["metric"] = "multi_logloss"
    joblib.dump(params, "../../parameters/" + args.params)
