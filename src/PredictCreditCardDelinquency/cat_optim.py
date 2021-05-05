import argparse

import joblib
import optuna
from catboost import CatBoostClassifier
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

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
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

    model = CatBoostClassifier(**params_cat)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        use_best_model=True,
        early_stopping_rounds=100,
        verbose=False,
    )
    cat_preds = model.predict_proba(X_valid)
    log_score = log_loss(y_valid, cat_preds)

    return log_score


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
    params["random_state"] = 42
    params["eval_metric"] = "MultiClass"
    params["loss_function"] = "MultiClass"
    params["learning_rate"] = 0.01
    params["od_type"] = "Iter"
    params["od_wait"] = 500
    params["n_estimators"] = 10000
    params["cat_features"] = [
        "income_type",
        "edu_type",
        "family_type",
        "house_type",
        "occyp_type",
        "gender_car",
        "gender_car_reality",
    ]
    joblib.dump(params, "../../parameters/" + args.params)
