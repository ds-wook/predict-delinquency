import argparse

import pandas as pd

from data.dataset import load_dataset
from model.bagging import stratified_kfold_rf

train, test = load_dataset()
X = train.drop("credit", axis=1)
y = train["credit"]
X_test = test.copy()


if __name__ == "__main__":
    path = "../../input/predict-credit-card-delinquency/"
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../submission/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="submit.csv")
    parse.add_argument("--fold", type=int, default=10)
    args = parse.parse_args()

    rf_params = {
        "criterion": "gini",
        "n_estimators": 300,
        "min_samples_split": 10,
        "min_samples_leaf": 2,
        "max_features": "auto",
        "oob_score": True,
        "random_state": 42,
        "n_jobs": -1
    }
    rf_preds = stratified_kfold_rf(rf_params, args.fold, X, y, X_test)

    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = rf_preds
    submission.to_csv(args.path + args.file, index=False)
