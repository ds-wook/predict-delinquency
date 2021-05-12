import argparse

import pandas as pd

from data.dataset import load_dataset
from model.gbdt import stratified_kfold_cat

train, test = load_dataset()
X = train.drop("credit", axis=1)
y = train["credit"]
cat_cols = [c for c in X.columns if X[c].dtypes == "int64"]
X_test = test.copy()


if __name__ == "__main__":
    path = "../../input/predict-credit-card-delinquency/"
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../submission/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, default=10)
    args = parse.parse_args()
    cat_params = pd.read_pickle("../../parameters/best_cat_params.pkl")
    cat_preds = stratified_kfold_cat(cat_params, args.fold, X, y, X_test)
    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = cat_preds
    submission.to_csv(args.path + args.file, index=False)
