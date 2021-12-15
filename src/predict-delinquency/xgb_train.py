import argparse

import pandas as pd

from data.dataset import load_dataset
from model.gbdt import stratified_kfold_xgb


def define_argparser():
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--submit", type=str, help="Input data save path", default="../../submission/"
    )
    parse.add_argument(
        "--path", type=str, default="../../input/predict-credit-card-delinquency/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    parse.add_argument("--fold", type=int, default=10)

    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    path = args.path
    train, test = load_dataset(path)
    X = train.drop("credit", axis=1)
    y = train["credit"]
    X_test = test.copy()

    xgb_params = pd.read_pickle("../../parameters/best_xgb_params.pkl")
    xgb_oof, xgb_preds = stratified_kfold_xgb(xgb_params, args.fold, X, y, X_test, 100)
    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = xgb_preds
    submission.to_csv(args.path + args.file, index=False)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
