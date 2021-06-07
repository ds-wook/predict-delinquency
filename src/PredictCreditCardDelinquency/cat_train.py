import argparse

import pandas as pd

from data.dataset import category_income, load_dataset
from model.gbdt import stratified_kfold_cat


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
    # income_total
    train = category_income(train)
    test = category_income(test)
    X = train.drop("credit", axis=1)
    y = train["credit"]

    X_test = test.copy()

    cat_params = pd.read_pickle("../../parameters/best_cat_params.pkl")
    cat_oof, cat_preds = stratified_kfold_cat(cat_params, args.fold, X, y, X_test, 100)
    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = cat_preds
    submission.to_csv(args.path + args.file, index=False)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
