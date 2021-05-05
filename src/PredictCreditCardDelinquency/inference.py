import argparse

import pandas as pd

if __name__ == "__main__":
    path = "../../input/predict-credit-card-delinquency/"
    parse = argparse.ArgumentParser("Ensemble")
    parse.add_argument("--w1", type=float, default=0.5)
    parse.add_argument("--w2", type=float, default=0.2)
    parse.add_argument("--w3", type=float, default=0.05)
    parse.add_argument("--w4", type=float, default=0.4)
    parse.add_argument("--file", type=str, default="ensemble_model.csv")
    args = parse.parse_args()

    lgb_preds = pd.read_csv("../../submission/pre_lgbm_kmeans.csv")
    xgb_preds = pd.read_csv("../../submission/xgb_kmeans.csv")
    cat_preds = pd.read_csv("../../submission/cat_kmeans.csv")
    rf_preds = pd.read_csv("../../submission/rf_kmeans.csv")

    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = (
        args.w1 * lgb_preds.iloc[:, 1:]
        + args.w2 * xgb_preds.iloc[:, 1:]
        + args.w3 * cat_preds.iloc[:, 1:]
        + args.w4 * rf_preds.iloc[:, 1:]
    )
    submission.to_csv("../../submission/" + args.file, index=False)
