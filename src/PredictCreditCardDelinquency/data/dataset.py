from typing import Tuple

import numpy as np
import pandas as pd


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "../../input/predict-credit-card-delinquency/"

    train = pd.read_csv(path + "train.csv")
    train = train.drop(["index"], axis=1)
    train.fillna("NAN", inplace=True)

    test = pd.read_csv(path + "test.csv")
    test = test.drop(["index"], axis=1)
    test.fillna("NAN", inplace=True)

    train = train.drop(["child_num"], axis=1)
    test = test.drop(["child_num"], axis=1)
    # DAYS_BIRTH
    train["DAYS_BIRTH_month"] = np.floor((-train["DAYS_BIRTH"]) / 30) - (
        (np.floor((-train["DAYS_BIRTH"]) / 30) / 12).astype(int) * 12
    )
    train["DAYS_BIRTH_week"] = np.floor((-train["DAYS_BIRTH"]) / 7) - (
        (np.floor((-train["DAYS_BIRTH"]) / 7) / 4).astype(int) * 4
    )

    # DAYS_EMPLOYED
    train["DAYS_EMPLOYED_month"] = np.floor((-train["DAYS_EMPLOYED"]) / 30) - (
        (np.floor((-train["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
    )
    train["DAYS_EMPLOYED_week"] = np.floor((-train["DAYS_EMPLOYED"]) / 7) - (
        (np.floor((-train["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
    )

    # before_EMPLOYED
    train["before_EMPLOYED"] = train["DAYS_BIRTH"] - train["DAYS_EMPLOYED"]
    train["before_EMPLOYED_month"] = np.floor((-train["before_EMPLOYED"]) / 30) - (
        (np.floor((-train["before_EMPLOYED"]) / 30) / 12).astype(int) * 12
    )
    train["before_EMPLOYED_week"] = np.floor((-train["before_EMPLOYED"]) / 7) - (
        (np.floor((-train["before_EMPLOYED"]) / 7) / 4).astype(int) * 4
    )

    # DAYS_BIRTH
    test["DAYS_BIRTH_month"] = np.floor((-test["DAYS_BIRTH"]) / 30) - (
        (np.floor((-test["DAYS_BIRTH"]) / 30) / 12).astype(int) * 12
    )
    test["DAYS_BIRTH_week"] = np.floor((-test["DAYS_BIRTH"]) / 7) - (
        (np.floor((-test["DAYS_BIRTH"]) / 7) / 4).astype(int) * 4
    )

    # DAYS_EMPLOYED
    test["DAYS_EMPLOYED_month"] = np.floor((-test["DAYS_EMPLOYED"]) / 30) - (
        (np.floor((-test["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
    )
    test["DAYS_EMPLOYED_week"] = np.floor((-test["DAYS_EMPLOYED"]) / 7) - (
        (np.floor((-test["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
    )

    # before_EMPLOYED
    test["before_EMPLOYED"] = test["DAYS_BIRTH"] - test["DAYS_EMPLOYED"]
    test["before_EMPLOYED_month"] = np.floor((-test["before_EMPLOYED"]) / 30) - (
        (np.floor((-test["before_EMPLOYED"]) / 30) / 12).astype(int) * 12
    )
    test["before_EMPLOYED_week"] = np.floor((-test["before_EMPLOYED"]) / 7) - (
        (np.floor((-test["before_EMPLOYED"]) / 7) / 4).astype(int) * 4
    )

    # pseudo_label = pd.read_csv("../../res/pseudo_lgbm.csv")

    # test["credit"] = [x for x in pseudo_label.credit]
    # train = pd.concat([train, test], axis=0)

    train_ohe = pd.get_dummies(train)
    test_ohe = pd.get_dummies(test)

    # test_ohe.drop(["credit"], axis=1, inplace=True)

    return train_ohe, test_ohe
