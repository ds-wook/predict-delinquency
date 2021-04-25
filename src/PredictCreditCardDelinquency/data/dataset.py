from typing import Tuple

import numpy as np
import pandas as pd


def remove_outlier(data: pd.DataFrame, column: str) -> pd.DataFrame:
    df = data[column]
    # 1분위수
    quan_25 = np.percentile(df.values, 25)

    # 3분위수
    quan_75 = np.percentile(df.values, 75)

    iqr = quan_75 - quan_25

    lowest = quan_25 - iqr * 1.5
    highest = quan_75 + iqr * 1.5
    outlier_index = df[(df < lowest) | (df > highest)].index
    data.drop(outlier_index, axis=0, inplace=True)

    return data


def load_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "../../input/predict-credit-card-delinquency/"

    train = pd.read_csv(path + "train.csv")
    train = train.drop(["index"], axis=1)
    train.fillna("NAN", inplace=True)

    test = pd.read_csv(path + "test.csv")
    test = test.drop(["index"], axis=1)
    test.fillna("NAN", inplace=True)

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
