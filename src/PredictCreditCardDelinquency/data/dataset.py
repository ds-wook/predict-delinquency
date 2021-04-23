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

    train_ohe = pd.get_dummies(train)
    test_ohe = pd.get_dummies(test)

    candidate = ["income_total", "DAYS_EMPLOYED", "family_size"]

    for cand in candidate:
        train_ohe = remove_outlier(train_ohe, cand)

    return train_ohe, test_ohe
