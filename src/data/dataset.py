import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_dataset(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(get_original_cwd()) / config.dataset.path
    train = pd.read_csv(path / config.dataset.train)
    train = train.drop(["index"], axis=1)
    train.fillna("NAN", inplace=True)

    test = pd.read_csv(path / config.dataset.test)
    test = test.drop(["index"], axis=1)
    test.fillna("NAN", inplace=True)

    # absolute
    train["DAYS_EMPLOYED"] = train["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
    train["DAYS_EMPLOYED"] = np.abs(train["DAYS_EMPLOYED"])
    test["DAYS_EMPLOYED"] = test["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
    test["DAYS_EMPLOYED"] = np.abs(test["DAYS_EMPLOYED"])
    train["DAYS_BIRTH"] = np.abs(train["DAYS_BIRTH"])
    test["DAYS_BIRTH"] = np.abs(test["DAYS_BIRTH"])
    train["begin_month"] = np.abs(train["begin_month"]).astype(int)
    test["begin_month"] = np.abs(test["begin_month"]).astype(int)

    # DAYS_BIRTH
    train["DAYS_BIRTH_month"] = np.floor(train["DAYS_BIRTH"] / 30) - (
        (np.floor(train["DAYS_BIRTH"] / 30) / 12).astype(int) * 12
    )
    train["DAYS_BIRTH_month"] = train["DAYS_BIRTH_month"].astype(int)
    train["DAYS_BIRTH_week"] = np.floor(train["DAYS_BIRTH"] / 7) - (
        (np.floor(train["DAYS_BIRTH"] / 7) / 4).astype(int) * 4
    )
    train["DAYS_BIRTH_week"] = train["DAYS_BIRTH_week"].astype(int)
    test["DAYS_BIRTH_month"] = np.floor(test["DAYS_BIRTH"] / 30) - (
        (np.floor(test["DAYS_BIRTH"] / 30) / 12).astype(int) * 12
    )
    test["DAYS_BIRTH_month"] = test["DAYS_BIRTH_month"].astype(int)
    test["DAYS_BIRTH_week"] = np.floor(test["DAYS_BIRTH"] / 7) - (
        (np.floor(test["DAYS_BIRTH"] / 7) / 4).astype(int) * 4
    )
    test["DAYS_BIRTH_week"] = test["DAYS_BIRTH_week"].astype(int)

    # Age
    train["Age"] = np.abs(train["DAYS_BIRTH"]) // 360
    test["Age"] = np.abs(test["DAYS_BIRTH"]) // 360

    # DAYS_EMPLOYED
    train["DAYS_EMPLOYED_month"] = np.floor(train["DAYS_EMPLOYED"] / 30) - (
        (np.floor(train["DAYS_EMPLOYED"] / 30) / 12).astype(int) * 12
    )
    train["DAYS_EMPLOYED_month"] = train["DAYS_EMPLOYED_month"].astype(int)
    train["DAYS_EMPLOYED_week"] = np.floor(train["DAYS_EMPLOYED"] / 7) - (
        (np.floor(train["DAYS_EMPLOYED"] / 7) / 4).astype(int) * 4
    )
    train["DAYS_EMPLOYED_week"] = train["DAYS_EMPLOYED_week"].astype(int)
    test["DAYS_EMPLOYED_month"] = np.floor(test["DAYS_EMPLOYED"] / 30) - (
        (np.floor(test["DAYS_EMPLOYED"] / 30) / 12).astype(int) * 12
    )
    test["DAYS_EMPLOYED_month"] = test["DAYS_EMPLOYED_month"].astype(int)
    test["DAYS_EMPLOYED_week"] = np.floor(test["DAYS_EMPLOYED"] / 7) - (
        (np.floor(test["DAYS_EMPLOYED"] / 7) / 4).astype(int) * 4
    )
    test["DAYS_EMPLOYED_week"] = test["DAYS_EMPLOYED_week"].astype(int)

    # EMPLOYED
    train["EMPLOYED"] = train["DAYS_EMPLOYED"] / 360
    test["EMPLOYED"] = test["DAYS_EMPLOYED"] / 360

    # before_EMPLOYED
    train["before_EMPLOYED"] = train["DAYS_BIRTH"] - train["DAYS_EMPLOYED"]
    train["before_EMPLOYED_month"] = np.floor(train["before_EMPLOYED"] / 30) - (
        (np.floor(train["before_EMPLOYED"] / 30) / 12).astype(int) * 12
    )
    train["before_EMPLOYED_month"] = train["before_EMPLOYED_month"].astype(int)
    train["before_EMPLOYED_week"] = np.floor(train["before_EMPLOYED"] / 7) - (
        (np.floor(train["before_EMPLOYED"] / 7) / 4).astype(int) * 4
    )
    train["before_EMPLOYED_week"] = train["before_EMPLOYED_week"].astype(int)
    test["before_EMPLOYED"] = test["DAYS_BIRTH"] - test["DAYS_EMPLOYED"]
    test["before_EMPLOYED_month"] = np.floor(test["before_EMPLOYED"] / 30) - (
        (np.floor(test["before_EMPLOYED"] / 30) / 12).astype(int) * 12
    )
    test["before_EMPLOYED_month"] = test["before_EMPLOYED_month"].astype(int)
    test["before_EMPLOYED_week"] = np.floor(test["before_EMPLOYED"] / 7) - (
        (np.floor(test["before_EMPLOYED"] / 7) / 4).astype(int) * 4
    )
    test["before_EMPLOYED_week"] = test["before_EMPLOYED_week"].astype(int)

    # gender_car_reality
    train["user_code"] = (
        train["gender"].astype(str)
        + "_"
        + train["car"].astype(str)
        + "_"
        + train["reality"].astype(str)
    )
    test["user_code"] = (
        test["gender"].astype(str)
        + "_"
        + test["car"].astype(str)
        + "_"
        + test["reality"].astype(str)
    )

    del_cols = [
        "gender",
        "car",
        "reality",
        "email",
        "child_num",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ]
    train.drop(train.loc[train["family_size"] > 7, "family_size"].index, inplace=True)
    train.drop(del_cols, axis=1, inplace=True)
    test.drop(del_cols, axis=1, inplace=True)

    cat_cols = [
        "income_type",
        "edu_type",
        "family_type",
        "house_type",
        "occyp_type",
        "user_code",
    ]

    for col in tqdm(cat_cols):
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(train[col])
        train[col] = label_encoder.transform(train[col])
        test[col] = label_encoder.transform(test[col])

    return train, test
