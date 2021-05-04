import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


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
    train["Age"] = np.abs(train["DAYS_BIRTH"]) // 365

    # DAYS_EMPLOYED
    train["DAYS_EMPLOYED_month"] = np.floor((-train["DAYS_EMPLOYED"]) / 30) - (
        (np.floor((-train["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
    )
    train["DAYS_EMPLOYED_week"] = np.floor((-train["DAYS_EMPLOYED"]) / 7) - (
        (np.floor((-train["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
    )
    train["EMPLOYED"] = train["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
    train["EMPLOYED"] = np.abs(train["EMPLOYED"]) // 365

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
    test["Age"] = np.abs(test["DAYS_BIRTH"]) // 365

    # DAYS_EMPLOYED
    test["DAYS_EMPLOYED_month"] = np.floor((-test["DAYS_EMPLOYED"]) / 30) - (
        (np.floor((-test["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
    )
    test["DAYS_EMPLOYED_week"] = np.floor((-test["DAYS_EMPLOYED"]) / 7) - (
        (np.floor((-test["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
    )
    test["EMPLOYED"] = test["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
    test["EMPLOYED"] = np.abs(test["EMPLOYED"]) // 365

    # before_EMPLOYED
    test["before_EMPLOYED"] = test["DAYS_BIRTH"] - test["DAYS_EMPLOYED"]
    test["before_EMPLOYED_month"] = np.floor((-test["before_EMPLOYED"]) / 30) - (
        (np.floor((-test["before_EMPLOYED"]) / 30) / 12).astype(int) * 12
    )
    test["before_EMPLOYED_week"] = np.floor((-test["before_EMPLOYED"]) / 7) - (
        (np.floor((-test["before_EMPLOYED"]) / 7) / 4).astype(int) * 4
    )

    train["income_total_log"] = np.log1p(train["income_total"])
    test["income_total_log"] = np.log1p(test["income_total"])

    cat_cols = [
        "gender",
        "car",
        "reality",
        "income_type",
        "edu_type",
        "family_type",
        "house_type",
        "occyp_type",
    ]

    for col in cat_cols:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(train[col])
        train[col] = label_encoder.transform(train[col])
        test[col] = label_encoder.transform(test[col])

    kmeans_train = train.drop(["credit"], axis=1)
    kmeans = KMeans(n_clusters=35, random_state=42).fit(kmeans_train)
    train["cluster"] = kmeans.predict(kmeans_train)
    train["silhouette_coeff"] = silhouette_samples(kmeans_train, train.cluster)
    test["cluster"] = kmeans.predict(test)
    test["silhouette_coeff"] = silhouette_samples(test, test.cluster)

    train["identity"] = (
        train["gender"].astype(str)
        + train["income_total"].astype(str)
        + train["income_type"].astype(str)
        + train["DAYS_BIRTH"].astype(str)
        + train["DAYS_EMPLOYED"].astype(str)
    )
    id_train = np.unique(train["identity"]).tolist()
    labeling_train = {ide: i for i, ide in enumerate(id_train)}
    train["identity"] = train["identity"].map(labeling_train)

    test["identity"] = (
        test["gender"].astype(str)
        + test["income_total"].astype(str)
        + test["income_type"].astype(str)
        + test["DAYS_BIRTH"].astype(str)
        + test["DAYS_EMPLOYED"].astype(str)
    )
    id_test = np.unique(test["identity"]).tolist()
    labeling_test = {ide: i for i, ide in enumerate(id_test)}
    test["identity"] = train["identity"].map(labeling_test)

    del_cols = [
        "child_num",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
    ]
    train.drop(train.loc[train["family_size"] > 7, "family_size"].index, inplace=True)
    train.drop(del_cols, axis=1, inplace=True)
    test.drop(del_cols, axis=1, inplace=True)

    return train, test
