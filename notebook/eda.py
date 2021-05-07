# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
# %%
path = "../input/predict-credit-card-delinquency/"

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
train["Age"] = np.abs(train["DAYS_BIRTH"]) / 360
train["Age"] = train["Age"].astype(int)

# DAYS_EMPLOYED
train["DAYS_EMPLOYED_month"] = np.floor((-train["DAYS_EMPLOYED"]) / 30) - (
    (np.floor((-train["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
train["DAYS_EMPLOYED_week"] = np.floor((-train["DAYS_EMPLOYED"]) / 7) - (
    (np.floor((-train["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
)
train["EMPLOYED"] = train["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
train["EMPLOYED"] = np.abs(train["EMPLOYED"]) / 360

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
test["Age"] = np.abs(test["DAYS_BIRTH"]) / 360
test["Age"] = test["Age"].astype(int)

# DAYS_EMPLOYED
test["DAYS_EMPLOYED_month"] = np.floor((-test["DAYS_EMPLOYED"]) / 30) - (
    (np.floor((-test["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
test["DAYS_EMPLOYED_week"] = np.floor((-test["DAYS_EMPLOYED"]) / 7) - (
    (np.floor((-test["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
)
test["EMPLOYED"] = test["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
test["EMPLOYED"] = np.abs(test["EMPLOYED"]) / 360

# before_EMPLOYED
test["before_EMPLOYED"] = test["DAYS_BIRTH"] - test["DAYS_EMPLOYED"]
test["before_EMPLOYED_month"] = np.floor((-test["before_EMPLOYED"]) / 30) - (
    (np.floor((-test["before_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
test["before_EMPLOYED_week"] = np.floor((-test["before_EMPLOYED"]) / 7) - (
    (np.floor((-test["before_EMPLOYED"]) / 7) / 4).astype(int) * 4
)

train["gender_car_reality"] = (
    train["gender"].astype(str)
    + "_"
    + train["car"].astype(str)
    + "_"
    + train["reality"].astype(str)
)
test["gender_car_reality"] = (
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
    "gender_car_reality",
]

for col in cat_cols:
    target_encoder = LabelEncoder()
    train[col] = target_encoder.fit_transform(train[col])
    test[col] = target_encoder.transform(test[col])

# %%
train.head()
# %%
train.info()
# %%
train["EMPLOYED"].head()
# %%
