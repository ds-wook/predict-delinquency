# %%
import math

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder

# %%
path = "../input/predict-credit-card-delinquency/"

train = pd.read_csv(path + "train.csv")
train = train.drop(["index"], axis=1)
train.fillna("NAN", inplace=True)

test = pd.read_csv(path + "test.csv")
test = test.drop(["index"], axis=1)
test.fillna("NAN", inplace=True)

# %%
# absolute
train["DAYS_EMPLOYED"] = train["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
train["DAYS_EMPLOYED"] = np.abs(train["DAYS_EMPLOYED"])
test["DAYS_EMPLOYED"] = test["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
test["DAYS_EMPLOYED"] = np.abs(test["DAYS_EMPLOYED"])
train["DAYS_BIRTH"] = np.abs(train["DAYS_BIRTH"])
test["DAYS_BIRTH"] = np.abs(test["DAYS_BIRTH"])


# DAYS_BIRTH
train["DAYS_BIRTH_month"] = np.floor(train["DAYS_BIRTH"] / 30) - (
    (np.floor(train["DAYS_BIRTH"] / 30) / 12).astype(int) * 12
)
train["DAYS_BIRTH_week"] = np.floor(train["DAYS_BIRTH"] / 7) - (
    (np.floor(train["DAYS_BIRTH"] / 7) / 4).astype(int) * 4
)
test["DAYS_BIRTH_month"] = np.floor(test["DAYS_BIRTH"] / 30) - (
    (np.floor(test["DAYS_BIRTH"] / 30) / 12).astype(int) * 12
)
test["DAYS_BIRTH_week"] = np.floor(test["DAYS_BIRTH"] / 7) - (
    (np.floor(test["DAYS_BIRTH"] / 7) / 4).astype(int) * 4
)

# percentage
train["DAYS_EMPLOYED_PERC"] = train["DAYS_EMPLOYED"] / train["DAYS_BIRTH"]
test["DAYS_EMPLOYED_PERC"] = test["DAYS_EMPLOYED"] / test["DAYS_BIRTH"]

# Age
train["Age"] = np.abs(train["DAYS_BIRTH"]) // 365
test["Age"] = np.abs(test["DAYS_BIRTH"]) // 365

# DAYS_EMPLOYED
train["DAYS_EMPLOYED_month"] = np.floor(np.abs(train["DAYS_EMPLOYED"]) / 30) - (
    (np.floor((-train["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
train["DAYS_EMPLOYED_week"] = np.floor(np.abs(train["DAYS_EMPLOYED"]) / 7) - (
    (np.floor((-train["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
)
test["DAYS_EMPLOYED_month"] = np.floor(np.abs(test["DAYS_EMPLOYED"]) / 30) - (
    (np.floor((-test["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
test["DAYS_EMPLOYED_week"] = np.floor(np.abs(test["DAYS_EMPLOYED"]) / 7) - (
    (np.floor((-test["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
)

# EMPLOYED
train["EMPLOYED"] = np.abs(train["DAYS_EMPLOYED"]) // 365
test["EMPLOYED"] = np.abs(test["DAYS_EMPLOYED"]) // 365

# before_EMPLOYED
train["before_EMPLOYED"] = train["DAYS_BIRTH"] - train["DAYS_EMPLOYED"]
train["before_EMPLOYED_month"] = np.floor(np.abs(train["before_EMPLOYED"]) / 30) - (
    (np.floor((-train["before_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
train["before_EMPLOYED_week"] = np.floor(np.abs(train["before_EMPLOYED"]) / 7) - (
    (np.floor((-train["before_EMPLOYED"]) / 7) / 4).astype(int) * 4
)
test["before_EMPLOYED"] = test["DAYS_BIRTH"] - test["DAYS_EMPLOYED"]
test["before_EMPLOYED_month"] = np.floor(np.abs(test["before_EMPLOYED"]) / 30) - (
    (np.floor((-test["before_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
test["before_EMPLOYED_week"] = np.floor(np.abs(test["before_EMPLOYED"]) / 7) - (
    (np.floor((-test["before_EMPLOYED"]) / 7) / 4).astype(int) * 4
)


# gender_car_reality
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

# income_total_log
train["income_total_log"] = np.log1p(train["income_total"])
test["income_total_log"] = np.log1p(test["income_total"])

del_cols = [
    "email",
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
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(train[col])
    train[col] = label_encoder.transform(train[col])
    test[col] = label_encoder.transform(test[col])
# %%
kmeans_train = train.drop(["credit"], axis=1)
kmeans = KMeans(n_clusters=35, random_state=42).fit(kmeans_train)
train["kmeans_clusters"] = kmeans.predict(kmeans_train)
test["kmeans_clusters"] = kmeans.predict(test)

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
kmeans_train = scaler.fit_transform(kmeans_train)

pca = PCA(n_components=2)
principal_comp = pca.fit_transform(kmeans_train)
pca_df = pd.DataFrame(data=principal_comp, columns=["pca1", "pca2"])
pca_df = pd.concat([pca_df, pd.DataFrame({"cluster": train.kmeans_clusters})], axis=1)

# %%
fig, ax = plt.subplots(figsize=(18, 15))
sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=pca_df)
plt.show()
# %%
fig, ax = plt.subplots(figsize=(18, 15))
sns.histplot(train["Age"], kde=True)
plt.show()
# %%
fig, ax = plt.subplots(figsize=(18, 15))
sns.histplot(train["DAYS_EMPLOYED_week"], kde=True)
plt.show()
# %%
train["EMPLOYED"]
# %%
train.info()
# %%
train["before_EMPLOYED"]
# %%
train["DAYS_EMPLOYED_week"].head()
# %%
fig, ax = plt.subplots(figsize=(18, 15))
sns.heatmap(train.corr())
plt.show()
# %%
