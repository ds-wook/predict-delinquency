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
print(train.shape)
train.drop_duplicates(keep=False, inplace=True)
print(train.shape)
# %%
train.info()
# %%
train.fillna("NAN", inplace=True)

test = pd.read_csv(path + "test.csv")
test = test.drop(["index"], axis=1)
test.fillna("NAN", inplace=True)
# %%
train["identity"] = (
    train["gender"].astype(str)
    + train["income_total"].astype(str)
    + train["income_type"].astype(str)
    + train["DAYS_BIRTH"].astype(str)
    + train["DAYS_EMPLOYED"].astype(str)
)


test["identity"] = (
    test["gender"].astype(str)
    + test["income_total"].astype(str)
    + test["income_type"].astype(str)
    + test["DAYS_BIRTH"].astype(str)
    + test["DAYS_EMPLOYED"].astype(str)
)
# %%
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
test["Age"] = np.abs(test["DAYS_BIRTH"]) / 365
test["Age"] = test["Age"].astype(int)

# DAYS_EMPLOYED
test["DAYS_EMPLOYED_month"] = np.floor((-test["DAYS_EMPLOYED"]) / 30) - (
    (np.floor((-test["DAYS_EMPLOYED"]) / 30) / 12).astype(int) * 12
)
test["DAYS_EMPLOYED_week"] = np.floor((-test["DAYS_EMPLOYED"]) / 7) - (
    (np.floor((-test["DAYS_EMPLOYED"]) / 7) / 4).astype(int) * 4
)
test["EMPLOYED"] = test["DAYS_EMPLOYED"].map(lambda x: 0 if x > 0 else x)
test["EMPLOYED"] = np.abs(test["EMPLOYED"]) / 365

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
# %%
train.head()
# %%
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
train.info()
# %%
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
train.head()
# %%
len(train["occyp_type"].value_counts().values)

# %%
kmeans_train = train.drop(["credit", "identity"], axis=1)
kmeans = KMeans(n_clusters=8724, random_state=42).fit(kmeans_train)
train["cluster"] = kmeans.predict(kmeans_train)
train["silhouette_coeff"] = silhouette_samples(kmeans_train, train.cluster)
# %%
train.loc[train["identity"] == "F157500.0State servant-12676-1350", "cluster"]
# %%
train["identity"].value_counts()
# %%


def visualize_silhouette(cluster_lists, X_features):
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):

        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산.
        clusterer = KMeans(n_clusters=n_cluster, random_state=42)
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title(
            "Number of Cluster : " + str(n_cluster) + "\n"
            "Silhouette Score :" + str(round(sil_avg, 3))
        )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels == i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_sil_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")


# %%
visualize_silhouette([2, 3, 4, 35, 40], kmeans_train)
# %%
ides = np.unique(train["identity"]).tolist()

labeling = {ide: i for i, ide in enumerate(ides)}
train["identity"] = train["identity"].map(labeling)
# %%
train.info()
# %%
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(train.corr())
plt.show()
# %%
