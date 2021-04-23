# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%


path = "../input/predict-credit-card-delinquency/"

train = pd.read_csv(path + "train.csv")
train = train.drop(["index"], axis=1)
train.fillna("NAN", inplace=True)

test = pd.read_csv(path + "test.csv")
test = test.drop(["index"], axis=1)
test.fillna("NAN", inplace=True)

train_ohe = pd.get_dummies(train)
test_ohe = pd.get_dummies(test)

# %%


train_ohe.head()


# %%


train_ohe.info()


# %%


train["credit"].value_counts()


# %%
fig, axes = plt.subplots()
sns.histplot(train["DAYS_EMPLOYED"])
plt.show()
# %%
fig, axes = plt.subplots()
sns.boxplot(x="DAYS_EMPLOYED", data=train)
plt.show()
# %%
plt.figure(figsize=(9, 9))
corr = train.corr()
sns.heatmap(corr, cmap="RdBu")

# %%
train = train.drop("child_num", axis=1)

# %%
columns = ["income_total", "DAYS_BIRTH", "DAYS_EMPLOYED", "family_size", "begin_month"]
train[columns].describe()

# %%
plt.figure(figsize=(20, 15))
for i, name in enumerate(columns):
    plt.subplot(3, 2, i + 1)
    sns.histplot(train[name])
plt.show()
# %%
candidate = ["income_total", "DAYS_EMPLOYED", "family_size"]
for cand in candidate:
    train[cand] = (
        train[cand] - np.min(train[cand]) / np.min(train[cand]) - np.min(train[cand])
    )

# %%
plt.figure(figsize=(20, 15))
for i, name in enumerate(columns):
    plt.subplot(3, 2, i + 1)
    sns.histplot(train[name])
plt.show()

# %%


def remove_outlier(train: pd.DataFrame, column: str) -> pd.DataFrame:
    df = train[column]
    # 1분위수
    quan_25 = np.percentile(df.values, 25)

    # 3분위수
    quan_75 = np.percentile(df.values, 75)

    iqr = quan_75 - quan_25

    lowest = quan_25 - iqr * 1.5
    highest = quan_75 + iqr * 1.5
    outlier_index = df[(df < lowest) | (df > highest)].index
    print("outlier의 수 : ", len(outlier_index))
    train.drop(outlier_index, axis=0, inplace=True)

    return train


# %%

for cand in candidate:
    train = remove_outlier(train, cand)

# %%

train_ohe = pd.get_dummies(train)
# %%
train_ohe.head()
# %%
