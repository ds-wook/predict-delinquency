# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
path = "../input/predict-credit-card-delinquency/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
train.head()
# %%
train.info()
# %%
train.groupby(["gender", "credit"])["credit"].count()
# %%
train.groupby(["car", "credit"])["credit"].count()
# %%
train.groupby(["reality", "credit"])["credit"].count()
# %%
train["edu_type"].value_counts()
# %%
train.groupby(["edu_type", "credit"])["credit"].count()
# %%
train.groupby(["edu_type"])["credit"].mean()
# %%
