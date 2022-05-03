from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import KFold
from tqdm import tqdm


def category_income(data: DataFrame) -> DataFrame:
    """
    categorize income

    Args:
        data: data
    Returns:
        dataframe
    """
    data["income_total"] = data["income_total"] / 10000
    conditions = [
        (data["income_total"].le(18)),
        (data["income_total"].gt(18) & data["income_total"].le(33)),
        (data["income_total"].gt(33) & data["income_total"].le(49)),
        (data["income_total"].gt(49) & data["income_total"].le(64)),
        (data["income_total"].gt(64) & data["income_total"].le(80)),
        (data["income_total"].gt(80) & data["income_total"].le(95)),
        (data["income_total"].gt(95) & data["income_total"].le(111)),
        (data["income_total"].gt(111) & data["income_total"].le(126)),
        (data["income_total"].gt(126) & data["income_total"].le(142)),
        (data["income_total"].gt(142)),
    ]
    choices = [i for i in range(10)]

    data["income_total"] = np.select(conditions, choices)
    return data


def kfold_mean_encoding(
    train_x: DataFrame,
    test_x: DataFrame,
    train_y: Series,
    cat_features: List[str],
) -> Tuple[DataFrame, DataFrame]:
    """
    K-fold mean encoding

    Args:
        train_x: train data
        test_x: test data
        train_y: train label
        cat_features: categorical features
    Returns:
        encoded train data, encoded test data
    """
    for c in tqdm(cat_features):
        data_tmp = pd.DataFrame({c: train_x[c], "target": train_y})
        target_mean = data_tmp.groupby(c)["target"].mean()

        # 테스트 데이터의 카테고리 변경
        test_x[c] = test_x[c].map(target_mean)

        # 학습 데이터 변환 후 값을 저장하는 배열 준비
        tmp = np.repeat(np.nan, train_x.shape[0])

        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        for train_idx, valid_idx in kf.split(train_x):
            # out of fold 로 각 범주형 목적변수 평균 계산
            target_mean = data_tmp.iloc[train_idx].groupby(c)["target"].mean()
            # 변환 후의 값을 날짜 배열에 저장
            tmp[valid_idx] = train_x[c].iloc[valid_idx].map(target_mean)

        train_x[c] = tmp

    return train_x, test_x
