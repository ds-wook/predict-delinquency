import pandas as pd
import numpy as np


def category_income(data: pd.DataFrame) -> pd.DataFrame:
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
