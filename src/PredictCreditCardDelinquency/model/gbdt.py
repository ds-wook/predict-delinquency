from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, plot_importance
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold


def stratified_kfold_lgbm(
    params: Dict[str, int],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> np.ndarray:

    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    lgb_oof = np.zeros((X.shape[0], 3))
    lgb_preds = np.zeros((X_test.shape[0], 3))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        pre_model = LGBMClassifier(**params)

        pre_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=100,
        )
        params2 = params.copy()
        params2["learning_rate"] = params["learning_rate"] * 0.1

        model = LGBMClassifier(**params2)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=100,
            init_model=pre_model,
        )

        lgb_oof[valid_idx] = model.predict_proba(X_valid)
        lgb_preds += model.predict_proba(X_test) / n_fold

    fig, ax = plt.subplots(figsize=(20, 14))
    plot_importance(model, ax=ax, max_num_features=len(X_test.columns))
    plt.savefig("../../graph/lgbm_import.png")
    log_score = log_loss(y, lgb_oof)
    print(f"Log Loss Score: {log_score:.5f}")

    return lgb_preds


def group_kfold_lgbm(
    params: Dict[str, int],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    group: pd.Series,
) -> np.ndarray:

    folds = GroupKFold(n_splits=n_fold)
    splits = folds.split(X, y, groups=group)
    lgb_oof = np.zeros((X.shape[0], 3))
    lgb_preds = np.zeros((X_test.shape[0], 3))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"============ Fold {fold} ============")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=100,
        )

        lgb_oof[valid_idx] = model.predict_proba(X_valid)
        lgb_preds += model.predict_proba(X_test) / n_fold

    log_score = log_loss(y, lgb_oof)
    print(f"Score: {log_score}")
    return lgb_preds
