from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def stratified_kfold_rf(
    params: Dict[str, int],
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> np.ndarray:

    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    rf_oof = np.zeros((X.shape[0], 3))
    rf_preds = np.zeros((X_test.shape[0], 3))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = RandomForestClassifier(**params)

        model.fit(X_train, y_train)

        rf_oof[valid_idx] = model.predict_proba(X_valid)
        rf_preds += model.predict_proba(X_test) / n_fold
        print(f"Log Loss Score: {log_loss(y_valid, rf_oof[valid_idx]):.5f}")

    log_score = log_loss(y, rf_oof)
    print(f"Log Loss Score: {log_score:.5f}")

    return rf_preds


def stratified_kfold_et(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:

    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    et_oof = np.zeros((X.shape[0], 3))
    et_preds = np.zeros((X_test.shape[0], 3))

    for fold, (train_idx, valid_idx) in enumerate(splits):
        print(f"============ Fold {fold} ============\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_features=20,
            criterion="entropy",
            min_samples_split=5,
            max_depth=50,
            min_samples_leaf=5,
        )

        model.fit(X_train, y_train)

        et_oof[valid_idx] = model.predict_proba(X_valid)
        et_preds += model.predict_proba(X_test) / n_fold
        print(f"Log Loss Score: {log_loss(y_valid, et_oof[valid_idx]):.5f}")

    log_score = log_loss(y, et_oof)
    print(f"Log Loss Score: {log_score:.5f}")

    return et_oof, et_preds
