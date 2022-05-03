import warnings

import neptune.new.integrations.lightgbm as nep_lgbm_utils
import neptune.new.integrations.xgboost as nep_xgb_utils
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from neptune.new import Run
from xgboost import XGBClassifier

from models.base import BaseModel

warnings.filterwarnings("ignore")


class LightGBMTrainer(BaseModel):
    def __init__(self, run: Run, **kwargs):
        super().__init__(**kwargs)
        self.run = run

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
    ) -> LGBMClassifier:
        """
        load train model
        """
        neptune_callback = (
            nep_lgbm_utils.NeptuneCallback(run=self.run, base_namespace=f"fold_{fold}")
            if not self.config.model.search
            else self.run
        )

        model = LGBMClassifier(
            random_state=self.config.model.seed, **self.config.model.params
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=self.config.model.metric,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose=self.config.model.verbose,
            callbacks=[neptune_callback],
        )

        if not self.config.model.search:
            # Log summary metadata to the same run under the "lgbm_summary" namespace
            self.run[f"summary/{fold}"] = nep_lgbm_utils.create_booster_summary(
                booster=model,
                log_pickled_booster=False,
                y_pred=model.predict(X_valid),
                y_true=y_valid,
            )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, run: Run, **kwargs):
        super().__init__(**kwargs)
        self.run = run

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
    ) -> CatBoostClassifier:
        """
        load train model
        """
        train_data = Pool(
            data=X_train, label=y_train, cat_features=self.config.dataset.cat_features
        )
        valid_data = Pool(
            data=X_valid, label=y_valid, cat_features=self.config.dataset.cat_features
        )

        model = CatBoostClassifier(
            random_state=self.config.model.seed,
            cat_features=self.config.dataset.cat_features,
            **self.config.model.params,
        )
        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose=self.config.model.verbose,
        )
        if self.config.model.search:
            self.run[f"catboost/fold_{fold}/best_iteration"] = model.best_iteration_
            self.run[f"catboost/fold_{fold}/best_score"] = model.best_score_

        return model


class XGBoostTrainer(BaseModel):
    def __init__(self, run: Run, **kwargs):
        super().__init__(**kwargs)
        self.run = run

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        fold: int,
    ) -> XGBClassifier:
        """
        load train model
        """
        neptune_callback = (
            nep_xgb_utils.NeptuneCallback(
                run=self.run,
                base_namespace=f"fold_{fold}",
                log_tree=[0, 1, 2, 3],
                max_num_features=10,
            )
            if not self.config.model.search
            else self.run
        )

        model = XGBClassifier(
            random_state=self.config.model.seed, **self.config.model.params
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_metric=self.config.model.metric,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
            verbose=self.config.model.verbose,
            callbacks=[neptune_callback],
        )

        return model
