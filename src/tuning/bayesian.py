import logging
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import neptune.new.integrations.optuna as optuna_utils
import optuna
import pandas as pd
import yaml
from hydra.utils import get_original_cwd
from neptune.new import Run
from omegaconf import DictConfig, open_dict
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import log_loss

from models.boosting import LightGBMTrainer, XGBoostTrainer

warnings.filterwarnings("ignore")


class BayesianSearch:
    def __init__(
        self,
        config: DictConfig,
        objective_function: Callable[[Trial], Union[float, Sequence[float]]],
        run: Optional[Run] = None,
    ):
        self.config = config
        self.objective_function = objective_function
        self.run = run

    def build_study(self, verbose: bool = False) -> Study:
        """
        Build study
        Args:
            study_name: study name
        Returns:
            study
        """
        try:
            neptune_callback = optuna_utils.NeptuneCallback(
                self.run,
                plots_update_freq=1,
                log_plot_slice=False,
                log_plot_contour=False,
            )
            study = optuna.create_study(
                study_name=self.config.search.study_name,
                direction=self.config.search.direction,
                sampler=TPESampler(seed=self.config.search.seed),
                pruner=HyperbandPruner(
                    min_resource=self.config.search.min_resource,
                    max_resource=self.config.search.max_resource,
                    reduction_factor=self.config.search.reduction_factor,
                ),
            )
            study.optimize(
                self.objective_function,
                n_trials=self.config.search.n_trials,
                callbacks=[neptune_callback],
            )
            self.run.stop()

        except TypeError:
            study = optuna.create_study(
                study_name=self.config.search.study_name,
                direction=self.config.search.direction,
                sampler=TPESampler(seed=self.config.search.seed),
                pruner=HyperbandPruner(
                    min_resource=self.config.search.min_resource,
                    max_resource=self.config.search.max_resource,
                    reduction_factor=self.config.search.reduction_factor,
                ),
            )
            study.optimize(
                self.objective_function, n_trials=self.config.search.n_trials
            )

        if verbose:
            self.display_study(study)

        return study

    def save_hyperparameters(self, study: Study) -> None:
        """
        Save best hyperparameters to yaml file
        Args:
            study: study best hyperparameter object.
        """
        path = Path(get_original_cwd()) / self.config.search.path_name

        with open(path, "r") as f:
            update_params = yaml.load(f, Loader=yaml.FullLoader)

        update_params.model.params.update(study.best_trial.params)

        path = Path(get_original_cwd()) / self.config.search.params_name
        with open(path, "w") as f:
            yaml.dump(update_params, f)

    @staticmethod
    def display_study(study: Study) -> None:
        """
        Display best metric score and hyperparameters
        Args:
            study: study best hyperparameter object.
        """
        logging.info("Best trial:")
        trial = study.best_trial
        logging.info(f"  Value: {trial.value}")
        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info(f"    '{key}': {value},")


def xgb_objective(
    trial: FrozenTrial,
    config: DictConfig,
    train_x: pd.DataFrame,
    train_y: pd.Series,
) -> float:
    """
    Objective function for XGBoost
    Args:
        trial: trial object
        config: config object
        train_x: train x dataframe
        train_y: train y series
    Returns:
        float: objective function value
    """
    # trial parameters
    params = {
        "max_depth": trial.suggest_int("max_depth", *config.search.max_depth),
        "subsample": trial.suggest_float("subsample", *config.search.subsample),
        "gamma": trial.suggest_float("gamma", *config.search.gamma),
        "reg_alpha": trial.suggest_float("reg_alpha", *config.search.reg_alpha),
        "reg_lambda": trial.suggest_float("reg_lambda", *config.search.reg_lambda),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *config.search.colsample_bytree
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *config.search.min_child_weight
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *config.search.learning_rate
        ),
    }

    # config update
    with open_dict(config.model):
        config.model.params.update(params)

    # search parameters
    pruning_callback = XGBoostPruningCallback(trial, "validation_1-logloss")

    xgb_trainer = XGBoostTrainer(
        run=pruning_callback, search=True, config=config, metric=log_loss
    )
    result = xgb_trainer.train(train_x, train_y)
    score = log_loss(train_y.to_numpy(), result.oof_preds)

    return score


def lgbm_objective(
    trial: FrozenTrial,
    config: DictConfig,
    train_x: pd.DataFrame,
    train_y: pd.Series,
) -> float:
    """
    Objective function for XGBoost
    Args:
        trial: trial object
        config: config object
        train_x: train x dataframe
        train_y: train y series
    Returns:
        float: objective function value
    """
    # trial parameters
    params = {
        "max_depth": trial.suggest_int("max_depth", *config.search.max_depth),
        "subsample": trial.suggest_float("subsample", *config.search.subsample),
        "gamma": trial.suggest_float("gamma", *config.search.gamma),
        "reg_alpha": trial.suggest_float("reg_alpha", *config.search.reg_alpha),
        "reg_lambda": trial.suggest_float("reg_lambda", *config.search.reg_lambda),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", *config.search.colsample_bytree
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", *config.search.min_child_weight
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", *config.search.learning_rate
        ),
    }

    # config update
    with open_dict(config.model):
        config.model.params.update(params)

    # search parameters
    pruning_callback = LightGBMPruningCallback(trial, "multi_logloss", valid_name="valid_1")

    lgbm_trainer = LightGBMTrainer(
        run=pruning_callback, search=True, config=config, metric=log_loss
    )
    result = lgbm_trainer.train(train_x, train_y)
    score = log_loss(train_y.to_numpy(), result.oof_preds)

    return score
