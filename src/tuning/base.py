import logging
import warnings
from abc import abstractclassmethod, ABCMeta
from functools import partial
from pathlib import Path
from typing import Optional

import neptune.new.integrations.optuna as optuna_utils
import optuna
from hydra.utils import get_original_cwd
from neptune.new import Run
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import HyperbandPruner
from optuna.samplers import BaseSampler, CmaEsSampler, RandomSampler, TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial

warnings.filterwarnings("ignore")


class BaseTuner(metaclass=ABCMeta):
    def __init__(self, config: DictConfig, run: Optional[Run] = None):
        self.config = config
        self.run = run

    @abstractclassmethod
    def _objective(self, trial: FrozenTrial, config: DictConfig) -> float:
        """
        Objective function

        Args:
            trial: trial object
            config: config object
        Returns:
            metric score
        """
        raise NotImplementedError

    def build_study(self, verbose: bool = False) -> Study:
        """
        Build study

        Args:
            study_name: study name
        Returns:
            study
        """
        try:
            # define study
            neptune_callback = optuna_utils.NeptuneCallback(
                self.run,
                plots_update_freq=1,
                log_plot_slice=False,
                log_plot_contour=False,
            )
            study = optuna.create_study(
                study_name=self.config.search.study_name,
                direction=self.config.search.direction,
                sampler=create_sampler(
                    mode=self.config.search.mode, seed=self.config.search.seed
                ),
                pruner=HyperbandPruner(
                    min_resource=self.config.search.min_resource,
                    max_resource=self.config.search.max_resource,
                    reduction_factor=self.config.search.reduction_factor,
                ),
            )

            # define callbacks
            objective = partial(self._objective, config=self.config)

            # optimize
            study.optimize(
                objective,
                n_trials=self.config.search.n_trials,
                callbacks=[neptune_callback],
            )
            self.run.stop()

        except TypeError:
            # define study
            study = optuna.create_study(
                study_name=self.config.search.study_name,
                direction=self.config.search.direction,
                sampler=create_sampler(
                    mode=self.config.search.mode, seed=self.config.search.seed
                ),
                pruner=HyperbandPruner(
                    min_resource=self.config.search.min_resource,
                    max_resource=self.config.search.max_resource,
                    reduction_factor=self.config.search.reduction_factor,
                ),
            )

            # define callbacks
            objective = partial(self._objective, config=self.config)

            # optimize
            study.optimize(objective, n_trials=self.config.search.n_trials)

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
        update_params = OmegaConf.load(path)

        update_params.model.params.update(study.best_trial.params)

        OmegaConf.save(update_params, path)

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


def create_sampler(mode: str, seed: int) -> BaseSampler:
    """
    Create sampler

    Args:
        sampler_mode: sampler mode
        seed: seed
    Returns:
        BaseSampler: sampler
    """
    if mode == "random":
        sampler = RandomSampler(seed=seed)
    elif mode == "tpe":
        sampler = TPESampler(seed=seed)
    elif mode == "cma":
        sampler = CmaEsSampler(seed=seed)
    else:
        raise ValueError(f"Unknown sampler mode: {mode}")

    return sampler
