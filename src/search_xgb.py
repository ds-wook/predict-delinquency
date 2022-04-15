from functools import partial

import hydra
import neptune.new as neptune
from omegaconf import DictConfig

from data.dataset import load_dataset
from tuning.bayesian import BayesianSearch, xgb_objective


@hydra.main(config_path="../config/tuning/", config_name="xgb.yaml")
def _main(cfg: DictConfig):
    train, test = load_dataset(cfg)
    train_x = train.drop(columns=cfg.dataset.target)

    train_y = train[cfg.dataset.target]
    run = neptune.init(project=cfg.experiment.project, tags=[*cfg.experiment.tags])
    objective = partial(xgb_objective, config=cfg, train_x=train_x, train_y=train_y)
    bayesian_search = BayesianSearch(config=cfg, objective_function=objective, run=run)
    study = bayesian_search.build_study(cfg.search.verbose)
    bayesian_search.save_hyperparameters(study)


if __name__ == "__main__":
    _main()
