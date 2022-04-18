import hydra
import neptune.new as neptune
from omegaconf import DictConfig

from data.dataset import load_dataset
from tuning.boosting import XGBoostTuner


@hydra.main(config_path="../config/tuning/", config_name="xgb.yaml")
def _main(cfg: DictConfig):
    train, test = load_dataset(cfg)
    train_x = train.drop(columns=cfg.dataset.target)

    train_y = train[cfg.dataset.target]
    run = neptune.init(project=cfg.experiment.project, tags=[*cfg.experiment.tags])
    xgb_tuner = XGBoostTuner(train_x, train_y, cfg, run)

    study = xgb_tuner.build_study(cfg.search.verbose)
    xgb_tuner.save_hyperparameters(study)


if __name__ == "__main__":
    _main()
