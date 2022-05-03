import hydra
import neptune.new as neptune
from omegaconf import DictConfig
from sklearn.metrics import log_loss

from data.dataset import load_dataset
from features.build import category_income, kfold_mean_encoding
from models.boosting import LightGBMTrainer


@hydra.main(config_path="../config/training/", config_name="lgb.yaml")
def _main(cfg: DictConfig):
    train, test = load_dataset(cfg)
    train = category_income(train)
    train_x = train.drop(columns=cfg.dataset.target)
    train_y = train[cfg.dataset.target]

    # train_x, test_x = kfold_mean_encoding(
    #     train_x, test_x, train_y, cfg.dataset.cat_features
    # )
    run = neptune.init(
        project=cfg.experiment.project,
        tags=[*cfg.experiment.tags],
        capture_hardware_metrics=False,
    )

    lgb_trainer = LightGBMTrainer(config=cfg, run=run, metric=log_loss)
    lgb_trainer.train(train_x, train_y)
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
