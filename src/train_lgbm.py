import hydra
from omegaconf import DictConfig
from sklearn.metrics import log_loss

from data.dataset import load_dataset
from models.boosting import LightGBMTrainer


@hydra.main(config_path="../config/training/", config_name="lgb.yaml")
def _main(cfg: DictConfig):
    train, test = load_dataset(cfg)

    train_x = train.drop(columns=cfg.dataset.target)
    train_y = train[cfg.dataset.target]

    lgb_trainer = LightGBMTrainer(config=cfg, metric=log_loss)
    lgb_trainer.train(train_x, train_y)
    lgb_trainer.save_model()


if __name__ == "__main__":
    _main()
