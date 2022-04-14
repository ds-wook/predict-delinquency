import hydra
import neptune.new as neptune
from omegaconf import DictConfig
from sklearn.metrics import log_loss

from data.dataset import load_dataset
from features.build import category_income
from inference.infer import load_model, predict
from models.boosting import CatBoostTrainer


@hydra.main(config_path="../config/modeling/", config_name="cb.yaml")
def _main(cfg: DictConfig):
    train, test = load_dataset(cfg)
    train = category_income(train)
    train_x = train.drop(columns=cfg.dataset.target)
    test_x = category_income(test)
    train_y = train[cfg.dataset.target]

    run = neptune.init(
        project=cfg.experiment.project,
        tags=list(cfg.experiment.tags),
        capture_hardware_metrics=False,
    )

    cb_trainer = CatBoostTrainer(config=cfg, run=run, metric=log_loss)
    cb_trainer.train(train_x, train_y)
    cb_trainer.save_model()

    # model load
    cb_results = load_model(cfg.model.path)

    # infer test
    predict(cb_results, test_x)


if __name__ == "__main__":
    _main()
