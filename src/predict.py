import hydra
from omegaconf import DictConfig

from data.dataset import load_dataset
from features.build import category_income
from models.infer import load_model, predict


@hydra.main(config_path="../config/", config_name="predict.yaml")
def _main(cfg: DictConfig):
    train, test = load_dataset(cfg)
    train = category_income(train)

    test_x = category_income(test)

    # model load
    lgb_results = load_model(cfg.model.path)

    # infer test
    predict(lgb_results, test_x)


if __name__ == "__main__":
    _main()
