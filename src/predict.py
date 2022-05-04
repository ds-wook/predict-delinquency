from pathlib import Path

import hydra
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from data.dataset import load_dataset
from features.build import category_income
from models.infer import load_model, predict


@hydra.main(config_path="../config/", config_name="predict.yaml")
def _main(cfg: DictConfig):
    train, test_x = load_dataset(cfg)
    train = category_income(train)
    path = Path(get_original_cwd())
    submit = pd.read_csv(path / cfg.dataset.path / cfg.dataset.submit)
    # model load
    lgb_results = load_model(cfg.model.lightgbm)

    # infer test
    pred = predict(lgb_results, test_x)
    submit.iloc[:, 1:] = pred
    submit.to_csv(path / cfg.output.name, index=False)


if __name__ == "__main__":
    _main()
