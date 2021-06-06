import argparse

from data.dataset import load_dataset
from optim.bayesian import BayesianOptimizer, xgb_objective


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    train, test = load_dataset()

    X = train.drop("credit", axis=1)
    y = train["credit"]
    X_test = test.copy()

    objective = xgb_objective(X, y, X_test, args.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=args.trials)
    bayesian_optim.xgb_save_params(study, args.params)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
