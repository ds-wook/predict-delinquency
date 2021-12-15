import argparse

from data.dataset import category_income, load_dataset
from optim.bayesian import BayesianOptimizer, cat_objective


def define_argparser():
    parse = argparse.ArgumentParser("Optimize")
    parse.add_argument("--fold", type=int, default=10)
    parse.add_argument("--trials", type=int, default=360)
    parse.add_argument("--params", type=str, default="params.pkl")
    parse.add_argument(
        "--path", type=str, default="../../input/predict-credit-card-delinquency/"
    )
    args = parse.parse_args()
    return args


def _main(args: argparse.Namespace):
    path = args.path
    train, test = load_dataset(path)
    # income_total
    train = category_income(train)
    test = category_income(test)

    X = train.drop("credit", axis=1)
    y = train["credit"]
    cat_cols = [c for c in X.columns if X[c].dtypes == "int64"]
    X_test = test.copy()

    objective = cat_objective(X, y, X_test, cat_cols, args.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=args.trials)
    bayesian_optim.cat_save_params(study, args.params, cat_cols)


if __name__ == "__main__":
    args = define_argparser()
    _main(args)
