import json
import os
import optuna
from sklearn.model_selection import cross_val_score
from src.models.registry import build_pipeline
from src.data.cache import load_features
from src.utils.logging import init_wandb, log_wandb, finish_wandb


def _knn_params(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 25, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
    }


def _kde_params(trial):
    return {
        "bandwidth": trial.suggest_float("bandwidth", 0.05, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["gaussian", "tophat", "epanechnikov"]),
    }


def _logistic_params(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42,
    }


def _decision_tree_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
    }


def _gradient_boosting_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 30, 200, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "random_state": 42,
    }


def _mlp_params(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_categorical("units", [64, 128, 256, 512])
    return {
        "hidden_layer_sizes": tuple([units] * n_layers),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": "adam",
        "max_iter": 500,
        "early_stopping": True,
        "random_state": 42,
    }


SEARCH_SPACES = {
    "knn": _knn_params,
    "logistic": _logistic_params,
    "decision_tree": _decision_tree_params,
    "gradient_boosting": _gradient_boosting_params,
    "mlp_sklearn": _mlp_params,
    "kde": _kde_params,
}


def create_objective(feature_type, model_name, cache_dir="data/cache",
                     cv_folds=5, reducer_name="none", reducer_params=None,
                     use_wandb=False):
    X_train, y_train = load_features(cache_dir, feature_type, "train")

    param_fn = SEARCH_SPACES.get(model_name)
    if param_fn is None:
        raise ValueError(f"No search space defined for '{model_name}'. "
                         f"Available: {list(SEARCH_SPACES.keys())}")

    def objective(trial):
        params = param_fn(trial)
        pipeline = build_pipeline(model_name, params, reducer_name, reducer_params)
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv_folds, scoring="f1_macro")
        mean_f1 = scores.mean()

        if use_wandb:
            log_wandb({
                "trial_number": trial.number,
                "trial_f1": mean_f1,
                "trial_f1_std": scores.std(),
                **{f"param_{k}": v for k, v in params.items()
                   if not isinstance(v, (tuple, list))},
            })

        return mean_f1

    return objective, X_train, y_train
