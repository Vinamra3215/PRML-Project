import os
import time
import json
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.data.cache import load_features
from src.models.registry import build_pipeline, get_param_grid
from src.evaluation.metrics import evaluate
from src.evaluation.cross_val import stratified_cv, print_cv_results
from src.evaluation.comparison import append_result
from src.utils.logging import (
    init_wandb, log_wandb, finish_wandb, log_experiment_csv, Timer
)


def run_experiment(
    feature_type, model_name, model_params=None,
    reducer_name="none", reducer_params=None,
    cache_dir="data/cache", results_dir="results/metrics",
    cv_folds=5, run_id=None, use_wandb=False,
    output_csv="master.csv", use_grid_search=True,
):
    
    model_params = model_params or {}
    reducer_params = reducer_params or {}
    rid = run_id or f"{model_name}_{feature_type}_{reducer_name}"

    print(f"\n{'=' * 60}")
    print(f"Experiment: {model_name} | Features: {feature_type} | Reducer: {reducer_name}")
    print(f"GridSearch: {'ON' if use_grid_search else 'OFF'}")
    print(f"{'=' * 60}")

    

    return pipeline, test_metrics, y_pred, best_params
