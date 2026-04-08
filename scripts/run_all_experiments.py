import sys
sys.path.insert(0, ".")
import os
import argparse
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.utils.logging import init_wandb, log_wandb, finish_wandb, log_experiment_csv
from src.data.cache import load_features
from src.models.registry import build_pipeline, get_param_grid
from src.evaluation.metrics import evaluate
from src.evaluation.comparison import append_result
from src.analysis.model_insights import generate_auto_summary, print_model_ranking
from sklearn.model_selection import GridSearchCV

seed_everything(42)

CACHE_DIR = "data/cache"
RESULTS_DIR = "results/metrics"
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_TYPES = ["histogram", "hog", "lbp", "glcm", "fused"]

MODELS = [
    "knn",
    "logistic",
    "naive_bayes",
    "decision_tree",
    "perceptron",
    "mlp_sklearn",
    "kde",
]

PCA_COMPONENTS = 200
CV_FOLDS = 5


def run_single(model_name, feature_type, reducer, reducer_params,
               use_wandb, output_csv):
    X_train, y_train = load_features(CACHE_DIR, feature_type, "train")
    X_test, y_test = load_features(CACHE_DIR, feature_type, "test")

    rid = f"{model_name}_{feature_type}_{reducer}"
    pipeline = build_pipeline(model_name, {}, reducer, reducer_params)
    param_grid = get_param_grid(model_name)

    if use_wandb:
        config = {"model": model_name, "features": feature_type, "reducer": reducer}
        init_wandb(config=config, run_name=rid, tags=[model_name, feature_type, reducer])

    t0 = time.time()

    if param_grid:
        print(f"  GridSearchCV: searching {len(param_grid)} param groups...")
        grid = GridSearchCV(
            pipeline, param_grid,
            cv=CV_FOLDS, scoring="f1_macro",
            n_jobs=-1, refit=True, verbose=0,
        )
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        best_params = grid.best_params_
        cv_f1 = grid.best_score_

        from sklearn.model_selection import cross_val_score
        cv_acc = cross_val_score(
            pipeline, X_train, y_train, cv=CV_FOLDS,
            scoring="accuracy", n_jobs=-1
        ).mean()

        print(f"  Best params: {best_params}")
        print(f"  Best CV F1:  {cv_f1:.4f}")
    else:
        from src.evaluation.cross_val import stratified_cv
        cv_results = stratified_cv(pipeline, X_train, y_train, k=CV_FOLDS)
        cv_acc = cv_results["test_accuracy"].mean()
        cv_f1 = cv_results["test_f1_macro"].mean()
        best_params = {}
        pipeline.fit(X_train, y_train)

    test_metrics, y_pred, y_prob = evaluate(pipeline, X_test, y_test)
    elapsed = time.time() - t0

    acc = test_metrics["accuracy"]
    f1 = test_metrics["f1_macro"]
    print(f"  Test: acc={acc:.4f}  f1={f1:.4f}  time={elapsed:.1f}s")

    if use_wandb:
        log_wandb({
            "cv_accuracy": cv_acc, "cv_f1": cv_f1,
            "test_accuracy": acc, "test_f1": f1, "runtime": elapsed,
            **{f"best_{k}": v for k, v in best_params.items()},
        })
        finish_wandb()

    run_data = {
        "run_id": rid,
        "model": model_name, "feature": feature_type, "reducer": reducer,
        "cv_accuracy": round(cv_acc, 4), "cv_f1": round(cv_f1, 4),
        "test_accuracy": round(acc, 4), "test_f1": round(f1, 4),
        "time_seconds": round(elapsed, 2),
    }
    append_result(RESULTS_DIR, run_data, csv_name=output_csv)

    log_experiment_csv(RESULTS_DIR, {
        "run_id": rid, "model": model_name, "feature": feature_type,
        "reducer": reducer, "params": best_params,
        "cv_accuracy": round(cv_acc, 4), "cv_f1": round(cv_f1, 4),
        "test_accuracy": round(acc, 4), "test_f1": round(f1, 4),
        "runtime": round(elapsed, 2),
    })

    return run_data


def run_phase(reducer, reducer_params, output_csv, label, use_wandb):
    print(f"\n{'#' * 70}")
    print(f"  {label}")
    print(f"  Models: {MODELS}")
    print(f"  Features: {FEATURE_TYPES}")
    print(f"  Hyperparameters: Selected via GridSearchCV (not hardcoded)")
    print(f"{'#' * 70}", flush=True)

    csv_path = os.path.join(RESULTS_DIR, output_csv)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    results = []
    total = len(MODELS) * len(FEATURE_TYPES)
    combos = [(f, m) for f in FEATURE_TYPES for m in MODELS]

    pbar = tqdm(combos, desc=label, unit="exp", ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    for feature_type, model_name in pbar:
        pbar.set_postfix_str(f"{model_name} | {feature_type}", refresh=True)
        print(f"\n[{pbar.n+1}/{total}] {model_name} | {feature_type} | reducer={reducer}")
        print("-" * 50, flush=True)
        try:
            result = run_single(model_name, feature_type, reducer,
                                reducer_params, use_wandb, output_csv)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}\n", flush=True)
    pbar.close()

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\n{'=' * 60}")
        print(f"  {label} — TOP 5")
        print(f"{'=' * 60}")
        print_model_ranking(df)


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1-4). 0=all phases")
    args = parser.parse_args()

    use_wandb = args.wandb
    phase = args.phase

    print(f"\n{'=' * 70}")
    print(f"  FOOD IMAGE CLASSIFICATION — EXPERIMENT PIPELINE")
    print(f"  Features: {FEATURE_TYPES} (handcrafted only, no ResNet)")
    print(f"  Models:   {MODELS}")
    print(f"  Tuning:   GridSearchCV (no hardcoded parameters)")
    print(f"  W&B:      {'ENABLED' if use_wandb else 'DISABLED'}")
    print(f"  Phase:    {'ALL' if phase == 0 else phase}")
    print(f"{'=' * 70}")

    start = time.time()

    if phase in (0, 1):
        run_phase("none", None, "master_no_pca.csv",
                  "PHASE 1: No PCA (Raw Dimensions)", use_wandb)

    if phase in (0, 2):
        run_phase("pca", {"n_components": PCA_COMPONENTS}, "master_with_pca.csv",
                  f"PHASE 2: With PCA ({PCA_COMPONENTS} dimensions)", use_wandb)

    total_time = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"  ALL DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"  Results: {RESULTS_DIR}/")
    print(f"   - master_no_pca.csv      (Table 1)")
    print(f"   - master_with_pca.csv    (Table 2)")
    print(f"   - experiment_log.csv     (Detailed log)")


if __name__ == "__main__":
    main()
