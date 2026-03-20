# Implementation Guide

> **A Comparative Study of Classical and Modern PRML Techniques for Food Image Classification**

This document is the single source of truth for the project's architecture, code structure, experiment design, and execution strategy. Read this before touching any code.

---

## Table of Contents

1. [Project Philosophy](#1-project-philosophy)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Configuration System (Hydra)](#4-configuration-system-hydra)
5. [Data Pipeline](#5-data-pipeline)
6. [Feature Extraction](#6-feature-extraction)
7. [Model Registry](#7-model-registry)
8. [Experiment Runner](#8-experiment-runner)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Visualization Pipeline](#10-visualization-pipeline)
11. [Tooling Integration](#11-tooling-integration)
12. [Experiment Matrix](#12-experiment-matrix)
13. [Report & Presentation Mapping](#13-report--presentation-mapping)
14. [Contribution & Team Split](#14-contribution--team-split)

---

## 1. Project Philosophy

This project is not about building one good classifier. It is a **controlled experimental framework** that answers:

> *"How do different feature representations and PRML algorithms interact on a standardized food image benchmark?"*

Every design decision — code structure, config layout, logging format — is in service of that question. The framework is built so any experiment can be reproduced with a single command.

**Three non-negotiable principles:**

- **No data leakage.** All scaling, PCA, and LDA are fit only on training folds, never on validation or test data. Enforced via `sklearn.Pipeline`.
- **Every run is logged.** No experiment result lives only in a notebook. All metrics, hyperparameters, and artifacts go to wandb.
- **Config over code.** Changing an experiment means changing a YAML file, not editing Python.

---

## 2. Repository Structure

```
food-prml/
│
├── configs/                        # Hydra config tree
│   ├── config.yaml                 # Root config (experiment entrypoint)
│   ├── data/
│   │   └── food101.yaml
│   ├── features/
│   │   ├── histogram.yaml
│   │   ├── hog.yaml
│   │   └── cnn.yaml
│   ├── reduction/
│   │   ├── none.yaml
│   │   ├── pca.yaml
│   │   └── lda.yaml
│   ├── model/
│   │   ├── knn.yaml
│   │   ├── logistic.yaml
│   │   ├── naive_bayes.yaml
│   │   ├── svm_linear.yaml
│   │   ├── svm_rbf.yaml
│   │   ├── decision_tree.yaml
│   │   ├── random_forest.yaml
│   │   ├── mlp_sklearn.yaml
│   │   └── cnn.yaml
│   └── experiment/
│       ├── feature_comparison.yaml
│       ├── reduction_comparison.yaml
│       ├── classifier_comparison.yaml
│       ├── knn_sweep.yaml
│       ├── svm_sweep.yaml
│       └── clustering.yaml
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Food101Dataset, stratified splits
│   │   ├── preprocess.py           # Resize, normalize, augment
│   │   └── cache.py                # Feature matrix caching (HDF5)
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base.py                 # FeatureExtractor ABC
│   │   ├── histogram.py            # Color histogram (RGB + HSV)
│   │   ├── hog.py                  # HOG via skimage
│   │   ├── lbp.py                  # Local Binary Patterns
│   │   ├── cnn_embeddings.py       # timm backbone, embedding extraction
│   │   └── fusion.py               # Concatenate feature spaces
│   │
│   ├── reduction/
│   │   ├── __init__.py
│   │   ├── pca_reducer.py          # Wrapped sklearn PCA + variance plots
│   │   └── lda_reducer.py          # Wrapped sklearn LDA
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── registry.py             # MODEL_REGISTRY dict, build_model()
│   │   ├── classical.py            # kNN, LR, NB, SVM, DT, RF — unified interface
│   │   ├── mlp.py                  # MLP via sklearn + optional PyTorch MLP
│   │   └── cnn/
│   │       ├── __init__.py
│   │       ├── model.py            # LightningModule wrapper (ResNet/EfficientNet)
│   │       ├── datamodule.py       # LightningDataModule for Food-101
│   │       └── callbacks.py        # GradCAM callback, early stopping
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── kmeans.py               # KMeans + purity + ARI
│   │   └── gmm.py                  # GMM + BIC/AIC sweep + EM convergence
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # accuracy, f1, confusion, calibration
│   │   ├── cross_val.py            # Stratified K-Fold wrapper with wandb logging
│   │   └── comparison.py           # Master results table builder
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── umap_plot.py            # UMAP + Plotly interactive scatter
│   │   ├── confusion_plot.py       # Seaborn confusion matrix
│   │   ├── learning_curves.py      # Train/val curves, accuracy-vs-k, etc.
│   │   ├── gradcam.py              # Grad-CAM heatmaps for CNN
│   │   └── shap_analysis.py        # SHAP on best classical model
│   │
│   ├── experiment/
│   │   ├── __init__.py
│   │   ├── runner.py               # Core: build pipeline, run CV, log to wandb
│   │   ├── sweeper.py              # Optuna study wrapper
│   │   └── clustering_runner.py    # Separate runner for clustering experiments
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py                 # Global seed setter
│       ├── logging.py              # Python logger + wandb init
│       └── io.py                   # Load/save feature matrices, results CSV
│
├── scripts/
│   ├── extract_features.py         # Precompute + cache all feature matrices
│   ├── run_experiment.py           # Main Hydra entrypoint
│   ├── run_sweep.py                # Launch Optuna sweep
│   ├── run_clustering.py           # Clustering experiments
│   ├── run_cnn.py                  # CNN fine-tuning (Lightning)
│   └── generate_report_plots.py    # Regenerate all final report plots from CSV
│
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_feature_visualization.ipynb
│   ├── 02_pca_lda_analysis.ipynb
│   ├── 03_classifier_results.ipynb
│   ├── 04_clustering_analysis.ipynb
│   └── 05_final_report_figures.ipynb
│
├── results/
│   ├── metrics/                    # Auto-generated CSVs per experiment group
│   └── plots/                      # Auto-generated figures
│
├── app/
│   └── streamlit_demo.py           # Upload image → top-5 predictions
│
├── tests/
│   ├── test_features.py
│   ├── test_pipeline.py
│   └── test_metrics.py
│
├── environment.yml                  # Conda env with pinned versions
├── pyproject.toml                   # Project metadata + dev tools
├── .pre-commit-config.yaml          # black, isort, ruff
├── .gitignore
├── dvc.yaml                         # DVC pipeline stages
├── dvc.lock
└── README.md
```

---

## 3. Environment Setup

```bash
# Clone and enter
git clone https://github.com/<your-org>/food-prml.git
cd food-prml

# Create environment
conda env create -f environment.yml
conda activate food-prml

# Install project in editable mode
pip install -e .

# Login to wandb
wandb login

# Download Food-101
python -c "import torchvision; torchvision.datasets.Food101(root='data/', download=True)"

# Precompute and cache all feature matrices (run once)
python scripts/extract_features.py
```

**`environment.yml` key packages:**

```yaml
name: food-prml
channels: [pytorch, conda-forge, defaults]
dependencies:
  - python=3.11
  - pytorch=2.2.0
  - torchvision
  - pytorch-lightning=2.2.0
  - pip:
    - timm==0.9.16
    - wandb
    - hydra-core==1.3.2
    - optuna==3.5.0
    - scikit-learn==1.4.0
    - scikit-image
    - umap-learn
    - shap
    - plotly
    - yellowbrick
    - albumentations
    - h5py                  # HDF5 feature cache
    - dvc
    - streamlit
    - grad-cam
```

---

## 4. Configuration System (Hydra)

### Root config

```yaml
# configs/config.yaml
defaults:
  - data: food101
  - features: cnn          # histogram | hog | cnn | fused
  - reduction: none        # none | pca | lda
  - model: svm_rbf
  - _self_

seed: 42
cv_folds: 5
n_classes: 20              # subset of Food-101
log_wandb: true
save_results: true

wandb:
  project: food-prml
  entity: <your-team>
```

### Example model config

```yaml
# configs/model/svm_rbf.yaml
_target_: sklearn.svm.SVC
kernel: rbf
C: 1.0
gamma: scale
probability: true           # enables calibration + SHAP
```

### Example reduction config

```yaml
# configs/reduction/pca.yaml
_target_: sklearn.decomposition.PCA
n_components: 100
whiten: true
```

### Running experiments

```bash
# Single run
python scripts/run_experiment.py model=svm_rbf features=cnn reduction=pca

# Multi-run sweep (Hydra)
python scripts/run_experiment.py --multirun \
  model=knn,svm_rbf,logistic,naive_bayes \
  features=histogram,hog,cnn \
  reduction=none,pca

# Optuna hyperparameter sweep
python scripts/run_sweep.py model=svm_rbf features=cnn n_trials=50
```

---

## 5. Data Pipeline

### `src/data/dataset.py`

```python
class Food101Dataset:
    """
    Handles loading, stratified splitting, and class subsetting.
    Returns (image_path, label) tuples — actual loading deferred to feature extractors.
    """
    def __init__(self, root, n_classes=20, split="train", seed=42):
        ...

    def get_splits(self) -> tuple[list, list, list]:
        # Returns train, val, test path lists — stratified
        ...
```

### `src/data/cache.py`

Feature extraction is expensive. Cache feature matrices to HDF5 after first extraction:

```python
# Cache schema: cache/<feature_type>/<split>.h5
# Keys: 'X' (n_samples, n_features), 'y' (n_samples,), 'classes' (list)

def save_features(X, y, path): ...
def load_features(path) -> tuple[np.ndarray, np.ndarray]: ...
def cache_exists(feature_type, split) -> bool: ...
```

`extract_features.py` iterates all feature types × splits and populates the cache. **Run this once before any experiments.** All experiment scripts load from cache, not raw images.

---

## 6. Feature Extraction

### Abstract base

```python
# src/features/base.py
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single HWC uint8 image."""
        ...

    def extract_dataset(self, image_paths, labels, n_jobs=-1) -> tuple[np.ndarray, np.ndarray]:
        """Parallel extraction over full dataset via joblib."""
        ...
```

### Feature types

| Extractor | File | Output dim | Key library |
|---|---|---|---|
| Color Histogram (RGB+HSV) | `histogram.py` | 192 | OpenCV |
| HOG | `hog.py` | ~1764 | skimage |
| LBP | `lbp.py` | 256 | skimage |
| CNN Embedding (ResNet-50) | `cnn_embeddings.py` | 2048 | timm |
| Fused (HOG + Hist + LBP) | `fusion.py` | ~2212 | — |

### CNN embedding extraction

```python
# src/features/cnn_embeddings.py
class CNNEmbeddingExtractor(FeatureExtractor):
    def __init__(self, backbone="resnet50", device="cuda"):
        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)
        # num_classes=0 removes the classification head — returns pooled embedding
        self.model.eval()

    @torch.no_grad()
    def extract(self, image):
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze().cpu().numpy()
```

> **Design note:** `num_classes=0` in timm removes the FC head cleanly. The model returns the global average pooled representation — this is your 2048-dim embedding for ResNet-50, 1280-dim for EfficientNet-B0.

---

## 7. Model Registry

All classical models share one unified interface via `sklearn.Pipeline`. The registry makes swapping models a config change, not a code change.

```python
# src/models/registry.py

MODEL_REGISTRY = {
    "knn":           KNeighborsClassifier,
    "logistic":      LogisticRegression,
    "naive_bayes":   GaussianNB,
    "svm_linear":    SVC,
    "svm_rbf":       SVC,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "mlp_sklearn":   MLPClassifier,
    "perceptron":    Perceptron,
}

def build_pipeline(scaler_cfg, reducer_cfg, model_cfg) -> Pipeline:
    """
    Builds:  StandardScaler → [PCA / LDA / None] → Classifier
    All fit() calls are safe for cross-validation — no leakage.
    """
    steps = [("scaler", instantiate(scaler_cfg))]
    if reducer_cfg is not None:
        steps.append(("reducer", instantiate(reducer_cfg)))
    steps.append(("clf", instantiate(model_cfg)))
    return Pipeline(steps)
```

---

## 8. Experiment Runner

The runner is the core of the project. It orchestrates the pipeline, runs cross-validation, logs everything, and saves results.

```python
# src/experiment/runner.py

@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    seed_everything(cfg.seed)

    if cfg.log_wandb:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg))

    # 1. Load cached features
    X_train, y_train = load_features(cfg.features._target_, "train")
    X_test,  y_test  = load_features(cfg.features._target_, "test")

    # 2. Build sklearn Pipeline (scaler + reducer + model)
    pipeline = build_pipeline(cfg.scaler, cfg.reduction, cfg.model)

    # 3. Stratified K-Fold CV
    cv_results = stratified_cv(pipeline, X_train, y_train, k=cfg.cv_folds)
    log_cv_metrics(cv_results)          # → wandb + CSV

    # 4. Final evaluation on test set
    pipeline.fit(X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test)
    log_test_metrics(test_metrics)      # → wandb + CSV

    # 5. Artifacts
    save_confusion_matrix(pipeline, X_test, y_test)
    if cfg.features == "cnn":
        save_umap_plot(X_test, y_test)

    wandb.finish()
```

### Optuna sweep

```python
# src/experiment/sweeper.py

def objective(trial, cfg, X_train, y_train):
    # Example: SVM sweep
    C     = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    pipeline = build_pipeline(cfg, model_override={"C": C, "gamma": gamma})
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")

    wandb.log({"trial": trial.number, "cv_f1": scores.mean(), "C": C, "gamma": gamma})
    return scores.mean()

study = optuna.create_study(direction="maximize", sampler=TPESampler())
study.optimize(objective, n_trials=cfg.n_trials)
```

---

## 9. Evaluation Framework

### `src/evaluation/metrics.py`

```python
def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    return {
        "accuracy":        accuracy_score(y_test, y_pred),
        "f1_macro":        f1_score(y_test, y_pred, average="macro"),
        "f1_weighted":     f1_score(y_test, y_pred, average="weighted"),
        "top5_accuracy":   top_k_accuracy_score(y_test, y_prob, k=5) if y_prob else None,
    }
```

### Master results table

Every experiment appends a row to `results/metrics/master.csv`:

```
run_id | feature | reducer | n_dims | model | cv_acc | cv_f1 | test_acc | test_f1 | params
```

`generate_report_plots.py` reads this CSV and regenerates all comparison plots without re-running experiments.

### Cross-validation — important detail

```python
# src/evaluation/cross_val.py

def stratified_cv(pipeline, X, y, k=5):
    """
    Uses StratifiedKFold. The entire Pipeline (scaler + reducer + clf)
    is fit independently on each fold's training split.
    This is the correct way — PCA/LDA see only training data per fold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return cross_validate(pipeline, X, y, cv=skf,
                          scoring=["accuracy", "f1_macro"],
                          return_train_score=True)
```

> **Explicitly mention this in your report.** Many student projects fit PCA on the full dataset before CV — that's leakage. Yours doesn't.

---

## 10. Visualization Pipeline

### UMAP interactive scatter

```python
# src/visualization/umap_plot.py

def plot_umap_interactive(X_emb, y, class_names, title="UMAP of CNN Embeddings"):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(X_emb)

    fig = px.scatter(
        x=coords[:, 0], y=coords[:, 1],
        color=[class_names[i] for i in y],
        title=title,
        opacity=0.6,
        width=900, height=700
    )
    fig.write_html("results/plots/umap_cnn.html")
    wandb.log({"umap": wandb.Html("results/plots/umap_cnn.html")})
```

### Plots per experiment group

| Experiment | Plot | Library |
|---|---|---|
| PCA variance | Explained variance curve (elbow) | Matplotlib |
| PCA dims vs accuracy | Line plot across dims | Seaborn |
| kNN: k vs accuracy | Line plot with error bars (CV) | Seaborn |
| SVM grid search | C × gamma heatmap | Seaborn |
| Confusion matrix | Annotated heatmap | Yellowbrick |
| Clustering | Silhouette plot, GMM BIC/AIC | Yellowbrick |
| CNN training | Loss + accuracy curves | wandb / Lightning |
| UMAP | Interactive 2D scatter | Plotly |
| Grad-CAM | Image + heatmap overlay | grad-cam library |
| SHAP | Feature importance bar | SHAP |
| Master comparison | Grouped bar chart | Plotly |

---

## 11. Tooling Integration

### wandb — what to log

```python
wandb.log({
    # Per epoch (CNN)
    "train/loss": loss, "val/accuracy": acc,

    # Per experiment (classical)
    "cv/accuracy_mean": cv_acc.mean(),
    "cv/accuracy_std":  cv_acc.std(),
    "test/accuracy":    test_acc,
    "test/f1_macro":    test_f1,

    # Artifacts
    "confusion_matrix": wandb.plot.confusion_matrix(y_true, y_pred, class_names),
    "umap_plot":        wandb.Html("results/plots/umap_cnn.html"),
})
```

Create a **wandb Report** at the end (wandb's built-in report builder) that narrates all your results. This can literally be submitted as supplementary material or shown to your professor live.

### DVC pipeline

```yaml
# dvc.yaml
stages:
  extract_features:
    cmd: python scripts/extract_features.py
    deps: [src/features/, data/food101/]
    outs: [data/cache/]     # cached HDF5 files

  run_experiments:
    cmd: python scripts/run_experiment.py --multirun ...
    deps: [data/cache/, configs/]
    outs: [results/metrics/master.csv]

  generate_plots:
    cmd: python scripts/generate_report_plots.py
    deps: [results/metrics/master.csv]
    outs: [results/plots/]
```

Reproducing the entire project: `dvc repro`. That's it.

---

## 12. Experiment Matrix

All 30 planned runs, pre-specified. Every run is reproducible with the command shown.

### Group 1 — Feature Comparison (fix model: SVM-RBF)

| Run | Feature | Reducer | Expected insight |
|---|---|---|---|
| 1 | Raw pixels | None | Baseline floor |
| 2 | Color Histogram | None | Color alone |
| 3 | HOG | None | Texture/shape |
| 4 | HOG + Histogram | None | Fused handcrafted |
| 5 | CNN (ResNet-50) | None | Deep features |

```bash
python scripts/run_experiment.py --multirun model=svm_rbf \
  features=raw,histogram,hog,fused,cnn reduction=none
```

### Group 2 — Dimensionality Reduction (fix model: SVM-RBF, feature: fused)

| Run | Reducer | n_dims | Expected insight |
|---|---|---|---|
| 6 | None | 2212 | Full dim baseline |
| 7 | PCA | 50 | Heavy compression |
| 8 | PCA | 100 | Moderate |
| 9 | PCA | 200 | Light compression |
| 10 | LDA | 19 (C-1) | Supervised reduction |
| 11 | PCA→LDA | 200→19 | Pipeline combo |

```bash
python scripts/run_experiment.py --multirun model=svm_rbf features=fused \
  reduction=none,pca_50,pca_100,pca_200,lda,pca200_lda
```

### Group 3 — Classifier Comparison (fix feature: CNN, reducer: None)

| Run | Model | Expected insight |
|---|---|---|
| 12 | kNN (k=5, L2) | Distance-based |
| 13 | Naive Bayes | Probabilistic |
| 14 | Logistic Regression | Linear probabilistic |
| 15 | Perceptron | Linear hard margin |
| 16 | SVM (Linear) | Max margin linear |
| 17 | SVM (RBF) | Kernel nonlinear |
| 18 | Decision Tree | Axis-aligned splits |
| 19 | Random Forest | Ensemble |
| 20 | MLP (2 hidden layers) | Neural network |
| 21 | CNN (fine-tuned) | End-to-end deep |

```bash
python scripts/run_experiment.py --multirun features=cnn reduction=none \
  model=knn,logistic,naive_bayes,perceptron,svm_linear,svm_rbf,decision_tree,random_forest,mlp_sklearn
```

### Group 4 — Hyperparameter Sweeps (Optuna)

| Run | Model | Search space |
|---|---|---|
| 22 | kNN | k ∈ {1,3,5,7,10,15,21}, distance ∈ {L1,L2,cosine} |
| 23 | SVM RBF | C ∈ [0.01, 100], gamma ∈ {scale, auto, 0.001, 0.01} |
| 24 | MLP | hidden_layers ∈ {1,2,3}, units ∈ {64,128,256,512} |

```bash
python scripts/run_sweep.py model=knn n_trials=30
python scripts/run_sweep.py model=svm_rbf n_trials=50
python scripts/run_sweep.py model=mlp_sklearn n_trials=40
```

### Group 5 — Clustering (unsupervised)

| Run | Algorithm | Feature | Expected insight |
|---|---|---|---|
| 25 | KMeans (k=20) | Raw pixels | Poor cluster purity |
| 26 | KMeans (k=20) | Handcrafted | Moderate purity |
| 27 | KMeans (k=20) | CNN embeddings | High purity |
| 28 | GMM (diagonal) | CNN + PCA-50 | Probabilistic clusters |
| 29 | GMM BIC sweep | CNN + PCA-50 | Optimal k selection |
| 30 | GMM (full cov) | CNN + PCA-50 | Compare covariance types |

```bash
python scripts/run_clustering.py --multirun \
  algorithm=kmeans features=raw,handcrafted,cnn
python scripts/run_clustering.py algorithm=gmm features=cnn reduction=pca_50
```

---

## 13. Report & Presentation Mapping

Every section of your report maps to a specific set of runs and plots.

| Report Section | Runs used | Key plot |
|---|---|---|
| Feature Representations | 1–5 | Grouped bar: acc by feature type |
| Dimensionality Reduction | 6–11 | PCA elbow + acc-vs-dims + LDA comparison |
| Classifier Comparison | 12–21 | Master comparison bar chart |
| Hyperparameter Analysis | 22–24 | k-vs-acc, Optuna importance plot |
| Clustering | 25–30 | UMAP scatter, purity table, GMM BIC curve |
| CNN vs Classical | 17, 21 | Side-by-side confusion matrices |
| Interpretability | Best run | SHAP bar + Grad-CAM overlays |

**Key result to build your narrative around:**

> SVM with RBF kernel on CNN embeddings approaches CNN fine-tuning accuracy — showing that the right features matter more than the classifier. Naive Bayes on handcrafted features shows the cost of wrong feature-model pairing. LDA outperforms PCA at same dimensionality because it uses label information.

---

## 14. Contribution & Team Split

Suggested team responsibilities (adjust to your group size):

| Module | Owner | Files |
|---|---|---|
| Data pipeline + caching | Member A | `src/data/`, `scripts/extract_features.py` |
| Feature extraction | Member A+B | `src/features/` |
| Classical models + evaluation | Member B | `src/models/classical.py`, `src/evaluation/` |
| CNN + Lightning | Member C | `src/models/cnn/`, `scripts/run_cnn.py` |
| Clustering | Member B | `src/clustering/` |
| Visualization + SHAP + GradCAM | Member C | `src/visualization/` |
| Hydra configs + Optuna sweeps | Member A | `configs/`, `scripts/run_sweep.py` |
| wandb setup + report plots | All | `results/`, notebooks |
| Streamlit demo | Anyone | `app/streamlit_demo.py` |

---

## Key Design Decisions (Explain These in Your Report)

**Why HDF5 for feature cache?**
Feature extraction for 100k images with CNN takes 30–60 minutes. HDF5 allows memory-mapped access — you load only what you need, and it's compressed. Alternatives (pickle, numpy) don't scale.

**Why `sklearn.Pipeline` and not manual steps?**
Pipeline ensures the scaler and dimensionality reducer are fit only on training folds during cross-validation. Fitting PCA on the full dataset before CV is a form of data leakage that inflates reported accuracy. We avoid it by design.

**Why CNN embeddings for classical classifiers?**
This is the conceptual bridge of the project. A pretrained CNN has learned a rich feature space from ImageNet. By using these embeddings as inputs to kNN, SVM, etc., we ask: *"How much of a classifier's job is the feature representation?"* The answer — a lot — is your most important finding.

**Why Optuna over grid search?**
Grid search over SVM's (C, gamma) space with even 5 values each = 25 fits × 5 folds = 125 pipeline fits. Optuna's TPE sampler finds better hyperparameters in ~30 trials by using past results to guide the search. Mention this as *Bayesian hyperparameter optimization* in the report.

**Why LDA can't exceed (n_classes − 1) dimensions?**
LDA maximizes the ratio of between-class to within-class scatter. The between-class scatter matrix has rank at most C−1. This is a fundamental linear algebra constraint — explaining it shows theoretical understanding, not just API usage.

---

*All results are reproducible. All experiments are logged. All plots are auto-generated from saved CSVs.*
*To reproduce everything from scratch: `dvc repro`*