# Food Image Classification — Unified Implementation Guide

> **A Comparative Study of Classical and Modern PRML Techniques for Food Image Classification**
>
> **Dataset:** [Food-101 (Kaggle)](https://www.kaggle.com/datasets/kmader/food41) — 101 food categories, ~1,000 images per category
>
> This document is the single source of truth for the project's architecture, code structure, experiment design, and execution strategy.

---

## Table of Contents

1. [Course Topics Coverage Map](#1-course-topics-coverage-map)
2. [Project Philosophy](#2-project-philosophy)
3. [Tech Stack](#3-tech-stack)
4. [Project Architecture — Block Diagram](#4-project-architecture--block-diagram)
5. [Repository Structure](#5-repository-structure)
6. [Environment Setup](#6-environment-setup)
7. [Configuration System (Hydra)](#7-configuration-system-hydra)
8. [Data Pipeline](#8-data-pipeline)
9. [Feature Extraction](#9-feature-extraction)
10. [Model Registry](#10-model-registry)
11. [Experiment Runner](#11-experiment-runner)
12. [Evaluation Framework](#12-evaluation-framework)
13. [Visualization Pipeline](#13-visualization-pipeline)
14. [Tooling Integration](#14-tooling-integration)
15. [Phase-by-Phase Implementation Procedure](#15-phase-by-phase-implementation-procedure)
16. [Experiment Matrix](#16-experiment-matrix)
17. [Key Design Decisions](#17-key-design-decisions)

---

## 1. Course Topics Coverage Map

Every single course topic is mapped to a specific implementation step in this project:

| # | Course Topic | Where It's Used |
|---|-------------|-----------------|
| 1 | **Introduction** | Problem formulation, dataset overview, ML pipeline introduction |
| 2 | **Feature computation & classification using histogram** | Color histograms (RGB, HSV) as features; histogram-based baseline classifier |
| 3 | **Multi-dimensional features, Cost of error, Distribution** | Combined feature vectors (color+texture+shape); error distribution analysis |
| 4 | **Bayes classifier** | Gaussian Naive Bayes classifier on extracted features |
| 5 | **Distance & similarity measures, kNN classifier** | kNN with Euclidean, Manhattan, Cosine distance metrics |
| 6 | **Weighted kNN, Data representation, Data normalization** | Distance-weighted kNN; Min-Max / Z-score normalization |
| 7 | **Cross-validation** | Stratified k-Fold CV for all model evaluations |
| 8 | **Linear regression** | Discussed as foundation; SGD-based linear classifiers |
| 9 | **Gradient descent** | SGD classifier; GD-based training of MLP/CNN |
| 10 | **Logistic regression** | Multinomial logistic regression (softmax) as baseline |
| 11 | **Multi-class classification, Overfitting, Regularization** | One-vs-Rest / Softmax; L1/L2 regularization; Dropout; Early stopping; Augmentation |
| 12 | **Dimensionality reduction, Covariance, Correlation** | Covariance matrix analysis; correlation heatmaps; scree plots |
| 13 | **PCA & LDA** | PCA for compression; LDA for supervised reduction; 2D/3D visualizations |
| 14 | **Linear Classifier: Perceptron** | Single-layer Perceptron for multi-class classification |
| 15 | **Linear Classifier: SVM** | Linear SVM (LinearSVC) on feature vectors |
| 16 | **SVM variants, Kernel SVM** | RBF and Polynomial kernel SVMs |
| 17 | **Multi-layer Perceptron (MLP)** | MLP using sklearn and PyTorch |
| 18 | **Similarity: Perceptron ↔ Neuron, MLP** | Documented biological neuron vs perceptron analogy |
| 19 | **Backpropagation algorithm** | Backprop in MLP/CNN training; gradient flow visualization |
| 20 | **Convolution filter, Convolution operation** | Sobel, Gabor, Gaussian, Laplacian filter demos on food images |
| 21 | **CNN** | Custom CNN + Transfer Learning (ResNet/EfficientNet) |
| 22 | **Data Clustering** | Cluster food images by visual features; discover natural groupings |
| 23 | **KMeans Clustering (EM) and variants** | KMeans, KMeans++, Mini-Batch KMeans; GMM with EM |
| 24 | **Agglomerative clustering, Practices** | Hierarchical clustering; dendrograms; Silhouette/Elbow analysis |
| 25 | **Decision Trees** | Decision tree classifier; tree visualization; feature importance |
| 26 | **Variants of Decision Tree** | Random Forest classifier |
| 27 | **Combining Classifiers (Boosting)** | Gradient Boosting on extracted features |

> **Total: 27/27 course topics covered ✅**

---

## 2. Project Philosophy

This project is a **controlled experimental framework** that answers:

> *"How do different feature representations and PRML algorithms interact on a standardized food image benchmark?"*

**Three non-negotiable principles:**

- **No data leakage.** All scaling, PCA, and LDA are fit only on training folds, never on validation or test data. Enforced via `sklearn.Pipeline`.
- **Every run is logged.** All metrics, hyperparameters, and artifacts go to wandb.
- **Config over code.** Changing an experiment means changing a YAML file, not editing Python.

---

## 3. Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11+ | Primary language |
| **Deep Learning** | PyTorch 2.2 + PyTorch Lightning | CNN, MLP, Backpropagation |
| **Classical ML** | scikit-learn 1.4 | kNN, SVM, Logistic Regression, Decision Trees, PCA, LDA, Clustering |
| **Image Processing** | OpenCV, scikit-image, Albumentations | Feature extraction, augmentation |
| **CNN Backbones** | timm | Pre-trained model embeddings |
| **Data Handling** | NumPy, Pandas, h5py | Numerical operations, HDF5 caching |
| **Visualization** | Matplotlib, Seaborn, Plotly | Static + interactive plots |
| **Interpretability** | SHAP, grad-cam | Feature importance, CNN heatmaps |
| **Experiment Config** | Hydra | YAML-driven experiment configuration |
| **Experiment Tracking** | Weights & Biases (wandb) | Metrics logging, plots, reports |
| **Hyperparameters** | Optuna | Bayesian hyperparameter search |
| **Reproducibility** | DVC | Data/pipeline versioning |
| **Demo** | Streamlit | Interactive image classifier app |
| **Notebook** | Jupyter Notebook | Interactive development |

### Hardware
- **GPU recommended** for CNN (Google Colab T4 is sufficient)
- **RAM:** 16 GB+ (or Colab 12 GB)
- **Storage:** ~6 GB for dataset

---

## 4. Project Architecture — Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FOOD IMAGE CLASSIFICATION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌────────────┐    ┌──────────────┐                │
│  │ Raw Food │───▶│  Feature   │───▶│ Preprocessed │                │
│  │  Images  │    │ Extraction │    │   Features   │                │
│  └──────────┘    └────────────┘    └──────┬───────┘                │
│                   │ Color Hist │          │                          │
│                   │ HOG / LBP  │          ▼                          │
│                   │ GLCM       │   ┌──────────────┐                │
│                   │ CNN Embed  │   │  Dim. Reduc. │                │
│                   └────────────┘   │  PCA / LDA   │                │
│                                    └──────┬───────┘                │
│                                           │                         │
│                    ┌──────────────────────┼──────────────────────┐  │
│                    ▼                      ▼                      ▼  │
│             ┌────────────┐      ┌──────────────┐      ┌──────────┐ │
│             │ Classical  │      │   Neural     │      │ Tree &   │ │
│             │    ML      │      │  Networks    │      │ Ensemble │ │
│             ├────────────┤      ├──────────────┤      ├──────────┤ │
│             │ Bayes      │      │ Perceptron   │      │ Dec.Tree │ │
│             │ kNN        │      │ MLP          │      │ Rand.For │ │
│             │ SVM/Kernel │      │ CNN          │      │ GradBoost│ │
│             │ Log. Reg.  │      │ Transfer Lrn │      │          │ │
│             └────────────┘      └──────────────┘      └──────────┘ │
│                    │                      │                    │    │
│                    └──────────────────────┼────────────────────┘    │
│                                           ▼                         │
│                                  ┌──────────────────┐              │
│                                  │ Comparison Table │              │
│                                  │ & Final Analysis │              │
│                                  └──────────────────┘              │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │              UNSUPERVISED ANALYSIS (Parallel)                 │ │
│  │  KMeans │ GMM (EM) │ Agglomerative │ Silhouette │ Dendrograms│ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Repository Structure

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
│   │   ├── glcm.py                 # GLCM texture features
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
│   │   ├── mlp.py                  # MLP via sklearn + PyTorch MLP
│   │   └── cnn/
│   │       ├── __init__.py
│   │       ├── model.py            # LightningModule wrapper (ResNet/EfficientNet)
│   │       ├── datamodule.py       # LightningDataModule for Food-101
│   │       └── callbacks.py        # GradCAM callback, early stopping
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── kmeans.py               # KMeans + purity + ARI
│   │   ├── agglomerative.py        # Agglomerative + dendrograms
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
│   │   ├── filter_demo.py          # Convolution filter visualizations
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
│   └── generate_report_plots.py    # Regenerate all report plots from CSV
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_visualization.ipynb
│   ├── 03_pca_lda_analysis.ipynb
│   ├── 04_classifier_results.ipynb
│   ├── 05_clustering_analysis.ipynb
│   └── 06_final_report_figures.ipynb
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
├── environment.yml
├── pyproject.toml
├── .pre-commit-config.yaml
├── .gitignore
├── dvc.yaml
├── dvc.lock
└── README.md
```

---

## 6. Environment Setup

```bash
git clone https://github.com/Vinamra3215/PRML-Project.git
cd PRML-Project

conda env create -f environment.yml
conda activate food-prml
pip install -e .

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
    - h5py
    - dvc
    - streamlit
    - grad-cam
```

---

## 7. Configuration System (Hydra)

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
probability: true
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

## 8. Data Pipeline

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
def save_features(X, y, path): ...
def load_features(path) -> tuple[np.ndarray, np.ndarray]: ...
def cache_exists(feature_type, split) -> bool: ...
```

`extract_features.py` iterates all feature types × splits and populates the cache. **Run this once before any experiments.**

### Data Preprocessing & Normalization

```python
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)
```

---

## 9. Feature Extraction

### Abstract Base

```python
class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single HWC uint8 image."""
        ...

    def extract_dataset(self, image_paths, labels, n_jobs=-1):
        """Parallel extraction over full dataset via joblib."""
        ...
```

### Feature Types

| Extractor | File | Output dim | Library |
|---|---|---|---|
| Color Histogram (RGB+HSV) | `histogram.py` | 192 | OpenCV |
| HOG | `hog.py` | ~1764 | skimage |
| LBP | `lbp.py` | 256 | skimage |
| GLCM Texture | `glcm.py` | ~30 | skimage |
| CNN Embedding (ResNet-50) | `cnn_embeddings.py` | 2048 | timm |
| Fused (HOG+Hist+LBP+GLCM) | `fusion.py` | ~2242 | — |

### Color Histogram

```python
def extract_color_histogram(image, bins=32):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    return hist / hist.sum()
```

### GLCM Texture Features

```python
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    return np.concatenate([contrast, homogeneity, energy, correlation])
```

### CNN Embedding Extraction

```python
class CNNEmbeddingExtractor(FeatureExtractor):
    def __init__(self, backbone="resnet50", device="cuda"):
        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.model.eval()

    @torch.no_grad()
    def extract(self, image):
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return self.model(tensor).squeeze().cpu().numpy()
```

> **Design note:** `num_classes=0` in timm removes the FC head cleanly. The model returns the global average pooled representation — 2048-dim for ResNet-50, 1280-dim for EfficientNet-B0.

---

## 10. Model Registry

All classical models share one unified interface via `sklearn.Pipeline`. The registry makes swapping models a config change, not a code change.

```python
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
    All fit() calls are safe for cross-validation — no data leakage.
    """
    steps = [("scaler", instantiate(scaler_cfg))]
    if reducer_cfg is not None:
        steps.append(("reducer", instantiate(reducer_cfg)))
    steps.append(("clf", instantiate(model_cfg)))
    return Pipeline(steps)
```

---

## 11. Experiment Runner

```python
@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def run(cfg: DictConfig):
    seed_everything(cfg.seed)

    if cfg.log_wandb:
        wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg))

    # 1. Load cached features
    X_train, y_train = load_features(cfg.features._target_, "train")
    X_test,  y_test  = load_features(cfg.features._target_, "test")

    # 2. Build sklearn Pipeline
    pipeline = build_pipeline(cfg.scaler, cfg.reduction, cfg.model)

    # 3. Stratified K-Fold CV
    cv_results = stratified_cv(pipeline, X_train, y_train, k=cfg.cv_folds)
    log_cv_metrics(cv_results)

    # 4. Final evaluation on test set
    pipeline.fit(X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test)
    log_test_metrics(test_metrics)

    # 5. Artifacts
    save_confusion_matrix(pipeline, X_test, y_test)
    if cfg.features == "cnn":
        save_umap_plot(X_test, y_test)
    wandb.finish()
```

### Optuna Sweep

```python
def objective(trial, cfg, X_train, y_train):
    C     = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    pipeline = build_pipeline(cfg, model_override={"C": C, "gamma": gamma})
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
    wandb.log({"trial": trial.number, "cv_f1": scores.mean()})
    return scores.mean()
```

---

## 12. Evaluation Framework

### Metrics

```python
def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    return {
        "accuracy":      accuracy_score(y_test, y_pred),
        "f1_macro":      f1_score(y_test, y_pred, average="macro"),
        "f1_weighted":   f1_score(y_test, y_pred, average="weighted"),
        "top5_accuracy": top_k_accuracy_score(y_test, y_prob, k=5) if y_prob else None,
    }
```

### Master Results Table

Every experiment appends a row to `results/metrics/master.csv`:

```
run_id | feature | reducer | n_dims | model | cv_acc | cv_f1 | test_acc | test_f1 | params
```

`generate_report_plots.py` reads this CSV and regenerates all comparison plots without re-running experiments.

### Cross-Validation — Important Detail

```python
def stratified_cv(pipeline, X, y, k=5):
    """
    Uses StratifiedKFold. The entire Pipeline (scaler + reducer + clf)
    is fit independently on each fold's training split.
    PCA/LDA see only training data per fold — no leakage.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return cross_validate(pipeline, X, y, cv=skf,
                          scoring=["accuracy", "f1_macro"],
                          return_train_score=True)
```

---

## 13. Visualization Pipeline

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
| Grad-CAM | Image + heatmap overlay | grad-cam |
| SHAP | Feature importance bar | SHAP |
| Master comparison | Grouped bar chart | Plotly |
| Convolution filters | Sobel, Gabor, Gaussian, Laplacian demos | OpenCV + Matplotlib |
| Dendrograms | Hierarchical clustering tree | scipy |
| Learning curves | Train/val accuracy vs samples | Matplotlib |

### Convolution Filter Demo

```python
sample_image = X_train[0]
gray = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
gabor_kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0)
gabor_filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
filters = [gray, sobel_x, sobel_y, gaussian, gabor_filtered, laplacian]
titles = ['Original', 'Sobel-X', 'Sobel-Y', 'Gaussian', 'Gabor', 'Laplacian']
for ax, img, title in zip(axes.flatten(), filters, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
plt.suptitle('Convolution Filter Operations on Food Image')
plt.show()
```

### UMAP Interactive Scatter

```python
def plot_umap_interactive(X_emb, y, class_names, title="UMAP of CNN Embeddings"):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(X_emb)
    fig = px.scatter(x=coords[:, 0], y=coords[:, 1],
                     color=[class_names[i] for i in y],
                     title=title, opacity=0.6, width=900, height=700)
    fig.write_html("results/plots/umap_cnn.html")
    wandb.log({"umap": wandb.Html("results/plots/umap_cnn.html")})
```

---

## 14. Tooling Integration

### wandb — What to Log

```python
wandb.log({
    # Per experiment (classical)
    "cv/accuracy_mean": cv_acc.mean(),
    "cv/accuracy_std":  cv_acc.std(),
    "test/accuracy":    test_acc,
    "test/f1_macro":    test_f1,
    # Per epoch (CNN)
    "train/loss": loss, "val/accuracy": acc,
    # Artifacts
    "confusion_matrix": wandb.plot.confusion_matrix(y_true, y_pred, class_names),
    "umap_plot":        wandb.Html("results/plots/umap_cnn.html"),
})
```

### DVC Pipeline

```yaml
stages:
  extract_features:
    cmd: python scripts/extract_features.py
    deps: [src/features/, data/food101/]
    outs: [data/cache/]

  run_experiments:
    cmd: python scripts/run_experiment.py --multirun ...
    deps: [data/cache/, configs/]
    outs: [results/metrics/master.csv]

  generate_plots:
    cmd: python scripts/generate_report_plots.py
    deps: [results/metrics/master.csv]
    outs: [results/plots/]
```

Reproducing the entire project: `dvc repro`.

---

## 15. Phase-by-Phase Implementation Procedure

### Phase 1: Introduction & Data Loading
**Topics:** `#1 Introduction`

- Define food classification as a multi-class problem (input: RGB image → output: 1 of 101 categories)
- Load Food-101 via `torchvision.datasets.Food101` or HDF5
- Stratified train/val/test split; class distribution analysis

### Phase 2: Feature Engineering
**Topics:** `#2 Feature computation & histogram`, `#3 Multi-dimensional features, Cost of error, Distribution`

- Extract Color Histograms (RGB + HSV), HOG, LBP, GLCM features
- Build fused multi-dimensional feature vectors
- Analyze feature distributions per class; compute misclassification cost matrix

### Phase 3: Data Preprocessing & Normalization
**Topics:** `#6 Data representation, Data normalization`

- Apply Z-score and Min-Max normalization; compare impact on model accuracy
- Three data representations: flattened pixels, hand-crafted features, CNN tensors

### Phase 4: Exploratory Data Analysis
**Topics:** `#3 Distribution`, `#12 Covariance, Correlation`

- Feature distribution plots (KDE) across food classes
- Correlation heatmap; covariance matrix + eigenvalue scree plot

### Phase 5: Dimensionality Reduction
**Topics:** `#12 Dimensionality reduction`, `#13 PCA & LDA`

- PCA: retain 95% variance; visualize 2D projections; explained variance curve
- LDA: supervised reduction (max C−1 dims); 2D visualizations
- Compare PCA vs LDA classification accuracy at same dimensions

### Phase 6: Classical ML Classifiers
**Topics:** `#4 Bayes`, `#5 kNN`, `#6 Weighted kNN`, `#7 Cross-validation`, `#8 Linear regression`, `#9 Gradient descent`, `#10 Logistic regression`, `#11 Multi-class, Overfitting, Regularization`, `#15 SVM`, `#16 Kernel SVM`

- **Naive Bayes** — GaussianNB on PCA/LDA features
- **kNN** — k=5 with Euclidean, Manhattan, Cosine; k-vs-accuracy plot
- **Weighted kNN** — distance-weighted with hyperparameter tuning
- **SGD Classifier** — gradient descent demonstration
- **Logistic Regression** — multinomial softmax; L1/L2 regularization comparison
- **Linear SVM** — LinearSVC
- **Kernel SVM** — RBF and Polynomial kernels
- **Cross-validation** — Stratified 5-Fold CV for all models
- **Overfitting analysis** — learning curves (train vs val accuracy); regularization effects

### Phase 7: Neural Networks
**Topics:** `#14 Perceptron`, `#17 MLP`, `#18 Perceptron ↔ Neuron`, `#19 Backpropagation`, `#9 Gradient descent`

- **Perceptron** — single-layer linear classifier
- **Neuron analogy** — document dendrites↔inputs, synapses↔weights, activation↔step function
- **MLP (sklearn)** — 3-layer MLP with early stopping; training loss curve
- **MLP (PyTorch)** — custom FoodMLP with BatchNorm + Dropout; backprop + gradient descent training
- **Backprop visualization** — gradient magnitudes per layer across epochs

### Phase 8: CNN
**Topics:** `#20 Convolution filter & operation`, `#21 CNN`, `#11 Regularization`

- **Convolution filter demo** — Sobel, Gabor, Gaussian, Laplacian on food images
- **Custom CNN** — 3-block ConvNet with BatchNorm, MaxPool, Dropout
- **Transfer Learning** — ResNet-50/EfficientNet fine-tuning (freeze backbone, train FC head)
- **Regularization** — data augmentation, early stopping, dropout, weight decay
- **Interpretability** — Grad-CAM heatmaps; learned filter visualization

### Phase 9: Clustering
**Topics:** `#22 Data Clustering`, `#23 KMeans (EM) & variants`, `#24 Agglomerative clustering`

- **KMeans** — k=20 on raw pixels, handcrafted features, CNN embeddings; elbow method
- **GMM** — EM algorithm; BIC/AIC sweep for optimal k; full vs diagonal covariance
- **Agglomerative** — hierarchical clustering with dendrograms
- **Evaluation** — Silhouette score, ARI, NMI; cluster-vs-label visualizations

### Phase 10: Tree-Based Methods
**Topics:** `#25 Decision Trees`, `#26 DT Variants`, `#27 Combining Classifiers (Boosting)`

- **Decision Tree** — max_depth tuning; tree visualization; feature importance
- **Random Forest** — 200 estimators; compare with single tree
- **Gradient Boosting** — GradientBoostingClassifier on extracted features

### Phase 11: Model Comparison & Final Analysis

- Master comparison table: all models ranked by accuracy, F1, top-5 accuracy
- Best classical vs best deep learning: accuracy gap analysis
- Effect of dimensionality reduction, normalization, feature type
- Which food categories are hardest? Confusion matrix deep-dive
- Report-to-experiment mapping table

---

## 16. Experiment Matrix

### Group 1 — Feature Comparison (fix model: SVM-RBF)

| Run | Feature | Reducer | Expected insight |
|---|---|---|---|
| 1 | Raw pixels | None | Baseline floor |
| 2 | Color Histogram | None | Color alone |
| 3 | HOG | None | Texture/shape |
| 4 | HOG + Histogram + LBP + GLCM | None | Fused handcrafted |
| 5 | CNN (ResNet-50) | None | Deep features |

### Group 2 — Dimensionality Reduction (fix model: SVM-RBF, feature: fused)

| Run | Reducer | n_dims | Expected insight |
|---|---|---|---|
| 6 | None | ~2242 | Full dim baseline |
| 7 | PCA | 50 | Heavy compression |
| 8 | PCA | 100 | Moderate |
| 9 | PCA | 200 | Light compression |
| 10 | LDA | 19 (C-1) | Supervised reduction |
| 11 | PCA→LDA | 200→19 | Pipeline combo |

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

### Group 4 — Hyperparameter Sweeps (Optuna)

| Run | Model | Search space |
|---|---|---|
| 22 | kNN | k ∈ {1,3,5,7,10,15,21}, distance ∈ {L1,L2,cosine} |
| 23 | SVM RBF | C ∈ [0.01, 100], gamma ∈ {scale, auto, 0.001, 0.01} |
| 24 | MLP | hidden_layers ∈ {1,2,3}, units ∈ {64,128,256,512} |

### Group 5 — Clustering (unsupervised)

| Run | Algorithm | Feature | Expected insight |
|---|---|---|---|
| 25 | KMeans (k=20) | Raw pixels | Poor cluster purity |
| 26 | KMeans (k=20) | Handcrafted | Moderate purity |
| 27 | KMeans (k=20) | CNN embeddings | High purity |
| 28 | GMM (diagonal) | CNN + PCA-50 | Probabilistic clusters |
| 29 | GMM BIC sweep | CNN + PCA-50 | Optimal k selection |
| 30 | Agglomerative | CNN + PCA-50 | Hierarchical structure + dendrogram |

---

## 17. Key Design Decisions

**Why HDF5 for feature cache?**
Feature extraction for 100k images with CNN takes 30–60 minutes. HDF5 allows memory-mapped access — you load only what you need, and it's compressed.

**Why `sklearn.Pipeline` and not manual steps?**
Pipeline ensures the scaler and dimensionality reducer are fit only on training folds during cross-validation. Fitting PCA on the full dataset before CV is data leakage that inflates accuracy.

**Why CNN embeddings for classical classifiers?**
A pretrained CNN has learned a rich feature space from ImageNet. Using these embeddings as inputs to kNN, SVM, etc. answers: *"How much of a classifier's job is the feature representation?"* The answer — a lot — is the project's most important finding.

**Why Optuna over grid search?**
Grid search over SVM's (C, gamma) with 5 values each = 25 fits × 5 folds = 125 pipeline fits. Optuna's TPE sampler finds better hyperparameters in ~30 trials using Bayesian optimization.

**Why LDA can't exceed (n_classes − 1) dimensions?**
LDA maximizes the ratio of between-class to within-class scatter. The between-class scatter matrix has rank at most C−1. This is a fundamental linear algebra constraint — explaining it shows theoretical understanding.

**Why GLCM and LBP alongside HOG?**
HOG captures edge orientation but misses texture periodicity and local patterns. GLCM captures spatial co-occurrence of pixel intensities (texture roughness/smoothness), while LBP captures micro-texture patterns. Food images depend heavily on texture (grilled vs. fried vs. raw).

**Why Agglomerative clustering alongside KMeans?**
KMeans assumes spherical clusters. Agglomerative clustering makes no shape assumption and produces dendrograms that reveal hierarchical food similarity (e.g., all Italian dishes cluster before separating into pizza vs pasta).

---

*All results are reproducible. All experiments are logged. All plots are auto-generated from saved CSVs.*
*To reproduce everything from scratch: `dvc repro`*
