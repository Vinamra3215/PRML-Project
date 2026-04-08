# 🍕 Food-101 Image Classification — Classical ML Pipeline

A comprehensive Food-101 image classification system using **only classical machine learning** — no pretrained models. Implements 7 classifiers across 5 handcrafted feature types, with automated hyperparameter tuning via GridSearchCV and Optuna.

**Best Accuracy: 26.66%** (MLP on Color Histogram, 20 classes)

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Pipeline Overview](#-pipeline-overview)
- [Feature Extractors](#-feature-extractors)
- [Models](#-models)
- [Results](#-results)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Analysis Plots](#-analysis-plots)
- [PRML Course Topics](#-prml-course-topics-covered)
- [Design Decisions](#-key-design-decisions)

---

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/Vinamra3215/PRML-Project.git
cd PRML-Project
conda env create -f environment.yml
conda activate food-prml

# 2. Place Food-101 dataset
# Download from Kaggle and extract to data/
# Expected structure: data/images/<class_name>/<image_id>.jpg
#                     data/meta/meta/classes.txt, train.json, test.json

# 3. Extract features (one-time, cached as HDF5)
python scripts/extract_features.py

# 4. Run all experiments (Phase 1: No PCA + Phase 2: PCA-200)
python scripts/run_all_experiments.py

# 5. Generate analysis plots
python scripts/generate_report_plots.py

# 6. Run unit tests
python -m pytest tests/
```

---

## 📁 Project Structure

```
PRML-Project/
├── scripts/                         # Entry-point scripts
│   ├── extract_features.py          # Step 1: Feature extraction → HDF5 cache
│   ├── run_all_experiments.py       # Step 2: Full pipeline (Phase 1 + Phase 2)
│   ├── generate_report_plots.py     # Step 3: Generate 8 analysis plots
│   ├── run_experiment.py            # Run a single model-feature experiment
│   ├── run_sweep.py                 # Optuna hyperparameter sweep
│   └── run_clustering.py            # Clustering analysis (KMeans, GMM, Agglomerative)
│
├── src/
│   ├── data/
│   │   ├── dataset.py               # Food101Dataset: class loading, splits
│   │   ├── cache.py                 # HDF5 feature save/load
│   │   └── preprocess.py            # Image loading and preprocessing
│   │
│   ├── features/
│   │   ├── base.py                  # Abstract FeatureExtractor with parallel extraction
│   │   ├── histogram.py             # Color Histogram (HSV, 32 bins/channel)
│   │   ├── hog.py                   # HOG (Histogram of Oriented Gradients)
│   │   ├── lbp.py                   # LBP (Local Binary Patterns)
│   │   ├── glcm.py                  # GLCM (Gray-Level Co-occurrence Matrix)
│   │   └── fusion.py                # Fused features (Histogram + HOG + LBP + GLCM)
│   │
│   ├── models/
│   │   ├── registry.py              # MODEL_REGISTRY, build_pipeline, param grids
│   │   └── classical.py             # KDE (Parzen Window) classifier
│   │
│   ├── evaluation/
│   │   ├── metrics.py               # Accuracy, F1, precision, recall
│   │   ├── cross_val.py             # Stratified K-Fold cross-validation
│   │   └── comparison.py            # CSV/Markdown result comparison tables
│   │
│   ├── experiment/
│   │   ├── runner.py                # Experiment runner with GridSearchCV + W&B
│   │   ├── sweeper.py               # Optuna TPE-based hyperparameter sweeps
│   │   └── clustering_runner.py     # Clustering experiment runner
│   │
│   ├── clustering/
│   │   ├── kmeans.py                # KMeans + MiniBatch KMeans + elbow plot
│   │   ├── gmm.py                   # GMM clustering + BIC/AIC sweep
│   │   └── agglomerative.py         # Agglomerative clustering + dendrogram
│   │
│   ├── analysis/
│   │   └── model_insights.py        # Auto-summary, model ranking, confusion matrix
│   │
│   └── utils/
│       ├── seed.py                  # Reproducibility (numpy, random, torch)
│       ├── io.py                    # File I/O utilities
│       └── logging.py               # W&B integration, CSV logging, Timer
│
├── tests/
│   ├── test_features.py             # Unit tests for all feature extractors
│   ├── test_metrics.py              # Unit tests for evaluation metrics
│   └── test_pipeline.py             # Unit tests for model pipeline
│
├── results/
│   ├── metrics/                     # CSV tables, markdown reports, experiment logs
│   └── plots/                       # 8 analysis plots (PNG)
│
├── environment.yml                  # Conda environment specification
├── .gitignore
└── README.md
```

---

## ⚙️ Pipeline Overview

```
Food-101 Images (20 classes, 1000 images/class)
        │
        ▼
┌─────────────────────────────┐
│  Feature Extraction          │   scripts/extract_features.py
│  Histogram │ HOG │ LBP │    │
│  GLCM │ Fused               │
└─────────────┬───────────────┘
              │ HDF5 Cache
              ▼
┌─────────────────────────────┐
│  Phase 1: No PCA             │   scripts/run_all_experiments.py
│  7 models × 5 features      │
│  = 31 experiments            │   (skips high-dim combos)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 2: PCA-200            │
│  7 models × 2 features      │
│  (hog + fused only)          │
│  = 14 experiments            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  8 Analysis Plots            │   scripts/generate_report_plots.py
│  + CSV/Markdown Tables       │
└─────────────────────────────┘
```

---

## 🛠 Feature Extractors

| Feature | Method | Dimensions | Library |
|---------|--------|:----------:|---------|
| **Color Histogram** | HSV color distribution (32 bins/channel) | 96 | OpenCV |
| **HOG** | Edge orientation histograms (8×8 cells, 9 orientations) | ~8,100 | scikit-image |
| **LBP** | Local texture patterns (24 points, radius 3) | 26 | scikit-image |
| **GLCM** | Texture statistics (5 properties × 2 distances × 4 angles) | 40 | scikit-image |
| **Fused** | Concatenation of all above | ~8,262 | numpy |

---

## 🤖 Models

| Model | Type | PRML Topic |
|-------|------|------------|
| **KNN** | Instance-based | K-Nearest Neighbors |
| **Logistic Regression** | Linear discriminative | Maximum likelihood estimation |
| **Naive Bayes** | Generative | Bayesian classification |
| **Decision Tree** | Non-parametric | Hierarchical partitioning |
| **Perceptron** | Linear | Linear discriminant |
| **MLP** | Neural network | Multilayer perceptron (sklearn) |
| **KDE (Parzen Window)** | Density-based generative | Bayesian decision theory |

---

## 📊 Results

### Phase 1: No PCA (Raw Feature Dimensions)

| Model | Features | CV Acc | CV F1 | Test Acc | Test F1 | Time (s) |
|:------|:---------|-------:|------:|---------:|--------:|---------:|
| **mlp_sklearn** | **histogram** | 0.2466 | 0.2362 | **0.2666** | **0.2643** | 142.1 |
| mlp_sklearn | fused | 0.2472 | 0.2423 | 0.2636 | 0.2610 | 3199.1 |
| logistic | histogram | 0.2274 | 0.2181 | 0.2386 | 0.2299 | 9.0 |
| mlp_sklearn | hog | 0.2097 | 0.2110 | 0.2304 | 0.2276 | 3356.2 |
| logistic | lbp | 0.2211 | 0.2021 | 0.2252 | 0.2063 | 3.6 |
| naive_bayes | fused | 0.2094 | 0.1935 | 0.2120 | 0.1967 | 82.3 |
| knn | histogram | 0.1865 | 0.1697 | 0.2052 | 0.1866 | 39.2 |
| kde | histogram | 0.1638 | 0.1481 | 0.1864 | 0.1665 | 92.7 |
| decision_tree | histogram | 0.1531 | 0.1500 | 0.1560 | 0.1546 | 27.1 |
| perceptron | fused | 0.1491 | 0.1478 | 0.1550 | 0.1551 | 838.2 |

> 🏆 **Phase 1 Best:** MLP on Color Histogram — **26.66% test accuracy**

### Phase 2: With PCA-200

| Model | Features | CV Acc | CV F1 | Test Acc | Test F1 | Time (s) |
|:------|:---------|-------:|------:|---------:|--------:|---------:|
| **logistic** | **fused** | 0.2422 | 0.2391 | **0.2658** | **0.2592** | 251.8 |
| logistic | hog | 0.2189 | 0.2151 | 0.2424 | 0.2373 | 245.2 |
| mlp_sklearn | fused | 0.2166 | 0.2216 | 0.2406 | 0.2400 | 279.9 |
| mlp_sklearn | hog | 0.2035 | 0.1964 | 0.2170 | 0.2156 | 346.8 |
| naive_bayes | fused | 0.1871 | 0.1877 | 0.1928 | 0.1891 | 233.5 |
| knn | fused | 0.1446 | 0.1330 | 0.1520 | 0.1390 | 695.7 |
| kde | fused | 0.0742 | 0.0724 | 0.0716 | 0.0694 | 324.5 |

> 🏆 **Phase 2 Best:** Logistic Regression on Fused features — **26.58% test accuracy**

### Key Findings

| Insight | Detail |
|---------|--------|
| **Best overall model** | MLP (histogram, no PCA) — 26.66% accuracy |
| **Best with PCA** | Logistic Regression (fused, PCA-200) — 26.58% accuracy |
| **Fastest model** | Naive Bayes — 0.5s training time |
| **Best feature** | Color Histogram — consistently top across models |
| **PCA effect** | Reduces fused (8,262d → 200d) with minimal accuracy loss |
| **Curse of dimensionality** | KNN and Decision Tree degrade on high-dim HOG/fused features |
| **KDE (Parzen)** | Works well on low-dim features (histogram: 18.6%) but struggles with high dimensions |

---

## 🔧 Hyperparameter Tuning

**All hyperparameters are found via GridSearchCV** — nothing is hardcoded.

| Model | Search Space |
|-------|-------------|
| **KNN** | k ∈ {1,3,5,7,9,11,15,21}, weights ∈ {uniform, distance}, metric ∈ {euclidean, manhattan} |
| **Logistic Regression** | C ∈ {0.01, 0.1, 1.0, 10.0} |
| **Naive Bayes** | var_smoothing ∈ {1e-9, 1e-8, 1e-7, 1e-6} |
| **Decision Tree** | max_depth ∈ {5,10,20,30,None}, min_samples_split ∈ {2,5,10} |
| **MLP** | layers ∈ {(128,),(256,128),(512,256,128)}, activation ∈ {relu, tanh} |
| **Perceptron** | penalty ∈ {None, l1, l2}, alpha ∈ {1e-4, 1e-3, 1e-2} |
| **KDE (Parzen)** | bandwidth ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, kernel ∈ {gaussian, tophat, epanechnikov} |

**Optuna** (Bayesian optimization with TPE sampler) for deeper sweeps:

```bash
python scripts/run_sweep.py --model knn --features fused --n-trials 30
python scripts/run_sweep.py --model logistic --features hog
```

---

## 📈 Analysis Plots (8 Total)

| # | Plot | Purpose |
|---|------|---------|
| 1 | `model_comparison.png` | All models ranked by accuracy |
| 2 | `feature_comparison.png` | Best accuracy per feature type |
| 3 | `pca_explained_variance.png` | Variance vs PCA components |
| 4 | `accuracy_vs_f1.png` | Accuracy-F1 correlation |
| 5 | `cv_vs_test_accuracy.png` | Generalization gap (bias-variance) |
| 6 | `confusion_best_model.png` | 20-class confusion matrix for best model |
| 7 | `runtime_comparison.png` | Computational cost comparison |
| 8 | `cnn_loss_curve.png` | MLP training vs validation loss |

---

## 🎓 PRML Course Topics Covered

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Bayesian Decision Theory | KDE classifier (generative Bayesian model) |
| 2 | Density Estimation & Parzen Window | `KDEClassifier` with multiple kernels |
| 3 | Dimensionality Reduction (PCA) | PCA-200 applied uniformly; variance analysis |
| 4 | K-Nearest Neighbors | KNN with GridSearchCV over k, weights, metric |
| 5 | Naive Bayes Classifier | GaussianNB with variance smoothing tuning |
| 6 | Logistic Regression | Multinomial logistic with L2 regularization |
| 7 | Perceptron & Linear Models | Perceptron with L1/L2 penalty tuning |
| 8 | Neural Networks (MLP) | MLPClassifier with loss curve tracking |
| 9 | Decision Trees | Pruned via max_depth, min_samples tuning |
| 10 | Model Evaluation | Stratified K-Fold CV, confusion matrix, F1 |
| 11 | Hyperparameter Optimization | GridSearchCV + Optuna (Bayesian) |
| 12 | Feature Engineering | HOG, LBP, GLCM, Color Histogram, Fusion |
| 13 | Bias-Variance Tradeoff | CV vs Test accuracy analysis |
| 14 | Clustering | KMeans, GMM (EM), Agglomerative (hierarchical) |
| 15 | Experiment Tracking | W&B integration, CSV logging |

---

## 🧑‍💻 Usage

### Run single experiment
```bash
python scripts/run_experiment.py --model knn --features hog
python scripts/run_experiment.py --model logistic --features fused --reducer pca --pca-components 200
```

### Run clustering analysis
```bash
python scripts/run_clustering.py --features histogram
```

### Run unit tests
```bash
python -m pytest tests/ -v
```

---

## 📦 Dependencies

```
scikit-learn >= 1.3
scikit-image
numpy
pandas
matplotlib
seaborn
opencv-python-headless
h5py
tqdm
torch
torchvision
optuna
wandb (optional)
```

---

## 🔬 Key Design Decisions

1. **No pretrained models**: All features are handcrafted (HOG, LBP, GLCM, Color Histogram) — no ResNet, ViT, or transfer learning. This keeps the project within classical ML / PRML scope.

2. **GridSearchCV for all models**: No hyperparameter is hardcoded. Every model's parameters are discovered through cross-validated grid search.

3. **Two comparison tables**: Models are evaluated both with and without PCA to ensure fair comparison across different feature dimensionalities.

4. **KDE (Parzen Window) Classifier**: Custom implementation of density-estimation-based classification, directly implementing Bayesian decision theory from PRML course.

5. **Reproducibility**: Fixed seed (42), stratified splits, cached features in HDF5, detailed experiment logging.

6. **Modular architecture**: Each component (features, models, evaluation) is a separate package — easy to extend with new extractors or classifiers.

---
