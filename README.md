# 🍕 Food Image Classification — PRML Course Project

A comprehensive machine learning pipeline for classifying food images from **20 categories** of the [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101), built entirely with **classical ML algorithms and handcrafted features** — no pretrained models.

This project demonstrates core Pattern Recognition & Machine Learning (PRML) concepts including feature engineering, dimensionality reduction, kernel methods, density estimation, Bayesian classification, and neural networks.

---

## 📊 Project Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | Food-101 (20 classes, ~20K images) |
| **Features** | Color Histogram (HSV), HOG, LBP, GLCM, Fused |
| **Models** | KNN, Logistic Regression, Naive Bayes, Decision Tree, Gradient Boosting, Perceptron, MLP, KDE (Parzen Window) |
| **Tuning** | GridSearchCV with Stratified 5-Fold Cross-Validation |
| **Dim. Reduction** | PCA (200 components) |
| **Tracking** | Local CSV logging + optional Weights & Biases |
| **Optimization** | Optuna (Bayesian / TPE sampler) |

---

## 🎯 PRML Course Topics Covered

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Bayesian Decision Theory | Naive Bayes classifier, KDE posterior estimation |
| 2 | Maximum Likelihood Estimation | Logistic Regression via MLE optimization |
| 3 | Non-parametric Methods | KNN classifier, Parzen Window (KDE) density estimation |
| 4 | Dimensionality Reduction | PCA (200 components), LDA |
| 5 | Linear Classifiers | Logistic Regression, Perceptron, SGD |
| 6 | Kernel Methods | SVM with RBF kernel |
| 7 | Neural Networks | MLP (sklearn), CNN embeddings |
| 8 | Ensemble Methods | Gradient Boosting, Decision Trees |
| 9 | Model Selection | GridSearchCV, Stratified K-Fold CV |
| 10 | Clustering | K-Means, GMM (BIC/AIC), Agglomerative |
| 11 | Feature Engineering | HOG, LBP, GLCM, Color Histograms, Feature Fusion |

---

## 🏗 Project Architecture

```
PRML-Project/
├── configs/                     # Hydra YAML configurations
│   ├── config.yaml              # Root config
│   ├── model/                   # Per-model hyperparameter configs
│   ├── features/                # Feature extraction configs
│   ├── reduction/               # PCA / LDA / None configs
│   └── experiment/              # Experiment sweep definitions
│
├── scripts/                     # Entry-point scripts
│   ├── extract_features.py      # Precompute & cache all features (HDF5)
│   ├── run_all_experiments.py   # ★ Main pipeline: 4-phase experiment runner
│   ├── run_experiment.py        # Single model experiment
│   ├── run_sweep.py             # Optuna hyperparameter optimization
│   ├── run_clustering.py        # Clustering analysis (K-Means, GMM)
│   ├── run_cnn.py               # CNN training with PyTorch Lightning
│   └── generate_report_plots.py # Generate 8 analysis/report plots
│
├── src/                         # Core source code
│   ├── data/                    # Data loading & caching
│   │   ├── dataset.py           # Food101Dataset with stratified splits
│   │   ├── cache.py             # HDF5 feature caching (save/load)
│   │   └── preprocess.py        # Image preprocessing utilities
│   │
│   ├── features/                # Handcrafted feature extractors
│   │   ├── base.py              # Abstract base class with parallel extraction
│   │   ├── histogram.py         # Color Histogram (HSV, 32 bins/channel)
│   │   ├── hog.py               # Histogram of Oriented Gradients
│   │   ├── lbp.py               # Local Binary Patterns
│   │   ├── glcm.py              # Gray-Level Co-occurrence Matrix
│   │   ├── fusion.py            # Concatenated multi-feature vector
│   │   └── cnn_embeddings.py    # ResNet feature embeddings (optional)
│   │
│   ├── models/                  # Classifiers
│   │   ├── registry.py          # Model & reducer registry + GridSearchCV grids
│   │   ├── classical.py         # KDE (Parzen Window) classifier
│   │   ├── mlp.py               # Custom MLP wrapper
│   │   └── cnn/                 # CNN model (PyTorch Lightning)
│   │
│   ├── evaluation/              # Metrics & comparison
│   │   ├── metrics.py           # Accuracy, F1, precision, recall
│   │   ├── cross_val.py         # Stratified K-Fold cross-validation
│   │   └── comparison.py        # Result comparison & CSV logging
│   │
│   ├── experiment/              # Experiment orchestration
│   │   ├── runner.py            # Core experiment runner (GridSearchCV + W&B)
│   │   ├── sweeper.py           # Optuna hyperparameter sweep
│   │   └── clustering_runner.py # Clustering experiment runner
│   │
│   ├── analysis/                # Post-experiment analysis
│   │   └── model_insights.py    # Model ranking, confusion analysis, auto-summary
│   │
│   ├── reduction/               # Dimensionality reduction wrappers
│   │   ├── pca_reducer.py       # PCA wrapper
│   │   └── lda_reducer.py       # LDA wrapper
│   │
│   ├── visualization/           # Plotting utilities
│   │   ├── confusion_plot.py    # Confusion matrix heatmaps
│   │   ├── learning_curves.py   # Training/validation curves
│   │   ├── umap_plot.py         # UMAP embedding plots
│   │   ├── shap_analysis.py     # SHAP feature importance
│   │   ├── gradcam.py           # GradCAM for CNN
│   │   └── filter_demo.py       # Convolution filter visualization
│   │
│   └── utils/                   # Shared utilities
│       ├── seed.py              # Reproducibility (random, numpy, torch)
│       ├── logging.py           # W&B integration + CSV experiment logger
│       └── io.py                # File I/O helpers
│
├── results/                     # Generated outputs (not tracked in git)
│   ├── plots/                   # 8 analysis plots (PNG)
│   └── metrics/                 # CSV results, classification reports
│
├── tests/                       # Unit tests
├── app/                         # Streamlit demo application
├── environment.yml              # Conda environment specification
├── pyproject.toml               # Build & project metadata
└── dvc.yaml                     # DVC pipeline definition
```

---

## 🔬 Experiment Pipeline

The main pipeline (`scripts/run_all_experiments.py`) runs **4 phases**:

### Phase 1 — No PCA (Raw Feature Dimensions)
> 7 models × 5 features = 35 experiments  
> Each tuned via GridSearchCV with Stratified 5-Fold CV

### Phase 2 — With PCA (200 Dimensions)
> Same 35 experiments, but features are reduced to 200 dimensions via PCA  
> Tests the effect of dimensionality reduction on classifier performance

### Phase 3 — MLP Loss Curve
> Trains an MLP on fused features for 50 epochs  
> Records train/validation loss per epoch for convergence analysis

### Phase 4 — Auto Summary
> Generates markdown comparison tables, classification reports  
> Identifies the best model–feature combination automatically

---

## 📈 Generated Plots

The pipeline produces **8 analysis plots** in `results/plots/`:

| Plot | Description |
|------|-------------|
| `model_comparison.png` | Grouped bar chart — accuracy by model across all features |
| `feature_comparison.png` | Feature-level performance comparison |
| `pca_explained_variance.png` | Cumulative explained variance vs. number of PCA components |
| `accuracy_vs_f1.png` | Scatter plot — accuracy vs F1 score for all experiments |
| `cv_vs_test_accuracy.png` | Cross-validation accuracy vs test accuracy (overfitting check) |
| `confusion_best_model.png` | 20×20 confusion matrix for the best model |
| `runtime_comparison.png` | Training time comparison across models and features |
| `cnn_loss_curve.png` | MLP training/validation loss curve over 50 epochs |

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
conda env create -f environment.yml
conda activate food-prml
```

### 2. Download Data
Download [Food-101 from Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101) and extract to:
```
data/
├── images/
│   ├── apple_pie/
│   ├── baby_back_ribs/
│   └── ... (101 food categories)
└── meta/
    └── meta/
        ├── train.json
        └── test.json
```

### 3. Run the Pipeline
```bash
# Step 1: Extract & cache features (HOG, LBP, GLCM, Histogram, Fused)
python scripts/extract_features.py

# Step 2: Run all experiments (GridSearchCV, 4 phases)
python scripts/run_all_experiments.py

# Step 3: Generate report plots
python scripts/generate_report_plots.py
```

### 4. Run Specific Phases
```bash
python scripts/run_all_experiments.py --phase 1   # Phase 1: No PCA
python scripts/run_all_experiments.py --phase 2   # Phase 2: PCA-200
python scripts/run_all_experiments.py --phase 3   # Phase 3: MLP loss curve
python scripts/run_all_experiments.py --phase 4   # Phase 4: Auto summary
```

---

## 🛠 Tech Stack

- **Python 3.11** — Core language
- **scikit-learn** — ML models, GridSearchCV, metrics, PCA
- **scikit-image** — HOG, LBP, GLCM feature extraction
- **OpenCV** — Image I/O and color space conversion
- **NumPy / Pandas** — Data processing
- **Matplotlib / Seaborn** — Visualization
- **h5py** — HDF5 feature caching
- **tqdm** — Progress bars
- **Optuna** — Bayesian hyperparameter optimization
- **Weights & Biases** — Optional experiment tracking
- **PyTorch / torchvision** — CNN embeddings (optional)

---

## 📂 Results

Results are saved to `results/`:
- `results/metrics/master_no_pca.csv` — Phase 1 results (all experiments)
- `results/metrics/master_with_pca.csv` — Phase 2 results (PCA-200)
- `results/metrics/experiment_log.csv` — Full experiment log
- `results/metrics/cnn_history.json` — MLP loss curve data
- `results/plots/*.png` — 8 analysis plots

---

## 📝 License

This project is developed as part of the **Pattern Recognition & Machine Learning** course at IIT Jodhpur.

---

## 🔜 Roadmap

- [x] Project structure & configuration
- [x] Utility layer (seed, I/O, logging)
- [ ] Data loading & caching layer
- [ ] Feature extraction (Histogram, HOG, LBP, GLCM, Fusion)
- [ ] Model registry & classifiers (KNN, LR, NB, DT, GB, MLP, KDE)
- [ ] Evaluation & cross-validation
- [ ] Experiment pipeline (4-phase runner)
- [ ] Visualization & report plots
- [ ] Clustering analysis (K-Means, GMM, Agglomerative)
- [ ] Tests & Streamlit demo app