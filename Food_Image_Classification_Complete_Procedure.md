# 🍕 Food Image Classification — Complete Project Procedure

> **Course:** Pattern Recognition & Machine Learning (PRML)
> **Project Title:** Food Image Classification
> **Dataset:** [Food-101 (Kaggle)](https://www.kaggle.com/datasets/kmader/food41) — 101 food categories, ~1,000 images per category
> **Goal:** Develop a machine learning pipeline that classifies food images, integrating the **maximum number of course topics** to demonstrate comprehensive understanding.

---

## 📋 Table of Contents

1. [Course Topics Coverage Map](#1-course-topics-coverage-map)
2. [Tech Stack](#2-tech-stack)
3. [Project Architecture Overview](#3-project-architecture-overview)
4. [Phase-by-Phase Implementation Procedure](#4-phase-by-phase-implementation-procedure)
   - [Phase 1: Introduction & Data Loading](#phase-1-introduction--data-loading)
   - [Phase 2: Feature Engineering](#phase-2-feature-engineering)
   - [Phase 3: Data Preprocessing & Normalization](#phase-3-data-preprocessing--normalization)
   - [Phase 4: Exploratory Data Analysis](#phase-4-exploratory-data-analysis)
   - [Phase 5: Dimensionality Reduction](#phase-5-dimensionality-reduction)
   - [Phase 6: Classical ML Classifiers](#phase-6-classical-ml-classifiers)
   - [Phase 7: Neural Network Classifiers](#phase-7-neural-network-classifiers)
   - [Phase 8: Convolutional Neural Network (CNN)](#phase-8-convolutional-neural-network-cnn)
   - [Phase 9: Unsupervised Learning & Clustering](#phase-9-unsupervised-learning--clustering)
   - [Phase 10: Tree-Based & Ensemble Methods](#phase-10-tree-based--ensemble-methods)
   - [Phase 11: Model Comparison & Final Analysis](#phase-11-model-comparison--final-analysis)
5. [Directory Structure](#5-directory-structure)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Deliverables](#7-deliverables)

---

## 1. Course Topics Coverage Map

Every single course topic is mapped to a specific phase and implementation step in this project:

| # | Course Topic | Project Phase | How It's Used |
|---|-------------|--------------|---------------|
| 1 | **Introduction** | Phase 1 | Problem formulation, dataset overview, ML pipeline introduction |
| 2 | **Feature computation & classification using histogram** | Phase 2 | Extract color histograms (RGB, HSV) as features; use histogram-based baseline classifier |
| 3 | **Multi-dimensional features, Cost of error, Distribution** | Phase 2 & 4 | Combine multiple feature types (color, texture, shape); analyze error distributions; study feature distributions |
| 4 | **Bayes classifier** | Phase 6 | Gaussian Naive Bayes classifier on extracted features |
| 5 | **Distance & similarity measures, kNN classifier** | Phase 6 | kNN with Euclidean, Manhattan, and Cosine distance on feature vectors |
| 6 | **Weighted kNN, Data representation, Data normalization** | Phase 3 & 6 | Distance-weighted kNN; Min-Max / Z-score normalization of features |
| 7 | **Cross-validation** | Phase 6–10 | k-Fold and Stratified k-Fold CV for all model evaluations |
| 8 | **Linear regression** | Phase 6 | Ridge regression adapted for classification (multi-output regression → argmax) |
| 9 | **Gradient descent** | Phase 6 & 7 | SGD classifier; Manual GD implementation for logistic regression; GD-based training of MLP/CNN |
| 10 | **Logistic regression** | Phase 6 | Multinomial logistic regression (softmax regression) as baseline |
| 11 | **Multi-class classification, Overfitting, Regularization** | Phase 6–8 | One-vs-Rest / Softmax strategies; L1/L2 regularization; Dropout; Early stopping; Data augmentation |
| 12 | **Dimensionality reduction, Covariance, Correlation** | Phase 5 | Covariance matrix of features; Feature correlation heatmap; Dimensionality reduction pipeline |
| 13 | **PCA & LDA** | Phase 5 | PCA for feature compression; LDA for supervised dimensionality reduction; Visualization in 2D/3D |
| 14 | **Linear Classifier: Perceptron** | Phase 7 | Single-layer Perceptron for binary & OvR multi-class classification |
| 15 | **Linear Classifier: SVM** | Phase 6 | Linear SVM (LinearSVC) on feature vectors |
| 16 | **SVM variants, Kernel SVM** | Phase 6 | RBF, Polynomial, and Sigmoid kernel SVMs |
| 17 | **Multi-layer Perceptron (MLP)** | Phase 7 | Fully-connected MLP using both sklearn and PyTorch/TensorFlow |
| 18 | **Similarity: Perceptron ↔ Neuron, MLP** | Phase 7 | Documented comparison of biological neuron and perceptron; MLP architecture analysis |
| 19 | **Backpropagation algorithm** | Phase 7 | Backprop in MLP training; gradient flow visualization; manual backprop implementation |
| 20 | **Convolution filter, Convolution operation** | Phase 8 | Apply Sobel, Gabor, Gaussian filters; visualize convolution operations on food images |
| 21 | **CNN** | Phase 8 | Custom CNN architecture + Transfer Learning (ResNet, VGG, EfficientNet) |
| 22 | **Data Clustering** | Phase 9 | Cluster food images by visual features; discover natural groupings |
| 23 | **KMeans Clustering (EM) and variants** | Phase 9 | KMeans, KMeans++, Mini-Batch KMeans on image features |
| 24 | **Agglomerative clustering, Practices related to clustering** | Phase 9 | Hierarchical clustering with dendrograms; Silhouette analysis; Elbow method |
| 25 | **Decision Trees** | Phase 10 | Decision tree classifier on extracted features; tree visualization |
| 26 | **Variants of Decision Tree** | Phase 10 | Random Forest classifier |
| 27 | **Combining Classifiers (Boosting)** | Phase 10 | AdaBoost, Gradient Boosting (XGBoost/LightGBM); Voting ensemble of all models |

> **Total: 27/27 course topics covered ✅**

---

## 2. Tech Stack

### Core Languages & Frameworks

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.10+ | Primary programming language |
| **Deep Learning** | PyTorch / TensorFlow-Keras | CNN, MLP, Backpropagation |
| **Classical ML** | scikit-learn | kNN, SVM, Logistic Regression, Decision Trees, PCA, LDA, Clustering |
| **Boosting** | XGBoost / LightGBM | Gradient Boosting classifiers |
| **Image Processing** | OpenCV, PIL/Pillow | Feature extraction, image preprocessing |
| **Data Handling** | NumPy, Pandas, h5py | Numerical operations, data management, HDF5 reading |
| **Visualization** | Matplotlib, Seaborn, Plotly | Plots, charts, confusion matrices, t-SNE embeddings |
| **Notebook** | Jupyter Notebook | Interactive development & documentation |
| **Experiment Tracking** | TensorBoard / Weights & Biases (optional) | Training monitoring |

### Hardware Requirements
- **GPU recommended** for CNN training (Google Colab free tier with T4 GPU is sufficient)
- **RAM:** 16 GB+ recommended (or use Colab's 12 GB)
- **Storage:** ~6 GB for dataset

---

## 3. Project Architecture Overview

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
│                   │ HOG        │          ▼                          │
│                   │ GLCM       │   ┌──────────────┐                │
│                   │ Edge/Gabor │   │  Dim. Reduc. │                │
│                   └────────────┘   │  PCA / LDA   │                │
│                                    └──────┬───────┘                │
│                                           │                         │
│                    ┌──────────────────────┼──────────────────────┐  │
│                    ▼                      ▼                      ▼  │
│             ┌────────────┐      ┌──────────────┐      ┌──────────┐ │
│             │ Classical  │      │   Neural     │      │ Ensemble │ │
│             │    ML      │      │  Networks    │      │ Methods  │ │
│             ├────────────┤      ├──────────────┤      ├──────────┤ │
│             │ Bayes      │      │ Perceptron   │      │ Dec.Tree │ │
│             │ kNN        │      │ MLP          │      │ RF       │ │
│             │ SVM/Kernel │      │ CNN          │      │ AdaBoost │ │
│             │ Log. Reg.  │      │ Transfer Lrn │      │ XGBoost  │ │
│             └────────────┘      └──────────────┘      └──────────┘ │
│                    │                      │                    │    │
│                    └──────────────────────┼────────────────────┘    │
│                                           ▼                         │
│                                  ┌──────────────────┐              │
│                                  │ Comparison Table │              │
│                                  │ & Final Analysis │              │
│                                  └──────────────────┘              │
│                                                                     │
│  ┌───────────────────────────────────────────────────┐             │
│  │           UNSUPERVISED ANALYSIS (Parallel)        │             │
│  │  KMeans │ Agglomerative │ Silhouette │ Dendrograms│             │
│  └───────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phase-by-Phase Implementation Procedure

---

### Phase 1: Introduction & Data Loading
**📘 Course Topics Covered:** `#1 Introduction`

#### 1.1 Problem Statement
- Define the food image classification task formally as a **multi-class classification problem**
- Input: RGB food image → Output: one of 101 food categories
- Discuss real-world applications: dietary monitoring, calorie tracking, restaurant automation

#### 1.2 Dataset Loading & Exploration
```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load the HDF5 dataset
# Option A: Using HDF5 files from the Kaggle dataset
h5_file = h5py.File('food_c101_n1000_r64x64x3.h5', 'r')
X = np.array(h5_file['images'])
y = np.array(h5_file['category'])

# Option B: Using raw image folders (if downloaded full Food-101)
from torchvision import datasets
dataset = datasets.ImageFolder('food-101/images/')

print(f"Total images: {len(X)}")
print(f"Image shape: {X[0].shape}")
print(f"Number of classes: {len(np.unique(y))}")
```

#### 1.3 Class Distribution Analysis
- Plot bar charts of samples per class
- Identify class imbalance (if any) and discuss handling strategies
- Display sample images from each category (grid of 10×10)

#### 1.4 Train-Test Split
```python
from sklearn.model_selection import train_test_split

# Stratified split to preserve class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Further split training into train + validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
```

---

### Phase 2: Feature Engineering
**📘 Course Topics Covered:** `#2 Feature computation & histogram`, `#3 Multi-dimensional features, Cost of error, Distribution`

#### 2.1 Color Histogram Features
```python
import cv2

def extract_color_histogram(image, bins=32):
    """Extract color histogram in HSV space"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    return hist / hist.sum()  # Normalize
```

- Extract histograms for **RGB** and **HSV** color spaces
- Demonstrate a **simple histogram-based classifier**: classify by nearest histogram match
- Show how different foods have distinctive color distributions (e.g., sushi vs. pizza)

#### 2.2 Texture Features (GLCM — Gray-Level Co-occurrence Matrix)
```python
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[1, 3], angles=[0, np.pi/4, np.pi/2],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
```

#### 2.3 Edge & Shape Features (HOG — Histogram of Oriented Gradients)
```python
from skimage.feature import hog

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), visualize=True)
    return features, hog_image
```

#### 2.4 Combined Multi-Dimensional Feature Vector
```python
def extract_all_features(image):
    color_hist = extract_color_histogram(image)       # ~96 dims
    glcm_feat = extract_glcm_features(image)          # ~30 dims
    hog_feat, _ = extract_hog_features(image)          # ~variable
    return np.concatenate([color_hist, glcm_feat, hog_feat])
```

- **Multi-dimensional features**: Combining all feature types creates a rich, multi-dimensional representation
- **Cost of error analysis**: Compute misclassification costs for confusing similar food types (e.g., misclassifying steak as hamburger is less costly than misclassifying steak as ice cream)
- **Distribution analysis**: Plot feature distributions (histograms, box plots) for different food classes using select features

---

### Phase 3: Data Preprocessing & Normalization
**📘 Course Topics Covered:** `#6 Data representation, Data normalization`

#### 3.1 Image Preprocessing
```python
from torchvision import transforms

# Standard preprocessing pipeline
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

#### 3.2 Feature Normalization (for classical ML)
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Z-Score Normalization (StandardScaler)
scaler_zscore = StandardScaler()
X_train_zscore = scaler_zscore.fit_transform(X_train_features)
X_test_zscore = scaler_zscore.transform(X_test_features)

# Min-Max Normalization
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train_features)
X_test_minmax = scaler_minmax.transform(X_test_features)
```

- **Compare** model performance with and without normalization
- **Discuss** why normalization is critical for distance-based (kNN), gradient-based (SVM, MLP), and scale-sensitive algorithms

#### 3.3 Data Representation
- **Flattened pixel vectors** — raw image as 1D array (for classical ML baselines)
- **Hand-crafted feature vectors** — from Phase 2 (for classical ML)
- **Tensor format** — 4D tensors (batch × channels × height × width) for CNNs
- Document each representation format and when to use which

---

### Phase 4: Exploratory Data Analysis
**📘 Course Topics Covered:** `#3 Multi-dimensional features, Cost of error, Distribution`, `#12 Covariance, Correlation`

#### 4.1 Feature Distribution Analysis
```python
import seaborn as sns

# Distribution of individual features across classes
for feature_idx in selected_features:
    plt.figure(figsize=(10, 4))
    for cls in sample_classes:
        sns.kdeplot(X_features[y == cls, feature_idx], label=class_names[cls])
    plt.title(f'Feature {feature_idx} Distribution Across Classes')
    plt.legend()
    plt.show()
```

#### 4.2 Correlation Analysis
```python
# Feature correlation heatmap
corr_matrix = np.corrcoef(X_train_features.T)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()
```

#### 4.3 Covariance Analysis
```python
# Covariance matrix computation
cov_matrix = np.cov(X_train_features.T)
print(f"Covariance matrix shape: {cov_matrix.shape}")

# Eigenvalue analysis of covariance matrix (leads into PCA)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
explained_variance_ratio = eigenvalues[::-1] / eigenvalues.sum()

plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot — Variance Explained by Eigenvectors')
plt.show()
```

---

### Phase 5: Dimensionality Reduction
**📘 Course Topics Covered:** `#12 Dimensionality reduction`, `#13 PCA & LDA`

#### 5.1 PCA (Principal Component Analysis)
```python
from sklearn.decomposition import PCA

# Fit PCA — retain 95% variance
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_features)
X_test_pca = pca.transform(X_test_features)

print(f"Original dimensions: {X_train_features.shape[1]}")
print(f"Reduced dimensions: {X_train_pca.shape[1]}")
print(f"Variance retained: {sum(pca.explained_variance_ratio_):.2%}")

# 2D visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_train_features)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='tab20', alpha=0.5, s=5)
plt.title('PCA 2D Projection of Food Features')
plt.colorbar()
plt.show()
```

#### 5.2 LDA (Linear Discriminant Analysis)
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# LDA — supervised dimensionality reduction (max components = n_classes - 1)
lda = LDA(n_components=50)  # Up to 100 for 101 classes
X_train_lda = lda.fit_transform(X_train_features, y_train)
X_test_lda = lda.transform(X_test_features)

print(f"LDA reduced dimensions: {X_train_lda.shape[1]}")

# 2D LDA visualization
lda_2d = LDA(n_components=2)
X_lda_2d = lda_2d.fit_transform(X_train_features, y_train)
plt.scatter(X_lda_2d[:, 0], X_lda_2d[:, 1], c=y_train, cmap='tab20', alpha=0.5, s=5)
plt.title('LDA 2D Projection of Food Features')
plt.colorbar()
plt.show()
```

#### 5.3 PCA vs LDA Comparison
- Compare classification accuracy using PCA-reduced vs LDA-reduced features
- Discuss **unsupervised** (PCA) vs **supervised** (LDA) reduction
- Show 2D/3D scatter plots for both, highlighting separation quality

---

### Phase 6: Classical ML Classifiers
**📘 Course Topics Covered:** `#4 Bayes`, `#5 kNN`, `#6 Weighted kNN`, `#7 Cross-validation`, `#8 Linear regression`, `#9 Gradient descent`, `#10 Logistic regression`, `#11 Multi-class, Overfitting, Regularization`, `#15 SVM`, `#16 Kernel SVM`

> **Note:** Use a **subset of classes** (e.g., 10–20 food categories) for classical ML methods to keep computation tractable. Use both original and PCA/LDA-reduced features.

#### 6.1 Bayes Classifier
```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train_pca, y_train)
y_pred_bayes = gnb.predict(X_test_pca)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_bayes):.4f}")
```

#### 6.2 kNN Classifier
```python
from sklearn.neighbors import KNeighborsClassifier

# Standard kNN with different distance metrics
for metric in ['euclidean', 'manhattan', 'cosine']:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_pca, y_train)
    acc = knn.score(X_test_pca, y_test)
    print(f"kNN (k=5, {metric}): {acc:.4f}")
```

#### 6.3 Weighted kNN
```python
# Distance-weighted kNN
knn_weighted = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
knn_weighted.fit(X_train_pca, y_train)
y_pred_wknn = knn_weighted.predict(X_test_pca)

# Hyperparameter tuning: optimal k
k_values = range(1, 31)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    knn.fit(X_train_pca, y_train)
    accuracies.append(knn.score(X_test_pca, y_test))

plt.plot(k_values, accuracies)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Weighted kNN — Accuracy vs k')
plt.show()
```

#### 6.4 Cross-Validation
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Stratified k-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Naive Bayes': GaussianNB(),
    'kNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'Weighted kNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_pca, y_train, cv=skf, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

#### 6.5 Linear Regression (for Classification)
```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer

# One-hot encode labels, apply ridge regression, then argmax
lb = LabelBinarizer()
y_train_onehot = lb.fit_transform(y_train)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_pca, y_train_onehot)
y_pred_ridge = lb.classes_[ridge.predict(X_test_pca).argmax(axis=1)]
print(f"Ridge Regression Accuracy: {accuracy_score(y_test, y_pred_ridge):.4f}")
```

#### 6.6 Gradient Descent — SGD Classifier
```python
from sklearn.linear_model import SGDClassifier

# Stochastic Gradient Descent classifier
sgd = SGDClassifier(loss='hinge', max_iter=1000, learning_rate='optimal', random_state=42)
sgd.fit(X_train_pca, y_train)
y_pred_sgd = sgd.predict(X_test_pca)
print(f"SGD Classifier Accuracy: {accuracy_score(y_test, y_pred_sgd):.4f}")

# Manual GD implementation for Logistic Regression (simplified, 2 classes for demonstration)
def gradient_descent_logistic(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    W = np.zeros(n)
    b = 0
    losses = []
    for epoch in range(epochs):
        z = X @ W + b
        sigmoid = 1 / (1 + np.exp(-z))
        loss = -np.mean(y * np.log(sigmoid + 1e-8) + (1-y) * np.log(1 - sigmoid + 1e-8))
        losses.append(loss)
        dW = (1/m) * X.T @ (sigmoid - y)
        db = (1/m) * np.sum(sigmoid - y)
        W -= lr * dW
        b -= lr * db
    return W, b, losses
```

#### 6.7 Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Multinomial Logistic Regression (Softmax)
log_reg = LogisticRegression(
    multi_class='multinomial', solver='lbfgs', max_iter=2000,
    C=1.0, random_state=42
)
log_reg.fit(X_train_pca, y_train)
y_pred_lr = log_reg.predict(X_test_pca)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
```

#### 6.8 Multi-class, Overfitting & Regularization Analysis
```python
# Demonstrate Overfitting: train vs test accuracy curves
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    LogisticRegression(max_iter=2000), X_train_pca, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
)
plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Samples')
plt.ylabel('Accuracy')
plt.title('Learning Curve — Overfitting Analysis')
plt.legend()
plt.show()

# Regularization comparison (L1 vs L2)
for penalty, C_val in [('l1', 0.01), ('l1', 1.0), ('l2', 0.01), ('l2', 1.0)]:
    lr = LogisticRegression(penalty=penalty, C=C_val, solver='saga', max_iter=2000)
    lr.fit(X_train_pca, y_train)
    print(f"LogReg ({penalty}, C={C_val}): Train={lr.score(X_train_pca, y_train):.4f}, "
          f"Test={lr.score(X_test_pca, y_test):.4f}")
```

#### 6.9 SVM — Linear
```python
from sklearn.svm import LinearSVC, SVC

# Linear SVM
linear_svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
linear_svm.fit(X_train_pca, y_train)
print(f"Linear SVM Accuracy: {linear_svm.score(X_test_pca, y_test):.4f}")
```

#### 6.10 SVM — Kernel Variants
```python
# Kernel SVMs (on smaller subset for speed)
for kernel in ['rbf', 'poly', 'sigmoid']:
    svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train_pca[:5000], y_train[:5000])
    acc = svm.score(X_test_pca, y_test)
    print(f"SVM ({kernel}): {acc:.4f}")
```

---

### Phase 7: Neural Network Classifiers
**📘 Course Topics Covered:** `#14 Perceptron`, `#17 MLP`, `#18 Perceptron ↔ Neuron similarity`, `#19 Backpropagation`, `#9 Gradient descent`

#### 7.1 Perceptron
```python
from sklearn.linear_model import Perceptron

# Single-layer Perceptron
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(X_train_pca, y_train)
print(f"Perceptron Accuracy: {perceptron.score(X_test_pca, y_test):.4f}")
```

#### 7.2 Biological Neuron vs Perceptron Analysis
- **Document and visualize** the analogy:
  - Dendrites ↔ Input features (x₁, x₂, ...)
  - Synaptic weights ↔ Learnable weights (w₁, w₂, ...)
  - Cell body ↔ Weighted sum (Σ wᵢxᵢ + b)
  - Axon hillock ↔ Activation function (step/sigmoid/ReLU)
  - Axon output ↔ Prediction
- Draw architecture comparison diagrams in the notebook

#### 7.3 Multi-Layer Perceptron (MLP) — sklearn
```python
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu', solver='adam',
    max_iter=500, random_state=42,
    early_stopping=True, validation_fraction=0.1
)
mlp.fit(X_train_pca, y_train)
print(f"MLP Accuracy: {mlp.score(X_test_pca, y_test):.4f}")

# Plot loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('MLP Training Loss Curve')
plt.show()
```

#### 7.4 MLP — PyTorch (with Backpropagation)
```python
import torch
import torch.nn as nn
import torch.optim as optim

class FoodMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Training with Backpropagation and Gradient Descent
model = FoodMLP(input_dim=X_train_pca.shape[1], num_classes=101)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization

# Training loop — backpropagation in action
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()          # ← BACKPROPAGATION
    optimizer.step()          # ← GRADIENT DESCENT UPDATE
```

#### 7.5 Backpropagation Visualization
- Visualize gradient magnitudes per layer across training epochs
- Discuss vanishing/exploding gradients
- Show the computational graph of the forward and backward pass

---

### Phase 8: Convolutional Neural Network (CNN)
**📘 Course Topics Covered:** `#20 Convolution filter & operation`, `#21 CNN`, `#11 Overfitting, Regularization`

#### 8.1 Convolution Filter Demonstrations
```python
import cv2

# Apply and visualize classical convolution filters on food images
sample_image = X_train[0]
gray = cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY)

# Sobel filters (edge detection)
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Gaussian blur
gaussian = cv2.GaussianBlur(gray, (5, 5), 0)

# Gabor filter (texture detection)
gabor_kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi/4, 10.0, 0.5, 0)
gabor_filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)

# Laplacian (second derivative edge detection)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Visualize all filters side by side
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
filters = [gray, sobel_x, sobel_y, gaussian, gabor_filtered, laplacian]
titles = ['Original', 'Sobel-X', 'Sobel-Y', 'Gaussian', 'Gabor', 'Laplacian']
for ax, img, title in zip(axes.flatten(), filters, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
plt.suptitle('Convolution Filter Operations on Food Image')
plt.show()
```

#### 8.2 Custom CNN Architecture
```python
class FoodCNN(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),           # ← REGULARIZATION
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

#### 8.3 Transfer Learning (Pre-trained CNN)
```python
import torchvision.models as models

# ResNet-50 Transfer Learning
resnet = models.resnet50(pretrained=True)

# Freeze early layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final FC layer
resnet.fc = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 101)
)

# Only train the final layers
optimizer = optim.Adam(resnet.fc.parameters(), lr=0.001)
```

#### 8.4 CNN Training with Regularization Techniques
```python
# Data Augmentation (reduces overfitting)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

# Training loop with all regularization
early_stopping = EarlyStopping(patience=7)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
```

#### 8.5 CNN Feature Map Visualization
- Visualize what each convolutional layer learns (filters from Layer 1, 2, 3)
- Show activation maps / Grad-CAM heatmaps highlighting which regions the CNN focuses on
- Compare learned CNN filters with hand-crafted filters from Phase 8.1

---

### Phase 9: Unsupervised Learning & Clustering
**📘 Course Topics Covered:** `#22 Data Clustering`, `#23 KMeans (EM) and variants`, `#24 Agglomerative clustering, Practices`

#### 9.1 Feature Extraction for Clustering
```python
# Use CNN features (from penultimate layer) or PCA-reduced features for clustering
# Extract CNN embeddings
def extract_cnn_features(model, dataloader):
    features = []
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            # Get features before the final FC layer
            x = model.features(images)  # For custom CNN
            features.append(x.view(x.size(0), -1).numpy())
    return np.vstack(features)
```

#### 9.2 KMeans Clustering
```python
from sklearn.cluster import KMeans, MiniBatchKMeans

# Standard KMeans
kmeans = KMeans(n_clusters=101, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_features_pca)

# Mini-Batch KMeans (faster for large data)
mb_kmeans = MiniBatchKMeans(n_clusters=101, batch_size=1000, random_state=42)
mb_cluster_labels = mb_kmeans.fit_predict(X_features_pca)

# Elbow Method — optimal k
inertias = []
K_range = range(10, 150, 10)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    km.fit(X_features_pca)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
```

#### 9.3 Agglomerative (Hierarchical) Clustering
```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=20, linkage='ward')  # Use subset
agg_labels = agg.fit_predict(X_subset_pca)

# Dendrogram
Z = linkage(X_subset_pca[:500], method='ward')  # Small subset for visualization
plt.figure(figsize=(15, 7))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.title('Hierarchical Clustering Dendrogram of Food Features')
plt.xlabel('Cluster')
plt.ylabel('Distance')
plt.show()
```

#### 9.4 Clustering Evaluation & Practices
```python
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Silhouette Score
sil_score = silhouette_score(X_features_pca, cluster_labels, sample_size=5000)
print(f"Silhouette Score: {sil_score:.4f}")

# Adjusted Rand Index (comparing clusters to true labels)
ari = adjusted_rand_score(y_true, cluster_labels)
nmi = normalized_mutual_info_score(y_true, cluster_labels)
print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")

# Visualize clusters vs true labels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap='tab20', s=3, alpha=0.5)
ax1.set_title('KMeans Clusters')
ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab20', s=3, alpha=0.5)
ax2.set_title('True Labels')
plt.show()
```

- **Discuss**: How well does unsupervised clustering align with true food categories?
- **Insight**: Which food types cluster together naturally? (e.g., similar-looking desserts)

---

### Phase 10: Tree-Based & Ensemble Methods
**📘 Course Topics Covered:** `#25 Decision Trees`, `#26 Variants of Decision Tree`, `#27 Combining Classifiers (Boosting)`

#### 10.1 Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

dt = DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42)
dt.fit(X_train_pca, y_train)
print(f"Decision Tree Accuracy: {dt.score(X_test_pca, y_test):.4f}")

# Visualize the tree (top levels)
plt.figure(figsize=(25, 10))
plot_tree(dt, max_depth=3, filled=True, feature_names=[f'PC{i}' for i in range(X_train_pca.shape[1])],
          fontsize=8)
plt.title('Decision Tree (Top 3 Levels)')
plt.show()

# Feature importance from Decision Tree
importances = dt.feature_importances_
top_features = np.argsort(importances)[::-1][:20]
plt.barh(range(20), importances[top_features])
plt.yticks(range(20), [f'PC{i}' for i in top_features])
plt.title('Top 20 Feature Importances — Decision Tree')
plt.show()
```

#### 10.2 Random Forest (Decision Tree Variant)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=30, min_samples_split=5,
                             n_jobs=-1, random_state=42)
rf.fit(X_train_pca, y_train)
print(f"Random Forest Accuracy: {rf.score(X_test_pca, y_test):.4f}")
```

#### 10.3 Boosting — Combining Classifiers
```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb

# AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=200, learning_rate=0.1, random_state=42
)
ada.fit(X_train_pca, y_train)
print(f"AdaBoost Accuracy: {ada.score(X_test_pca, y_test):.4f}")

# XGBoost
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=101,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42
)
xgb_clf.fit(X_train_pca, y_train)
print(f"XGBoost Accuracy: {xgb_clf.score(X_test_pca, y_test):.4f}")
```

#### 10.4 Voting Ensemble (Combining All Classifiers)
```python
from sklearn.ensemble import VotingClassifier

# Soft voting ensemble of top performers
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=2000)),
        ('svm', SVC(kernel='rbf', probability=True)),
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', xgb_clf)
    ],
    voting='soft'
)
ensemble.fit(X_train_pca, y_train)
print(f"Voting Ensemble Accuracy: {ensemble.score(X_test_pca, y_test):.4f}")
```

---

### Phase 11: Model Comparison & Final Analysis
**📘 Course Topics Covered:** All topics — comprehensive comparison

#### 11.1 Results Summary Table
```python
import pandas as pd

results = pd.DataFrame({
    'Model': ['Naive Bayes', 'kNN (k=5)', 'Weighted kNN', 'Linear Regression',
              'SGD Classifier', 'Logistic Regression', 'Perceptron',
              'Linear SVM', 'RBF SVM', 'Poly SVM',
              'MLP (sklearn)', 'MLP (PyTorch)', 'Decision Tree',
              'Random Forest', 'AdaBoost', 'XGBoost',
              'Custom CNN', 'ResNet-50 (Transfer)', 'Voting Ensemble'],
    'Accuracy': [acc_bayes, acc_knn, acc_wknn, acc_ridge,
                 acc_sgd, acc_lr, acc_perceptron,
                 acc_lsvm, acc_rbf_svm, acc_poly_svm,
                 acc_mlp_sk, acc_mlp_pt, acc_dt,
                 acc_rf, acc_ada, acc_xgb,
                 acc_cnn, acc_resnet, acc_ensemble],
    'Course Topics Used': ['Bayes', 'kNN, Distance measures', 'Weighted kNN, Normalization',
                           'Linear Regression', 'Gradient Descent', 'Logistic Regression, Multi-class',
                           'Perceptron', 'SVM (Linear)', 'SVM (Kernel)', 'SVM variants',
                           'MLP', 'MLP, Backprop, GD', 'Decision Trees',
                           'DT Variants', 'Boosting', 'Boosting',
                           'Conv filters, CNN', 'CNN, Transfer Learning', 'Combining Classifiers']
})

results = results.sort_values('Accuracy', ascending=False)
print(results.to_string(index=False))
```

#### 11.2 Confusion Matrix (Best Model)
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# For the best model (likely ResNet-50)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(25, 25))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix — Best Model')
plt.show()

# Per-class classification report
print(classification_report(y_test, y_pred_best, target_names=class_names))
```

#### 11.3 Key Findings & Analysis
Document these as part of the final report:
- **Best classical ML model** vs **Best deep learning model** — what's the accuracy gap?
- **Effect of dimensionality reduction** — PCA vs LDA vs raw features
- **Effect of normalization** — with vs without
- **Overfitting analysis** — train vs test gaps for each model
- **Which food categories are hardest to classify?** — confusion matrix insights
- **Clustering insights** — do unsupervised clusters align with food categories?
- **Ensemble benefit** — does combining classifiers improve accuracy?

---

## 5. Directory Structure

```
PRML Project/
├── Food_Image_Classification_Complete_Procedure.md   ← This document
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_preprocessing_normalization.ipynb
│   ├── 04_dimensionality_reduction.ipynb
│   ├── 05_classical_ml_classifiers.ipynb
│   ├── 06_neural_networks.ipynb
│   ├── 07_cnn_and_transfer_learning.ipynb
│   ├── 08_clustering_analysis.ipynb
│   ├── 09_tree_ensemble_methods.ipynb
│   └── 10_final_comparison.ipynb
├── src/
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── models/
│   │   ├── classical_ml.py
│   │   ├── neural_networks.py
│   │   └── cnn_models.py
│   └── utils.py
├── data/
│   ├── food-101/                     ← Raw dataset (downloaded from Kaggle)
│   └── processed/                    ← Preprocessed features (HDF5/NumPy)
├── models/                           ← Saved trained models
├── results/
│   ├── figures/                      ← All plots and visualizations
│   └── metrics/                      ← CSV files with results
├── report/
│   └── PRML_Food_Classification_Report.pdf
└── requirements.txt
```

---

## 6. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall classification correctness |
| **Precision** | Per-class, how many predicted positives are actually correct |
| **Recall** | Per-class, how many actual positives are correctly identified |
| **F1-Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | Visual error analysis across all classes |
| **Top-5 Accuracy** | Whether correct class is in the top 5 predictions (important for 101 classes) |
| **Training Time** | Computational cost comparison |
| **Silhouette Score** | Clustering quality metric |
| **ARI / NMI** | Clustering agreement with true labels |

---

## 7. Deliverables

| # | Deliverable | Description |
|---|------------|-------------|
| 1 | **Jupyter Notebooks (10)** | Phased implementation, well-documented with markdown explanations |
| 2 | **Source Code** | Clean, modular Python code in `src/` |
| 3 | **Trained Models** | Saved model weights for best performers |
| 4 | **Results Report** | Comprehensive PDF/report with all results, tables, and plots |
| 5 | **Visualizations** | All plots saved in `results/figures/` |
| 6 | **Model Comparison Table** | Final ranking of all 19+ models |
| 7 | **This Procedure Document** | Complete implementation guide |

---

## Summary — Course Topics Integration

```
╔══════════════════════════════════════════════════════════════════╗
║              COURSE TOPICS COVERAGE: 27 / 27 ✅                 ║
╠══════════════════════════════════════════════════════════════════╣
║ ✅ Introduction                    ✅ Perceptron               ║
║ ✅ Feature computation, Histogram   ✅ SVM (Linear)             ║
║ ✅ Multi-dim features, Error, Dist  ✅ SVM variants, Kernel     ║
║ ✅ Bayes classifier                 ✅ MLP                      ║
║ ✅ Distance measures, kNN           ✅ Perceptron ↔ Neuron      ║
║ ✅ Weighted kNN, Normalization      ✅ Backpropagation          ║
║ ✅ Cross-validation                 ✅ Convolution filter/op    ║
║ ✅ Linear regression                ✅ CNN                      ║
║ ✅ Gradient descent                 ✅ Data Clustering          ║
║ ✅ Logistic regression              ✅ KMeans (EM) & variants   ║
║ ✅ Multi-class, Overfit, Regularize ✅ Agglomerative clustering ║
║ ✅ Dim. reduction, Cov, Corr       ✅ Decision Trees           ║
║ ✅ PCA & LDA                        ✅ DT Variants (RF)         ║
║                                     ✅ Boosting (Ensemble)      ║
╚══════════════════════════════════════════════════════════════════╝
```

---

> **🎯 This project is designed to maximize grading by demonstrating ALL 27 course topics in a single, cohesive food image classification pipeline — from raw pixel features to deep CNNs, with clustering analysis and ensemble methods layered on top.**
