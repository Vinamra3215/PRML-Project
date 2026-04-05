from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.models.classical import KDEClassifier


MODEL_REGISTRY = {
    "knn":               KNeighborsClassifier,
    "logistic":          LogisticRegression,
    "naive_bayes":       GaussianNB,
    "decision_tree":     DecisionTreeClassifier,
    "mlp_sklearn":       MLPClassifier,
    "perceptron":        Perceptron,
    "sgd":               SGDClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "kde":               KDEClassifier,
}

REDUCER_REGISTRY = {
    "pca": PCA,
    "lda": LinearDiscriminantAnalysis,
    "none": None,
}

PARAM_GRIDS = {
    "knn": {
        "clf__n_neighbors": [3, 7, 11, 21],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "logistic": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
        "clf__max_iter": [2000],
    },
    "naive_bayes": {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "decision_tree": {
        "clf__max_depth": [5, 10, 20, None],
        "clf__min_samples_split": [2, 10],
        "clf__min_samples_leaf": [1, 5],
    },
}
