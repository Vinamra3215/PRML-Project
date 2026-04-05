import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


class KDEClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        n_total = len(y)

        for cls in self.classes_:
            X_cls = X[y == cls]
            self.priors_[cls] = len(X_cls) / n_total
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(X_cls)
            self.models_[cls] = kde

        return self

    def predict(self, X):
        log_probs = self._compute_log_probs(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        log_probs = self._compute_log_probs(X)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _compute_log_probs(self, X):
        log_probs = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            log_likelihood = self.models_[cls].score_samples(X)
            log_prior = np.log(self.priors_[cls])
            log_probs[:, i] = log_likelihood + log_prior
        return log_probs
