import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
import matplotlib.pyplot as plt

def stratified_cv(pipeline, X, y, k=5, scoring=None):
    
    scoring = scoring or ["accuracy", "f1_macro"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline, X, y, cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )
    return results
