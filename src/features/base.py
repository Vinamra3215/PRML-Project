from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.preprocess import load_image


class FeatureExtractor(ABC):

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        ...

    def extract_from_path(self, path: str, size: int = 224) -> np.ndarray:
        image = load_image(path, size=size)
        return self.extract(image)
    
    def extract_dataset(
        self,
        image_paths: list,
        labels: np.ndarray,
        size: int = 224,
        n_jobs: int = -1,
    ) -> tuple:

        if n_jobs == 1:
            features = []
            valid_labels = []
            for i, path in enumerate(tqdm(image_paths, desc=self.__class__.__name__)):
                try:
                    feat = self.extract_from_path(path, size)
                    features.append(feat)
                    valid_labels.append(labels[i])
                except Exception:
                    continue
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._safe_extract)(path, size)
                for path in tqdm(image_paths, desc=self.__class__.__name__)
            )
            features = []
            valid_labels = []
            for i, feat in enumerate(results):
                if feat is not None:
                    features.append(feat)
                    valid_labels.append(labels[i])

        X = np.array(features, dtype=np.float32)
        y = np.array(valid_labels)
        return X, y
