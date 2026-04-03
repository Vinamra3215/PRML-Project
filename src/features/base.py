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