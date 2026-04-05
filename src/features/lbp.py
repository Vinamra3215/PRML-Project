import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from src.features.base import FeatureExtractor


class LBPExtractor(FeatureExtractor):

    def __init__(self, n_points=24, radius=3, method="uniform"):
        self.n_points = n_points
        self.radius = radius
        self.method = method
