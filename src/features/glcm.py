import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from src.features.base import FeatureExtractor


class GLCMExtractor(FeatureExtractor):

    def __init__(self, distances=None, angles=None, properties=None):
        self.distances = distances or [1, 3]
        self.angles = angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.properties = properties or [
            "contrast", "dissimilarity", "homogeneity", "energy", "correlation"
        ]
