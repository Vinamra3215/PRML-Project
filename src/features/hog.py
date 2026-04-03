import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
from src.features.base import FeatureExtractor


class HOGExtractor(FeatureExtractor):

    def __init__(self, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
        self.orientations = orientations
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)
