import cv2
import numpy as np
from src.features.base import FeatureExtractor


class ColorHistogramExtractor(FeatureExtractor):

    def __init__(self, bins: int = 32, color_spaces: list = None):
        self.bins = bins
        self.color_spaces = color_spaces or ["rgb", "hsv"]
