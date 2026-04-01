import os
import json
import random
from pathlib import Path

import numpy as np


class Food101Dataset:

    def __init__(
        self,
        root: str = "data/",
        n_classes: int = 20,
        val_split: float = 0.15,
        seed: int = 42,
    ):
        self.root = root
        self.n_classes = n_classes
        self.val_split = val_split
        self.seed = seed

        self.images_dir = os.path.join(root, "images")
        self.meta_dir = os.path.join(root, "meta", "meta")

        self.class_names, self.class_to_idx = self._load_classes()

    def _load_classes(self):
        classes_file = os.path.join(self.meta_dir, "classes.txt")
        with open(classes_file, "r") as f:
            all_classes = sorted([line.strip() for line in f if line.strip()])

        random.seed(self.seed)
        selected = sorted(random.sample(all_classes, min(self.n_classes, len(all_classes))))
        class_to_idx = {c: i for i, c in enumerate(selected)}
        return selected, class_to_idx
