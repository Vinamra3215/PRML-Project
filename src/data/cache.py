import os

import h5py
import numpy as np


def get_cache_path(cache_dir: str, feature_type: str, split: str) -> str:
    return os.path.join(cache_dir, feature_type, f"{split}.h5")


def save_features(
    X: np.ndarray,
    y: np.ndarray,
    cache_dir: str,
    feature_type: str,
    split: str,
    class_names: list = None,
):
    path = get_cache_path(cache_dir, feature_type, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=y)
        if class_names is not None:
            f.attrs["classes"] = class_names
        f.attrs["n_samples"] = X.shape[0]
        f.attrs["n_features"] = X.shape[1]

    print(f"Cached {split} features ({feature_type}): {X.shape} -> {path}")
