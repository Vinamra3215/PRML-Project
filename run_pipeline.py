import os
import sys
import zipfile
import subprocess
import urllib.request
import shutil

sys.path.insert(0, ".")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DATASET_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
