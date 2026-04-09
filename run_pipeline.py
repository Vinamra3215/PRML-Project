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


def download_dataset():
    images_dir = os.path.join(DATA_DIR, "images")
    meta_dir = os.path.join(DATA_DIR, "meta", "meta")

    if os.path.isdir(images_dir) and os.path.isdir(meta_dir):
        n_classes = len(os.listdir(images_dir))
        print(f"[SKIP] Dataset already exists: {images_dir} ({n_classes} classes)")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        import kaggle
        print("[DOWNLOAD] Downloading Food-101 from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "dansbecker/food-101", path=DATA_DIR, unzip=False
        )
        zip_path = os.path.join(DATA_DIR, "food-101.zip")
        if os.path.exists(zip_path):
            print("[EXTRACT] Extracting zip...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_DIR)
            os.remove(zip_path)
    except Exception as e:
        print(f"[INFO] Kaggle API not available ({e}), trying direct download...")
        _download_direct()

    _organize_extracted()


def _download_direct():
    archive_path = os.path.join(DATA_DIR, "food-101.tar.gz")

    if not os.path.exists(archive_path):
        print(f"[DOWNLOAD] Downloading Food-101 from ETH Zurich (~4.6 GB)...")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
            print(f"\r  Progress: {pct}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

        urllib.request.urlretrieve(DATASET_URL, archive_path, reporthook=_progress)
        print()

    print("[EXTRACT] Extracting tar.gz...")
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    os.remove(archive_path)


def _organize_extracted():
    extracted_root = os.path.join(DATA_DIR, "food-101")
    if os.path.isdir(extracted_root):
        for item in os.listdir(extracted_root):
            src = os.path.join(extracted_root, item)
            dst = os.path.join(DATA_DIR, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(extracted_root, ignore_errors=True)

    images_dir = os.path.join(DATA_DIR, "images")
    if os.path.isdir(images_dir):
        n_classes = len(os.listdir(images_dir))
        print(f"[OK] Dataset ready: {n_classes} classes in {images_dir}")
    else:
        print("[ERROR] Dataset extraction failed.")
        print(f"  Expected: {images_dir}/<class>/<image>.jpg")
        sys.exit(1)


def run_step(description, script_path, extra_args=None):
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"  Running: python {script_path}")
    print(f"{'=' * 70}\n")

    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, cwd=PROJECT_DIR)

    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with exit code {result.returncode}")
        print(f"  Re-run manually: python {script_path}")
        sys.exit(result.returncode)

    print(f"\n[OK] {description} completed successfully!")
