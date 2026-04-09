"""
One-shot pipeline: Download dataset → Extract features → Run experiments → Generate plots.

Usage:
    pip install -r requirements.txt
    python run_pipeline.py
"""
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
    archive_path = os.path.join(DATA_DIR, "food-101.tar.gz")

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

        if not os.path.exists(archive_path):
            print(f"[DOWNLOAD] Downloading Food-101 from ETH Zurich (~4.6 GB)...")
            print(f"  URL: {DATASET_URL}")

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

    extracted_root = os.path.join(DATA_DIR, "food-101")
    if os.path.isdir(extracted_root):
        for item in os.listdir(extracted_root):
            src = os.path.join(extracted_root, item)
            dst = os.path.join(DATA_DIR, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(extracted_root, ignore_errors=True)

    if os.path.isdir(images_dir):
        n_classes = len(os.listdir(images_dir))
        print(f"[OK] Dataset ready: {n_classes} classes in {images_dir}")
    else:
        print("[ERROR] Dataset extraction failed. Expected structure:")
        print(f"  {images_dir}/<class_name>/<image_id>.jpg")
        print(f"  {meta_dir}/classes.txt, train.json, test.json")
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
        print(f"  You can re-run this step manually: python {script_path}")
        sys.exit(result.returncode)

    print(f"\n[OK] {description} completed successfully!")


def print_results_summary():
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE!")
    print(f"{'=' * 70}")

    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    plots_dir = os.path.join(RESULTS_DIR, "plots")

    print(f"\n  Results saved to: {RESULTS_DIR}/")

    if os.path.isdir(metrics_dir):
        print(f"\n  Metrics:")
        for f in sorted(os.listdir(metrics_dir)):
            fpath = os.path.join(metrics_dir, f)
            size = os.path.getsize(fpath)
            print(f"    {f:<40s} ({size:>6,} bytes)")

    if os.path.isdir(plots_dir):
        print(f"\n  Plots:")
        for f in sorted(os.listdir(plots_dir)):
            fpath = os.path.join(plots_dir, f)
            size = os.path.getsize(fpath)
            print(f"    {f:<40s} ({size:>6,} bytes)")

    no_pca = os.path.join(metrics_dir, "master_no_pca.csv")
    if os.path.exists(no_pca):
        import pandas as pd
        df = pd.read_csv(no_pca)
        best = df.loc[df["test_accuracy"].idxmax()]
        print(f"\n  Best Result (No PCA):")
        print(f"    Model: {best['model']}  |  Features: {best['feature']}  |  Accuracy: {best['test_accuracy']:.4f}")

    with_pca = os.path.join(metrics_dir, "master_with_pca.csv")
    if os.path.exists(with_pca):
        import pandas as pd
        df = pd.read_csv(with_pca)
        best = df.loc[df["test_accuracy"].idxmax()]
        print(f"\n  Best Result (PCA-200):")
        print(f"    Model: {best['model']}  |  Features: {best['feature']}  |  Accuracy: {best['test_accuracy']:.4f}")


def main():
    print(f"\n{'#' * 70}")
    print(f"  FOOD-101 CLASSIFICATION — FULL PIPELINE")
    print(f"  No pretrained models. Classical ML only.")
    print(f"{'#' * 70}")

    print(f"\n  Steps:")
    print(f"    1. Download & extract Food-101 dataset")
    print(f"    2. Extract features (Histogram, HOG, LBP, GLCM, Fused)")
    print(f"    3. Run all experiments (7 models × 5 features × 2 phases)")
    print(f"    4. Generate analysis plots (8 total)")
    print(f"\n")

    download_dataset()

    run_step(
        "STEP 2: Feature Extraction (Histogram, HOG, LBP, GLCM, Fused)",
        "scripts/extract_features.py",
    )

    run_step(
        "STEP 3: Run All Experiments (Phase 1: No PCA + Phase 2: PCA-200)",
        "scripts/run_all_experiments.py",
    )

    run_step(
        "STEP 4: Generate Analysis Plots (8 plots)",
        "scripts/generate_report_plots.py",
    )

    print_results_summary()


if __name__ == "__main__":
    main()
