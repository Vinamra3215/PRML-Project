import sys
sys.path.insert(0, ".")
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.seed import seed_everything
from src.data.cache import load_features
from src.data.dataset import Food101Dataset
from src.models.registry import build_pipeline
from src.evaluation.metrics import evaluate, get_classification_report

seed_everything(42)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
sns.set_theme(style="whitegrid", font_scale=1.05)

PLOTS_DIR = "results/plots"
METRICS_DIR = "results/metrics"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

no_pca_path = os.path.join(METRICS_DIR, "master_no_pca.csv")
with_pca_path = os.path.join(METRICS_DIR, "master_with_pca.csv")
legacy_path = os.path.join(METRICS_DIR, "master.csv")

df_no_pca, df_with_pca = None, None

if os.path.exists(no_pca_path):
    df_no_pca = pd.read_csv(no_pca_path).dropna(subset=["test_accuracy"])
    df_no_pca = df_no_pca.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    print(f"Loaded Table 1 (No PCA): {len(df_no_pca)} experiments")

if os.path.exists(with_pca_path):
    df_with_pca = pd.read_csv(with_pca_path).dropna(subset=["test_accuracy"])
    df_with_pca = df_with_pca.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    print(f"Loaded Table 2 (PCA):    {len(df_with_pca)} experiments")

if df_no_pca is not None:
    df = df_no_pca
elif df_with_pca is not None:
    df = df_with_pca
elif os.path.exists(legacy_path):
    df = pd.read_csv(legacy_path).dropna(subset=["test_accuracy"])
    df = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    print(f"Loaded legacy master.csv: {len(df)} experiments")
else:
    print("ERROR: No results CSV found. Run experiments first.")
    sys.exit(1)

dataset = Food101Dataset(root="data/", n_classes=20, seed=42)
class_names = dataset.class_names
print(f"Classes: {len(class_names)}\n")

# Find best model for confusion matrix
best_row = df.iloc[0]
BEST_MODEL = best_row["model"]
BEST_FEATURE = best_row["feature"]
print(f"Best model: {BEST_MODEL} on {BEST_FEATURE} ({best_row['test_accuracy']:.2%})\n")


def save_comparison_table(df_in, label, csv_name, md_name):
    df_out = df_in[["model", "feature", "cv_accuracy", "cv_f1",
                     "test_accuracy", "test_f1", "time_seconds"]].copy()
    df_out.columns = ["Model", "Features", "CV Acc", "CV F1",
                       "Test Acc", "Test F1", "Time (s)"]
    for col in ["CV Acc", "CV F1", "Test Acc", "Test F1"]:
        df_out[col] = df_out[col].apply(lambda x: f"{x:.4f}")
    df_out["Time (s)"] = df_out["Time (s)"].apply(lambda x: f"{x:.1f}")
    df_out.to_csv(os.path.join(METRICS_DIR, csv_name), index=False)
    md = [f"# {label}\n", f"**Dataset:** Food-101 (20 classes)",
          "**Cross-validation:** Stratified 5-Fold\n\n",
          df_out.to_markdown(index=False),
          f"\n\n**Best:** {df_in.iloc[0]['model']} on {df_in.iloc[0]['feature']} "
          f"— {df_in.iloc[0]['test_accuracy']:.2%} accuracy\n"]
    with open(os.path.join(METRICS_DIR, md_name), "w") as f:
        f.write("\n".join(md))
    print(f"  Saved {csv_name} + {md_name}")


print("[1/8] model_comparison.png")
fig, ax = plt.subplots(figsize=(13, max(7, len(df) * 0.45)))
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(df)))
labels = df["model"].str.replace("_", " ").str.title() + "  (" + df["feature"] + ")"
bars = ax.barh(range(len(df)), df["test_accuracy"], color=cmap, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Test Accuracy")
ax.set_title("Model Comparison — Test Accuracy (20 Food Classes)")
ax.invert_yaxis()
ax.set_xlim(0, max(df["test_accuracy"]) * 1.15)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
for bar, acc in zip(bars, df["test_accuracy"]):
    ax.text(acc + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}", va="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"))
plt.close()


print("[2/8] feature_comparison.png")
feat_df = df.groupby("feature")["test_accuracy"].max().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, max(4, len(feat_df) * 0.8)))
feat_colors = plt.cm.Set2(np.linspace(0, 1, len(feat_df)))
bars = ax.barh(feat_df.index.str.upper(), feat_df.values, color=feat_colors,
               edgecolor="white", linewidth=0.5, height=0.6)
ax.set_xlabel("Best Test Accuracy")
ax.set_title("Feature Ablation — Best Accuracy per Feature Type")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
for bar, val in zip(bars, feat_df.values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_comparison.png"))
plt.close()
