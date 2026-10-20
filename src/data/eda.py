"""Exploratory data analysis: saves figures to reports/figures/."""
from pathlib import Path
import logging

import matplotlib
matplotlib.use("Agg")  # no display needed, also works in CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.clean_data import CATEGORICAL_FEATURES, CLEAN_PATH, NUMERIC_FEATURES, RAW_PATH, TARGET

FIG_DIR = Path("reports/figures")
sns.set_theme(style="whitegrid", palette="Set2")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    log.info("Saved %s", FIG_DIR / name)


def plot_class_balance(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(x="num", data=raw, ax=axes[0])
    axes[0].set_title("Original severity label (0-4)")
    sns.countplot(x=TARGET, data=df, ax=axes[1])
    axes[1].set_title("Binary target (0 = no disease, 1 = disease)")
    for ax in axes:
        for c in ax.containers:
            ax.bar_label(c)
    save(fig, "01_class_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, element="step")
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    save(fig, "02_numeric_histograms.png")

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
        ax.set_title(col)
    save(fig, "03_numeric_boxplots.png")


def plot_categorical_vs_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
        ax.set_title(f"{col} by target")
    save(fig, "04_categorical_vs_target.png")


def plot_correlation(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    save(fig, "05_correlation_heatmap.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    df = pd.read_csv(CLEAN_PATH)

    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nTarget proportions:\n", df[TARGET].value_counts(normalize=True).round(3))
    print("\nCorrelation with target:\n",
          df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False).round(3))

    plot_class_balance(df, raw)
    plot_numeric_distributions(df)
    plot_categorical_vs_target(df)
    plot_correlation(df)


if __name__ == "__main__":
    main()