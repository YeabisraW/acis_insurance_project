import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Paths
DATA_PATH = "data/MachineLearningRating_v3.txt"
OUTPUT_DIR = "notebooks/eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    """Load dataset and handle errors."""
    try:
        df = pd.read_csv(path, sep='|', low_memory=False)
        print(f"Dataset loaded successfully!\nShape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File not found: {path}")
        exit(1)

def compute_basic_stats(df):
    """Compute basic stats for numeric columns and save summary."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[numeric_cols].describe()
    stats_file = os.path.join(OUTPUT_DIR, "numeric_summary.csv")
    stats.to_csv(stats_file)
    print(f"Basic statistics saved to {stats_file}")
    return stats

def plot_histogram(df, col, save_path=None):
    """Plot histogram for a numeric column."""
    if col not in df.columns:
        print(f"Column {col} not found!")
        return
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Histogram of {col}')
    if save_path:
        file = os.path.join(save_path, f"{col}_hist.png")
        plt.savefig(file)
        print(f"Histogram saved to {file}")
    plt.show()

def plot_scatter(df, x_col, y_col, save_path=None):
    """Plot scatter plot between two numeric columns."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Columns {x_col} or {y_col} not found!")
        return
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'{y_col} vs {x_col}')
    if save_path:
        file = os.path.join(save_path, f"{y_col}_vs_{x_col}_scatter.png")
        plt.savefig(file)
        print(f"Scatter plot saved to {file}")
    plt.show()

def plot_box(df, col, save_path=None):
    """Plot box plot and identify outliers."""
    if col not in df.columns:
        print(f"Column {col} not found!")
        return
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[col].dropna())
    plt.title(f'Box plot of {col}')
    if save_path:
        file = os.path.join(save_path, f"{col}_boxplot.png")
        plt.savefig(file)
        print(f"Box plot saved to {file}")
    plt.show()

def plot_correlation_matrix(df, save_path=None):
    """Plot correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    corr_file = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    corr.to_csv(corr_file)
    print(f"Correlation matrix saved to {corr_file}")
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    if save_path:
        file = os.path.join(save_path, "correlation_matrix.png")
        plt.savefig(file)
        print(f"Correlation matrix plot saved to {file}")
    plt.show()

def save_outliers(df, col):
    """Save outliers for a numeric column."""
    if col not in df.columns:
        return
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
    outlier_file = os.path.join(OUTPUT_DIR, f"{col}_outliers.csv")
    outliers.to_csv(outlier_file, index=False)
    print(f"Outliers for {col} saved to {outlier_file}")
    return outliers

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    stats = compute_basic_stats(df)

    numeric_cols = ["TotalPremium", "TotalClaims"]
    for col in numeric_cols:
        plot_histogram(df, col, save_path=OUTPUT_DIR)
        plot_box(df, col, save_path=OUTPUT_DIR)
        save_outliers(df, col)

    # Example scatter plot
    plot_scatter(df, "TotalPremium", "TotalClaims", save_path=OUTPUT_DIR)

    plot_correlation_matrix(df, save_path=OUTPUT_DIR)

    # Example LossRatio analysis if column exists
    if "LossRatio" in df.columns:
        plot_histogram(df, "LossRatio", save_path=OUTPUT_DIR)
        plot_box(df, "LossRatio", save_path=OUTPUT_DIR)
        save_outliers(df, "LossRatio")
    else:
        print("Column 'LossRatio' not found, skipping related plots.")
