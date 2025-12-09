# notebooks/eda-task1.py
"""
Full EDA and Preprocessing for ACIS Insurance Project
Generates:
- Descriptive statistics
- Histograms and bar plots
- Box plots and outlier detection
- Correlation matrices
- Cleaned CSV for further analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# -----------------------------
# Parameters
# -----------------------------
RAW_DATA_PATH = "data/raw/MachineLearningRating_v3.txt"
OUTPUT_DIR = "notebooks/eda_outputs"
CLEANED_CSV_PATH = "data/processed/insurance_cleaned.csv"
NUMERIC_COLS = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']

# -----------------------------
# Helper Functions
# -----------------------------
def create_output_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, delimiter='|')
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")

def clean_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=cols)
    return df

def compute_loss_ratio(df):
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    df['LossRatio'] = df['LossRatio'].fillna(0)
    return df

def save_summary(df, output_dir):
    summary = df.describe(include='all')
    summary.to_csv(os.path.join(output_dir, 'numeric_summary.csv'))
    print("Saved summary statistics")

# -----------------------------
# Plotting Functions
# -----------------------------
def plot_histograms(df, numeric_cols, output_dir):
    for col in numeric_cols + ['LossRatio']:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f"{col} Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
        plt.close()

def plot_boxplots(df, numeric_cols, output_dir):
    for col in numeric_cols + ['LossRatio']:
        plt.figure(figsize=(6,4))
        sns.boxplot(y=df[col])
        plt.title(f"{col} Boxplot")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_boxplot.png"))
        plt.close()

def plot_scatter(df, output_dir):
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', data=df)
    plt.title("TotalClaims vs TotalPremium")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "TotalClaims_vs_TotalPremium_scatter.png"))
    plt.close()
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='TotalPremium', y='LossRatio', data=df)
    plt.title("LossRatio vs TotalPremium")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "LossRatio_vs_TotalPremium_scatter.png"))
    plt.close()
    
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='TotalClaims', y='LossRatio', data=df)
    plt.title("LossRatio vs TotalClaims")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "LossRatio_vs_TotalClaims_scatter.png"))
    plt.close()

def correlation_matrix(df, numeric_cols, output_dir):
    corr = df[numeric_cols + ['LossRatio']].corr()
    corr.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()
    print("Saved correlation matrix")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    create_output_dir(OUTPUT_DIR)
    create_output_dir(os.path.dirname(CLEANED_CSV_PATH))
    
    df = load_data(RAW_DATA_PATH)
    df = clean_numeric(df, NUMERIC_COLS)
    df = compute_loss_ratio(df)
    
    save_summary(df, OUTPUT_DIR)
    
    plot_histograms(df, NUMERIC_COLS, OUTPUT_DIR)
    plot_boxplots(df, NUMERIC_COLS, OUTPUT_DIR)
    plot_scatter(df, OUTPUT_DIR)
    correlation_matrix(df, NUMERIC_COLS, OUTPUT_DIR)
    
    # Save cleaned CSV for Task 3
    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Cleaned CSV saved at: {CLEANED_CSV_PATH}")
    print("EDA Completed Successfully!")
