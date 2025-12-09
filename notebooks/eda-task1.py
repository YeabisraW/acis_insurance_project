# notebooks/eda-task1.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------
# 1. Load data
# ----------------------------
def load_data(file_path="data/raw/MachineLearningRating_v3.txt", delimiter="|"):
    """Load dataset with error handling."""
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File not found -> {file_path}")
        return None
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
        return None

# ----------------------------
# 2. Compute derived columns
# ----------------------------
def compute_loss_ratio(df, premium_col="TotalPremium", claims_col="TotalClaims", loss_col="LossRatio"):
    """Compute LossRatio and handle divide-by-zero."""
    if df is not None:
        df[loss_col] = df[claims_col] / df[premium_col]
        df[loss_col] = df[loss_col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ----------------------------
# 3. Basic stats
# ----------------------------
def compute_basic_stats(df, numeric_vars):
    """Return descriptive statistics."""
    if df is not None:
        return df[numeric_vars].describe()
    return None

# ----------------------------
# 4. Univariate analysis
# ----------------------------
def univariate_analysis(df, numeric_vars, output_dir="artifacts/plots/univariate"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("artifacts/reports", exist_ok=True)

    outlier_report = {}

    for col in numeric_vars:
        # Histogram
        plt.figure(figsize=(7, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{col}_histogram.png"))
        plt.close()

        # Boxplot + outlier detection
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_report[col] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "LowerBound": lower,
            "UpperBound": upper,
            "OutlierCount": len(outliers)
        }

        plt.figure(figsize=(7, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(os.path.join(output_dir, f"{col}_boxplot.png"))
        plt.close()

    # Save outlier summary
    outlier_df = pd.DataFrame(outlier_report).T
    outlier_df.to_csv("artifacts/reports/outlier_summary.csv")
    return outlier_df

# ----------------------------
# 5. Bivariate analysis
# ----------------------------
def bivariate_analysis(df, pairs, output_dir="artifacts/plots/bivariate"):
    os.makedirs(output_dir, exist_ok=True)

    for x, y in pairs:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=df[x], y=df[y])
        plt.title(f"Scatter Plot: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.savefig(os.path.join(output_dir, f"{x}_vs_{y}.png"))
        plt.close()

# ----------------------------
# 6. Multivariate analysis
# ----------------------------
def multivariate_analysis(df, numeric_vars, output_dir="artifacts/plots/multivariate"):
    os.makedirs(output_dir, exist_ok=True)
    corr = df[numeric_vars].corr()
    corr.to_csv("artifacts/reports/correlation_matrix.csv")

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_matrix_heatmap.png"))
    plt.close()
    return corr

# ----------------------------
# Main execution
# ----------------------------
def main(file_path="data/raw/MachineLearningRating_v3.txt",
         numeric_vars=["TotalPremium", "TotalClaims", "LossRatio"]):
    df = load_data(file_path)
    if df is None:
        return

    df = compute_loss_ratio(df, numeric_vars[0], numeric_vars[1], numeric_vars[2])
    stats = compute_basic_stats(df, numeric_vars)
    print("\nBasic statistics:\n", stats)

    univariate_analysis(df, numeric_vars)

    pairs = [("TotalPremium", "TotalClaims"),
             ("TotalPremium", "LossRatio"),
             ("TotalClaims", "LossRatio")]
    bivariate_analysis(df, pairs)

    multivariate_analysis(df, numeric_vars)
    print("\nEDA completed successfully. Check artifacts/ folder for outputs.\n")

if __name__ == "__main__":
    main()
