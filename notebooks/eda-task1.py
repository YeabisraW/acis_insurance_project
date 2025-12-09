import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
df = pd.read_csv("data/raw/MachineLearningRating_v3.txt", delimiter="|")

# -----------------------------------------------------------------------------
# 2. CREATE NECESSARY VARIABLES
# -----------------------------------------------------------------------------

# Create Loss Ratio = TotalClaims / TotalPremium
df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"]

# Handle division by zero, inf, or missing values
df["LossRatio"] = df["LossRatio"].replace([np.inf, -np.inf], np.nan)
df["LossRatio"] = df["LossRatio"].fillna(0)

# Numeric variables required for the EDA
numeric_vars = ["TotalPremium", "TotalClaims", "LossRatio"]

# -----------------------------------------------------------------------------
# 3. CREATE OUTPUT FOLDERS
# -----------------------------------------------------------------------------
os.makedirs("artifacts/plots/univariate", exist_ok=True)
os.makedirs("artifacts/plots/bivariate", exist_ok=True)
os.makedirs("artifacts/plots/multivariate", exist_ok=True)
os.makedirs("artifacts/reports", exist_ok=True)

# -----------------------------------------------------------------------------
# 4. UNIVARIATE ANALYSIS
# -----------------------------------------------------------------------------
outlier_report = {}

# --- HISTOGRAMS ---
for col in numeric_vars:
    plt.figure(figsize=(7, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(f"artifacts/plots/univariate/{col}_histogram.png")
    plt.close()

# --- BOXPLOTS + OUTLIER ANALYSIS ---
for col in numeric_vars:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Detect outliers
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
    plt.savefig(f"artifacts/plots/univariate/{col}_boxplot.png")
    plt.close()

# Save outlier summary
pd.DataFrame(outlier_report).T.to_csv("artifacts/reports/outlier_summary.csv")

# -----------------------------------------------------------------------------
# 5. BIVARIATE ANALYSIS
# -----------------------------------------------------------------------------

pairs = [
    ("TotalPremium", "TotalClaims"),
    ("TotalPremium", "LossRatio"),
    ("TotalClaims", "LossRatio")
]

for x, y in pairs:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=df[x], y=df[y])
    plt.title(f"Scatter Plot: {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(f"artifacts/plots/bivariate/{x}_vs_{y}.png")
    plt.close()

# -----------------------------------------------------------------------------
# 6. MULTIVARIATE ANALYSIS
# -----------------------------------------------------------------------------

# Correlation matrix
corr = df[numeric_vars].corr()

# Save correlation matrix CSV
corr.to_csv("artifacts/reports/correlation_matrix.csv")

# Heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.savefig("artifacts/plots/multivariate/correlation_matrix_heatmap.png")
plt.close()

print("\nEDA Completed Successfully. Check the 'artifacts' folder for outputs.\n")

