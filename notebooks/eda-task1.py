import os
import pandas as pd
import numpy as np

# ------------------------------
# Paths
# ------------------------------
# Project root (one level above notebooks)
project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Data path
data_file = os.path.join(project_root, 'data', 'MachineLearningRating_v3.txt')

# Output folder
output_dir = os.path.join(project_root, 'notebooks', 'eda_outputs')
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv(data_file, sep='|', low_memory=False)
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# ------------------------------
# Basic EDA
# ------------------------------
# Missing values
missing = df.isna().sum()
missing.to_csv(os.path.join(output_dir, 'missing_values.csv'))
print("--- Missing Values ---")
print(missing.head(10))

# Statistics
stats = df.describe(include='all')
stats.to_csv(os.path.join(output_dir, 'basic_stats.csv'))
print("--- Basic Statistics ---")
print(stats.head(10))

# ------------------------------
# Loss Ratio calculation
# ------------------------------
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
df['LossRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)

# Overall loss ratio
overall_lr = df['LossRatio'].mean()
print(f"\nOverall Loss Ratio: {overall_lr:.2f}")

# By Province
lr_by_province = df.groupby('Province')['LossRatio'].mean()
lr_by_province.to_csv(os.path.join(output_dir, 'lossratio_by_province.csv'))
print("\nLoss Ratio by Province:")
print(lr_by_province)

# By VehicleType
lr_by_vehicle = df.groupby('VehicleType')['LossRatio'].mean()
lr_by_vehicle.to_csv(os.path.join(output_dir, 'lossratio_by_vehicle.csv'))
print("\nLoss Ratio by VehicleType:")
print(lr_by_vehicle)

# By Gender
lr_by_gender = df.groupby('Gender')['LossRatio'].mean()
lr_by_gender.to_csv(os.path.join(output_dir, 'lossratio_by_gender.csv'))
print("\nLoss Ratio by Gender:")
print(lr_by_gender)

# Top 5 Vehicle Makes by Loss Ratio
top_makes = df.groupby('make')['LossRatio'].mean().sort_values(ascending=False).head(5)
top_makes.to_csv(os.path.join(output_dir, 'top5_lossratio_makes.csv'))
print("\nTop 5 Vehicle Makes by Loss Ratio:")
print(top_makes)

# ------------------------------
# Save sample rows
# ------------------------------
df.sample(5).to_csv(os.path.join(output_dir, 'sample_rows.csv'))
print("\nSample rows saved.")
