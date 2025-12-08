# Task 1: End-to-End EDA – ACIS Insurance Risk Analytics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Dataset ---

file_path = r"C:\Users\sciec\acis_insurance_project\data\MachineLearningRating_v3.txt"

df = pd.read_csv(file_path, sep='|', low_memory=False)
print("Dataset loaded successfully!")
print("Shape:", df.shape)

# --- Basic Info ---

print("\n--- Dataset Info ---")
print(df.info())

# --- Missing Values ---

print("\n--- Missing Values ---")
print(df.isna().sum())

# --- Basic Statistics ---
print("\n--- Basic Statistics ---")
print(df.describe(include='all'))

# --- Data Cleaning & Preprocessing ---

# Convert numeric-like columns with object dtype

numeric_cols = ['CapitalOutstanding', 'CustomValueEstimate', 'Cylinders',
'cubiccapacity', 'kilowatts', 'NumberOfDoors']
for col in numeric_cols:
  df[col] = pd.to_numeric(df[col], errors='coerce')

# Create Loss Ratio column 
  df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
  df['LossRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)

# --- Overall Loss Ratio ---

overall_loss_ratio = df['LossRatio'].mean()
print("\nOverall Loss Ratio:", round(overall_loss_ratio, 2))

# --- Loss Ratio by Province, VehicleType, Gender ---
province_loss = df.groupby('Province')['LossRatio'].mean()
vehicle_loss = df.groupby('VehicleType')['LossRatio'].mean()
gender_loss = df.groupby('Gender')['LossRatio'].mean()

print("\nLoss Ratio by Province:\n", province_loss)
print("\nLoss Ratio by VehicleType:\n", vehicle_loss)
print("\nLoss Ratio by Gender:\n", gender_loss)

# --- Visualizations ---

# 1. Loss Ratio by Province

plt.figure(figsize=(10,6))
sns.barplot(x=province_loss.index, y=province_loss.values)
plt.title("Average Loss Ratio by Province")
plt.ylabel("Loss Ratio")
plt.xticks(rotation=45)
plt.show()

# 2. Loss Ratio by VehicleType

plt.figure(figsize=(10,6))
sns.barplot(x=vehicle_loss.index, y=vehicle_loss.values)
plt.title("Average Loss Ratio by Vehicle Type")
plt.ylabel("Loss Ratio")
plt.xticks(rotation=45)
plt.show()

# 3. Loss Ratio by Gender

plt.figure(figsize=(6,4))
sns.barplot(x=gender_loss.index, y=gender_loss.values)
plt.title("Average Loss Ratio by Gender")
plt.ylabel("Loss Ratio")
plt.show()

# 4. Distribution of TotalPremium

plt.figure(figsize=(8,4))
sns.histplot(df['TotalPremium'], bins=50, kde=True)
plt.title("Distribution of Total Premium")
plt.show()

# 5. Distribution of TotalClaims

plt.figure(figsize=(8,4))
sns.histplot(df['TotalClaims'], bins=50, kde=True)
plt.title("Distribution of Total Claims")
plt.show()

# 6. Outlier Detection for TotalClaims

plt.figure(figsize=(8,4))
sns.boxplot(x=df['TotalClaims'])
plt.title("Boxplot: Total Claims")
plt.show()

# 7. Correlation Heatmap for numeric columns

numeric_features = df.select_dtypes(include=np.number)
plt.figure(figsize=(12,8))
sns.heatmap(numeric_features.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# --- Trend Analysis over TransactionMonth ---

df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
monthly_summary = df.groupby('TransactionMonth')[['TotalPremium','TotalClaims']].sum()
monthly_summary['LossRatio'] = monthly_summary['TotalClaims'] / monthly_summary['TotalPremium']

plt.figure(figsize=(12,6))
plt.plot(monthly_summary.index, monthly_summary['LossRatio'], marker='o')
plt.title("Monthly Loss Ratio Trend")
plt.ylabel("Loss Ratio")
plt.xlabel("Transaction Month")
plt.xticks(rotation=45)
plt.show()

# --- Top 5 Vehicle Makes with Highest Average Loss Ratio ---

vehicle_make_loss = df.groupby('make')['LossRatio'].mean().sort_values(ascending=False).head(5)
print("\nTop 5 Vehicle Makes by Loss Ratio:\n", vehicle_make_loss)

plt.figure(figsize=(8,4))
sns.barplot(x=vehicle_make_loss.index, y=vehicle_make_loss.values)
plt.title("Top 5 Vehicle Makes by Average Loss Ratio")
plt.ylabel("Loss Ratio")
plt.show()

# --- Sample Rows to Inspect ---

print("\n--- Sample Rows ---")
print(df.sample(5))