import pandas as pd
import os

# --- File path ---

file_path = "data/MachineLearningRating_v3.txt"
# --- Load dataset ---

try:
    df = pd.read_csv(file_path, sep='|', low_memory=False)  # Pipe separator
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# --- Basic EDA ---

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Statistics ---")
print(df.describe(include='all'))

# --- Value counts for selected categorical columns ---

columns_to_check = ['VehicleType', 'Gender', 'CoverType']
for col in columns_to_check:
    if col in df.columns:
        print(f"\n--- {col} Value Counts ---")
        print(df[col].value_counts())

# --- Sample rows ---

print("\n--- Sample Rows ---")
print(df.sample(5))
