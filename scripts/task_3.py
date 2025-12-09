# scripts/task_3.py
"""
Task 3: Hypothesis Testing on Insurance Risk Data
- Quantify risk by Claim Frequency, Claim Severity, and Margin
- Test differences across Provinces, PostalCodes, and Gender
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import os

# -----------------------------
# Parameters
# -----------------------------
CLEANED_CSV_PATH = "data/processed/insurance_cleaned.csv"
OUTPUT_DIR = "scripts/task3_outputs"

# -----------------------------
# Helper Functions
# -----------------------------
def create_output_dir(path):
    os.makedirs(path, exist_ok=True)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")

def calculate_kpis(df):
    """Add KPI columns: ClaimFrequency, ClaimSeverity, Margin"""
    df['ClaimFrequency'] = np.where(df['TotalClaims'] > 0, 1, 0)
    # ClaimSeverity: average claim given claim occurred
    df['ClaimSeverity'] = df.apply(lambda x: x['TotalClaims'] if x['TotalClaims'] > 0 else np.nan, axis=1)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def t_test_metric(df, group_col, metric):
    """Perform t-test between two groups of a metric"""
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        print(f"Skipping {group_col} for t-test: not exactly 2 groups")
        return None
    
    group1 = df[df[group_col] == groups[0]][metric].dropna()
    group2 = df[df[group_col] == groups[1]][metric].dropna()
    
    stat, p_value = ttest_ind(group1, group2, equal_var=False)
    return p_value

def chi2_test_categorical(df, cat_col, target_col):
    """Chi-squared test for categorical target"""
    contingency = pd.crosstab(df[cat_col], df[target_col])
    chi2, p, dof, ex = chi2_contingency(contingency)
    return p

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    create_output_dir(OUTPUT_DIR)
    
    # Load data
    df = load_data(CLEANED_CSV_PATH)
    
    # Calculate KPIs
    df = calculate_kpis(df)
    
    results = []

    # --- H0: No risk difference across provinces ---
    provinces = df['Province'].dropna().unique()
    if len(provinces) > 1:
        for metric in ['ClaimFrequency', 'ClaimSeverity', 'Margin']:
            p_values = []
            for province in provinces:
                rest = df[df['Province'] != province][metric].dropna()
                current = df[df['Province'] == province][metric].dropna()
                stat, p = ttest_ind(current, rest, equal_var=False)
                p_values.append((province, metric, p))
            results.extend(p_values)
    
    # --- H0: No risk difference between genders ---
    if 'Gender' in df.columns:
        for metric in ['ClaimFrequency', 'ClaimSeverity', 'Margin']:
            p = t_test_metric(df, 'Gender', metric)
            results.append(('Gender', metric, p))
    
    # --- H0: No margin difference between zip codes (PostalCode) ---
    zipcodes = df['PostalCode'].dropna().unique()
    if len(zipcodes) > 1:
        for metric in ['Margin']:
            # Use t-test between top 2 most frequent zip codes for simplicity
            top2 = df['PostalCode'].value_counts().index[:2]
            group1 = df[df['PostalCode'] == top2[0]][metric].dropna()
            group2 = df[df['PostalCode'] == top2[1]][metric].dropna()
            stat, p = ttest_ind(group1, group2, equal_var=False)
            results.append(('Top2ZipCodes', metric, p))
    
    # Save results
    results_df = pd.DataFrame(results, columns=['Feature', 'Metric', 'p_value'])
    results_df['Reject_H0'] = results_df['p_value'] < 0.05
    results_df.to_csv(os.path.join(OUTPUT_DIR, "hypothesis_testing_results.csv"), index=False)
    
    print("Task 3 completed successfully!")
    print(f"Results saved at: {os.path.join(OUTPUT_DIR, 'hypothesis_testing_results.csv')}")
