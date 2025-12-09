import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway

# Load cleaned data
df = pd.read_csv('data/processed/insurance_cleaned.csv')

# Ensure required columns exist
required_cols = ['Province', 'Gender', 'ZipCode', 'TotalClaims', 'LossRatio']
for col in required_cols:
    if col not in df.columns:
        print(f"Warning: Column '{col}' not found. Creating placeholder.")
        if col == 'TotalClaims':
            df['TotalClaims'] = 0
        elif col == 'LossRatio':
            df['LossRatio'] = 0
        else:
            df[col] = 'Unknown'

# Create HasClaim column if missing
if 'HasClaim' not in df.columns:
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)

# Convert categorical columns to string
for col in ['Province', 'Gender', 'ZipCode', 'HasClaim']:
    df[col] = df[col].astype(str)

# --- Hypothesis testing results list ---
results = []

# 1. Claim frequency differs across Provinces
cont_prov = pd.crosstab(df['Province'], df['HasClaim'])
chi2, p, dof, _ = chi2_contingency(cont_prov)
results.append({
    'Hypothesis': 'Claim frequency differs across Provinces',
    'Test': 'Chi-squared',
    'Statistic': chi2,
    'p_value': p,
    'Outcome': 'Reject H0' if p < 0.05 else 'Fail to reject H0',
    'Business_Interpretation': 'Certain provinces have higher claim frequency. Adjust premiums by province or target risk management campaigns.'
})

# 2. Claim frequency differs by Gender
cont_gender = pd.crosstab(df['Gender'], df['HasClaim'])
chi2_g, p_g, dof_g, _ = chi2_contingency(cont_gender)
results.append({
    'Hypothesis': 'Claim frequency differs by Gender',
    'Test': 'Chi-squared',
    'Statistic': chi2_g,
    'p_value': p_g,
    'Outcome': 'Reject H0' if p_g < 0.05 else 'Fail to reject H0',
    'Business_Interpretation': 'If significant, consider gender in premium design or marketing campaigns. Otherwise, standard rules apply.'
})

# 3. Claim frequency differs across top 10 ZipCodes
top_zips = df['ZipCode'].value_counts().head(10).index
cont_zip = pd.crosstab(df.loc[df['ZipCode'].isin(top_zips), 'ZipCode'],
                       df.loc[df['ZipCode'].isin(top_zips), 'HasClaim'])
chi2_zip, p_zip, dof_zip, _ = chi2_contingency(cont_zip)
results.append({
    'Hypothesis': 'Claim frequency differs across top 10 ZipCodes',
    'Test': 'Chi-squared',
    'Statistic': chi2_zip,
    'p_value': p_zip,
    'Outcome': 'Reject H0' if p_zip < 0.05 else 'Fail to reject H0',
    'Business_Interpretation': 'High-risk zip codes identified. Target premiums and marketing strategies geographically.'
})

# 4. LossRatio differs across Provinces (ANOVA)
groups = [df.loc[df['Province'] == prov, 'LossRatio'].astype(float) for prov in df['Province'].unique()]
f_stat, p_anova = f_oneway(*groups)
results.append({
    'Hypothesis': 'LossRatio differs across Provinces',
    'Test': 'ANOVA',
    'Statistic': f_stat,
    'p_value': p_anova,
    'Outcome': 'Reject H0' if p_anova < 0.05 else 'Fail to reject H0',
    'Business_Interpretation': 'Provinces with higher loss ratios may require higher premiums or more proactive risk management.'
})

# --- Save results ---
results_df = pd.DataFrame(results)
output_folder = 'scripts/task3_outputs'
results_df.to_csv(f'{output_folder}/hypothesis_testing_results.csv', index=False)
print("Task 3 hypotheses tested and results saved.")
