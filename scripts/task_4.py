import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

# --- Load cleaned dataset ---
df = pd.read_csv('data/processed/insurance_cleaned.csv')

# Filter only rows with claims > 0 for claim severity modeling
target_severity = 'TotalClaims'
df = df[df[target_severity] > 0]

# --- Feature selection ---
numeric_features = df.select_dtypes(include=['number']).columns.tolist()
if target_severity in numeric_features:
    numeric_features.remove(target_severity)

# Include categorical features
categorical_features = ['Province', 'Gender']  # example
for cat in categorical_features:
    if cat not in df.columns:
        df[cat] = 'Unknown'
df_encoded = pd.get_dummies(df[categorical_features], drop_first=True)

# Combine numeric and encoded categorical features
X = pd.concat([df[numeric_features].fillna(0), df_encoded], axis=1)
y = df[target_severity].fillna(0)

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Model definitions ---
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse')
}

# --- Ensure output folder exists ---
output_folder = 'notebooks/eda_outputs/task4_outputs'
os.makedirs(output_folder, exist_ok=True)

# --- Model training and evaluation ---
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'R2': r2})
    
    print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # --- SHAP analysis ---
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test, check_additivity=False)
        shap.summary_plot(shap_values, X_test, show=False)
        plot_file = os.path.join(output_folder, f'shap_summary_{name}.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"SHAP summary plot saved for {name}")
        
        # Top features interpretation
        shap_mean = pd.DataFrame({
            'Feature': X_train.columns,
            'MeanAbsSHAP': np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by='MeanAbsSHAP', ascending=False)
        
        print("\nTop 5 influential features (business interpretation):")
        for i, row in shap_mean.head(5).iterrows():
            print(f"- {row['Feature']}: contributes to claim severity prediction. Higher values may increase expected claims and suggest higher premiums.")
        
    except Exception as e:
        print(f"SHAP analysis failed for {name}: {e}")

# --- Summary table ---
results_df = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print(results_df)
results_df.to_csv(os.path.join(output_folder, 'model_performance_comparison.csv'), index=False)
