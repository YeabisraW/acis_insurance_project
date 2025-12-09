import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from xgboost import XGBRegressor, XGBClassifier  # Uncomment if installed
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Load cleaned data
df = pd.read_csv("data/processed/insurance_cleaned.csv")

# -------------------------
# 1. Claim Severity Model
# -------------------------
df_claims = df[df['TotalClaims'] > 0].copy()
numeric_features = df_claims.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('TotalClaims')
X_sev = df_claims[numeric_features]
y_sev = df_claims['TotalClaims'].squeeze()

X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_sev, y_sev, test_size=0.3, random_state=42)

# Ensure numeric
X_train_sev = X_train_sev.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_sev = X_test_sev.apply(pd.to_numeric, errors='coerce').fillna(0)

models_sev = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    # "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

results_sev = {}
for name, model in models_sev.items():
    model.fit(X_train_sev, y_train_sev)
    y_pred = model.predict(X_test_sev)
    rmse = np.sqrt(mean_squared_error(y_test_sev, y_pred))
    r2 = r2_score(y_test_sev, y_pred)
    results_sev[name] = {"RMSE": rmse, "R2": r2}
    print(f"{name} - Claim Severity RMSE: {rmse:.2f}, R2: {r2:.4f}")

# -------------------------
# 2. Premium Prediction Model
# -------------------------
df_premium = df.copy()
numeric_features = df_premium.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('CalculatedPremiumPerTerm')
X_prem = df_premium[numeric_features]
y_prem = df_premium['CalculatedPremiumPerTerm'].squeeze()

X_train_prem, X_test_prem, y_train_prem, y_test_prem = train_test_split(X_prem, y_prem, test_size=0.3, random_state=42)

X_train_prem = X_train_prem.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_prem = X_test_prem.apply(pd.to_numeric, errors='coerce').fillna(0)

models_prem = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    # "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

results_prem = {}
for name, model in models_prem.items():
    model.fit(X_train_prem, y_train_prem)
    y_pred = model.predict(X_test_prem)
    rmse = np.sqrt(mean_squared_error(y_test_prem, y_pred))
    r2 = r2_score(y_test_prem, y_pred)
    results_prem[name] = {"RMSE": rmse, "R2": r2}
    print(f"{name} - Premium RMSE: {rmse:.2f}, R2: {r2:.4f}")

# -------------------------
# 3. Claim Occurrence Model (Binary Classification)
# -------------------------
df_binary = df.copy()
df_binary['HasClaim'] = (df_binary['TotalClaims'] > 0).astype(int)

numeric_features = df_binary.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('HasClaim')
X_bin = df_binary[numeric_features]
y_bin = df_binary['HasClaim'].squeeze()

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

X_train_bin = X_train_bin.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_bin = X_test_bin.apply(pd.to_numeric, errors='coerce').fillna(0)

models_bin = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
    # "XGBoostClassifier": XGBClassifier(n_estimators=100, random_state=42)
}

results_bin = {}
for name, model in models_bin.items():
    model.fit(X_train_bin, y_train_bin)
    y_pred = model.predict(X_test_bin)
    acc = accuracy_score(y_test_bin, y_pred)
    prec = precision_score(y_test_bin, y_pred)
    rec = recall_score(y_test_bin, y_pred)
    f1 = f1_score(y_test_bin, y_pred)
    results_bin[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    print(f"{name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# -------------------------
# Optional: Feature importance / SHAP
# -------------------------
# Add SHAP analysis after installing shap: pip install shap
# import shap
# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test)

