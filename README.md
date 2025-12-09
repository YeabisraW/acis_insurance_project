# ACIS Insurance Analytics Project
# 
# This project performs exploratory data analysis (EDA), statistical hypothesis testing, 
# feature engineering, and predictive modeling for car insurance risk data using a fully 
# reproducible, DVC-powered ML pipeline. The repository follows best practices in version control, 
# reproducibility, documentation, and code quality.
#
# Project Structure:
# acis_insurance_project/
# │
# ├── data/
# │   ├── raw/                     # Original dataset (DVC-tracked)
# │   ├── processed/               # Cleaned dataset outputs
# │
# ├── notebooks/
# │   ├── eda-task1.py             # Main EDA script
# │   ├── eda_outputs/             # Plots, statistics, correlation matrices
# │   ├── task3_outputs/           # Hypothesis testing outputs
# │   ├── task4_outputs/           # Model evaluation & SHAP plots
# │
# ├── scripts/
# │   ├── task_3.py                # Statistical hypothesis testing
# │   ├── task_4.py                # Predictive modeling & SHAP analysis
# │
# ├── dvc.yaml                     # Pipeline definition
# ├── dvc.lock                     # Auto-generated dependency lock
# ├── requirements.txt             # Python dependencies
# └── README.md                    # Project documentation
#
# Getting Started:
# 1. Clone the Repository:
#    git clone https://github.com/YeabisraW/acis_insurance_project.git
#    cd acis_insurance_project
#
# 2. Create and Activate Environment:
#    python -m venv venv
#    source venv/bin/activate        # macOS/Linux
#    venv\Scripts\activate           # Windows
#
# 3. Install Dependencies:
#    pip install -r requirements.txt
#
# Data Management with DVC:
#    pip install dvc
#    dvc pull
#
# Tasks Overview:
# Task 1 – Exploratory Data Analysis (EDA):
# - Descriptive statistics: TotalPremium, TotalClaims, LossRatio
# - Histograms and boxplots
# - Outlier detection (IQR method)
# - Scatter plots for bivariate analysis
# - Correlation matrix
# Outputs: notebooks/eda_outputs/
#
# Task 2 – Data Version Control (DVC):
# - DVC initialized and configured
# - Raw dataset and EDA outputs tracked
# Commands:
#    dvc init
#    dvc add data/raw/MachineLearningRating_v3.txt
#    dvc repro eda
#    dvc push
#
# Task 3 – Statistical Hypothesis Testing:
# Hypotheses:
# - Claim frequency differs across Provinces (Chi-squared)
# - Claim frequency differs by Gender (Chi-squared)
# - Claim frequency differs across top 10 ZipCodes (Chi-squared)
# - LossRatio differs across Provinces (ANOVA)
# Business Insights:
# - Provinces with higher claim frequency or loss ratios may require premium adjustments
# - Top-risk zip codes identified for location-based pricing and marketing targeting
# - Gender effects inform marketing strategy
# Outputs: scripts/task3_outputs/hypothesis_testing_results.csv
#
# Task 4 – Predictive Modeling & Risk-Based Premiums:
# Objectives:
# - Predict claim severity (TotalClaims) using Linear Regression, Random Forest, and XGBoost
# - Analyze feature importance using SHAP
# Modeling Steps:
# - Filter dataset to policies with claims > 0
# - Select numeric features dynamically
# - Handle missing values
# - Train-test split (70/30)
# - Model evaluation: R-squared scores
# - SHAP analysis for top features
# Business Insights:
# - Features like VehicleAge, PolicyTerm, and prior claims have the highest impact
# - SHAP plots guide premium adjustments
# Outputs: notebooks/eda_outputs/task4_outputs/
#
# DVC Pipeline:
# Stage: eda             - Runs EDA and generates plots and summaries
# Stage: feature_engineering - Prepares model-ready features
# Stage: modeling         - Trains ML models and outputs performance & SHAP plots
#
# Git Workflow Guidelines:
# - Branches: main (stable), task-1 → task-4 (feature branches)
# - Commit prefixes: feat:, fix:, docs:, chore:
# - Pull Requests: descriptive titles, small focused commits, testing instructions
#
# GitHub Actions (CI/CD):
# Optional workflow: runs linting and tests on push/PR
# File path: .github/workflows/python-lint-test.yml
# - Checks code automatically before merging
