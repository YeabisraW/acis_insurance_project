ACIS Insurance Analytics Project

This project performs exploratory data analysis (EDA), feature engineering, statistical analysis, and modeling for insurance risk data using a fully reproducible, DVC-powered ML pipeline. The repository follows engineering best practices in version control, reproducibility, documentation, and code quality.

Project Structure
acis_insurance_project/
│
├── data/
│   ├── raw/                     # Original dataset (DVC-tracked)
│   ├── processed/               # Cleaned dataset outputs
│
├── notebooks/
│   ├── eda-task1.py             # Main EDA script (modular, function-based)
│   ├── eda_outputs/             # Histograms, boxplots, scatterplots, correlations
│   ├── task3_analysis.ipynb     # Statistical analysis & hypothesis testing outputs
│   └── task4_modeling.ipynb     # Predictive modeling outputs
│
├── scripts/
│   ├── task_3.py                # Statistical analysis & hypothesis testing
│   ├── task_4.py                # Predictive modeling & premium optimization
│
├── dvc.yaml                     # Pipeline definition (EDA, task3, task4 stages included)
├── dvc.lock                     # Auto-generated DVC dependency lock
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

Getting Started
1. Clone the Repository
git clone https://github.com/<your-username>/acis_insurance_project.git
cd acis_insurance_project

Environment Setup
2. Create and Activate Environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

3. Install Dependencies
pip install -r requirements.txt

Data Retrieval (DVC)

This project uses DVC to manage large datasets.

4. Pull Data from Remote Storage

Make sure DVC is installed:

pip install dvc


Then pull the dataset:

dvc pull


This downloads all required data files into the data/ directory.

Running the EDA Pipeline

The EDA stage generates:

Histograms

Boxplots

Outlier analysis (IQR method)

Scatter plots

Correlation matrix (CSV + heatmap)

Summary statistics

All outputs are saved in:

notebooks/eda_outputs/

Run EDA via DVC:
dvc repro eda

Or run manually:
python notebooks/eda-task1.py

Task 3 – Statistical Analysis & Hypothesis Testing

Objective: Statistically validate key hypotheses about risk drivers.

Key Activities:

Calculate claim frequency and severity by Province, ZipCode, and Gender

Conduct ANOVA and t-tests to accept/reject hypotheses

Visualize distributions and differences

Outputs:

scripts/task_3.py
notebooks/eda_outputs/task3_outputs/
    ├── LossRatio_by_Province.png
    ├── TotalClaims_by_Gender_boxplot.png
    ├── Margin_by_ZipCode_boxplot.png
scripts/task3_outputs/hypothesis_testing_results.csv


Sample Findings:

Claim frequency differs significantly across provinces – Reject H₀

No significant difference in claim frequency by gender – Fail to reject H₀

Margin differences across top 10 zip codes not significant – Fail to reject H₀

Task 4 – Predictive Modeling & Premium Optimization

Objective: Build predictive models for claim severity and risk-based premium estimation.

Key Activities:

Data preparation: handle missing values, feature engineering, encode categorical variables

Train-test split (70:30)

Model implementation: Linear Regression, Random Forest, XGBoost

Evaluate using RMSE and R² for regression; accuracy, precision, recall, F1-score for classification

Model interpretability using SHAP or LIME

Outputs:

scripts/task_4.py
notebooks/eda_outputs/task4_outputs/
    ├── ModelEvaluation_Regression.png
    ├── ModelEvaluation_Classification.png
    ├── FeatureImportance_SHAP.png
    └── RiskBasedPremium_Predictions.csv


Sample Insights:

Older vehicles increase predicted claim amount.

Certain vehicle makes/models have higher risk.

SHAP analysis identifies top 5–10 influential features affecting premium prediction.

Pipeline (DVC)

The pipeline now includes:

Stage	Description
eda	Runs eda-task1.py, generates plots and summaries
task3_analysis	Runs task_3.py, performs statistical tests, generates plots
task4_modeling	Runs task_4.py, trains models, evaluates, and generates outputs
feature_engineering	Prepares model-ready features
modeling	Trains and evaluates ML models

Visualize the pipeline graph:

dvc dag

Git Workflow Guidelines

Follow the same clean Git workflow as before:

main → stable

Feature branches: task-1, task-2, task-3, task-4

Conventional commits: feat:, fix:, docs:, chore:

Pull Requests: include description, list of changes, test steps