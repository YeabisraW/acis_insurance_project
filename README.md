# ACIS Insurance Analytics Project

This project performs exploratory data analysis (EDA), feature engineering, and modeling for insurance risk data using a fully reproducible, DVC-powered ML pipeline. The repository follows engineering best practices in version control, reproducibility, documentation, and code quality.

---

## Project Structure

```
acis_insurance_project/
│
├── data/
│   ├── raw/                     # Original dataset (DVC-tracked)
│   ├── processed/               # Cleaned dataset outputs
│
├── notebooks/
│   ├── eda-task1.py             # Main EDA script (modular, function-based)
│   ├── eda_outputs/             # Histograms, boxplots, scatterplots, correlations
│
├── dvc.yaml                     # Pipeline definition (EDA stage included)
├── dvc.lock                     # Auto-generated DVC dependency lock
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/acis_insurance_project.git
cd acis_insurance_project
```

---

## Environment Setup

### 2. Create and Activate Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Data Retrieval (DVC)

This project uses **DVC** to manage large datasets.

### 4. Pull Data from Remote Storage

Make sure DVC is installed:

```bash
pip install dvc
```

Then pull the dataset:

```bash
dvc pull
```

This downloads all required data files into the `data/` directory.

---

## Running the EDA Pipeline

The EDA stage generates:

* Histograms
* Boxplots
* Outlier analysis (IQR method)
* Scatter plots
* Correlation matrix (CSV + heatmap)
* Summary statistics

All outputs are saved in:

```
notebooks/eda_outputs/
```

### Run EDA via DVC:

```bash
dvc repro eda
```

### Or run manually:

```bash
python notebooks/eda-task1.py
```

---

## Pipeline (DVC)

The pipeline consists of the following stages:

| Stage                   | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| **eda**                 | Runs `eda-task1.py`, generates plots, summaries, outlier files |
| **feature_engineering** | (Planned) Will create model-ready features                     |
| **modeling**            | (Planned) ML model training + evaluation                       |

To see the full pipeline graph:

```bash
dvc dag
```

---

## Git Workflow Guidelines

This repository follows a clean Git strategy:

### Conventional Commit Prefixes:

* `feat:` new feature
* `fix:` bug fix
* `docs:` documentation changes
* `chore:` non-code maintenance

### Branching:

* `main` → stable
* `task-1`, `task-2` → feature branches

### Creating a Pull Request:

* Include a short description
* List changes made
* Add test steps
* Keep commits small and focused

---

## EDA Outputs

The EDA script automatically generates:

```
numeric_summary.csv
TotalPremium_hist.png
TotalPremium_boxplot.png
TotalPremium_outliers.csv
TotalClaims_hist.png
TotalClaims_boxplot.png
TotalClaims_outliers.csv
TotalClaims_vs_TotalPremium_scatter.png
LossRatio_hist.png
LossRatio_boxplot.png
LossRatio_outliers.csv
LossRatio_vs_TotalPremium_scatter.png
LossRatio_vs_TotalClaims_scatter.png
correlation_matrix.csv
correlation_matrix.png
```

All files appear inside:

```
notebooks/eda_outputs/
```

---

## Next Steps (Future Stages)

The next pipeline components will include:

### 🔹 Feature Engineering

* Handling missing values
* Encoding categorical variables
* Feature scaling
* Feature selection

### 🔹 Modeling

* Training baseline ML models
* Hyperparameter tuning
* Model evaluation metrics

Each stage will be fully added to `dvc.yaml` for reproducibility.

---