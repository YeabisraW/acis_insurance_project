# ACIS Insurance Project

## Overview

This project involves the analysis of insurance datasets and establishing a reproducible and auditable data pipeline using **Data Version Control (DVC)**. The workflow ensures that all analyses and results are reproducible and compliant with standard data management practices, which is critical in regulated industries such as finance and insurance.

---

## Task 1: Exploratory Data Analysis (EDA)

**Objective:** Perform a comprehensive exploratory data analysis on the `MachineLearningRating_v3.txt` dataset.

**Key Steps:**

* Load and inspect the dataset.
* Analyze missing values and basic statistics.
* Compute metrics like **Loss Ratio** overall, by province, vehicle type, and gender.
* Identify top vehicle makes with high loss ratios.
* Save sample rows and analysis outputs for reproducibility.

**Files:**

* `notebooks/eda-task1.py` — Python script performing EDA.
* `notebooks/eda_outputs/` — Directory containing outputs from the EDA script (sample rows, summary statistics).

**Run EDA:**

```bash
python notebooks/eda-task1.py
```

---

## Task 2: Data Version Control (DVC) Pipeline

**Objective:** Set up a reproducible and auditable data pipeline using DVC.

**Key Steps:**

1. **Install DVC:**

```bash
pip install dvc
```

2. **Initialize DVC:**

```bash
dvc init
```

3. **Set up local DVC remote storage:**

```bash
mkdir .dvc_storage
dvc remote add -d localstorage .dvc_storage
```

4. **Add data to DVC:**

```bash
dvc add data/MachineLearningRating_v3.txt
```

5. **Commit DVC metadata to Git:**

```bash
git add dvc.yaml dvc.lock .gitignore
git commit -m "Add DVC stage for Task 1 EDA"
```

6. **Run DVC pipeline:**

```bash
dvc repro
```

7. **Push DVC-tracked data to remote storage:**

```bash
dvc push
```

**Files:**

* `.dvc/config` — DVC configuration file.
* `dvc.yaml` — DVC pipeline stages (including `run_eda`).
* `dvc.lock` — Locked versions of stages and data.
* `.dvc_storage/` — Local remote storage for data files (ignored in Git, tracked by DVC).

---

## Notes

* Large data files are managed using DVC to avoid GitHub file size limitations.
* The workflow ensures any dataset and analysis can be reproduced exactly by re-running the DVC pipeline.
* This setup provides transparency and compliance for auditing and regulatory purposes.

---

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/YeabisraW/acis_insurance_project.git
cd acis_insurance_project
```

2. Checkout `task-2` branch for the latest DVC pipeline:

```bash
git checkout task-2
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

4. Pull data from DVC remote storage:

```bash
dvc pull
```

5. Reproduce EDA results:

```bash
dvc repro
```

