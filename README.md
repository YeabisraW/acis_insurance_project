# ACIS Insurance Project

This repository contains the ACIS Insurance dataset analysis project, including exploratory data analysis (EDA) and a reproducible data pipeline using Data Version Control (DVC). The project is structured to support reproducible research and versioned datasets, in line with standard practices in regulated industries such as finance and insurance.

## Table of Contents

* [Project Overview](#project-overview)
* [Task 1: Exploratory Data Analysis (EDA)](#task-1-exploratory-data-analysis-eda)
* [Task 2: Reproducible Data Pipeline (DVC)](#task-2-reproducible-data-pipeline-dvc)
* [Setup and Installation](#setup-and-installation)
* [Usage](#usage)
* [Repository Structure](#repository-structure)
* [Contact](#contact)

---

## Project Overview

The goal of this project is to analyze the ACIS Insurance dataset to understand trends, missing values, and key statistics across multiple features. Additionally, we implement a reproducible data pipeline to ensure that all analysis can be reproduced at any time, which is critical for auditing, regulatory compliance, and debugging in financial and insurance industries.

---

## Task 1: Exploratory Data Analysis (EDA)

**Objective:** Perform a comprehensive analysis of the dataset, identifying missing values, distributions, and key metrics such as the loss ratio.

**Key Steps:**

1. Load the dataset `MachineLearningRating_v3.txt`.
2. Analyze missing values across all features.
3. Compute basic statistics (mean, median, standard deviation, min, max) for numeric columns.
4. Calculate and analyze the overall Loss Ratio, as well as by Province, Vehicle Type, and Gender.
5. Identify top vehicle makes by Loss Ratio.
6. Save summarized outputs to `notebooks/eda_outputs/`.

**Key File:**

* `notebooks/eda-task1.py`: Python script containing the full EDA workflow.

---

## Task 2: Reproducible Data Pipeline (DVC)

**Objective:** Set up a version-controlled data pipeline to track datasets and analysis outputs.

**Key Steps:**

1. Install DVC: `pip install dvc`
2. Initialize DVC in the repository: `dvc init`
3. Create a local storage folder for dataset versioning: `.dvc_storage/`
4. Add dataset to DVC: `dvc add data/MachineLearningRating_v3.txt`
5. Create a pipeline stage to reproduce EDA:

   ```bash
   dvc stage add -n run_eda \
       -d notebooks/eda-task1.py \
       -d data/MachineLearningRating_v3.txt \
       -o notebooks/eda_outputs \
       python notebooks/eda-task1.py
   ```
6. Commit DVC metadata and stage:

   ```bash
   git add dvc.yaml dvc.lock .gitignore
   git commit -m "Add DVC stage for Task 1 EDA"
   ```
7. Push dataset to local DVC remote: `dvc push`

**Benefits:**

* Data versioning ensures any analysis is reproducible.
* The `.dvc` and `.gitignore` setup prevents large datasets from bloating Git.
* Pipelines automate running scripts and updating outputs.

---

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/YeabisraW/acis_insurance_project.git
cd acis_insurance_project
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install dvc
```

3. Initialize DVC (if not done already):

```bash
dvc init
dvc remote add -d localstorage .dvc_storage
```

4. Pull dataset from DVC remote:

```bash
dvc pull
```

---

## Usage

To reproduce the EDA pipeline:

```bash
dvc repro
```

To push updates to DVC remote storage:

```bash
dvc push
```

---

## Repository Structure

```
acis_insurance_project/
├── data/
│   └── MachineLearningRating_v3.txt.dvc
├── notebooks/
│   ├── eda-task1.py
│   └── eda_outputs/
├── .dvc/
├── .dvc_storage/
├── dvc.yaml
├── dvc.lock
├── .gitignore
└── README.md
```

**Note:** Ensure `data/MachineLearningRating_v3.txt` is not committed to Git. All large datasets are versioned via DVC.
