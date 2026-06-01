# MLOps Customer Churn Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)
![Airflow](https://img.shields.io/badge/Airflow-Orchestration-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end MLOps pipeline for predicting customer churn, built on the [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3). The focus is on pipeline orchestration with Airflow, experiment tracking with MLflow, automated model promotion, and REST API serving — not on modeling performance.

## Overview

Most ML projects stop at a trained model. This one starts there. The goal is to demonstrate that a model can be operated reliably: retrained automatically when new data arrives, tracked across runs, promoted only when it improves, and served through a live API endpoint.

The dataset is a telecom customer churn prediction task (binary classification, evaluated by ROC-AUC). The modeling is intentionally straightforward — LightGBM with standard preprocessing — so the engineering layer stays in focus.

### Key Results

| Run | Training data | Val ROC-AUC |
|-----|--------------|-------------|
| 1   | Batch 1      | —           |
| 2   | Batch 1 + 2  | —           |
| 3   | Batch 1 + 2 + 3 | —        |


## How It Works

### Problem

Predict `Churn` probability for each customer in the test set. Evaluated by ROC-AUC between predicted probability and observed binary target.

### Retraining Story

Training data is split into three sequential batches to simulate a real-world scenario where new customer data arrives over time. The pipeline is triggered once per batch:

- **Run 1** — train on batch 1, log to MLflow, register as champion
- **Run 2** — train on batch 1+2, compare AUC to Run 1, promote if better
- **Run 3** — train on batch 1+2+3, compare AUC to Run 2, promote if better

Each run is fully tracked in MLflow: parameters, metrics, and the serialized model artifact.

### Pipeline Steps

Each Airflow DAG run executes the following steps in order:

1. **Ingest** — load the correct CSV batch for this run
2. **Preprocess** — encode categoricals, handle nulls, train/validation split
3. **Train** — fit LightGBM on the current batch
4. **Track** — log parameters, metrics, and model artifact to MLflow
5. **Evaluate** — compare validation AUC against the current champion
6. **Promote** — register new model as champion if AUC improves
7. **Serve** — FastAPI endpoint loads the champion model and exposes `/predict`

## Dataset

[Playground Series S6E3 — Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3/data) from Kaggle.

- Source: synthetic data generated from the IBM Telco Customer Churn dataset
- Evaluation metric: ROC-AUC
- Target: `Churn` (binary)

| Property | Value |
|----------|-------|
| Training rows | — |
| Features | — |
| Class balance (churn rate) | — |

## Project Structure

```
predict-customer-churn/
│
├── dags/
│   └── churn_pipeline.py        # Airflow DAG — orchestrates all steps
│
├── pipeline/
│   ├── ingest.py                # Load batch CSV for current run
│   ├── preprocess.py            # Encoding, null handling, train/val split
│   ├── train.py                 # LightGBM training
│   └── evaluate.py              # Compare vs champion, promote if better
│
├── serving/
│   ├── app.py                   # FastAPI app with /predict endpoint
│   └── Dockerfile               # Container for the serving layer
│
├── data/
│   ├── raw/                     # Original Kaggle CSVs (not committed)
│   └── batches/                 # Simulated temporal splits (not committed)
│
├── notebooks/
│   └── eda.ipynb                # Exploratory data analysis
│
├── docker-compose.yml           # Runs Airflow, MLflow, and FastAPI together
├── requirements.txt
├── .gitignore
└── README.md
```

## Kaggle Submission

The Run 3 model (trained on full data) is used to generate predictions on `test.csv` for Kaggle submission.

| Submission | Public leaderboard ROC-AUC |
|------------|---------------------------|
| Run 3 model | - |

## License

MIT