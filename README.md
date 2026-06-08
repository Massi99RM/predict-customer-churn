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

| Run | Training data      | Val ROC-AUC |
|-----|--------------------|-------------|
| 1   | Batch 1            | 0.9156      |
| 2   | Batch 1 + 2        | 0.9156      |
| 3   | Batch 1 + 2 + 3    | 0.9163      |


## How It Works

### Problem

Predict `Churn` probability for each customer in the test set. Evaluated by ROC-AUC between predicted probability and observed binary target.

### Retraining Story

Training data is split into three sequential batches to simulate a real-world scenario where new customer data arrives over time. The pipeline is triggered once per batch:

- **Run 1** — train on batch 1, log to MLflow, register as champion (no previous baseline)
- **Run 2** — train on batch 1+2, compare AUC to Run 1, promote if better
- **Run 3** — train on batch 1+2+3, compare AUC to Run 2, promote if better

Each run is fully tracked in MLflow: parameters, metrics, and the serialized model artifact. The next batch number is determined automatically by counting completed MLflow runs — no manual configuration needed between triggers.

### Pipeline Steps

Each Airflow DAG run executes the following steps in order:

1. **Ingest** — load the correct CSV batch for this run
2. **Preprocess** — encode categoricals, train/validation split
3. **Train** — fit LightGBM on the current batch, log artifact to MLflow
4. **Evaluate** — compare validation AUC against the current champion
5. **Promote** — register new model as champion if AUC improves
6. **Serve** — FastAPI endpoint loads the champion model and exposes `/predict`

## Dataset

[Playground Series S6E3 — Predict Customer Churn](https://www.kaggle.com/competitions/playground-series-s6e3/data) from Kaggle.

- Source: synthetic data generated from the IBM Telco Customer Churn dataset
- Evaluation metric: ROC-AUC
- Target: `Churn` (binary)

| Property | Value |
|----------|-------|
| Training rows | 594,194 |
| Features | 19 (after dropping id and target) |
| Class balance (churn rate) | 22.5% positive — mildly imbalanced, not handled explicitly |

## Project Structure

```
predict-customer-churn/
│
├── dags/
│   └── churn_pipeline.py        # Airflow DAG — orchestrates all steps
│
├── pipeline/
│   ├── ingest_data.py           # Load batch CSV for current run
│   ├── preprocess.py            # Encoding, train/val split
│   ├── train.py                 # LightGBM training + MLflow artifact logging
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
├── create_batches.py            # One-off script to split raw data into batches
├── docker-compose.yml           # Runs Airflow, MLflow, and FastAPI together
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Project

### Prerequisites

- Docker and Docker Compose installed
- Kaggle dataset downloaded to `data/raw/train.csv`

### Setup

**1. Generate the batch files** (one-time, run from repo root):
```bash
python create_batches.py
```

**2. Start all services:**
```bash
docker-compose up -d
```

This starts four containers:
- `mlflow-init` — sets permissions on the shared artifact volume (exits after)
- `mlflow` — tracking server at `http://localhost:5000`
- `airflow` — scheduler + webserver at `http://localhost:8080`
- `serving` — FastAPI endpoint at `http://localhost:8000`

Default Airflow credentials: `admin` / `admin`

**3. Run the pipeline:**

Trigger the `churn_pipeline` DAG from the Airflow UI (`http://localhost:8080`). Each trigger processes the next batch automatically. Trigger it three times to complete all runs.

**4. Test the serving endpoint:**

Navigate to `http://localhost:8000/docs` and use the `/predict` endpoint with a feature dictionary. Example request body:

```json
{
  "features": {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 29,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 60.1,
    "TotalCharges": 1653.85
  }
}
```

**Note:** The serving container must be rebuilt after any changes to `app.py`:
```bash
docker-compose build serving && docker-compose up -d
```

### Resetting

To rerun from scratch, delete all runs from the `churn_pipeline` experiment and the `champion` model from the Model Registry in the MLflow UI (`http://localhost:5000`), then retrigger the DAG.


## License

MIT