from datetime import datetime
import mlflow
from airflow import DAG
from airflow.operators.python import PythonOperator # type: ignore

# Adds the project root to the Python path
# so that imports from pipeline/ work inside the DAG.
import sys
sys.path.append('/opt/airflow')

from pipeline.ingest_data import load_batch
from pipeline.preprocess import preprocess
from pipeline.train import train
from pipeline.evaluate import evaluate


def get_next_batch_number() -> int:
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("churn_pipeline")

    # Experiment doesn't exist yet, start from batch 1.
    if experiment is None:
        return 1
    
    # Only count runs that completed successfully — failed runs are excluded
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )
    next_batch = len(runs) + 1
    if next_batch > 3:
        raise ValueError("All 3 batches have already been processed.")
    return next_batch


def run_pipeline():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("churn_pipeline")


    # Batch number is determined automatically from MLflow run history
    batch_number = get_next_batch_number()
    df = load_batch(batch_number)
    X_train, X_val, y_train, y_val = preprocess(df)

    with mlflow.start_run():
        mlflow.log_param("batch_number", batch_number)
        model = train(X_train, y_train)
        evaluate(model, X_val, y_val)


with DAG(
    dag_id="churn_pipeline",
    start_date=datetime(2026, 1, 1),
    # Triggered manually from the Airflow UI, once per batch.
    schedule_interval=None,
    catchup=False,
) as dag:

    task = PythonOperator(
        task_id="run_pipeline",
        python_callable=run_pipeline,
    )