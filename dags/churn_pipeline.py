from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
sys.path.append('/opt/airflow')

from pipeline.ingest_data import load_batch
from pipeline.preprocess import preprocess
from pipeline.train import train
from pipeline.evaluate import evaluate


def run_pipeline(batch_number: int):
    """
    Full pipeline run for a given batch number.

    Args:
        batch_number: integer 1, 2, or 3
    """
    
    df = load_batch(batch_number)
    X_train, X_val, y_train, y_val = preprocess(df)
    model = train(X_train, y_train)
    evaluate(model, X_val, y_val)


with DAG(
    dag_id="churn_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    task = PythonOperator(
        task_id="run_pipeline",
        python_callable=run_pipeline,
        op_kwargs={"batch_number": 1},
    )