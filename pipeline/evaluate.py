import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.metrics import roc_auc_score


def evaluate(model: lgb.LGBMClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """
    Evaluate model on validation set, compare to current champion,
    and promote if better.

    Args:
        model: trained LGBMClassifier from train.py
        X_val: validation features
        y_val: validation target

    Returns:
        val_auc: ROC-AUC score on validation set
    """
    
    y_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_proba)

    mlflow.log_metric("val_auc", val_auc)

    client = mlflow.MlflowClient()
    runs = client.search_runs(experiment_ids=["0"], order_by=["metrics.val_auc DESC"])
    best_auc = runs[0].data.metrics["val_auc"] if runs else 0.0
    
    run_id = mlflow.active_run().info.run_id
    if val_auc > best_auc: 
        mlflow.register_model(f"runs:/{run_id}/model", "champion")

    return val_auc