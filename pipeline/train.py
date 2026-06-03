import pandas as pd
import lightgbm as lgb
import mlflow


def train(X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
    """
    Train a LightGBM classifier and log the run to MLflow.

    Args:
        X_train: training features
        y_train: training target

    Returns:
        Trained LGBMClassifier model

    """
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42
    }

    model = lgb.LGBMClassifier(**params)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.log_params(params)
        mlflow.lightgbm.log_model(model, "model")
    
    return model