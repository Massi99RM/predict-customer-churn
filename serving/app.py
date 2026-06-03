import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class PredictRequest(BaseModel):
    features: dict


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    """
    Load the champion model from MLflow and return churn probability.

    Args:
        request: PredictRequest with a features dictionary

    Returns:
        dict with churn_probability
    """
    
    # Model is loaded on every request since it's a learning project.
    # In production, load once at startup using a lifespan event handler.
    model = mlflow.lightgbm.load_model("models:/champion/latest")
    
    df = pd.DataFrame([request.features])
    churn_probability = model.predict_proba(df)[:, 1][0]

    return {"churn_probability": churn_probability}