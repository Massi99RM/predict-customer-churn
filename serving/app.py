import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class PredictRequest(BaseModel):
    # Accepts any feature dictionary — keys must match the training feature names.
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

    # Apply the same categorical encoding used during training in preprocess.py.
    # pd.Categorical().codes maps each string category to a consistent integer.
    for col in df.select_dtypes('object').columns:
        df[col] = pd.Categorical(df[col]).codes
    churn_probability = model.predict_proba(df)[:, 1][0]

    return {"churn_probability": churn_probability}