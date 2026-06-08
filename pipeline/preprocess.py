import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(df: pd.DataFrame) -> tuple:
    """
    Encode features and split into train/validation sets.

    Args:
        df: raw DataFrame from ingest_data.py

    Returns:
        X_train, X_val, y_train, y_val
    """

    X = df.drop(['id', 'Churn'], axis=1)
    y = df['Churn'].map({'No': 0, 'Yes': 1})


    # Encode string columns as integer category codes.
    # pd.Categorical().codes assigns a consistent integer to each unique value.
    for col in X.select_dtypes('object').columns:
        X[col] = pd.Categorical(X[col]).codes

    # stratify=y preserves the class balance (22.5% churn) in both splits.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_val, y_train, y_val