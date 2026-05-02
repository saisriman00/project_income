import pandas as pd
import joblib
import os

from models.model_registry import get_models


def train_models_pipeline():
    os.makedirs("artifacts/models", exist_ok=True)

    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    models = get_models()

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, f"artifacts/models/{name}.pkl")
