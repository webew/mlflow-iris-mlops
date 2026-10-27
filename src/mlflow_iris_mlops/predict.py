"""Prédiction à partir d'un modèle enregistré dans le registre MLflow."""

import argparse

import mlflow
import pandas as pd
from dotenv import load_dotenv

from mlflow_iris_mlops.data import load_data
from mlflow_iris_mlops.train import MODEL_NAME

CLASSES = ["setosa", "versicolor", "virginica"]


def predict(features: pd.DataFrame, version: str = "latest") -> list[str]:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    return [CLASSES[int(p)] for p in model.predict(features)]


def main() -> None:
    load_dotenv()  # MLFLOW_TRACKING_URI pointe vers le serveur MLflow
    parser = argparse.ArgumentParser(description="Prédit l'espèce d'iris (modèle enregistré).")
    parser.add_argument("--version", default="latest")
    args = parser.parse_args()

    X, _ = load_data()
    sample = X.sample(5, random_state=0)
    for row, label in zip(sample.to_dict("records"), predict(sample, args.version), strict=True):
        print(f"{row} -> {label}")


if __name__ == "__main__":
    main()
