"""Entraînement d'un classifieur Iris avec suivi MLflow."""

import argparse

import mlflow
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from mlflow_iris_mlops.data import load_data, split_data

EXPERIMENT_NAME = "iris-classification"
MODEL_NAME = "iris-classifier"


def train(n_estimators: int = 100, max_depth: int | None = None, random_state: int = 42) -> dict:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=random_state)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(5),
            # MLflow sérialise via skops : les arbres internes doivent être déclarés sûrs
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            registered_model_name=MODEL_NAME,
        )

        print(f"Run {run.info.run_id} — {metrics}")
        return metrics


def main() -> None:
    load_dotenv()  # MLFLOW_TRACKING_URI pointe vers le serveur MLflow
    parser = argparse.ArgumentParser(description="Entraîne un modèle Iris et le logue dans MLflow.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth, args.random_state)


if __name__ == "__main__":
    main()
