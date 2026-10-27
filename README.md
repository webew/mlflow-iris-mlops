# mlflow-iris-mlops

Classification du jeu de données Iris (RandomForest, scikit-learn), avec le suivi des expériences et le registre de modèles MLflow, géré avec [uv](https://docs.astral.sh/uv/).

## Architecture

- **PostgreSQL** stocke les métadonnées MLflow : expériences, runs, paramètres, métriques et registre de modèles.
- **MLflow server** expose l'API et l'interface web, et sert les artefacts (modèles) depuis le volume Docker `mlflow-artifacts`.
- Le code client (`train`, `predict`) parle uniquement au serveur, via `MLFLOW_TRACKING_URI` défini dans `.env`.

## Installation

```bash
uv sync
cp .env.example .env      # puis changer POSTGRES_PASSWORD
docker compose up -d      # PostgreSQL + MLflow server
```

L'interface MLflow est disponible sur <http://localhost:5001>. Le port 5000 est évité car AirPlay l'occupe sur macOS.

## Utilisation

```bash
# Entraîner un modèle (paramètres, métriques et modèle logués dans MLflow)
uv run train --n-estimators 200 --max-depth 5

# Prédire avec la dernière version du modèle enregistré
uv run predict            # ou : uv run predict --version 1
```

Arrêter les services : `docker compose down`. Ajouter `-v` supprime aussi les données PostgreSQL et les artefacts.

## Développement

```bash
uv run pytest             # utilise une base SQLite temporaire, sans le serveur
uv run ruff check . && uv run ruff format .
```

## Structure

```text
docker-compose.yml           # PostgreSQL + MLflow server
docker/mlflow/Dockerfile     # image MLflow + pilote psycopg (installé avec uv)
src/mlflow_iris_mlops/
├── data.py      # chargement et split train/test
├── train.py     # entraînement + logging MLflow + enregistrement du modèle
└── predict.py   # inférence depuis le registre de modèles
tests/
└── test_train.py
```
