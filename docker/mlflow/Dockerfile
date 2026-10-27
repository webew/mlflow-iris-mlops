FROM ghcr.io/mlflow/mlflow:v3.17.2

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /usr/local/bin/uv

# Pilote PostgreSQL pour le backend store
RUN uv pip install --system --no-cache "psycopg[binary]"
