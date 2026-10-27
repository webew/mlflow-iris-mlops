import mlflow

from mlflow_iris_mlops.data import load_data, split_data
from mlflow_iris_mlops.train import train


def test_split_is_stratified():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 120 and len(X_test) == 30
    assert set(y_test.value_counts()) == {10}


def test_train_logs_run(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    metrics = train(n_estimators=10)
    assert metrics["accuracy"] > 0.85
