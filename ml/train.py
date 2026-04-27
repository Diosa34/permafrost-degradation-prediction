import os
import joblib
import mlflow
import numpy as np
import pandas as pd

from db.postgres import load_from_postgres
from config import TABLE_NAME
from ml.dataset import prepare_dataset, split_data
from ml.models import get_model
from ml.evaluate import evaluate, plot_predictions, plot_feature_importance


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("permafrost-alt-prediction")


def train_experiment(model_name: str, params: dict):

    print("\n" + "=" * 80)
    print(f"TRAINING STARTED | MODEL: {model_name}")
    print("=" * 80)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    df = load_from_postgres(TABLE_NAME)

    df = prepare_dataset(df)

    print("AFTER PREPARE_DATASET")
    print("Shape:", df.shape)
    print("Target stats:")
    print(df["target"].describe())

    X_train, X_test, y_train, y_test = split_data(df)

    baseline_pred = np.full_like(y_test, y_train.mean())

    baseline_mae = np.mean(np.abs(y_test - baseline_pred))
    print("BASELINE MODEL (mean predictor)")
    print(f"MAE baseline: {baseline_mae:.4f}")

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model = get_model(model_name, params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        metrics = evaluate(y_test, preds)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        os.makedirs("artifacts", exist_ok=True)

        pred_path = "artifacts/predictions.png"
        imp_path = "artifacts/importance.png"

        plot_predictions(y_test, preds, pred_path)
        mlflow.log_artifact(pred_path)

        plot_feature_importance(model, X_train.columns, imp_path)
        mlflow.log_artifact(imp_path)

        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_train.columns)
            print("\nFEATURE IMPORTANCE")
            print(importances.sort_values(ascending=False))

        os.makedirs("models", exist_ok=True)
        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="permafrost_alt_model",
        )

        print("\n COMPLETED")
        print("=" * 80)

        return metrics
