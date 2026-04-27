import os
import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from db.postgres import load_from_postgres
from config import TABLE_NAME
from ml.dataset import split_data
from ml.models import get_model
from ml.evaluate import (
    evaluate,
    plot_predictions,
    plot_feature_importance,
    plot_validation_curve_n_estimators,
    plot_residual_distribution,
)


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("permafrost-alt-prediction")


def train_experiment(model_name: str, params: dict):
    _print_run_header(model_name)
    X_train, X_test, y_train, y_test = _load_split_data()
    baseline_mae = _evaluate_baseline(X_train, y_train, X_test, y_test)

    with mlflow.start_run(run_name=model_name):
        _log_params(model_name, params)

        model = get_model(model_name, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = evaluate(y_test, preds)
        metrics["baseline_mae"] = baseline_mae
        _log_metrics(metrics)

        _save_common_artifacts(model_name, model, X_train, y_test, preds)
        _save_model_specific_artifacts(model_name, model, X_train, y_train, X_test, y_test, preds)
        _save_trained_model(model_name, model)

        print("\n COMPLETED")
        print("=" * 80)
        return metrics


def _print_run_header(model_name: str):
    print("\n" + "=" * 80)
    print(f"TRAINING STARTED | MODEL: {model_name}")
    print("=" * 80)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


def _load_split_data():
    df = load_from_postgres(TABLE_NAME)
    print("AFTER LOAD DATASET")
    print("Shape:", df.shape)
    print("Target stats:")
    print(df["target"].describe())
    return split_data(df)


def _evaluate_baseline(X_train, y_train, X_test, y_test):
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    print("BASELINE MODEL (Linear Regression)")
    print(f"MAE baseline: {baseline_mae:.4f}")
    return baseline_mae


def _log_params(model_name: str, params: dict):
    mlflow.log_param("model", model_name)
    for key, value in params.items():
        mlflow.log_param(key, value)


def _log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def _save_common_artifacts(model_name, model, X_train, y_test, preds):
    os.makedirs("artifacts", exist_ok=True)

    pred_path = f"artifacts/predictions_{model_name}.png"
    imp_path = f"artifacts/importance_{model_name}.png"

    plot_predictions(y_test, preds, pred_path)
    mlflow.log_artifact(pred_path)

    plot_feature_importance(model, X_train.columns, imp_path)
    if os.path.exists(imp_path):
        mlflow.log_artifact(imp_path)

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        sorted_importances = importances.sort_values(ascending=False)
        print("\nFEATURE IMPORTANCE")
        print(sorted_importances)

        importance_csv_path = f"artifacts/feature_importance_{model_name}.csv"
        sorted_importances.to_csv(importance_csv_path, header=["importance"])
        mlflow.log_artifact(importance_csv_path)


def _save_model_specific_artifacts(model_name, model, X_train, y_train, X_test, y_test, preds):
    if model_name in {"rf", "gboost"}:
        validation_curve_path = f"artifacts/validation_curve_{model_name}.png"
        n_estimators_grid = [20, 50, 100, 150, 200, 300]
        plot_validation_curve_n_estimators(
            base_model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_estimators_grid=n_estimators_grid,
            path=validation_curve_path,
        )
        mlflow.log_artifact(validation_curve_path)

    if model_name == "extra_trees":
        residuals_path = "artifacts/residual_distribution_extra_trees.png"
        plot_residual_distribution(y_true=y_test, y_pred=preds, path=residuals_path)
        mlflow.log_artifact(residuals_path)


def _save_trained_model(model_name: str, model):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="permafrost_alt_model",
    )
