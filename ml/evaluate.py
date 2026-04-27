import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
import numpy as np


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def plot_predictions(y_true, y_pred, path):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted")
    plt.savefig(path)
    plt.close()


def plot_feature_importance(model, feature_names, path):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances, y=feature_names)
        plt.title("Feature Importance")
        plt.savefig(path)
        plt.close()


def plot_validation_curve_n_estimators(base_model, X_train, y_train, X_test, y_test, n_estimators_grid, path):
    train_mae_scores = []
    test_mae_scores = []

    for n_estimators in n_estimators_grid:
        candidate_model = clone(base_model)
        candidate_model.set_params(n_estimators=n_estimators)
        candidate_model.fit(X_train, y_train)

        train_pred = candidate_model.predict(X_train)
        test_pred = candidate_model.predict(X_test)

        train_mae_scores.append(mean_absolute_error(y_train, train_pred))
        test_mae_scores.append(mean_absolute_error(y_test, test_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(n_estimators_grid, train_mae_scores, marker="o", label="Train MAE")
    plt.plot(n_estimators_grid, test_mae_scores, marker="o", label="Test MAE")
    plt.xlabel("n_estimators")
    plt.ylabel("MAE")
    plt.title("Validation Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_residual_distribution(y_true, y_pred, path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, bins=25, kde=True)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.savefig(path)
    plt.close()