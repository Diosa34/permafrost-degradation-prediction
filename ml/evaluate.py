import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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