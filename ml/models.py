from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression


def get_model(model_name: str, params: dict):
    if model_name == "rf":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )

    elif model_name == "gboost":
        return GradientBoostingRegressor(
            random_state=42,
            **params
        )

    elif model_name == "linear":
        return LinearRegression(**params)

    elif model_name == "extra_trees":
        return ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")