from ml.train import train_experiment


def run_all():
    experiments = [

        {
            "model": "rf",
            "params": {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 8,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
            },
        },

        {
            "model": "rf",
            "params": {
                "n_estimators": 300,
                "max_depth": 6,
                "min_samples_split": 8,
                "min_samples_leaf": 6,
                "max_features": "sqrt",
            },
        },

        {
            "model": "gboost",
            "params": {
                "n_estimators": 420,
                "max_depth": 2,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "min_samples_leaf": 3,
            },
        },
        {
            "model": "extra_trees",
            "params": {
                "n_estimators": 120,
                "max_depth": 7,
                "min_samples_split": 8,
                "min_samples_leaf": 4,
                "max_features": "sqrt",
            },
        },
    ]

    for exp in experiments:
        print(f"Running {exp}")

        train_experiment(
            exp["model"],
            exp["params"],
        )


if __name__ == "__main__":
    run_all()
