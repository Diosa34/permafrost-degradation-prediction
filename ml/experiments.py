from ml.train import train_experiment


def run_all():
    experiments = [

        {
            "model": "rf",
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
            },
        },

        {
            "model": "rf",
            "params": {
                "n_estimators": 50,
                "max_depth": 20,
            },
        },

        {
            "model": "gboost",
            "params": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
        },
        {
            "model": "extra_trees",
            "params": {
                "n_estimators": 200,
                "max_depth": 20,
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
