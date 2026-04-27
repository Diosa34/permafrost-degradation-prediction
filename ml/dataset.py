import pandas as pd


def prepare_dataset(df: pd.DataFrame):

    if "site_code" not in df.columns:
        raise KeyError("Missing required column: site_code")

    if "year" not in df.columns:
        raise KeyError("Missing required column: year")

    if "alt" not in df.columns:
        raise KeyError("Missing required column: alt")

    df = df.sort_values(["site_code", "year"])

    # Panel forecasting (по сайтам):
    # Превращаем временной ряд в supervised-таблицу:
    df["target"] = df.groupby("site_code")["alt"].shift(-1)
    df["target_year"] = df["year"] + 1

    # последняя запись в каждом site не имеет таргета (t+1)
    df = df.dropna(subset=["target"]).copy()

    return df


def split_data(df: pd.DataFrame):

    # Time-based split по ГОДУ ТАРГЕТА.
    # обучаемся на 2001–2015 и предсказываем 2016–2020.
    if "target_year" not in df.columns:
        raise KeyError("Missing required column: target_year (call prepare_dataset first)")

    train = df[df["target_year"] <= 2015]
    test = df[(df["target_year"] >= 2016) & (df["target_year"] <= 2020)]

    # не используем ALT_t и alt_lag* как признаки.
    drop_cols = ["target", "alt", "alt_lag1", "alt_lag2", "target_year"]
    X_train = train.drop(columns=drop_cols, errors="ignore")
    y_train = train["target"]

    X_test = test.drop(columns=drop_cols, errors="ignore")
    y_test = test["target"]

    return X_train, X_test, y_train, y_test