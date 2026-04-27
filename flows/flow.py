from prefect import flow, task

import config
from ml.experiments import run_all

from transformations.loaders import load_csv
from transformations.cleaners import preprocess_site_dataframe, preprocess_timeseries_dataframe
from transformations.transformers import merge_all
from transformations.features import build_features, handle_missing, suggest_features_to_drop, \
    encode_categorical_as_codes
from ml.dataset import prepare_dataset

from db.postgres import add_to_postgres, save_to_postgres

POSTGRES_URI = config.POSTGRES_URI
TABLE_NAME = config.TABLE_NAME
INITIAL_DATA_PATH = getattr(config, "INITIAL_DATA_PATH", getattr(config, "DATA_PATH", None))
APPEND_DATA_PATH = getattr(config, "APPEND_DATA_PATH", getattr(config, "DATA_PATH", None))


@task(retries=3, retry_delay_seconds=5)
def load_data(data_path: str):
    site = load_csv(data_path + "SITE.csv")
    alt = load_csv(data_path + "ALT.csv")
    ttop = load_csv(data_path + "TTOP.csv")
    pt10 = load_csv(data_path + "PT10m.csv")
    pt15 = load_csv(data_path + "PT15m.csv")

    return site, alt, ttop, pt10, pt15


@task
def preprocess_site(site):
    return preprocess_site_dataframe(site)


@task
def preprocess_timeseries(df, value_name):
    return preprocess_timeseries_dataframe(df, value_name)


@task
def merge_data(alt, ttop, pt10, pt15, site):
    df = merge_all(alt, ttop, pt10, pt15, site)
    return df


@task
def encode_categorical_features(df):
    return encode_categorical_as_codes(df)


@task
def feature_engineering(df):
    df = build_features(df)
    df = handle_missing(df)
    df = prepare_dataset(df)
    return df


@task
def drop_selected_features(df):
    return df.drop(columns=["pt15m", "magtoc", "geomorphic_unit_code"], errors="ignore")


@task
def combine_coordinates(df):
    if "latitude" in df.columns and "longitude" in df.columns:
        # географический индекс на основе декартовой нормы координат.
        df["geo_coord_norm"] = (df["latitude"] ** 2 + df["longitude"] ** 2) ** 0.5
        df = df.drop(columns=["latitude", "longitude"])
    return df


@task
def eda(df):
    to_drop, diagnostics = suggest_features_to_drop(
        df, target_col="alt",
        corr_threshold=0.9,
        missing_threshold=0.3,
        importance_threshold=0.01,
        outlier_threshold=0.05
    )

    print("Корреляции:", diagnostics["correlated_pairs"])
    print("Значимость:", diagnostics["importance"])


@task
def save_to_db(df):
    save_to_postgres(df, TABLE_NAME)


@task
def append_to_db(df):
    add_to_postgres(df, TABLE_NAME)


def data_preprocessing(data_path: str):
    site, alt, ttop, pt10, pt15 = load_data(data_path)

    site = preprocess_site(site)
    alt = preprocess_timeseries(alt, "alt")
    ttop = preprocess_timeseries(ttop, "ttop")
    pt10 = preprocess_timeseries(pt10, "pt10m")
    pt15 = preprocess_timeseries(pt15, "pt15m")

    df = merge_data(alt, ttop, pt10, pt15, site)
    df = encode_categorical_features(df)

    return df


def initial_build_features(df):
    eda(df)
    df = drop_selected_features(df)
    df = combine_coordinates(df)
    eda(df)
    df = feature_engineering(df)
    return df


def incremental_build_features(df):
    df = drop_selected_features(df)
    df = combine_coordinates(df)
    df = feature_engineering(df)
    return df


@flow(name="Permafrost ETL initial Pipeline")
def initial_pipeline():
    df = data_preprocessing(INITIAL_DATA_PATH)
    df = initial_build_features(df)
    save_to_db(df)


@flow(name="Permafrost ETL append Pipeline")
def append_pipeline():
    df = data_preprocessing(APPEND_DATA_PATH)
    df = incremental_build_features(df)
    append_to_db(df)


@flow
def train_pipeline():
    run_all()


if __name__ == "__main__":
    # Однократно загружаем исторические данные при старте приложения. Если таблицы не пуста, то содержимое заменяется.
    initial_pipeline()
    # Запускаем cron для инкрементальной дозагрузки (по задумке раз в год).
    # append_pipeline.serve(
    #     name="permafrost-etl-append-hourly",
    #     cron="*/5 * * * *",
    # )

    train_pipeline()
