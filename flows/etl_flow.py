from prefect import flow, task

import config

from transformations.loaders import load_csv
from transformations.cleaners import clean_numeric, normalize_columns, clean_site_column
from transformations.transformers import merge_all
from transformations.features import build_features, handle_missing, suggest_features_to_drop

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
    site = normalize_columns(site)
    site = clean_site_column(site)

    site["site"] = (
        site["site"]
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[()%]", "", regex=True)
    )
    return site


@task
def preprocess_timeseries(df, value_name):
    df = normalize_columns(df)
    df = clean_numeric(df)

    df_long = df.melt(id_vars=["year"], var_name="site", value_name=value_name)

    df_long["site"] = df_long["site"].str.strip()

    return df_long


@task
def merge_data(alt, ttop, pt10, pt15, site):
    df = merge_all(alt, ttop, pt10, pt15, site)
    return df


@task
def feature_engineering(df):
    df = handle_missing(df)
    df = build_features(df)
    return df


@task
def eda(df):
    to_drop, diagnostics = suggest_features_to_drop(
        df, target_col="permafrost_depth",
        corr_threshold=0.9,
        missing_threshold=0.3,
        importance_threshold=0.01,
        outlier_threshold=0.05
    )

    print("Рекомендуемые признаки для удаления:", to_drop)
    print("Признаки с выбросами:", diagnostics["outlier_features"])


@task
def save_to_db(df):
    save_to_postgres(df, POSTGRES_URI, TABLE_NAME)


@task
def append_to_db(df):
    add_to_postgres(df, POSTGRES_URI, TABLE_NAME)


def load_and_transform_data(data_path: str):
    site, alt, ttop, pt10, pt15 = load_data(data_path)

    site = preprocess_site(site)
    alt = preprocess_timeseries(alt, "alt")
    ttop = preprocess_timeseries(ttop, "ttop")
    pt10 = preprocess_timeseries(pt10, "pt10m")
    pt15 = preprocess_timeseries(pt15, "pt15m")

    df = merge_data(alt, ttop, pt10, pt15, site)

    eda(df)
    df = feature_engineering(df)
    return df


@flow(name="Permafrost ETL initial Pipeline")
def initial_pipeline():
    df = load_and_transform_data(INITIAL_DATA_PATH)

    save_to_db(df)


@flow(name="Permafrost ETL append Pipeline")
def append_pipeline():
    df = load_and_transform_data(APPEND_DATA_PATH)

    append_to_db(df)


if __name__ == "__main__":
    # Однократно загружаем исторические данные при старте приложения.
    initial_pipeline()
    # Затем запускаем часовой cron для инкрементальной дозагрузки.
    append_pipeline.serve(
        name="permafrost-etl-append-hourly",
        cron="*/20 * * * *",
    )
