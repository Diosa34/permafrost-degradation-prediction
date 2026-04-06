from sqlalchemy import create_engine
import pandas as pd


def save_to_postgres(df: pd.DataFrame, uri: str, table: str):
    engine = create_engine(uri)

    df.to_sql(
        table,
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000
    )


def add_to_postgres(df: pd.DataFrame, uri: str, table: str):
    engine = create_engine(uri)

    df.to_sql(
        table,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000
    )
