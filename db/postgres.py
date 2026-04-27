from sqlalchemy import create_engine
import pandas as pd

from config import POSTGRES_URI


def get_engine():
    return create_engine(POSTGRES_URI)


def save_to_postgres(df: pd.DataFrame, table: str):
    engine = get_engine()

    df.to_sql(
        table,
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000
    )


def add_to_postgres(df: pd.DataFrame, table: str):
    engine = get_engine()

    df.to_sql(
        table,
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000
    )


def load_from_postgres(table: str) -> pd.DataFrame:
    engine = get_engine()

    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, engine)

    return df
