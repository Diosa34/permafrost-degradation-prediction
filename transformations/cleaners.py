import pandas as pd


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace("####", pd.NA)

    def _to_numeric_ignore(col: pd.Series) -> pd.Series:
        converted = pd.to_numeric(col, errors="coerce")
        return converted.where(~converted.isna() | col.isna(), col)

    return df.apply(_to_numeric_ignore)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[()%]", "", regex=True)
    )
    return df


def clean_site_column(df: pd.DataFrame) -> pd.DataFrame:
    if "site" in df.columns:
        df["site"] = df["site"].astype(str).str.strip()
    return df
