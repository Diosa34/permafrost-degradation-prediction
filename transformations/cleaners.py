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


def normalize_identifier(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[()%]", "", regex=True)
    )


def preprocess_site_dataframe(site: pd.DataFrame) -> pd.DataFrame:
    site = normalize_columns(site)
    site = clean_site_column(site)
    site["site"] = normalize_identifier(site["site"])

    category_replacements = {
        "ecological_type": {
            "alpine__steppe": "alpine_steppe",
            "apine_meadow": "alpine_meadow",
        },
        "geomorphic_unit": {
            "vally": "valley",
        },
    }

    for col in ["region", "ecological_type", "geomorphic_unit", "soil_type"]:
        if col in site.columns:
            site[col] = normalize_identifier(site[col])
            replacements = category_replacements.get(col)
            if replacements:
                site[col] = site[col].replace(replacements)

    return site.drop_duplicates(subset=["site"], keep="first")


def preprocess_timeseries_dataframe(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = normalize_columns(df)
    df = clean_numeric(df)

    df_long = df.melt(id_vars=["year"], var_name="site", value_name=value_name)
    df_long["site"] = normalize_identifier(df_long["site"])

    median_value = df_long[value_name].median(skipna=True)
    if pd.notna(median_value):
        df_long[value_name] = df_long[value_name].fillna(median_value)

    return df_long
