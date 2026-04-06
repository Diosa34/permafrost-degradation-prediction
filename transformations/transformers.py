import pandas as pd


def melt_timeseries(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    df = df.rename(columns={"year": "year"})
    df_long = df.melt(id_vars=["year"], var_name="site", value_name=value_name)
    df_long["site"] = df_long["site"].str.strip()
    return df_long


def merge_all(alt, ttop, pt10, pt15, site):
    df = alt.merge(ttop, on=["site", "year"], how="left") \
            .merge(pt10, on=["site", "year"], how="left") \
            .merge(pt15, on=["site", "year"], how="left") \
            .merge(site, on="site", how="left")

    return df


def encode_categorical_as_codes(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = ["region", "ecological_type", "geomorphic_unit", "soil_type"]
    existing_cols = [col for col in categorical_cols if col in df.columns]
    if not existing_cols:
        return df

    encoded = df.copy()
    for col in existing_cols:
        encoded[f"{col}_code"] = pd.Categorical(encoded[col]).codes

    return encoded.drop(columns=existing_cols)
