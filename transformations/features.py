import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression


def feature_correlations(df, threshold=0.9):
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) < 2:
        return []
    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_pairs = [(col1, col2, upper.loc[col1, col2])
                        for col1 in upper.columns
                        for col2 in upper.index
                        if pd.notna(upper.loc[col1, col2]) and upper.loc[col1, col2] > threshold]
    return correlated_pairs


def feature_importance(df, target_col, method="random_forest", random_state=42):
    if target_col not in df.columns:
        return pd.Series(dtype=float)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols = X.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return pd.Series(dtype=float)
    X_numeric = X[numeric_cols]

    valid_rows = y.notna()
    X_numeric = X_numeric.loc[valid_rows]
    y = y.loc[valid_rows]
    if X_numeric.empty:
        return pd.Series(dtype=float)

    if method == "random_forest":
        model = RandomForestRegressor(random_state=random_state)
        model.fit(X_numeric, y)
        importance = pd.Series(model.feature_importances_, index=numeric_cols)
    elif method == "mutual_info":
        mi = mutual_info_regression(X_numeric, y, random_state=random_state)
        importance = pd.Series(mi, index=numeric_cols)

    return importance.sort_values(ascending=False)


def missing_values_analysis(df, time_col="date", group_col="site", threshold=0.3):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return df.copy(), pd.DataFrame(columns=["feature", "overall_missing"])

    missing_summary = pd.DataFrame({
        "feature": numeric_cols,
        "overall_missing": df[numeric_cols].isna().mean().values
    })

    df_filled = df.copy()
    for col in numeric_cols:
        median_value = df_filled[col].median(skipna=True)
        if pd.notna(median_value):
            df_filled[col] = df_filled[col].fillna(median_value)

    return df_filled, missing_summary


def detect_outliers(df, group_col="site", threshold=0.05):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return []

    outlier_features = []

    for col in numeric_cols:
        if group_col in df.columns:
            outlier_flag = df.groupby(group_col)[col].apply(
                lambda x: ((x < (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)))) |
                           (x > (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))).mean()
            )
            has_many_outliers = (outlier_flag > threshold).any()
        else:
            series = df[col]
            iqr = series.quantile(0.75) - series.quantile(0.25)
            lower = series.quantile(0.25) - 1.5 * iqr
            upper = series.quantile(0.75) + 1.5 * iqr
            has_many_outliers = ((series < lower) | (series > upper)).mean() > threshold

        if has_many_outliers:
            outlier_features.append(col)

    return outlier_features


def suggest_features_to_drop(df, target_col, corr_threshold=0.9, missing_threshold=0.3,
                             importance_threshold=0.01, outlier_threshold=0.05):
    df_for_analysis, missing_summary = missing_values_analysis(df, threshold=missing_threshold)

    correlated_pairs = feature_correlations(df, threshold=corr_threshold)
    corr_drop_features = set([pair[1] for pair in correlated_pairs])

    importance = feature_importance(df_for_analysis, target_col)
    low_importance_features = set(importance[importance < importance_threshold].index)

    outlier_drop_features = detect_outliers(df_for_analysis, threshold=outlier_threshold)

    suggested_drop = (corr_drop_features
                      .union(low_importance_features)
                      .union(outlier_drop_features)
                      )

    return list(suggested_drop), {
        "correlated_pairs": correlated_pairs,
        "importance": importance,
        "missing_summary": missing_summary,
        "outlier_features": outlier_drop_features
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["site_code", "year"]).copy()

    df["alt_lag1"] = df.groupby("site_code")["alt"].shift(1)
    df["alt_lag2"] = df.groupby("site_code")["alt"].shift(2)

    df["temp_offset"] = df["ttop"] - df["pt10m"]

    # лаги/роллинги
    for col in ["ttop", "pt10m", "temp_offset"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("site_code")[col].shift(1)
            df[f"{col}_lag2"] = df.groupby("site_code")[col].shift(2)
            # 3-летнее скользящее среднее по прошлым значениям (без утечки будущего)
            df[f"{col}_roll3"] = (
                df.groupby("site_code")[col]
                .shift(1)
                .rolling(window=3, min_periods=1)
                .mean()
            )

    df["year_idx"] = df["year"] - df["year"].min()

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["site_code", "year"]).copy()

    lag_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_lag2")]

    # только прямое заполнение прошлым (ffill).
    df[lag_cols] = df.groupby("site_code")[lag_cols].transform(lambda x: x.ffill())

    # Для первых лет в каждом site лаги могут оставаться NaN — заполняем текущими значениями
    if "alt_lag1" in df.columns and "alt" in df.columns:
        df["alt_lag1"] = df["alt_lag1"].fillna(df["alt"])
    if "alt_lag2" in df.columns:
        df["alt_lag2"] = df["alt_lag2"].fillna(df.get("alt_lag1", df.get("alt")))

    for base in ["ttop", "pt10m", "temp_offset"]:
        if f"{base}_lag1" in df.columns and base in df.columns:
            df[f"{base}_lag1"] = df[f"{base}_lag1"].fillna(df[base])
        if f"{base}_lag2" in df.columns:
            df[f"{base}_lag2"] = df[f"{base}_lag2"].fillna(df.get(f"{base}_lag1", df.get(base)))

    numeric_cols = df.select_dtypes(include=["number"]).columns
    non_lag_cols = [c for c in numeric_cols if c not in lag_cols]

    # медианное заполнение (устойчиво к выбросам)
    df[non_lag_cols] = df[non_lag_cols].fillna(df[non_lag_cols].median())

    df = df.dropna()

    return df


def encode_categorical_as_codes(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()

    if "site" in encoded.columns:
        encoded["site_code"] = pd.Categorical(encoded["site"]).codes
        encoded = encoded.drop(columns=["site"])

    categorical_cols = ["region", "ecological_type", "geomorphic_unit", "soil_type"]
    existing_cols = [col for col in categorical_cols if col in encoded.columns]

    for col in existing_cols:
        encoded[f"{col}_code"] = pd.Categorical(encoded[col]).codes

    encoded = encoded.drop(columns=existing_cols)

    return encoded
