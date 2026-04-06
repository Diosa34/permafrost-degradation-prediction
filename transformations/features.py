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
    else:
        raise ValueError("method should be 'random_forest' or 'mutual_info'")

    return importance.sort_values(ascending=False)


def missing_values_analysis(df, time_col="date", group_col="site", threshold=0.3):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return [], pd.DataFrame(columns=["feature", "overall_missing"])

    drop_features = []
    if group_col in df.columns:
        for col in numeric_cols:
            group_missing_ratios = df.groupby(group_col)[col].apply(lambda x: x.isna().mean())
            if (group_missing_ratios > threshold).any():
                drop_features.append(col)
    else:
        for col in numeric_cols:
            if df[col].isna().mean() > threshold:
                drop_features.append(col)

    missing_summary = pd.DataFrame({
        "feature": numeric_cols,
        "overall_missing": df[numeric_cols].isna().mean().values
    })
    return drop_features, missing_summary


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


# Рекомендации по удалению признаков с учетом пропусков, корреляции, значимости и выбросов
def suggest_features_to_drop(df, target_col, corr_threshold=0.9, missing_threshold=0.3,
                             importance_threshold=0.01, outlier_threshold=0.05):
    correlated_pairs = feature_correlations(df, threshold=corr_threshold)
    corr_drop_features = set([pair[1] for pair in correlated_pairs])

    importance = feature_importance(df, target_col)
    low_importance_features = set(importance[importance < importance_threshold].index)

    missing_drop_features, missing_summary = missing_values_analysis(df, threshold=missing_threshold)

    outlier_drop_features = detect_outliers(df, threshold=outlier_threshold)

    suggested_drop = (corr_drop_features
                      .union(low_importance_features)
                      .union(missing_drop_features)
                      .union(outlier_drop_features)
                      )

    return list(suggested_drop), {
        "correlated_pairs": correlated_pairs,
        "importance": importance,
        "missing_summary": missing_summary,
        "outlier_features": outlier_drop_features
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["site", "year"])

    # лаги
    df["alt_lag1"] = df.groupby("site")["alt"].shift(1)
    df["alt_lag2"] = df.groupby("site")["alt"].shift(2)

    # температурные признаки
    df["temp_gradient_10_15"] = df["pt10m"] - df["pt15m"]
    df["temp_offset"] = df["ttop"] - df["pt10m"]

    # тренд времени
    df["year_idx"] = df["year"] - df["year"].min()

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["site", "year"])
    df = df.infer_objects(copy=False)

    def process_group(group):
        numeric_cols = group.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            group[numeric_cols] = group[numeric_cols].interpolate(limit_direction="both")
        return group

    sites = df['site']

    df_grouped = df.drop(columns=['site']).groupby(sites, group_keys=False).apply(process_group)

    df_grouped['site'] = sites.values
    return df_grouped
