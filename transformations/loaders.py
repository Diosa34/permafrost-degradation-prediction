import pandas as pd


def load_csv(path: str, sep=";") -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=sep,
        engine="python"
    )

    # полностью пустые строки
    df = df.dropna(how="all")

    # пустые колонки
    df = df.dropna(axis=1, how="all")

    # пробелы в строках
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    return df
